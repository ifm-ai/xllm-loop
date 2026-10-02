"""Learned categorical depth prior for the Table 2 "Learned" rows of Part II.

The prior is a categorical distribution over recurrence depths 1..64, stored
as float64 logits on the CPU, outside the model and its optimizer. Every
training forward draws one depth from it; after each optimizer step it takes a
REINFORCE step toward depths whose inverse perplexity exp(-CE) beats an EMA
baseline; the cost is -exp(-CE) of the drawn forward. The advantage is rescaled
into cross-entropy units by the ratio of the EMA standard deviations of CE and of
the cost. A quadratic penalty holds the mean depth at the target, and an entropy
term (coefficient lambda_H) keeps the distribution spread. The arithmetic follows
the paper's training code operation by operation: given the same per-forward
losses, the prior takes the same steps.
"""

from __future__ import annotations

import copy
import math
from dataclasses import asdict, dataclass
from typing import Sequence

import torch
import torch.distributed as dist

from xllm.models.looped_depth import PLN_SEED_OFFSET


MAX_DEPTH = 64
SUPPORT = torch.arange(1, MAX_DEPTH + 1, dtype=torch.float64)


@dataclass(frozen=True)
class DepthPriorConfig:
    """Hyperparameters of the learned prior (defaults are the paper's)."""

    entropy_coefficient: float
    lr: float = 1e-3
    mean_penalty: float = 1.0
    target_mean: float = 5.0
    baseline_decay: float = 0.95
    burnin_steps: int = 32
    seed: int = PLN_SEED_OFFSET
    std_floor: float = 1e-12


class LearnedDepthPrior:
    """Categorical depth prior trained alongside the model (not an nn.Module).

    `baseline` and `cost_variance` are EMAs of the weighted mean and variance of the
    cost -exp(-CE) over optimizer steps; `raw_baseline` and `raw_variance` track the
    same for CE itself. These names are part of the saved state and the logged metrics.
    """

    def __init__(self, config: DepthPriorConfig, initial_pmf: Sequence[float]) -> None:
        p = torch.as_tensor(initial_pmf, dtype=torch.float64, device="cpu")
        if p.shape != (MAX_DEPTH,) or not torch.isfinite(p).all() or not (p > 0).all():
            raise ValueError(f"initial PMF must be a positive distribution over depths 1..{MAX_DEPTH}")
        if abs(p.sum().item() - 1) >= 1e-10 or abs(p.dot(SUPPORT).item() - config.target_mean) >= 1e-10:
            raise ValueError("initial PMF must sum to 1 and have the target mean")
        self.config = config
        self.logits = torch.nn.Parameter(p.log())
        self.optimizer = torch.optim.Adam([self.logits], lr=config.lr, betas=(0.9, 0.999), weight_decay=0.0)
        self.rng = torch.Generator().manual_seed(config.seed)
        self.baseline = 0.0
        self.raw_baseline = 0.0
        self.raw_variance = 0.0
        self.cost_variance = 0.0
        self.updates = 0
        self.pending_depths: list[int] = []
        self.pending_losses: list[float] = []
        self.pending_counts: list[float] = []

    def sample(self) -> int:
        """Draw the depth of the next training forward (identical on every rank)."""
        if len(self.pending_depths) != len(self.pending_losses):
            raise RuntimeError("the previous forward was not observed")
        depth = int(torch.multinomial(self.probabilities(), 1, generator=self.rng).item()) + 1
        self.pending_depths.append(depth)
        return depth

    def observe(self, token_losses: torch.Tensor, token_mask: torch.Tensor | None = None) -> None:
        """Record the global mean token loss of the forward that used the last draw."""
        if len(self.pending_depths) != len(self.pending_losses) + 1:
            raise RuntimeError("observe() must follow sample()")
        total = token_losses.detach().double().sum()
        count = (
            torch.tensor(token_losses.numel(), device=total.device, dtype=torch.float64)
            if token_mask is None
            else token_mask.detach().double().sum()
        )
        pair = torch.stack((total, count))
        if dist.is_initialized():
            dist.all_reduce(pair)
        if not torch.isfinite(pair).all() or pair[1] <= 0:
            raise ValueError("nonfinite loss or empty forward")
        self.pending_losses.append(float((pair[0] / pair[1]).cpu()))
        self.pending_counts.append(float(pair[1].cpu()))

    def finish_step(self) -> dict[str, float]:
        """Update the prior once per optimizer step and return its metrics."""
        if not self.pending_depths or len(self.pending_depths) != len(self.pending_losses):
            raise RuntimeError("finish_step() needs every drawn forward observed")
        raw = torch.tensor(self.pending_losses, dtype=torch.float64)
        costs = -(-raw).exp()
        counts = torch.tensor(self.pending_counts, dtype=torch.float64)
        weights = counts / counts.sum()
        scale = self._advantage_scale()
        loss = self._loss(costs, weights, scale)
        if self.updates >= self.config.burnin_steps:
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if not torch.isfinite(self.logits.grad).all():
                raise FloatingPointError("nonfinite depth-prior gradient")
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
        raw_observed, observed = self._update_baselines(raw, costs, weights)
        self.updates += 1

        metrics = self.metrics()
        metrics.update(
            signal_scale=scale,
            observed_raw_ce=raw_observed,
            observed_cost=observed,
            sampled_step_mean=sum(self.pending_depths) / len(self.pending_depths),
        )
        self.pending_depths.clear()
        self.pending_losses.clear()
        self.pending_counts.clear()
        after = self.probabilities()
        if not torch.isfinite(after).all() or not (after > 0).all():
            raise FloatingPointError("the depth prior lost full support")
        return metrics

    def _advantage_scale(self) -> float:
        """Rescale the cost advantage into CE units: EMA std of CE over EMA std of the cost."""
        if self.updates == 0:
            return 1.0
        floor = self.config.std_floor
        return max(math.sqrt(self.raw_variance), floor) / max(math.sqrt(self.cost_variance), floor)

    def _loss(self, costs: torch.Tensor, weights: torch.Tensor, scale: float) -> torch.Tensor:
        """REINFORCE term plus the mean-depth penalty and the entropy term."""
        config = self.config
        logp = self.logits.log_softmax(0)
        p = logp.exp()
        reinforce_term = (((costs - self.baseline) * scale).detach() * weights
                          * logp[torch.tensor(self.pending_depths) - 1]).sum()
        mean_penalty_term = config.mean_penalty * (p.dot(SUPPORT) - config.target_mean).square()
        entropy_term = config.entropy_coefficient * (p * logp).sum()
        return sum((reinforce_term, mean_penalty_term, entropy_term))

    def _update_baselines(self, raw: torch.Tensor, costs: torch.Tensor, weights: torch.Tensor) -> tuple[float, float]:
        """Fold this step's weighted CE and cost statistics into the EMAs; return the step's means."""
        raw_observed = float((raw * weights).sum())
        observed = float((costs * weights).sum())
        raw_within = float((weights * (raw - raw_observed).square()).sum())
        cost_within = float((weights * (costs - observed).square()).sum())
        decay = self.config.baseline_decay
        if self.updates == 0:
            self.raw_variance, self.cost_variance = raw_within, cost_within
            self.raw_baseline, self.baseline = raw_observed, observed
        else:
            self.raw_variance = (decay * self.raw_variance + (1 - decay) * raw_within
                                 + decay * (1 - decay) * (raw_observed - self.raw_baseline) ** 2)
            self.cost_variance = (decay * self.cost_variance + (1 - decay) * cost_within
                                  + decay * (1 - decay) * (observed - self.baseline) ** 2)
            self.raw_baseline = decay * self.raw_baseline + (1 - decay) * raw_observed
            self.baseline = decay * self.baseline + (1 - decay) * observed
        return raw_observed, observed

    def probabilities(self) -> torch.Tensor:
        return self.logits.detach().softmax(0)

    def metrics(self) -> dict[str, float]:
        p = self.probabilities()
        result = {f"p_{i + 1:02d}": float(v) for i, v in enumerate(p)}
        result.update(
            mean=float(p.dot(SUPPORT)),
            entropy=float(-(p * p.log()).sum()),
            baseline=self.baseline,
            raw_ce_baseline=self.raw_baseline,
            raw_ce_variance=self.raw_variance,
            cost_variance=self.cost_variance,
            updates=self.updates,
        )
        return result

    _STATE_KEYS = ("baseline", "raw_baseline", "raw_variance", "cost_variance", "updates")

    def state_dict(self) -> dict:
        if self.pending_depths or self.pending_losses:
            raise RuntimeError("checkpoint the prior only at an optimizer-step boundary")
        return copy.deepcopy(dict(
            version=1, config=asdict(self.config), logits=self.logits.detach(),
            optimizer=self.optimizer.state_dict(), rng=self.rng.get_state(),
            **{key: getattr(self, key) for key in self._STATE_KEYS},
        ))

    def load_state_dict(self, state: dict) -> None:
        if state["version"] != 1 or state["config"] != asdict(self.config):
            raise ValueError("depth-prior state does not match this configuration")
        with torch.no_grad():
            self.logits.copy_(state["logits"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.rng.set_state(state["rng"])
        for key in self._STATE_KEYS:
            setattr(self, key, copy.deepcopy(state[key]))
