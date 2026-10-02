"""Training FLOPs accounting for `DepthControlledHuginn`, as the paper's training code counted them.

Work is counted in multiply-accumulates ("units"): 2 FLOPs each, tripled for backpropagated
work (6x, or 8x with layerwise activation checkpointing, which recomputes the forward) and
doubled for forward-only recurrences outside the backward window (2x). The prelude, coda and
output head run once per token; each recurrence runs the recurrent blocks and the input
injection. The diagonal injection adds 6d elementwise units to its d x d projection, and OrthoInj
3d + 1 more for the projection. The exit norm of Huginn-Linear is not counted.

This differs from `Transformer.tflops_per_token`, which `LoopedTransformer` and `Huginn` use.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional


@dataclass(frozen=True)
class HuginnShape:
    """Dimensions that determine a dense Huginn model's FLOPs."""

    model_dim: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    ffn_hidden_dim: int
    vocab_size: int
    output_size: int
    prelude_layers: int
    recurrent_layers: int
    coda_layers: int
    input_injection: str = "diagonal"
    apply_rmsnorm: bool = True
    qknorm: bool = False
    apply_attn_gate: bool = False
    swiglu: bool = True
    layerwise_ckpt: bool = False
    prelude_norm: bool = False  # RMSNorm on the injected prelude output, counted once like a layer norm
    orthogonal_injection: bool = False  # OrthoInj projection, 3d + 1 units per injection

    # Every unit count below is an integer-valued float far below 2**53, so it is exact.

    def _layer_units(self, seq_len: int) -> tuple[float, int]:
        norm = self.model_dim * (4 if self.apply_rmsnorm else 8)
        if self.qknorm:
            norm += self.head_dim * (self.num_heads + self.num_kv_heads) * 2
        attention = self.model_dim * (self.num_heads + self.num_kv_heads) * self.head_dim * 2
        if self.apply_attn_gate:
            attention += self.model_dim * self.num_heads * self.head_dim
        attention += self.num_heads * self.head_dim * seq_len
        ffn = self.model_dim * self.ffn_hidden_dim * (3 if self.swiglu else 2)
        return float(norm + attention + ffn), norm

    def _injection_units(self) -> float:
        if self.input_injection == "none":
            return 0.0
        if self.input_injection == "linear":
            return float(2 * self.model_dim * self.model_dim)
        if self.input_injection == "diagonal":
            units = float(self.model_dim * self.model_dim + 6 * self.model_dim)
            return units + (3 * self.model_dim + 1) if self.orthogonal_injection else units
        raise ValueError(f"unknown input injection: {self.input_injection!r}")

    def _static_units(self, seq_len: int) -> float:
        layer, norm = self._layer_units(seq_len)
        head = float(self.model_dim + self.vocab_size + self.model_dim * self.output_size + norm)
        return (
            head
            + self.prelude_layers * layer
            + (norm if self.prelude_norm else 0.0)
            + self.coda_layers * layer
        )

    def _recurrence_units(self, seq_len: int, depth: int, backprop_depth: Optional[int]) -> tuple[float, float]:
        """Return (backpropagated, forward-only) units of `depth` recurrences; None backpropagates all."""
        layer, _ = self._layer_units(seq_len)
        span = self.recurrent_layers * layer
        injection = self._injection_units()
        steps_with_grad = depth if backprop_depth is None else min(depth, backprop_depth)
        steps_no_grad = depth - steps_with_grad
        train = steps_with_grad * span + steps_with_grad * injection
        forward_only = steps_no_grad * span + steps_no_grad * injection
        return train, forward_only

    def _to_tflops(self, train_units: float, forward_only_units: float) -> float:
        return (
            train_units * (8 if self.layerwise_ckpt else 6) / 10**12
            + forward_only_units * 2 / 10**12
        )

    def expected_tflops_per_token(
        self,
        depth_distribution: Iterable[tuple[int, float]],
        backprop_depth: Optional[int],
        seq_len: int,
    ) -> float:
        """Return training TFLOPs per token averaged over a (depth, probability) distribution."""
        train_units = self._static_units(seq_len)
        forward_only_units = 0.0
        for depth, probability in depth_distribution:
            train, forward_only = self._recurrence_units(seq_len, depth, backprop_depth)
            train_units += probability * train
            forward_only_units += probability * forward_only
        return self._to_tflops(train_units, forward_only_units)

    def depth_tflops_per_token(self, depth: int, backprop_depth: Optional[int], seq_len: int) -> float:
        """Return training TFLOPs per token of one forward at `depth` recurrences."""
        train, forward_only = self._recurrence_units(seq_len, depth, backprop_depth)
        return self._to_tflops(self._static_units(seq_len) + train, forward_only)
