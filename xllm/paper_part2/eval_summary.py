"""Assemble one model's Table 2-4 row from its raw evaluation outputs.

An evaluation writes, per model, the Wiki PPL sums `ppl.json`, sharded
choice-task records `tasks/<task>-shard<i>of<n>.jsonl`, and generation shards
`gen/<task>/shard<i>of<n>/` with `generations.jsonl` and `result.json`.
"""
from __future__ import annotations

import json
from pathlib import Path

from xllm.paper_part2.choice_tasks import load_choice_examples, reduce_task
from xllm.paper_part2.eval_data import WIKI_SEQUENCES
from xllm.paper_part2.eval_protocol import CHOICE_TASKS, GENERATION
from xllm.paper_part2.evaluator import ppl_metrics


def choice_scores(results_dir: str | Path, examples: dict[str, list[dict]]) -> dict[str, float]:
    """Reduce every task's option records to its primary score (percent), AVG and LAMBADA PPL."""
    tasks_dir = Path(results_dir) / "tasks"
    done = sorted(tasks_dir.glob("done-shard*.json"))
    if not done:
        raise ValueError(f"no choice-task results in {tasks_dir}; run the tasks stage first")
    shards = int(done[0].name.split("of")[1].split(".")[0])
    if len(done) != shards:
        raise ValueError(f"incomplete choice-task shards in {tasks_dir}")
    scores = {}
    for task in CHOICE_TASKS:
        records = [json.loads(line) for shard in range(shards)
                   for line in (tasks_dir / f"{task}-shard{shard}of{shards}.jsonl").read_text().splitlines()]
        metrics, _ = reduce_task(task, examples[task], records)
        scores[task] = 100 * metrics["primary_score"]
        if task == "lambada_openai":
            scores["lambada_ppl"] = metrics["perplexity"]
    scores["avg7"] = sum(scores[task] for task in CHOICE_TASKS) / len(CHOICE_TASKS)
    return scores


def generation_score(task: str, results_dir: str | Path) -> float:
    """Example-weighted table metric over a task's generation shards."""
    parts = sorted((Path(results_dir) / "gen" / task).glob("shard*/result.json"))
    shards = int(parts[0].parent.name.split("of")[1])
    if len(parts) != shards:
        raise ValueError(f"incomplete {task} shards in {results_dir}")
    results = [json.loads(path.read_text()) for path in parts]
    key = GENERATION[task].metric
    return sum(r["metrics"][key] * r["examples"] for r in results) / sum(r["examples"] for r in results)


def table_row(out: str | Path, data: dict) -> dict:
    """Wiki PPL, the choice-task scores and AVG, and the generation metrics present in `out`."""
    out = Path(out)
    path = out / "ppl.json"
    if not path.exists():
        raise ValueError(f"no PPL results in {out}; run the ppl stage first")
    wiki = json.loads(path.read_text())["wiki"]
    if wiki["sequences"] != WIKI_SEQUENCES:
        raise ValueError(f"incomplete PPL results in {path}")
    ppl = {"wiki": ppl_metrics(wiki)}
    row = {"wiki_ppl": ppl["wiki"]["continued"], "ppl": ppl}
    examples = {task: load_choice_examples(task, data["tasks"][task]) for task in CHOICE_TASKS}
    row.update(choice_scores(out, examples))
    for task in GENERATION:
        if list((out / "gen" / task).glob("shard*/result.json")):
            row[task] = generation_score(task, out)
    return row
