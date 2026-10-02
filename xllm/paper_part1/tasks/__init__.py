"""Evaluation tasks of the Part 1 paper protocol.

The task modules (``arc``, ``bbh_cot``, ``drop``, ``gsm8k``, ``hellaswag``,
``human_eval``, ``math``, ``mbpp``, ``mmlu``, ``mmlu_pro`` and ``tqa``, on the
classes in ``base``) build prompts, extract answers and score items as the
paper code did. ``task_iterator`` turns a task's JSONL file into processed
items with the reader and padding helper in ``data``; ``utils`` holds the
scoring, answer-parsing, seeding and code-execution helpers of these modules.
``evalplus`` and ``evalplus_rescore`` rescore HumanEval and MBPP generations
with EvalPlus 0.3.1, and ``task_paths`` maps task names to data directories.
"""
