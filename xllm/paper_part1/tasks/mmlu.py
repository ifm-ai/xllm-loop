from typing import List, Tuple, Dict, Optional, Type
from collections import defaultdict
import os
from pathlib import Path

from xllm.data.dataset_streamer.tokenizer import Tokenizer
from xllm.paper_part1.tasks.base import ChoiceTask, Example, BaseTask

# Abstract Algebra example
# {
#     "question": "Find the degree for the given field extension Q(sqrt(2), sqrt(3), sqrt(18)) over Q.",
#     "choices": {
#         "A": "0",
#         "B": "4",
#         "C": "2",
#         "D": "6"
#     },
#     "answer": "B"
# }


class ClassPropertyDescriptor(object):
    def __init__(self, fget, fset=None):
        self.fget = fget
        self.fset = fset

    def __get__(self, obj, klass=None):
        if klass is None:
            klass = type(obj)
        return self.fget.__get__(obj, klass)()

    def __set__(self, obj, value):
        if not self.fset:
            raise AttributeError("can't set attribute")
        type_ = type(obj)
        return self.fset.__get__(obj, type_)(value)

    def setter(self, func):
        if not isinstance(func, (classmethod, staticmethod)):
            func = classmethod(func)
        self.fset = func
        return self


def classproperty(func):
    if not isinstance(func, (classmethod, staticmethod)):
        func = classmethod(func)

    return ClassPropertyDescriptor(func)


class MMLUTask(ChoiceTask):
    dataset_dirs: List[str] = ["mmlu/"]
    fewshot_mode = "index"
    fewshot_index: List[int] = [0, 1, 2, 3, 4]
    fewshot_file: str = "dev.jsonl"
    eval_file: str = "test.jsonl"
    sep: str = "\n\n"
    max_text_len: int = 2048
    _nb_samples: Optional[int] = None
    task_dir_root: Optional[str] = None

    def __init__(self, tokenizer: Tokenizer, task_dir: str):
        super().__init__(tokenizer=tokenizer, task_dir=task_dir)
        if type(self).task_dir_root is None and type(self) is not MMLUTask:
            MMLUTask.task_dir_root = str(Path(task_dir).parent.parent)

    @classproperty
    def mmlu_tasks(cls) -> Dict[str, "MMLUTask"]:
        tasks = {
            k: v
            for k, v in BaseTask.tasks.items()
            if k.startswith("mmlu/") and v is not MMLUTask
        }
        return tasks  # type: ignore

    @classproperty
    def nb_samples(cls) -> int:
        if cls._nb_samples is None:
            if cls.dataset_dirs[0] == "mmlu/":
                cls._nb_samples = 0
                for task, task_cls in MMLUTask.mmlu_tasks.items():
                    if task != "mmlu/" and task.startswith("mmlu/"):
                        assert issubclass(task_cls, MMLUTask)
                        cls._nb_samples += task_cls.nb_samples  # type: ignore
            else:
                assert cls.task_dir_root is not None
                with open(
                    os.path.join(cls.task_dir_root, cls.dataset_dirs[0], "test.jsonl")
                ) as f:
                    cls._nb_samples = len([l for l in f])
        nb_samples = cls._nb_samples
        assert nb_samples is not None
        return nb_samples

    def context(self, example: Example) -> str:
        question = example["question"]
        choices = example["choices"]
        choice_str = "\n".join([f"{l}. {choices[l]}" for l in ["A", "B", "C", "D"]])
        return f"{question}\n{choice_str}\nAnswer:"

    def textify_example(self, example: Example) -> str:
        context = self.context(example)
        answer = example["answer"]
        return f"{context} {answer}"

    def get_text_completion(self, example: Example) -> List[Tuple[str, str]]:
        context = self.context(example)
        return [(f"{context} {l}", l) for l in ["A", "B", "C", "D"]]

    def label(self, example: Example) -> int:
        mapping = {"A": 0, "B": 1, "C": 2, "D": 3}
        return mapping[example["answer"]]


MMLU_TASKS_DOMAINS = {
    "abstract_algebra": "STEM",
    "anatomy": "STEM",
    "astronomy": "STEM",
    "business_ethics": "Other",
    "clinical_knowledge": "Other",
    "college_biology": "STEM",
    "college_chemistry": "STEM",
    "college_computer_science": "STEM",
    "college_mathematics": "STEM",
    "college_medicine": "Other",
    "college_physics": "STEM",
    "computer_security": "STEM",
    "conceptual_physics": "STEM",
    "econometrics": "Social Science",
    "electrical_engineering": "STEM",
    "elementary_mathematics": "STEM",
    "formal_logic": "Humanities",
    "global_facts": "Other",
    "high_school_biology": "STEM",
    "high_school_chemistry": "STEM",
    "high_school_computer_science": "STEM",
    "high_school_european_history": "Humanities",
    "high_school_geography": "Social Science",
    "high_school_government_and_politics": "Social Science",
    "high_school_macroeconomics": "Social Science",
    "high_school_mathematics": "STEM",
    "high_school_microeconomics": "Social Science",
    "high_school_physics": "STEM",
    "high_school_psychology": "Social Science",
    "high_school_statistics": "STEM",
    "high_school_us_history": "Humanities",
    "high_school_world_history": "Humanities",
    "human_aging": "Other",
    "human_sexuality": "Social Science",
    "international_law": "Humanities",
    "jurisprudence": "Humanities",
    "logical_fallacies": "Humanities",
    "machine_learning": "STEM",
    "management": "Other",
    "marketing": "Other",
    "medical_genetics": "Other",
    "miscellaneous": "Other",
    "moral_disputes": "Humanities",
    "moral_scenarios": "Humanities",
    "nutrition": "Other",
    "philosophy": "Humanities",
    "prehistory": "Humanities",
    "professional_accounting": "Other",
    "professional_law": "Humanities",
    "professional_medicine": "Other",
    "professional_psychology": "Social Science",
    "public_relations": "Social Science",
    "security_studies": "Social Science",
    "sociology": "Social Science",
    "us_foreign_policy": "Social Science",
    "virology": "Other",
    "world_religions": "Humanities",
}
MMLU_TASKS: Dict[str, Type[MMLUTask]] = {}
for task_name in MMLU_TASKS_DOMAINS:
    cap_task_name = "".join([s.capitalize() for s in task_name.split("_")])
    task_cls = type(
        cap_task_name,
        (MMLUTask,),
        {"dataset_dirs": [f"mmlu/{task_name}"]},
    )
    assert issubclass(task_cls, MMLUTask)
    MMLU_TASKS[task_name] = task_cls


def get_mmlu_scores(scores: Dict[str, float]) -> Dict[str, float]:
    metrics_ls_all: Dict[str, List[Tuple[int, float]]] = defaultdict(list)
    metrics_ls_domain: Dict[str, Dict[str, List[Tuple[int, float]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for k, v in scores.items():
        if k.split("/")[1] == "mmlu":
            task_name = k.split("/")[2]
            domain = MMLU_TASKS_DOMAINS[task_name]
            task_cls = MMLUTask.mmlu_tasks["mmlu/" + task_name]
            metrics_ls_all["/".join(k.split("/")[3:])].append((task_cls.nb_samples, v))
            metrics_ls_domain[domain]["/".join(k.split("/")[3:])].append(
                (task_cls.nb_samples, v)
            )
    assert all(len(ls) == len(MMLUTask.mmlu_tasks) for ls in metrics_ls_all.values())
    metrics: Dict[str, float] = {}
    for domain, metrics_ls in metrics_ls_domain.items():
        nb_samples = sum(
            [
                task_cls.nb_samples
                for task, task_cls in MMLUTask.mmlu_tasks.items()
                if MMLU_TASKS_DOMAINS[task[len("mmlu/") :]] == domain
            ]
        )
        for k, ls in metrics_ls.items():
            assert sum([nb_samples for nb_samples, _ in ls]) == nb_samples
            metrics[f"{domain}/macro_avg/{k}"] = sum([v for _, v in ls]) / len(ls)
            metrics[f"{domain}/micro_avg/{k}"] = (
                sum([nb_samples * v for nb_samples, v in ls]) / nb_samples
            )
    for k, ls in metrics_ls_all.items():
        nb_samples = MMLUTask.nb_samples
        assert sum([nb_samples for nb_samples, _ in ls]) == nb_samples
        metrics[f"macro_avg/{k}"] = sum([v for _, v in ls]) / len(ls)
        metrics[f"micro_avg/{k}"] = (
            sum([nb_samples * v for nb_samples, v in ls]) / nb_samples
        )
    return metrics
