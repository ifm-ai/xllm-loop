import os
import contextlib
import faulthandler
import io
import shutil
import logging
import multiprocessing
import platform
import re
import signal
import string
import tempfile
from collections import Counter
from typing import List
from typing import Optional, Callable
from numpy.random import RandomState

logger = logging.getLogger()
# SQuAD evaluation script (MIT); see THIRD_PARTY_NOTICES.md.
"""
Normalization and score functions from SQuAD evaluation script
https://worksheets.codalab.org/rest/bundles/0x6b567e1cf2e041ec80d7098f031c5c9e/contents/blob/
"""


def remove_articles(text: str) -> str:
    return re.sub(r"\b(a|an|the)\b", " ", text)


def white_space_fix(text: str) -> str:
    return " ".join(text.split())


def remove_punc(text: str) -> str:
    exclude = set(string.punctuation)
    return "".join(ch for ch in text if ch not in exclude)


def normalize_answer(s: str) -> str:
    return white_space_fix(remove_articles(remove_punc(s.lower())))


def em(prediction: str, ground_truth: str, normalize_fn: Callable[[str], str]):
    return float(normalize_fn(prediction) == normalize_fn(ground_truth))


def f1(prediction: str, ground_truth: str, normalize_fn: Callable[[str], str]):
    prediction_tokens = normalize_fn(prediction).split()
    ground_truth_tokens = normalize_fn(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def f1_score(prediction: str, ground_truths: List[str], normalize_fn: Callable[[str], str]):
    return max([f1(prediction, gt, normalize_fn) for gt in ground_truths])


def exact_match_score(prediction: str, ground_truths: List[str], normalize_fn: Callable[[str], str]):
    return max([em(prediction, gt, normalize_fn) for gt in ground_truths])


def math_accuracy(prediction: str, ground_truth: List[str], normalize_fn: Callable[[str], str]):
    return float(is_equiv(normalize_fn(prediction), normalize_fn(ground_truth)))


def math_accuracy_score(prediction: str, ground_truths: List[str], normalize_fn: Callable[[str], str]):
    return max([math_accuracy(prediction, gt, normalize_fn) for gt in ground_truths])


def get_task_rng(
    base_seed: int = 42,
    data_parallel_rank: Optional[int] = None,
    different_seed_for_tasks_in_job_array: bool = False,
) -> RandomState:
    """
    Please use this RandomState for creating the data in your eval tasks
    Base seed needs to be the same within a model parallel group

    different_seed_for_mp_groups: bool -> If you want different GPUs to
        have different seed, please set it to True (it will make sure that it is
        the same within the same mp group, but different between mp groups)
    different_seed_for_tasks_in_job_array -> If you want the seed to be different
       between tasks in job array (to have different prompts sampled for majority
       voting for instance)

    """
    seed = [base_seed]
    if data_parallel_rank is not None:
        seed.append(data_parallel_rank)
    if different_seed_for_tasks_in_job_array:
        # will give different seed for jobs arrays
        seed.append(different_seed_when_job_array(0))

    logger.info(f"Creating rng with seed: {tuple(seed)}")

    return RandomState(tuple(seed))


def different_seed_when_job_array(seed: int) -> int:
    # when using array jobs, we want different seed for each job
    return seed + int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))


# is_valid_code through reliability_guard: adapted from openai/human-eval (MIT);
# see THIRD_PARTY_NOTICES.md.
def is_valid_code(
    code: str,
    test: str,
    timeout: Optional[float] = 1.0,
) -> bool:
    actual_code = "\n".join([l for l in code.split("\n")])
    actual_test = "\n".join([l for l in test.split("\n")])
    script = f"""{actual_code}

{actual_test}
    """

    def unsafe_execute():
        execute(script, result, timeout)

    manager = multiprocessing.Manager()
    result = manager.list()

    p = multiprocessing.Process(target=unsafe_execute)
    p.start()
    p.join(timeout=timeout + 1 if timeout is not None else None)
    if p.is_alive():
        p.kill()

    if not result:
        result.append("timed out")

    return result[0] == "passed"


def execute(script, result, timeout):
    with create_tempdir():
        # These system calls are needed when cleaning up tempdir.
        import os
        import shutil

        rmtree = shutil.rmtree
        rmdir = os.rmdir
        chdir = os.chdir

        # Disable functionalities that can make destructive changes to the test.
        reliability_guard()

        # Construct the check program and run it.
        check_program = script

        try:
            exec_globals = {}
            with swallow_io():
                with time_limit(timeout):
                    exec(check_program, exec_globals)
            result.append("passed")
        except TimeoutException:
            result.append("timed out")
        except BaseException as e:
            result.append(f"failed: {e}")
        # Needed for cleaning up.
        shutil.rmtree = rmtree
        os.rmdir = rmdir
        os.chdir = chdir


@contextlib.contextmanager
def time_limit(seconds: float):
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")

    signal.setitimer(signal.ITIMER_REAL, seconds)
    signal.signal(signal.SIGALRM, signal_handler)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


@contextlib.contextmanager
def swallow_io():
    stream = WriteOnlyStringIO()
    with contextlib.redirect_stdout(stream):
        with contextlib.redirect_stderr(stream):
            with redirect_stdin(stream):
                yield


@contextlib.contextmanager
def create_tempdir():
    with tempfile.TemporaryDirectory() as dirname:
        with chdir(dirname):
            yield dirname


class TimeoutException(Exception):
    pass


class WriteOnlyStringIO(io.StringIO):
    """StringIO that throws an exception when it's read from"""

    def read(self, *args, **kwargs):
        raise IOError

    def readline(self, *args, **kwargs):
        raise IOError

    def readlines(self, *args, **kwargs):
        raise IOError

    def readable(self, *args, **kwargs):
        """Returns True if the IO object can be read."""
        return False


class redirect_stdin(contextlib._RedirectStream):  # type: ignore
    _stream = "stdin"


@contextlib.contextmanager
def chdir(root):
    if root == ".":
        yield
        return
    cwd = os.getcwd()
    os.chdir(root)
    try:
        yield
    except BaseException as exc:
        raise exc
    finally:
        os.chdir(cwd)


def reliability_guard(maximum_memory_bytes: Optional[int] = None):
    """
    This disables various destructive functions and prevents the generated code
    from interfering with the test (e.g. fork bomb, killing other processes,
    removing filesystem files, etc.)
    WARNING
    This function is NOT a security sandbox. Untrusted code, including, model-
    generated code, should not be blindly executed outside of one. See the
    Codex paper for more information about OpenAI's code sandbox, and proceed
    with caution.
    """

    if maximum_memory_bytes is not None:
        import resource

        resource.setrlimit(
            resource.RLIMIT_AS, (maximum_memory_bytes, maximum_memory_bytes)
        )
        resource.setrlimit(
            resource.RLIMIT_DATA, (maximum_memory_bytes, maximum_memory_bytes)
        )
        if not platform.uname().system == "Darwin":
            resource.setrlimit(
                resource.RLIMIT_STACK, (maximum_memory_bytes, maximum_memory_bytes)
            )

    faulthandler.disable()

    import builtins

    builtins.exit = None  # type: ignore
    builtins.quit = None  # type: ignore

    import os

    os.environ["OMP_NUM_THREADS"] = "1"

    os.kill = None  # type: ignore
    os.system = None  # type: ignore
    os.putenv = None  # type: ignore
    os.remove = None  # type: ignore
    os.removedirs = None  # type: ignore
    os.rmdir = None  # type: ignore
    os.fchdir = None  # type: ignore
    os.setuid = None  # type: ignore
    os.fork = None  # type: ignore
    os.forkpty = None  # type: ignore
    os.killpg = None  # type: ignore
    os.rename = None  # type: ignore
    os.renames = None  # type: ignore
    os.truncate = None  # type: ignore
    os.replace = None  # type: ignore
    os.unlink = None  # type: ignore
    os.fchmod = None  # type: ignore
    os.fchown = None  # type: ignore
    os.chmod = None  # type: ignore
    os.chown = None  # type: ignore
    os.chroot = None  # type: ignore
    os.fchdir = None  # type: ignore
    os.lchflags = None  # type: ignore
    os.lchmod = None  # type: ignore
    os.lchown = None  # type: ignore
    os.getcwd = None  # type: ignore
    os.chdir = None  # type: ignore

    import shutil

    shutil.rmtree = None  # type: ignore
    shutil.move = None  # type: ignore
    shutil.chown = None  # type: ignore

    import subprocess

    subprocess.Popen = None  # type: ignore

    __builtins__["help"] = None  # type: ignore

    import sys

    sys.modules["ipdb"] = None  # type: ignore
    sys.modules["joblib"] = None  # type: ignore
    sys.modules["resource"] = None  # type: ignore
    sys.modules["psutil"] = None  # type: ignore
    sys.modules["tkinter"] = None  # type: ignore


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


# extract_answer, _strip_string, _fix_sqrt, _fix_fracs and _fix_a_slash_b: adapted from
# microsoft/ToRA and hendrycks/math (MIT); see THIRD_PARTY_NOTICES.md.
def extract_answer(pred_str):
    pred_str = str(pred_str).replace("\u00a0", " ")
    pred_str = _fix_latex_fracs(pred_str)

    if 'boxed' in pred_str:
        try:
            ans = pred_str.split('boxed')[-1]
            if ans.startswith("{"):
                stack = 1
                a = ''
                for c in ans[1:]:
                    if c == '{': stack += 1
                    elif c == '}': stack -= 1
                    if stack == 0: break
                    a += c
                return _strip_string(a)
        except:
            pass
    
    patterns = [
        r'Final Answer?\s*(.*)',
        r'final answer?\s*(.*)',
        r'he answer is\s*(.*)',
        r'result is\s*(.*)',
    ]
    candidate_text = pred_str
    for p in patterns:
        matches = re.findall(p, pred_str, re.IGNORECASE)
        if matches:
            candidate_text = matches[-1]
            break
            
    num_pattern = r"(-?\d+(?:\.\d+)?(?:/\d+(?:\.\d+)?)?%?)"
    
    clean_text = candidate_text.replace(",", "")
    numbers = re.findall(num_pattern, clean_text)
    
    if numbers:
        return _strip_string(numbers[-1])
    
    return ""


def _strip_string(string):
    string = str(string).strip()
    # linebreaks
    string = string.replace("\n", "")

    # right "."
    string = string.rstrip(".")

    # remove inverse spaces
    string = string.replace("\\!", "")
    string = string.replace("\\ ", "")

    # replace \\ with \
    string = string.replace("\\\\", "\\")
    string = string.replace("\\\\", "\\")

    # replace tfrac and dfrac with frac
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")

    # remove \left and \right
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")

    # Remove unit: miles, dollars if after is not none
    _string = re.sub(r"\\text{.*?}$", "", string).strip()
    if _string != "" and _string != string:
        # print("Warning: unit not removed: '{}' -> '{}'".format(string, _string))
        string = _string

    units = [
        "dollars", "dollar", "miles", "mile", "hours", "hour", 
        "gallons", "gallon", "units", "unit", "degrees", "degree", 
        "minutes", "minute", "seconds", "second", "lbs", "pounds",
        "GB", "MB", "KB", "TB", "kg", "lb", "m", "cm", "mm", "km", "meters", "meter",
        "gumballs", "roti"
    ]
    pattern = r"\s(" + "|".join(units) + r")\b"
    string = re.sub(pattern, "", string, flags=re.IGNORECASE)

    # Remove circ (degrees)
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")

    # remove dollar signs
    string = string.replace("\\$", "")
    string = string.replace("$", "")

    string = string.replace("\\text", "")
    string = string.replace("x\\in", "")

    # remove percentage
    string = string.replace("\\%", "")
    string = string.replace("%", "")

    # " 0." equivalent to " ." and "{0." equivalent to "{." Alternatively, add "0" if "." is the start of the string
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")

    # cdot
    string = string.replace("\\cdot", "")

    # inf
    string = string.replace("infinity", "\\infty")
    if "\\infty" not in string:
        string = string.replace("inf", "\\infty")
    string = string.replace("+\\inity", "\\infty")

    # and 
    string = string.replace("and", "")
    string = string.replace("\\mathbf", "")

    # use regex to remove \mbox{...}
    string = re.sub(r"\\mbox{.*?}", "", string)

    # quote
    string.replace("'", "")
    string.replace("\"", "")
    
    # i, j
    if "j" in string and "i" not in string:
        string = string.replace("j", "i")

    # replace a.000b where b is not number or b is end, with ab, use regex
    string = re.sub(r"(\d+)\.0+([^\d])", r"\1\2", string)
    string = re.sub(r"(\d+)\.0+$", r"\1", string)

    # if empty, return empty string
    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string

    # to consider: get rid of e.g. "k = " or "q = " at beginning
    if len(string.split("=")) == 2:
        if len(string.split("=")[0]) <= 2:
            string = string.split("=")[1]

    string = _fix_sqrt(string)
    string = string.replace(" ", "")

    string = _fix_latex_fracs(string)

    return string

def _fix_sqrt(string):
    string = re.sub(r"\\sqrt\s*([0-9a-zA-Z]+)", r"\\sqrt{\1}", string)
    return string

def _fix_latex_fracs(string):
    """
    \frac{1}{2} -> 1/2
    \frac12 -> 1/2
    """
    string = re.sub(r"\\frac\s*\{(\d+)\}\s*\{(\d+)\}", r"\1/\2", string)
    string = re.sub(r"\\frac\s*(\d)\s*(\d)", r"\1/\2", string)
    return string


def _fix_fracs(string):
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += "\\frac"
            if len(substr) > 0 and substr[0] == "{":
                new_str += substr
            else:
                try:
                    assert len(substr) >= 2
                except:
                    return string
                a = substr[0]
                b = substr[1]
                if b != "{":
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}{" + b + "}" + post_substr
                    else:
                        new_str += "{" + a + "}{" + b + "}"
                else:
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}" + b + post_substr
                    else:
                        new_str += "{" + a + "}" + b
    string = new_str
    return string


def _fix_a_slash_b(string):
    if len(string.split("/")) != 2:
        return string
    a = string.split("/")[0]
    b = string.split("/")[1]
    try:
        if "sqrt" not in a:
            a = int(a)
        if "sqrt" not in b:
            b = int(b)
        assert string == "{}/{}".format(a, b)
        new_string = "\\frac{" + str(a) + "}{" + str(b) + "}"
        return new_string
    except:
        return string

def parse_value(text):
    if not text:
        return None
        
    text = str(text).strip()
    
    if "/" in text:
        try:
            parts = text.split("/")
            if len(parts) == 2:
                return float(parts[0]) / float(parts[1])
        except:
            return None

    try:
        return float(text.replace(",", ""))
    except:
        return None
    
def is_equiv(pred, gold):

    if pred == gold:
        return True

    pred_val = parse_value(pred)
    gold_val = parse_value(gold)

    if pred_val is not None and gold_val is not None:
        return abs(pred_val - gold_val) < 1e-6

    return False
