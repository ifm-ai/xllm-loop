"""Answer normalization from the paper's evaluation runtime, copied unchanged.

em and f1 follow the SQuAD evaluation script; extract_answer, _strip_string and
_fix_sqrt are adapted from microsoft/ToRA and hendrycks/math (all MIT);
see THIRD_PARTY_NOTICES.md.
"""
import re
from collections import Counter
from typing import Callable, List


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

