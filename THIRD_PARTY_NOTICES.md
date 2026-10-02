# Third-Party Notices

xLLM is released under the [Apache License 2.0](LICENSE). Some of the
evaluation code in `xllm/eval`, `xllm/paper_part1` and `xllm/paper_part2` is
copied or adapted from the projects listed below. Each entry names the source,
its license and copyright notice, and the files that use it; the license texts
follow. Other third-party code in the repository carries its notice in the
files themselves, for example the PyTorch-derived kernels in `xllm/csrc` and
the Transformer Engine code in `xllm/csrc/transformer_engine`. Python packages
installed as dependencies are not part of this repository and are covered by
their own licenses.

## Code

### OpenAI HumanEval

- Source: [openai/human-eval](https://github.com/openai/human-eval), `human_eval/execution.py`
- License: MIT, Copyright (c) OpenAI (https://openai.com)
- Used in: `xllm/eval/coding/execute.py` and `xllm/paper_part1/tasks/utils.py`,
  the sandboxed code-execution helpers (`time_limit`, `swallow_io`,
  `create_tempdir`, `reliability_guard` and related), adapted to run the
  code-generation tasks.

### SQuAD evaluation script

- Source: the official SQuAD evaluation script, distributed by
  [SQuAD-explorer](https://github.com/rajpurkar/SQuAD-explorer)
- License: MIT, Copyright (c) 2020 Pranav Rajpurkar
- Used in: `xllm/eval/utils.py`, `xllm/paper_part1/tasks/utils.py` and
  `xllm/paper_part2/_answer_helpers.py`, the answer normalization and the exact
  match and F1 scores, which take the normalizer as an argument.

### Hendrycks MATH

- Source: [hendrycks/math](https://github.com/hendrycks/math), `modeling/math_equivalence.py`
- License: MIT, Copyright (c) 2021 Dan Hendrycks
- Used in: `xllm/paper_part1/tasks/math.py` (`_fix_fracs`, `_fix_a_slash_b`,
  `_remove_right_units`, `_fix_sqrt`, `_normalise_result`), modified. The ToRA
  helpers below also build on it.

### Microsoft ToRA

- Source: [microsoft/ToRA](https://github.com/microsoft/ToRA), `src/utils/parser.py`
- License: MIT, Copyright (c) Microsoft Corporation.
- Used in: `xllm/eval/math/parse.py`, `xllm/paper_part1/tasks/utils.py` and
  `xllm/paper_part2/_answer_helpers.py` (`extract_answer`, `_strip_string`,
  `_fix_sqrt`, and in `xllm/paper_part1/tasks/utils.py` also `_fix_fracs` and
  `_fix_a_slash_b`), modified.

### Minerva answer normalization

- Source: A. Lewkowycz et al., "Solving Quantitative Reasoning Problems with
  Language Models", 2022, appendix D, <https://arxiv.org/abs/2206.14858>; the
  same code is distributed in EleutherAI lm-evaluation-harness,
  `lm_eval/tasks/minerva_math/utils.py`
- License: the paper is under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/);
  lm-evaluation-harness is under the MIT license listed below
- Used in: `xllm/paper_part1/tasks/math.py` (`SUBSTITUTIONS`,
  `REMOVED_EXPRESSIONS` and the body of `normalize_final_answer`). The answer
  extraction around them is new.

### EleutherAI lm-evaluation-harness

- Source: [EleutherAI/lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) v0.4.2
- License: MIT, Copyright (c) 2020 EleutherAI
- Used in: `xllm/eval/task/hellaswag.py` and `xllm/paper_part1/tasks/hellaswag.py`
  (`preprocess`, from `lm_eval/tasks/hellaswag/utils.py`).
  `xllm/paper_part1/tasks/hellaswag.py` and `xllm/paper_part2/choice_tasks.py`
  also follow its multiple-choice prompt templates and its context and
  continuation encoding.

### AllenNLP DROP evaluation

- Source: [allenai/allennlp](https://github.com/allenai/allennlp) v0.9.0, `allennlp/tools/drop_eval.py`
- License: Apache License 2.0 (text in [LICENSE](LICENSE))
- Used in: `xllm/paper_part2/drop_eval.py`. Changes: the shebang line is
  removed, and a module docstring and a SciPy-free `linear_sum_assignment`
  fallback are added.

## Prompt text

### Chain-of-thought exemplars for GSM8K

- Source: J. Wei et al., "Chain-of-Thought Prompting Elicits Reasoning in Large
  Language Models", 2022, <https://arxiv.org/abs/2201.11903>
- License: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
- Used in: `xllm/eval/task/gsm8k.py`, `xllm/paper_part1/tasks/gsm8k.py` and
  `xllm/paper_part2/generation_tasks.py`, the eight few-shot exemplars. Changes:
  each answer starts with "Let's think step by step." and ends with the final
  answer as `#### N`.

## License texts

### MIT License

Applies to the code above under these notices:

```
Copyright (c) OpenAI (https://openai.com)
Copyright (c) 2020 Pranav Rajpurkar
Copyright (c) 2021 Dan Hendrycks
Copyright (c) Microsoft Corporation.
Copyright (c) 2020 EleutherAI
```

```
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

### Apache License 2.0

The full text is in [LICENSE](LICENSE).

### Creative Commons Attribution 4.0 International (CC BY 4.0)

The full text is at <https://creativecommons.org/licenses/by/4.0/legalcode>.
