from typing import List, Dict
import re
import numpy as np

from xllm.data.dataset_streamer.tokenizer import Tokenizer
from xllm.paper_part1.tasks.base import GenerationTask, Example
from xllm.paper_part1.tasks.utils import exact_match_score, f1_score, extract_answer, math_accuracy_score, is_equiv

# GSM8K example
# {
#     "question": "Janet\u2019s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?",
#     "answer": "Janet sells 16 - 3 - 4 = 9 duck eggs a day.\nShe makes 9 * 2 = $18 every day at the farmer\u2019s market.\n#### 18"
# }


class GSM8KTask(GenerationTask):
    dataset_dirs = ["gsm8k"]
    metrics = ["acc", "em", "f1", "nll"]
    sep = "\n\n"
    eval_file = "test.jsonl"
    answer_format: str = "{solution} The answer is {result}."
    max_text_len: int = 2048
    max_gen_len: int = 512

    def __init__(self, tokenizer: Tokenizer, task_dir: str):
        self.tokenizer = tokenizer
        self.task_dir = task_dir

    # Chain-of-thought exemplars of Wei et al. (2022, CC BY 4.0), modified;
    # see THIRD_PARTY_NOTICES.md.
    def get_fewshot(
        self, example: Example, rng: np.random.RandomState
    ) -> List[Example]:
        return [
            {
                "question": "There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?",
                "answer": "Let's think step by step. There are 15 trees originally. Then there were 21 trees after some more were planted. So there must have been 21 - 15 = 6.\n#### 6",
            },
            {
                "question": "If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?",
                "answer": "Let's think step by step. There are originally 3 cars. 2 more cars arrive. 3 + 2 = 5.\n#### 5",
            },
            {
                "question": "Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?",
                "answer": "Let's think step by step. Originally, Leah had 32 chocolates. Her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39.\n#### 39",
            },
            {
                "question": "Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?",
                "answer": "Let's think step by step. Jason started with 20 lollipops. Then he had 12 after giving some to Denny. So he gave Denny 20 - 12 = 8.\n#### 8",
            },
            {
                "question": "Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?",
                "answer": "Let's think step by step. Shawn started with 5 toys. If he got 2 toys each from his mom and dad, then that is 4 more toys. 5 + 4 = 9.\n#### 9",
            },
            {
                "question": "There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?",
                "answer": "Let's think step by step. There were originally 9 computers. For each of 4 days, 5 more computers were added. So 5 * 4 = 20 computers were added. 9 + 20 is 29.\n#### 29",
            },
            {
                "question": "Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?",
                "answer": "Let's think step by step. Michael started with 58 golf balls. After losing 23 on tuesday, he had 58 - 23 = 35. After losing 2 more, he had 35 - 2 = 33 golf balls.\n#### 33",
            },
            {
                "question": "Olivia has $23. She bought five bagels for $3 each. How much money does she have left?",
                "answer": "Let's think step by step. Olivia had 23 dollars. 5 bagels for 3 dollars each will be 5 x 3 = 15 dollars. So she has 23 - 15 dollars left. 23 - 15 is 8.\n#### 8",
            },
        ]

    def textify_example(self, example: Example) -> str:
        return f"{self.get_prompt(example)} {self.get_target(example)}"

    def get_prompt(self, example: Example) -> str:
        return f"Q: {example['question']}\nA:"

    def get_target(self, example: Example) -> str:
        split = example["answer"].rsplit("\n#### ", maxsplit=1)
        if len(split) != 2:
            return ""
        solution, result = split
        return f"{solution} The answer is {result}."

    def postprocess(self, tokens: List[int]) -> str:
        generation = self.tokenizer.decode(tokens, cut_at_eos=True)
        generation = generation.split("Q: ")[0].split("A:")[0]
        return generation

    def evaluate(self, prediction: str, example: Example) -> Dict[str, float]:
        ground_truths = [self.get_target(example)]
        sample_metrics = {
            "acc": 100 * math_accuracy_score(prediction, ground_truths, self.eval_normalizer),
            "em": 100 * exact_match_score(prediction, ground_truths, self.eval_normalizer),
            "f1": 100 * f1_score(prediction, ground_truths, self.eval_normalizer),
        }
        return sample_metrics

    @staticmethod
    def eval_normalizer(answer: str) -> str:
        return extract_answer(answer)


if __name__ == "__main__":
    """
    python -m src.eval.task.gsm8k
    """

    # fmt: off
    TESTS = [
        ("She makes 16 - 3 - 4 - 1 = 8 dollars per day. The answer is 8.", "8"),
        ("60 mph is 1 mile per hour. 5.5 hours is 5.5 * 1 = 5.5 miles. The answer is 5.50 miles.", "5.5"),
        ("Then they travel northwards for 150 miles. The answer is 230.0 miles.", "230"),
        ("15.625 is 15.625 / 1.25 = 12.5. 19.50 - 12.5 = 7.5. The original price of the book was \\frac{15}{2} dollars.", "7.5"),
        ("15.625 is 15.625 / 1.25 = 12.5. 19.50 - 12.5 = 7.5. The original price of the book was 7.50000.", "\\frac{15}{2}"),
        ("So he spends .5 * 10 = 5 hours a day. 5 hours a day * 7 days a week = 35 hours a week. The answer is 35 hours a week.", "35"),
        ("The owner's manual says that her tank holds 12 gallons. So she has 12 - 96 = -84 gallons left. The answer is -84 miles.", "-84.0"),
        ("So he has 20 x 3 + 19 x 4 = 126 dollars. The answer is 126 dollars.", "126"),
        ("Her mom placed 1/3 of the remaining pieces. So 1/4 + 1/3 = 1/2. 1/2 of the pieces are left. The answer is 1/2.", "1/2"),
        ("1000080 is 1000080 / 365 = 278.5. The answer is 278.5.", "278.5"),
        ("Profit is the difference between total income and total expenses, so profit is 57 - 35 = 22 dollars.", "22"),
        ("Carl pays with a $10 bill. So he gets $10 - $6 = $4.00 in change.", "4.00"),
        ("a week as a coach, then she will make 20 * 35 * 50 * 35 * 15 = 10,000 dollars. The answer is 10,000 dollars.", "10000"),
        ("12 + 3 * 5 + 1.5 * 4 + 8.5 * ? = 50. 12 + 3 * 5 + 1.5 * 4 + 8.5 * ? = 50. 12 + 15 + 6 + 36 = 65. 65 - 36 = 29. The answer is 29.", "29"),
        ("There are 30 more gold coins than silver coins. So 110 - 30 = 80. 80 / 30 = 2.67. The answer is 8/3.", "2.666667"),
        ("Each floor contains 8 units. 15 x 8 = 120 units. 120 - 3/4 = 120 - 1/2 = 60 units. The answer is 60.", "60"),
        ("The empty truck weighs 3600 pounds. The answer is 3600 - 150 - 3755 = 150. The maximum number of boxes is 150.", "150"),
        ("Christina needs 16 * .75 = 12 gift bags. 12 * 2 = 24. So she will spend 24 dollars.", "24"),
        ("After cooking, she had 16 ounces of sauce. 32 - 16 = 16. The answer is **16**.", "16"),
        ("So the answer is 14 + 42 + 2/3 - 1 = 69.", "69"),
        ("The value of the house increased by 150%. 80,000 + 50,000 = 130,000. 130,000 x 1.5 = 195,000. 195,000 - 80,000 = 115,000. 115,000 is the profit.", "115000"),
        ("1 dozen eggs x 28 days = 28 dozens of eggs. The answer is 28 dozens of eggs.", "28"),
        ("If Ted wants to have enough to feed everyone, he needs to bring 250 lbs of potato salad.", "250"),
        ("So he completed 100 / 2 = 50 questions. He left 25 questions incomplete.", "25"),
        ("1800 calories / 6 servings = 300 calories. So you can eat 300 calories from the bag of chips.", "300"),
        ("There are 22 more pink gumballs than blue gumballs. 22 + 12 = 34. So there are 34 pink gumballs.", "34"),
        ("how long will they be in feet? The answer is 300 + 120 + 60 = 480.", "480"),
        ("24 liters is 2/3 + 3/5 = 5/3. The answer is 5/3.", "5/3"),
        ("60 yogurts for 4 yogurts for $5.00 is 60 x 4 = 240 dollars. 240 dollars is the answer.", "240"),
        ("Charlie's net profit is 2000 - 40 = 1960 dollars.", "1,960"),
        ("12 / 20 = 60%. 4 / 20 = 20%. 60% + 20% = 80%. The answer is 80%.", "80"),
        ("The total number of hours she spent writing articles in the three days is 10 + 40 + 10 = 60 hours.", "60"),
        ("The mechanic earned 480 - 240 = 240 more revenue on Friday.", "240"),
        ("Allen's age 10 years from now is 0145.", "145"),
        ("The profit will be 5,000 - 8,000 = -3,000. The answer is -3,000.0.", "-3000"),
        ("13/20 of 120 = 78 teaspoonfuls of water. The answer is 42 + 78 = 120 teaspoonfuls of sugar.", "120"),
        ("The answer is 138.6, which rounds to 139.", "139"),
        ("the total cost with tax is 15 + 1.5 = 16.5 dollars. The answer is 16.5.", "16.5"),
    ]

    # Case we can not handle well yet:
    # =========CASE 1=========
    # GenerationA.1: The total cost is 3500 + 21000 = 24500 cents. The answer is 24500 cents, which is $245. GroundTruth: 24500
    # GenerationA.2: The answer is -80, which means Margareth has 80 more beads than Elizabeth. GroundTruth: 80


    
    # If we choose the first number after "The answer is", we get -80 instead of 80.
    # If we choose the last number in the whole text, we get 245 instead of 24500.
    # In this script, we choose the last number in the whole text.


    print("Running Tests...")
    passed = 0
    for test in TESTS:
        pred_str, gold_answer = test
        gold_extracted = extract_answer(gold_answer)
        pred_extracted = extract_answer(pred_str)
        
        is_match = is_equiv(pred_extracted, gold_extracted)
        
        if not is_match:
            print(f"Fail:")
            print(f"Original: {pred_str}")
            print(f"Pred: {pred_extracted} | Gold: {gold_extracted}")
        else:
            passed += 1

    print(f"-----\nTotal: {len(TESTS)}, Passed: {passed}")

