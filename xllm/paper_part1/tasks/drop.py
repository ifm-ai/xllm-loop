from typing import Dict, List
import numpy as np

from xllm.paper_part1.tasks.base import GenerationTask, Example
from xllm.paper_part1.tasks.utils import exact_match_score, f1_score, normalize_answer

# DROP example
# {
#     "section_id": "nfl_1184",
#     "query_id": "f37e81fa-ef7b-4583-b671-762fc433faa9",
#     "passage": " Hoping to rebound from their loss to the Patriots, the Raiders stayed at home for a Week 16 duel with the Houston Texans.  Oakland would get the early lead in the first quarter as quarterback JaMarcus Russell completed a 20-yard touchdown pass to rookie wide receiver Chaz Schilens.  The Texans would respond with fullback Vonta Leach getting a 1-yard touchdown run, yet the Raiders would answer with kicker Sebastian Janikowski getting a 33-yard and a 30-yard field goal.  Houston would tie the game in the second quarter with kicker Kris Brown getting a 53-yard and a 24-yard field goal. Oakland would take the lead in the third quarter with wide receiver Johnnie Lee Higgins catching a 29-yard touchdown pass from Russell, followed up by an 80-yard punt return for a touchdown.  The Texans tried to rally in the fourth quarter as Brown nailed a 40-yard field goal, yet the Raiders' defense would shut down any possible attempt.",
#     "question": "Who scored the first touchdown of the game?",
#     "answers_spans": {
#         "spans": [
#             "Chaz Schilens",
#             "JaMarcus Russell"
#         ],
#         "types": [
#             "span",
#             "span"
#         ]
#     }
# }


class DROPTask(GenerationTask):
    dataset_dirs = ["drop"]
    metrics = ["em", "f1", "nll"]
    fewshot_mode = "first"
    sep = "\n\n"
    eval_file = "test.jsonl"
    question_prefix = "Question:"
    passage_prefix = "Text:"
    target_prefix = "Answer:"
    prompt_format = "{passage_prefix}\n{passage}\n\n{question_prefix}\n{question}\n{target_prefix}"
    max_text_len = 2500
    max_gen_len = 64

    def get_fewshot(
        self, example: Example, rng: np.random.RandomState
    ) -> List[Example]:
        return [
            {
                "passage": "Trunajaya rebellion or Trunajaya War was the ultimately unsuccessful rebellion waged by the Madurese prince Trunajaya and fighters from Makassar against the Mataram Sultanate and its Dutch East India Company supporters in Java during the 1670s. The rebellion was initially successful: the rebels defeated the royal army at Gegodog, captured most of the Javanese north coast, and took the Mataram capital Plered. King Amangkurat I died during the retreat of the royal court. His son and successor, Amangkurat II, requested help from the VOC in exchange for financial remuneration and geopolitical concessions. The VOC\'s subsequent involvement turned the tide of the war. VOC and Mataram forces recovered lost territories and overran Trunajaya\'s new capital at Kediri. However, the rebellion continued until the capture of Trunajaya at the end of 1679, and the defeat, death, or surrender of the other rebel leaders. Trunajaya was killed by Amangkurat II personally in 1680 while a prisoner of the VOC. After his father\'s death in 1677, Amangkurat II also faced rival claims to the throne. The most serious rival was his brother Pangeran Puger, who took the capital Plered in 1677 and did not surrender until 1681.",
                "question": "How many years was it between Trunajaya\'s capture and his death while prisoner of the VOC?",
                "answers_spans": {
                    "spans": [
                        "1"
                    ],
                    "types": [
                        "span"
                    ]
                }
            },
            {
                "passage": "Led by former Giant Kurt Warner, the defending NFC champions took the field at Giants Stadium against a Giants team still reeling from their bad loss in New Orleans. The Giants scored first, sending Jacobs in for a 4-yard touchdown run following a Terrell Thomas interception. Later, Arizona running back Beanie Wells scored his first career touchdown on a 13-yard rush. Manning responded by throwing a 62-yard touchdown to Nicks for his longest reception of the year. In the second half, the Cardinals\' Tim Hightower and Jason Wright scored touchdowns. But it was turnovers that decided this game; Manning\'s 3 interceptions were as many as he had thrown all season. The Giants scored only 3 points in the second half, ending the game on an interception to Antrel Rolle. The Giants notable streak of 38 consecutive starts by the same offensive line unit was ended here, as offensive tackle Kareem McKenzie missed the game with a groin injury. McKenzie returned the following week.",
                "question": "Which player made the first score of the game?",
                "answers_spans": {
                    "spans": [
                        "Jacobs"
                    ],
                    "types": [
                        "span"
                    ]
                }
            },
            {
                "passage": "Hoping to rebound from their road loss to the Bills, the Chargers flew to Wembley Stadium for the 2008 International Series game with the New Orleans Saints. In the first quarter, San Diego trailed early as kicker Taylor Mehlhaff got a 23-yard field goal. The \'Bolts would respond with kicker Nate Kaeding getting a 33-yard field goal. In the second quarter, New Orleans regained the lead as QB Drew Brees (a former Charger) completed a 12-yard TD pass to WR Devery Henderson (with a failed PAT) and RB Deuce McAllister getting a 1-yard TD run. San Diego answered as QB Philip Rivers completed a 12-yard TD pass to RB LaDainian Tomlinson, but the Saints replied with Brees completing a 30-yard TD pass to WR Lance Moore. The Chargers closed out the half with Rivers completing a 12-yard TD pass to TE Antonio Gates. In the third quarter, New Orleans increased its lead Brees completing a 1-yard TD pass to TE Mark Campbell, after a very controversial Pass interference call on cornerback Cletis Gordon put the Saints on the 1-yard line. The \'Bolts would answer with Kaeding getting a 24-yard field goal. In the fourth quarter, the Saints continued to build its lead as FB Mike Karney got a 1-yard TD run. San Diego tried to rally as Kaeding nailed a 31-yard field goal, Rivers completed a 14-yard TD pass to WR Vincent Jackson, and Brees giving the \'Bolts a safety via an incomplete pass thrown into the back of his own endzone. However, New Orleans\' defense stiffened for the win. With the loss, the Chargers went into their bye week at 3-5.",
                "question": "How many total yards of touchdown passes did Drew Brees make?",
                "answers_spans": {
                    "spans": [
                        "43"
                    ],
                    "types": [
                        "span"
                    ]
                }
            },
        ]

    def textify_example(self, example: Example) -> str:
        context = self.get_prompt(example)
        answer = self.get_target(example)
        return f"{context} {answer}"

    def get_prompt(self, example: Example) -> str:
        passage = example["passage"].strip()
        question = example["question"].strip().rstrip("?") + "?"
        return self.prompt_format.format(
            passage_prefix=self.passage_prefix,
            passage=passage,
            question_prefix=self.question_prefix,
            question=question,
            target_prefix=self.target_prefix,
        )

    def get_target(self, example: Example) -> str:
        answers = example.get("answers_spans", {}).get("spans", [])
        return answers[0] if answers else ""

    def postprocess(self, tokens: List[int]) -> str:
        generation = self.tokenizer.decode(tokens, cut_at_eos=True)
        generation = generation.split(self.question_prefix)[0]
        generation = generation.split(self.target_prefix)[0]
        generation = generation.split("\n")[0].strip(" .")
        return generation

    def evaluate(self, prediction: str, example: Example) -> Dict[str, float]:
        ground_truths = example.get("answers_spans", {}).get("spans", [])
        sample_metrics = {
            "em": 100 * exact_match_score(prediction, ground_truths, normalize_answer),
            "f1": 100 * f1_score(prediction, ground_truths, normalize_answer),
        }
        return sample_metrics
