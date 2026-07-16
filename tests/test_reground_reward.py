"""Offline tests for the paper-aligned Stage-2 reward."""

import unittest

from grpo.protocol import REGROUND_USER_PROMPT
from grpo.reward import answers_match, compute_score


DIRECT_CORRECT = """<think>I inspect the chart labels carefully.</think>
<answer>15</answer>"""

DIRECT_INCORRECT = """<think>I inspect the chart labels carefully.</think>
<answer>12</answer>"""

REGROUND_PREFIX = """<think>I should count the marked objects before deciding.</think>
<reground>Wait, I need to recheck each visible object and confirm the count.</reground>"""

REGROUND_CORRECT = f"""{REGROUND_PREFIX}
{REGROUND_USER_PROMPT}
<think>I count all three groups again and confirm five objects in each.</think>
<answer>15</answer>"""

REGROUND_INCORRECT = f"""{REGROUND_PREFIX}
{REGROUND_USER_PROMPT}
<think>I count the groups again but still miss one object.</think>
<answer>12</answer>"""


class ReGroundRewardTest(unittest.TestCase):
    def test_four_reward_quadrants_match_supplement_after_rounding(self) -> None:
        cases = [
            (REGROUND_CORRECT, 1.0),
            (DIRECT_CORRECT, 0.6),
            (REGROUND_INCORRECT, 0.4),
            (DIRECT_INCORRECT, -0.1),
        ]
        for solution, expected in cases:
            with self.subTest(expected=expected):
                result = compute_score("reground", solution, "15")
                self.assertAlmostEqual(round(result["score"], 1), expected)
                self.assertEqual(result["format_ok"], 1.0)

    def test_format_penalty_is_applied_separately(self) -> None:
        malformed = "<think>unfinished<answer>15</answer>"
        result = compute_score("reground", malformed, "15")
        self.assertEqual(result["format_ok"], 0.0)
        self.assertAlmostEqual(result["format_component"], -0.01)

    def test_hollow_marker_is_not_genuine_diagnosis(self) -> None:
        hollow = f"""<think>short</think>
<reground>look again</reground>
{REGROUND_USER_PROMPT}
<think>I inspect the image again.</think>
<answer>15</answer>"""
        result = compute_score("reground", hollow, "15")
        self.assertEqual(result["trigger_rate"], 1.0)
        self.assertEqual(result["genuine_reground"], 0.0)

    def test_multiple_choice_accepts_label_or_option_text(self) -> None:
        extra = {"choices": ["red", "blue", "green"], "accepted_answers": ["B", "blue"]}
        self.assertTrue(answers_match("B", "B", extra))
        self.assertTrue(answers_match("blue", "B", extra))
        self.assertFalse(answers_match("green", "B", extra))

    def test_numeric_comparison_handles_equivalent_values(self) -> None:
        self.assertTrue(answers_match("1.500000", "1.5"))
        self.assertTrue(answers_match("15%", "0.15"))


if __name__ == "__main__":
    unittest.main()
