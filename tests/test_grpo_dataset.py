"""Offline tests for generic veRL dataset construction."""

import argparse
from pathlib import Path
import tempfile
import unittest

from grpo.prepare_dataset import build_record


class GRPODatasetTest(unittest.TestCase):
    def test_multiple_choice_record_uses_reground_agent(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image = root / "sample.jpg"
            image.write_bytes(b"placeholder")
            args = argparse.Namespace(
                question_field="question",
                answer_field="answer",
                choices_field="choices",
                image_field="image",
                source_field="source",
                ability_field="ability",
                id_field="id",
                answer_index_base="zero",
                image_root=None,
            )
            raw = {
                "id": "sample-1",
                "question": "Which color is shown?",
                "choices": ["red", "blue"],
                "answer": 1,
                "image": image.name,
                "source": "demo",
            }
            record = build_record(raw, root, 0, args)

        self.assertEqual(record["agent_name"], "reground_agent")
        self.assertEqual(record["reward_model"]["ground_truth"], "B")
        self.assertEqual(record["extra_info"]["accepted_answers"], ["1", "B", "blue"])
        self.assertIn("<image>", record["prompt"][0]["content"])
        self.assertEqual(record["images"][0]["image"], str(image.resolve()))


if __name__ == "__main__":
    unittest.main()
