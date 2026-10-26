from __future__ import annotations

import json
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_json(relative_path: str) -> dict:
    return json.loads((PROJECT_ROOT / relative_path).read_text(encoding="utf-8"))


class ArtifactContractTests(unittest.TestCase):
    def test_subject_split_and_folds_are_disjoint_and_complete(self) -> None:
        split = load_json("baseline/artifacts/split_meta.json")
        folds = load_json("baseline/artifacts/development_folds.json")

        train = set(split["train_user_ids"])
        test = set(split["test_user_ids"])
        folded = [
            subject
            for subjects in folds["validation_subject_ids_by_fold"].values()
            for subject in subjects
        ]

        self.assertEqual(len(train), 41)
        self.assertEqual(len(test), 10)
        self.assertTrue(train.isdisjoint(test))
        self.assertEqual(len(folded), len(set(folded)))
        self.assertEqual(set(folded), train)

    def test_preprocessing_contract_matches_frozen_split(self) -> None:
        split = load_json("baseline/artifacts/split_meta.json")
        preprocessing = load_json(
            "modified/watch_pair_20hz/preprocessing_meta.json"
        )

        self.assertEqual(preprocessing["subjects"]["train"], split["train_user_ids"])
        self.assertEqual(preprocessing["subjects"]["test"], split["test_user_ids"])
        self.assertEqual(
            preprocessing["counts"]["train"]["paired_output_examples"], 25_585
        )
        self.assertEqual(
            preprocessing["counts"]["test"]["paired_output_examples"], 6_161
        )
        self.assertEqual(preprocessing["resampling"]["target_rate_hz"], 20)
        self.assertEqual(preprocessing["resampling"]["target_timesteps"], 100)
        self.assertEqual(preprocessing["resampling"]["normalization"], "none")

    def test_final_eighteen_class_result_matches_frozen_specification(self) -> None:
        specification = load_json("modeling/artifacts/final_model_spec.json")
        result = load_json("modeling/artifacts/final_test_result.json")

        self.assertEqual(specification["status"], "frozen_before_reserved_test_access")
        self.assertEqual(specification["features"]["count"], 222)
        self.assertEqual(specification["final_fit"]["training_examples"], 25_585)
        self.assertEqual(specification["final_fit"]["reserved_test_examples"], 6_161)
        self.assertEqual(
            result["status"], "completed_once_without_post_test_model_changes"
        )
        self.assertEqual(result["population"]["training_subjects"], 41)
        self.assertEqual(result["population"]["test_subjects"], 10)
        self.assertEqual(len(result["activity_metrics"]), 18)
        self.assertAlmostEqual(result["reserved_test"]["macro_f1"], 0.6850628393)

    def test_grouped_task_is_recorded_as_a_separate_closed_evaluation(self) -> None:
        mapping = load_json("business_case/artifacts/activity_group_mapping.json")
        development = load_json("business_case/artifacts/grouped_target_result.json")
        test_result = load_json("business_case/artifacts/grouped_test_result.json")

        self.assertEqual(len(mapping["activity_code_to_business_label"]), 18)
        self.assertEqual(len(set(mapping["activity_code_to_business_label"].values())), 3)
        self.assertEqual(development["selection"]["selected_approach"], "direct_3_class")
        self.assertEqual(
            test_result["status"],
            "final_grouped_target_test_complete_no_post_test_changes",
        )
        self.assertIn("previously evaluated", test_result["history_disclosure"])
        self.assertEqual(test_result["test_population"]["examples"], 6_161)
        self.assertEqual(test_result["test_population"]["subjects"], 10)


if __name__ == "__main__":
    unittest.main()
