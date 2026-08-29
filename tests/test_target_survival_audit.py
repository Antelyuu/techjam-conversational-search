import json
import tempfile
import unittest
from pathlib import Path

from scripts.target_survival_audit import audit


class TargetSurvivalAuditTest(unittest.TestCase):
    def test_audit_handles_override_sessions_and_counts_target_removals(self):
        products = [
            {
                "parent_asin": "shoe-target",
                "title": "Running shoes",
                "categories": ["Shoes"],
                "features": ["lightweight"],
                "details": {},
                "price": 120,
            },
            {
                "parent_asin": "jewelry-target",
                "title": "Gold necklace",
                "categories": ["Jewelry"],
                "features": ["gold"],
                "details": {},
                "price": None,
            },
        ]
        samples = [
            {
                "sample_id": "buying-over-budget",
                "scenario_type": "buying",
                "user_profile": {},
                "ground_truth": {"parent_asin": "shoe-target"},
                "intent_card": {
                    "target_category": "Running shoes",
                    "hard_constraints": ["under $100"],
                    "soft_preferences": [],
                },
                "behavior": {"scenario_type": "buying"},
            },
            {
                "sample_id": "override-without-budget",
                "scenario_type": "intent_override",
                "user_profile": {},
                "ground_truth": {"parent_asin": "jewelry-target"},
                "intent_card": {
                    "target_category": "Gold necklace",
                    "hard_constraints": ["gold"],
                    "soft_preferences": ["silver"],
                },
                "behavior": {
                    "scenario_type": "intent_override",
                    "override": {
                        "turn": 3,
                        "old_value": "silver",
                        "new_value": "gold",
                        "message": "Actually, what I need is gold.",
                    },
                },
            },
        ]

        with tempfile.TemporaryDirectory() as directory:
            catalog_path = Path(directory) / "catalog.jsonl"
            dataset_path = Path(directory) / "public_set.jsonl"
            catalog_path.write_text(
                "".join(json.dumps(product) + "\n" for product in products),
                encoding="utf-8",
            )
            dataset_path.write_text(
                "".join(json.dumps(sample) + "\n" for sample in samples),
                encoding="utf-8",
            )

            report = audit(str(catalog_path), str(dataset_path))

        self.assertEqual(report["sessions_checked"], 2)
        self.assertEqual(report["targets_survived"], 1)
        self.assertEqual(report["targets_removed_by_hard_filters"], 1)
        self.assertEqual(report["price_filter"]["hard_constraints_evaluated"], 1)
        self.assertEqual(report["price_filter"]["targets_excluded"], 1)
        self.assertEqual(report["category_signal"]["constraints_evaluated"], 2)
        self.assertEqual(report["category_signal"]["hard_exclusions"], 0)

    def test_audit_materializes_missing_override_behavior(self):
        product = {
            "parent_asin": "override-target",
            "title": "Blue running shoes",
            "categories": ["Shoes"],
            "features": ["lightweight", "blue"],
            "details": {},
            "price": None,
        }
        sample = {
            "sample_id": "generated-override",
            "scenario_type": "intent_override",
            "user_profile": {},
            "ground_truth": {"parent_asin": "override-target"},
        }

        with tempfile.TemporaryDirectory() as directory:
            catalog_path = Path(directory) / "catalog.jsonl"
            dataset_path = Path(directory) / "public_set.jsonl"
            catalog_path.write_text(json.dumps(product) + "\n", encoding="utf-8")
            dataset_path.write_text(json.dumps(sample) + "\n", encoding="utf-8")

            report = audit(str(catalog_path), str(dataset_path))

        self.assertEqual(report["sessions_checked"], 1)
        self.assertEqual(report["targets_survived"], 1)


if __name__ == "__main__":
    unittest.main()
