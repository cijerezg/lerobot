"""Focused tests for carry/release precision-hint pairing."""

import unittest

from lerobot.annotation.quality_v2 import grade_batches as make_grade_batches


def unit(uid, unit_id, action, precision, start=None, dataset="test", episode="ep0"):
    row = {
        "uid": uid,
        "dataset": dataset,
        "episode": episode,
        "unit": unit_id,
        "action": action,
        "precision_v1": precision,
    }
    if start is not None:
        row.update(from_index=start, to_index=start + 10)
    return row


class PrecisionHintsTest(unittest.TestCase):
    def test_diverse_carry_uses_first_later_release_across_intervening_unit(self):
        rows = [
            unit("release", "p0a3", "release", 4),
            unit("carry", "p0a1", "carry", 1),
            unit("middle", "p0a2", "return", 2),
            unit("grasp", "p0a0", "grasp", 3),
        ]
        hints = make_grade_batches.precision_hints(rows)
        self.assertEqual(hints["carry"], 4)

    def test_next_grasp_ends_pairing_search(self):
        rows = [
            unit("carry", "p0a0", "carry", 2),
            unit("next-grasp", "p0a1", "grasp", 3),
            unit("later-release", "p0a2", "release", 5),
        ]
        hints = make_grade_batches.precision_hints(rows)
        self.assertEqual(hints["carry"], 2)

    def test_frame_order_preserves_rebot_adjacent_release_behavior(self):
        rows = [
            unit("release", "seg2", "release", 3, start=30),
            unit("carry", "seg1", "carry", 1, start=20),
            unit("grasp", "seg0", "grasp", 4, start=10),
        ]
        hints = make_grade_batches.precision_hints(rows)
        self.assertEqual(hints["carry"], 3)
        self.assertEqual(hints["release"], 3)


if __name__ == "__main__":
    unittest.main()
