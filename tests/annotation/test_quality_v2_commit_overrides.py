"""Focused tests for reviewed pilot commit overrides."""

import unittest

import pytest

from lerobot.annotation.paths import QUALITY_V2
from lerobot.annotation.quality_v2.commit_overrides import apply_commit_override, load_commit_overrides

# The reviewed overrides are a result of the class pass: they live in its results folder, outside the repo.
if not (QUALITY_V2 / "pilot_commit_overrides.json").exists():
    pytest.skip("class-pass results folder not on this machine", allow_module_level=True)
OVERRIDES = load_commit_overrides(QUALITY_V2 / "pilot_commit_overrides.json")


class CommitOverridesTest(unittest.TestCase):
    def test_reviewed_overrides_have_expected_frames(self):
        expected = {
            "household__ep001155_p0a0": 56,
            "household__ep001157_p0a0": 92,
            "household__ep002118_p0a0": 48,
            "household__ep002124_p0a0": 50,
            "household__ep006058_p0a0": 35,
            "household__ep006138_p0a0": 38,
            "household__ep006735_p0a1": 159,
            "household__ep006747_p0a0": 200,
            "household__ep006771_p0a0": 132,
        }
        self.assertEqual({uid: row["frame"] for uid, row in OVERRIDES.items()}, expected)

    def test_known_override_replaces_default(self):
        actual = apply_commit_override(
            "household__ep002118_p0a0", 0, 68, (68, "end", True), OVERRIDES
        )
        self.assertEqual(actual, (48, "action_end", True))

    def test_no_press_uses_not_found_end_fallback(self):
        actual = apply_commit_override(
            "household__ep001157_p0a0", 0, 92, (92, "end", True), OVERRIDES
        )
        self.assertEqual(actual, (92, "end", False))

    def test_unknown_uid_preserves_default(self):
        default = (40, "end", True)
        self.assertIs(
            apply_commit_override("household__ep999999_p0a0", 0, 40, default, OVERRIDES),
            default,
        )

    def test_found_override_must_be_inside_half_open_unit(self):
        bad = {
            "household__ep999999_p0a0": {
                "frame": 40,
                "kind": "action_end",
                "found": True,
                "reason": "synthetic out-of-bounds test",
            }
        }
        with self.assertRaisesRegex(ValueError, "exclusive unit end"):
            apply_commit_override(
                "household__ep999999_p0a0", 0, 40, (40, "end", True), bad
            )

    def test_override_must_be_within_unit_bounds(self):
        with self.assertRaisesRegex(ValueError, "outside unit bounds"):
            apply_commit_override(
                "household__ep006058_p0a0", 36, 48, (48, "end", True), OVERRIDES
            )


if __name__ == "__main__":
    unittest.main()
