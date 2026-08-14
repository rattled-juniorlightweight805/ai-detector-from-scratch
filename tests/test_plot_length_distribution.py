import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "01_human-data"
    / "01_plot_length_distribution.py"
)
SPEC = importlib.util.spec_from_file_location("plot_length_distribution", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class LengthDistributionTests(unittest.TestCase):
    def test_assign_bin_uses_documented_boundaries(self):
        expected = {
            1: 0,
            50: 0,
            51: 1,
            100: 1,
            101: 2,
            250: 2,
            251: 3,
            500: 3,
            501: 4,
            1000: 4,
            1001: 5,
            2000: 5,
            2001: 6,
        }
        for word_count, bin_index in expected.items():
            self.assertEqual(MODULE.assign_bin(word_count), bin_index)

    def test_two_folders_render_as_side_by_side_svg(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            human = root / "human"
            ai = root / "ai"
            human.mkdir()
            ai.mkdir()
            (human / "1.txt").write_text("word " * 50, encoding="utf-8")
            (human / "2.txt").write_text("word " * 250, encoding="utf-8")
            (ai / "1.txt").write_text("word " * 100, encoding="utf-8")
            (ai / "2.txt").write_text("word " * 500, encoding="utf-8")

            distributions = [
                MODULE.read_distribution(human),
                MODULE.read_distribution(ai),
            ]
            svg = MODULE.render_svg(distributions)

            self.assertIn('width="1316"', svg)
            self.assertIn(">Human-written samples</text>", svg)
            self.assertIn(">AI-generated samples</text>", svg)
            self.assertIn(">Share of samples</text>", svg)
            self.assertIn(">Sample length (words)</text>", svg)
            self.assertIn(">words ≤50</text>", svg)
            self.assertIn(">n=1</text>", svg)
            self.assertIn("50.0%", svg)
            self.assertEqual(distributions[0].bin_counts, (1, 0, 1, 0, 0, 0, 0))
            self.assertEqual(distributions[1].bin_counts, (0, 1, 0, 1, 0, 0, 0))

    def test_axis_has_headroom_for_bar_labels(self):
        self.assertEqual(MODULE.nice_axis_max(24.5), 30)
        self.assertEqual(MODULE.nice_axis_max(7.0), 10)


if __name__ == "__main__":
    unittest.main()
