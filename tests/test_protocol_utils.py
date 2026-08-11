import json
import math
import tempfile
import unittest
from pathlib import Path

from experiments.scripts.analysis.generate_overlap import (
    generate_overlap_report,
    load_colmap_images_txt,
    load_visibility_json,
    subset_views,
)
from experiments.scripts.core.protocol_utils import (
    aggregate_overlap,
    budget_checkpoint,
    build_overlap_report,
    classify_delta,
    compute_tau,
    holm_adjust,
    scene_cluster_bootstrap_ci,
)


class ProtocolUtilsTest(unittest.TestCase):
    def test_overlap_keeps_non_edges_as_zero(self):
        summary = aggregate_overlap(
            {
                "v1": {1, 2, 3},
                "v2": {2, 3, 4},
                "v3": {9},
            }
        )

        self.assertEqual(summary["pair_count"], 3)
        self.assertEqual(summary["zero_pair_count"], 2)
        self.assertAlmostEqual(summary["mean_overlap"], (4 / 6) / 3)
        self.assertTrue(summary["has_isolated_view"])

    def test_budget_checkpoint_uses_last_checkpoint_before_budget(self):
        checkpoints = [
            {"wall_clock": 0.9, "test_psnr": 10.0},
            {"wall_clock": 10.0, "test_psnr": 9.0},
            {"wall_clock": 10.2, "test_psnr": 20.0},
        ]

        selected = budget_checkpoint(checkpoints, 10.0)
        self.assertEqual(selected["test_psnr"], 9.0)

    def test_tau_and_delta_classification(self):
        tau = compute_tau(seed_variability=0.2, practical_min_delta=0.5)
        self.assertEqual(tau, 0.5)
        self.assertEqual(classify_delta(0.6, tau), "feedforward_win")
        self.assertEqual(classify_delta(-0.6, tau), "optimization_win")
        self.assertEqual(classify_delta(0.2, tau), "tie")

    def test_scene_cluster_bootstrap_counts_scenes_not_runs(self):
        rows = [
            {"scene": "a", "value": 1.0},
            {"scene": "a", "value": 3.0},
            {"scene": "b", "value": 5.0},
        ]

        result = scene_cluster_bootstrap_ci(rows, lambda row: float(row["value"]), iterations=50)
        self.assertEqual(result["scene_count"], 2)
        self.assertAlmostEqual(result["mean"], 3.5)
        self.assertFalse(math.isnan(result["ci_low"]))

    def test_build_overlap_report_contains_pairwise_rows(self):
        report = build_overlap_report({"v1": {1}, "v2": {1, 2}})
        self.assertEqual(report["summary"]["view_count"], 2)
        self.assertEqual(len(report["pairwise"]), 1)

    def test_load_visibility_json_and_subset(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "visibility.json"
            path.write_text(json.dumps({"v1": [1, 2], "v2": [2]}), encoding="utf-8")

            view_points = load_visibility_json(path)
            self.assertEqual(view_points["v1"], {"1", "2"})
            self.assertEqual(subset_views(view_points, ["v2"]), {"v2": {"2"}})

    def test_load_colmap_images_txt_ignores_unmatched_points(self):
        content = """# Image list
1 1 0 0 0 0 0 0 1 img_a.png
0 0 10 1 1 -1
2 1 0 0 0 0 0 0 1 img_b.png
0 0 10 2 2 7
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "images.txt"
            path.write_text(content, encoding="utf-8")

            view_points = load_colmap_images_txt(path)
            self.assertEqual(view_points["img_a.png"], {"10"})
            self.assertEqual(view_points["img_b.png"], {"10", "7"})

    def test_generate_overlap_report_writes_outputs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            outdir = Path(tmpdir) / "overlap"
            report = generate_overlap_report(
                {"v1": {"1", "2"}, "v2": {"2"}, "v3": {"9"}},
                output_dir=outdir,
                scene="pilot_scene",
                view_count_label="3view",
            )

            self.assertTrue((outdir / "summary.json").exists())
            self.assertTrue((outdir / "pairwise_overlap.csv").exists())
            self.assertEqual(report["metadata"]["non_edge_policy"], "zero_included")

    def test_holm_adjust_preserves_input_order(self):
        self.assertEqual(holm_adjust([0.04, 0.01]), [0.04, 0.02])


if __name__ == "__main__":
    unittest.main()
