"""Exercise notebook cells at the interactive settings that exposed review bugs."""

import ast
import unittest
from itertools import combinations
from math import comb
from pathlib import Path
from types import SimpleNamespace

import marimo as mo
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

import companion_style
from companion_data import read_csv

ROOT = Path(__file__).resolve().parents[1]


def run_cell(chapter, definition, **dependencies):
    """Run the actual cell with supplied inputs and expose its local results."""
    path = next(ROOT.glob(f"{chapter:02d}_*.py"))
    tree = ast.parse(path.read_text())
    for cell in tree.body:
        if not isinstance(cell, ast.FunctionDef):
            continue
        definitions = {
            node.id
            for node in ast.walk(cell)
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)
        } | {node.name for node in cell.body if isinstance(node, ast.FunctionDef)}
        if definition in definitions:
            cell.decorator_list = []
            cell.name = "run"
            cell.body = [node for node in cell.body if not isinstance(node, ast.Return)]
            cell.body.append(
                ast.Return(
                    value=ast.Call(
                        func=ast.Name(id="locals", ctx=ast.Load()), args=[], keywords=[]
                    )
                )
            )
            module = ast.fix_missing_locations(ast.Module(body=[cell], type_ignores=[]))
            namespace = {}
            exec(compile(module, str(path), "exec"), namespace)  # noqa: S102 - trusted repository cells
            inputs = dict(
                vars(companion_style),
                mo=mo,
                np=np,
                pd=pd,
                plt=plt,
                stats=stats,
                comb=comb,
                combinations=combinations,
                read_csv=read_csv,
            )
            inputs.update(dependencies)
            return namespace["run"](
                **{arg.arg: inputs[arg.arg] for arg in cell.args.args}
            )
    raise AssertionError(f"No cell defines {definition}")


def control(value):
    return SimpleNamespace(value=value)


class NotebookRegressionTests(unittest.TestCase):
    def tearDown(self):
        plt.close("all")

    def test_height_histogram_counts_every_observation(self):
        heights = pd.Series([130.0, 144.9, 165.0, 175.0, 200.1, 220.0])
        summary = run_cell(1, "summary_table", height_subset=heights)["summary_table"]
        for bins in (5, 15, 30):
            result = run_cell(
                1,
                "histogram_values",
                height_subset=heights,
                summary_table=summary,
                bin_count=control(bins),
                density_scale=control(False),
                group_selector=mo.ui.dropdown(["All students"], value="All students"),
                normal_overlay=control(True),
                polygon_overlay=control(True),
                selected_group_label="All students",
            )
            self.assertEqual(result["histogram_values"].sum(), len(heights))
            low, high = result["histogram_axis"].get_xlim()
            self.assertLess(low, heights.min())
            self.assertGreater(high, heights.max())

    def test_scatter_extremes_remain_visible(self):
        helpers = run_cell(3, "generate_point_cloud")
        for slope in (-2, 2):
            result = run_cell(
                3,
                "cloud_figure",
                **helpers,
                cloud_slope=control(slope),
                cloud_noise=control(4),
                cloud_sample_size=control(200),
            )
            low, high = result["cloud_axis"].get_ylim()
            self.assertLess(low, result["cloud_y"].min())
            self.assertGreater(high, result["cloud_y"].max())

    def test_density_above_one_is_reachable_and_visible(self):
        widgets = run_cell(2, "normal_sd")
        minimum_sd = widgets["normal_sd"].start
        result = run_cell(
            2,
            "normal_pdf",
            normal_sd=control(minimum_sd),
            normal_mean=control(0),
            normal_interval=control([-1, 1]),
        )
        peak = result["normal_pdf"].max()
        self.assertGreater(peak, 1)
        self.assertGreater(result["normal_axes"][0].get_ylim()[1], peak)
        self.assertIn("exceeds one", result["density_note"])

    def test_posthoc_follows_both_illumination_subsets(self):
        helpers = run_cell(5, "holm_adjust")
        data = pd.read_csv(ROOT / "public/peroxidase.csv")
        results = []
        for illumination in ("light", "dark"):
            frame = data.loc[data.light_conditions == illumination]
            result = run_cell(
                5,
                "posthoc_frame",
                **helpers,
                selected_peroxidase=frame,
                anova_illumination=control(illumination),
                tissue_order=["root", "mesocotyl", "primary leaf"],
            )
            expected = stats.tukey_hsd(
                *[
                    frame.loc[frame.tissue == tissue, "peroxidase_amount"].to_numpy()
                    for tissue in ["root", "mesocotyl", "primary leaf"]
                ]
            )
            np.testing.assert_allclose(
                result["posthoc_tukey"], expected.pvalue[np.triu_indices(3, 1)]
            )
            low, high = result["posthoc_axis"].get_xlim()
            intervals = expected.confidence_interval()
            for first, second in result["posthoc_pairs"]:
                self.assertLessEqual(low, -intervals.high[first, second])
                self.assertGreaterEqual(high, -intervals.low[first, second])
            self.assertIn(illumination.capitalize(), result["posthoc_axis"].get_title())
            results.append(result)
        self.assertFalse(
            np.allclose(results[0]["posthoc_tukey"], results[1]["posthoc_tukey"])
        )
        self.assertNotEqual(
            results[0]["posthoc_supported"], results[1]["posthoc_supported"]
        )

    def test_interaction_profiles_match_their_names(self):
        helpers = run_cell(5, "holm_adjust")
        data = pd.read_csv(ROOT / "public/peroxidase.csv")
        for pattern in ("parallel", "converging", "diverging", "crossing"):
            result = run_cell(
                5,
                "pattern_profiles",
                **helpers,
                peroxidase_data=data,
                interaction_pattern=control(pattern),
            )
            lines = result["pattern_axis"].get_lines()
            difference = lines[0].get_ydata() - lines[1].get_ydata()
            if pattern == "parallel":
                np.testing.assert_allclose(np.diff(difference), 0, atol=1e-12)
            elif pattern == "crossing":
                self.assertLess(difference[0] * difference[-1], 0)
            else:
                self.assertTrue(np.all(difference > 0))
                changes = np.diff(difference)
                self.assertTrue(
                    np.all(changes < 0)
                    if pattern == "converging"
                    else np.all(changes > 0)
                )

    def test_dice_reveal_disabled_for_student_rolls(self):
        for source in ("lecture", "student"):
            result = run_cell(6, "dice_reveal", dice_sequence_source=control(source))
            self.assertEqual(
                result["dice_reveal"]._component_args["disabled"], source == "student"
            )


if __name__ == "__main__":
    unittest.main()
