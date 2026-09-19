"""Exercise notebook cells at the interactive settings that exposed review bugs."""

import ast
import unittest
from itertools import combinations, pairwise
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

    def test_repeated_testing_curves_and_independent_null_formula(self):
        simulate = run_cell(4, "simulate_pvalue_experiment")[
            "simulate_pvalue_experiment"
        ]
        for strategy in ("single", "interim", "outcomes"):
            result = simulate(20, 20_000, strategy, 0.05, 1729, max_tests=10)
            rates = result["curve_rates"]
            self.assertTrue(np.all(np.diff(rates) >= 0))
            self.assertEqual(rates[-1], result["rejection_rate"])
            self.assertAlmostEqual(rates[0], 0.05, delta=0.01)
            np.testing.assert_allclose(
                result["curve_mcse"], np.sqrt(rates * (1 - rates) / 20_000)
            )
            if strategy == "outcomes":
                expected = 1 - 0.95 ** result["test_counts"]
                self.assertTrue(
                    np.all(abs(rates - expected) < 5 * result["curve_mcse"])
                )
            if strategy == "interim":
                self.assertEqual(result["look_sizes"][0], 20)
                self.assertEqual(result["look_sizes"][-1], 5)
                self.assertEqual(len(set(result["look_sizes"])), 10)
        edge = simulate(5, 1000, "interim", 0.05, 1, max_tests=1)
        single = simulate(5, 1000, "single", 0.05, 1)
        np.testing.assert_array_equal(edge["p_values"], single["p_values"])

    def test_repeated_testing_slider_matches_available_distinct_checks(self):
        simulate = run_cell(4, "simulate_pvalue_experiment")[
            "simulate_pvalue_experiment"
        ]
        for n, limit in ((5, 1), (10, 6), (14, 10), (20, 10)):
            widgets = run_cell(
                4,
                "simulation_max_tests",
                simulation_sample_size=control(n),
                simulation_strategy=control("interim"),
            )
            self.assertEqual(widgets["simulation_max_tests"].stop, limit)
            for count in range(1, limit + 1):
                result = simulate(n, 100, "interim", 0.05, 42, max_tests=count)
                self.assertEqual(len(set(result["look_sizes"])), count)
                self.assertEqual(len(result["test_counts"]), count)
        with self.assertRaises(ValueError):
            simulate(10, 100, "interim", 0.05, 42, max_tests=7)
        for strategy, limit in (("single", 1), ("outcomes", 10)):
            widgets = run_cell(
                4,
                "simulation_max_tests",
                simulation_sample_size=control(5),
                simulation_strategy=control(strategy),
            )
            self.assertEqual(widgets["simulation_max_tests"].stop, limit)

    def test_repeated_testing_controls_wait_for_run_and_label_actual_results(self):
        helpers = run_cell(4, "simulate_pvalue_experiment")
        widgets = run_cell(4, "simulation_sample_size")
        saved = [helpers["simulate_pvalue_experiment"](20, 1000, "outcomes", 0.05, 4)]
        next_controls = widgets | {
            "simulation_strategy": control("interim"),
            "simulation_max_tests": control(1),
            "simulation_sample_size": control(5),
            "simulation_repetitions": control(1000),
        }
        callback = run_cell(
            4,
            "simulation_run_button",
            **helpers,
            **next_controls,
            set_simulation_result=lambda value: saved.__setitem__(0, value),
        )
        result = run_cell(
            4,
            "pvalue_simulation_result",
            **next_controls,
            get_simulation_result=lambda: saved[0],
            simulation_run_button=callback["simulation_run_button"],
        )
        self.assertEqual(
            result["pvalue_curve_axis"].get_ylabel(),
            "Fraction of studies with a false alarm",
        )
        self.assertEqual(len(result["pvalue_counts"]), 5)
        self.assertEqual(len(result["pvalue_curve_axis"].lines), 5)
        callback["run_pvalue_simulation"](None)
        self.assertNotIn("simulation_effect", widgets)
        self.assertEqual(saved[0]["strategy"], "interim")
        self.assertEqual(len(saved[0]["test_counts"]), 1)
        updated = run_cell(
            4,
            "pvalue_simulation_result",
            **next_controls,
            get_simulation_result=lambda: saved[0],
            simulation_run_button=callback["simulation_run_button"],
        )
        self.assertEqual(
            updated["pvalue_curve_axis"].get_ylabel(),
            "Fraction of studies with a false alarm",
        )
        self.assertEqual(len(updated["pvalue_curve_axis"].lines), 4)

    def test_mean_effect_precision_and_matched_intervals(self):
        mean_test = run_cell(4, "one_sample_mean_test")["one_sample_mean_test"]
        for test_type in ("z", "t"):
            small = mean_test(2200, 3000, 900, 10, test_type, "two-sided", 0.05)
            large = mean_test(2200, 3000, 900, 40, test_type, "two-sided", 0.05)
            self.assertEqual(small["standardized_effect"], large["standardized_effect"])
            self.assertAlmostEqual(small["standardized_effect"], -800 / 900)
            self.assertEqual(small["difference"], large["difference"])
            self.assertAlmostEqual(small["standard_error"], 2 * large["standard_error"])
            self.assertLess(large["interval_width"], small["interval_width"])
            self.assertLess(large["p_value"], small["p_value"])
            for n in (2, 10, 100):
                for alternative in ("less", "greater", "two-sided"):
                    for observed in (2200, 3000, 3500):
                        for alpha in (0.01, 0.05, 0.10):
                            result = mean_test(
                                observed, 3000, 900, n, test_type, alternative, alpha
                            )
                            low, high = result["difference_interval"]
                            self.assertEqual(result["reject"], low > 0 or high < 0)
                            self.assertEqual(np.isneginf(low), alternative == "less")
                            self.assertEqual(
                                np.isposinf(high), alternative == "greater"
                            )
                            if alternative != "two-sided":
                                self.assertTrue(np.isinf(result["interval_width"]))
            if test_type == "z":
                expected = stats.norm.interval(0.95, loc=-800, scale=900 / np.sqrt(10))
            else:
                expected = stats.t.interval(0.95, 9, loc=-800, scale=900 / np.sqrt(10))
            np.testing.assert_allclose(small["difference_interval"], expected)

    def test_mean_explorer_displays_matching_effect_and_interval(self):
        helpers = run_cell(4, "one_sample_mean_test")
        widgets = run_cell(4, "mean_test_type")
        for test_type in ("z", "t"):
            for alternative in ("less", "greater", "two-sided"):
                result = run_cell(
                    4,
                    "anatomy_result",
                    **helpers,
                    **(
                        widgets
                        | {
                            "mean_test_type": control(test_type),
                            "mean_alternative": control(alternative),
                        }
                    ),
                )
                self.assertIn(
                    result["anatomy_interval_text"],
                    result["anatomy_rows"]["Value"].tolist(),
                )
                self.assertIn(
                    "Cohen" if test_type == "t" else "population",
                    result["anatomy_effect_label"],
                )
                self.assertEqual(
                    "∞" in result["anatomy_interval_text"], alternative != "two-sided"
                )

    def test_lognormal_means_and_linked_densities(self):
        previous_ratio = 0
        for sigma in (0.1, 0.6, 1.5):
            for mu in (-1, 1, 3):
                controls = {
                    "lognormal_mu": control(mu),
                    "lognormal_sigma": control(sigma),
                }
                result = run_cell(1, "lognormal_geometric", **controls)
                population = result["lognormal_population"]
                self.assertAlmostEqual(
                    result["lognormal_arithmetic"], population.mean()
                )
                self.assertAlmostEqual(
                    result["lognormal_geometric"], population.median()
                )
                self.assertAlmostEqual(population.expect(np.log), mu, places=6)
                display = run_cell(
                    1,
                    "lognormal_raw_figure",
                    **(
                        result
                        | {
                            "lognormal_mu": mo.ui.slider(-1, 3, value=mu),
                            "lognormal_sigma": mo.ui.slider(
                                0.1, 1.5, step=0.1, value=sigma
                            ),
                        }
                    ),
                )
                raw_axis = display["lognormal_raw_axis"]
                self.assertEqual(raw_axis.get_xscale(), "linear")
                self.assertAlmostEqual(population.cdf(raw_axis.get_xlim()[1]), 0.99)
                for line, mean in zip(
                    raw_axis.lines,
                    (result["lognormal_geometric"], result["lognormal_arithmetic"]),
                    strict=True,
                ):
                    np.testing.assert_allclose(line.get_xdata(), mean)
                    self.assertLess(mean, raw_axis.get_xlim()[1])
                np.testing.assert_allclose(
                    display["lognormal_log_axis"].lines[0].get_xdata(), mu
                )
                plt.close("all")
            self.assertGreater(result["lognormal_ratio"], previous_ratio)
            previous_ratio = result["lognormal_ratio"]

    def test_precision_plan_is_minimal_and_fourfold_scaling_holds(self):
        for mode in ("Target SE", "Target margin of error"):
            for sd, target, level in ((10, 1, 0.95), (0.5, 5, 0.8), (20, 0.1, 0.99)):
                with self.subTest(mode=mode, sd=sd, target=target, level=level):
                    controls = {
                        "precision_sd": control(sd),
                        "precision_n": control(25),
                        "precision_mode": control(mode),
                        "precision_target": control(target),
                        "precision_confidence": control(level),
                    }
                    plan = run_cell(2, "planning_required_n", **controls)
                    required = plan["planning_required_n"]
                    multiplier = (
                        stats.norm.ppf((1 + level) / 2)
                        if mode == "Target margin of error"
                        else 1
                    )
                    self.assertLessEqual(multiplier * sd / np.sqrt(required), target)
                    if required > 1:
                        self.assertGreater(
                            multiplier * sd / np.sqrt(required - 1), target
                        )
                    larger = run_cell(
                        2,
                        "planning_required_n",
                        **(controls | {"precision_n": control(100)}),
                    )
                    self.assertAlmostEqual(
                        larger["planning_current_se"], plan["planning_current_se"] / 2
                    )
                    plot = run_cell(2, "scaling_figure", **plan)
                    low, high = plot["scaling_axis"].get_xlim()
                    self.assertLess(low, min(25, required))
                    self.assertGreater(high, max(100, required))
        base = {
            "precision_sd": control(10),
            "precision_n": control(25),
            "precision_mode": control("Target SE"),
            "precision_confidence": control(0.95),
        }
        whole = run_cell(2, "planning_required_n", **base, precision_target=control(1))
        half = run_cell(2, "planning_required_n", **base, precision_target=control(0.5))
        self.assertEqual(half["planning_required_n"], 4 * whole["planning_required_n"])

    def test_bootstrap_confidence_reuses_draws_and_matches_same_sample_t_interval(self):
        helpers = run_cell(2, "bootstrap_height_sample")
        values = np.array([140.0, 154.0, 160.0, 162.0, 163.0, 169.0, 190.0])
        for statistic in ("Mean", "Median", "Standard deviation", "IQR"):
            result = helpers["bootstrap_height_sample"](
                values, 1000, statistic, np.random.default_rng(93)
            )
            saved_draws = result["statistics"].copy()
            intervals = []
            for level in (0.80, 0.95, 0.99):
                calculation = run_cell(
                    2,
                    "bootstrap_t_interval",
                    bootstrap_confidence=control(level),
                    get_bootstrap_result=lambda result=result: result,
                    original_height_values=values,
                )
                interval = (
                    calculation["bootstrap_lower"],
                    calculation["bootstrap_upper"],
                )
                np.testing.assert_allclose(
                    interval,
                    np.percentile(saved_draws, [50 * (1 - level), 50 * (1 + level)]),
                )
                self.assertIs(calculation["bootstrap_statistics"], result["statistics"])
                np.testing.assert_array_equal(result["statistics"], saved_draws)
                if statistic == "Mean":
                    expected = stats.t.interval(
                        level,
                        len(values) - 1,
                        loc=values.mean(),
                        scale=stats.sem(values),
                    )
                    np.testing.assert_allclose(
                        calculation["bootstrap_t_interval"], expected
                    )
                else:
                    self.assertIsNone(calculation["bootstrap_t_interval"])
                intervals.append(interval)
            self.assertTrue(
                all(a[0] >= b[0] and a[1] <= b[1] for a, b in pairwise(intervals))
            )

    def test_bootstrap_settings_wait_for_button_and_display_completed_statistic(self):
        helpers = run_cell(2, "bootstrap_height_sample")
        values = np.array([150.0, 158.0, 162.0, 170.0, 190.0])
        runs = []
        controls = run_cell(
            2,
            "run_bootstrap",
            **helpers,
            original_height_values=values,
            set_bootstrap_result=runs.append,
        )
        self.assertEqual(runs, [])
        controls["bootstrap_statistic"]._update(["Median"])
        controls["bootstrap_repetitions"]._update(200)
        self.assertEqual(runs, [])
        controls["run_bootstrap_button"]._on_click(None)
        self.assertEqual(runs[-1]["statistic_name"], "Median")
        self.assertEqual(len(runs[-1]["statistics"]), 200)
        controls["bootstrap_statistic"]._update(["Mean"])
        confidence = run_cell(2, "bootstrap_confidence")["bootstrap_confidence"]
        calculation = run_cell(
            2,
            "bootstrap_t_interval",
            bootstrap_confidence=confidence,
            get_bootstrap_result=lambda: runs[-1],
            original_height_values=values,
        )
        display = run_cell(
            2,
            "bootstrap_figure",
            **(controls | calculation),
        )
        self.assertTrue(display["bootstrap_pending"])
        self.assertIsNone(calculation["bootstrap_t_interval"])
        self.assertIn("median", display["bootstrap_axis"].get_xlabel())
        previous = runs[-1]["statistics"].copy()
        controls["run_bootstrap_button"]._on_click(None)
        self.assertEqual(runs[-1]["statistic_name"], "Mean")
        self.assertFalse(np.array_equal(previous, runs[-1]["statistics"]))

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

    def test_dice_predictive_probabilities_match_known_cases(self):
        update = run_cell(6, "dice_update")["dice_update"]
        sides = [4, 6, 8, 12, 20]
        prior = np.ones(5) / 5
        initial = update(sides, prior, [])
        np.testing.assert_array_equal(initial["next_roll_outcomes"], np.arange(1, 21))
        np.testing.assert_allclose(
            initial["next_roll_probability"],
            [0.135] * 4 + [0.085] * 2 + [31 / 600] * 2 + [2 / 75] * 4 + [0.01] * 8,
        )
        after_seven = update(sides, prior, [7])
        np.testing.assert_allclose(after_seven["posterior"][:2], 0)
        self.assertTrue(np.all(after_seven["next_roll_probability"][:6] > 0))
        self.assertAlmostEqual(after_seven["next_roll_probability"][19], 3 / 310)
        after_twenty = update(sides, prior, [20])
        np.testing.assert_allclose(after_twenty["next_roll_probability"], 0.05)

        for weights in (
            prior,
            [0.10, 0.10, 0.10, 0.15, 0.55],
            [0.35, 0.35, 0.15, 0.10, 0.05],
        ):
            for observations in ([], [6], [6, 4, 7, 7, 8, 2], [20], [1] * 30):
                result = update(sides, weights, observations)
                probabilities = result["next_roll_probability"]
                self.assertAlmostEqual(probabilities.sum(), 1)
                self.assertTrue(np.all(probabilities >= 0))
                # A prediction before a new observation is its next update's evidence.
                for roll, probability in enumerate(probabilities, start=1):
                    next_update = update(sides, weights, [*observations, roll])
                    self.assertAlmostEqual(probability, next_update["evidence"])

    def test_dice_predictive_plot_follows_source_prior_and_reveal(self):
        helpers = run_cell(6, "dice_update")
        cases = [
            ("lecture", "uniform", 0, [20]),
            ("lecture", "uniform", 3, [20]),
            ("lecture", "favor_d20", 6, []),
            ("student", "uniform", 6, []),
            ("student", "favor_small", 0, [6, 7]),
            ("student", "favor_d20", 6, [20]),
        ]
        for source, prior, reveal, rolls in cases:
            with self.subTest(source=source, prior=prior, reveal=reveal, rolls=rolls):
                result = run_cell(
                    6,
                    "dice_predictive_figure",
                    **helpers,
                    dice_sequence_source=mo.ui.radio([source], value=source),
                    dice_prior_choice=mo.ui.radio([prior], value=prior),
                    dice_reveal=mo.ui.slider(0, 6, value=reveal),
                    get_student_dice_rolls=lambda rolls=rolls: rolls,
                    student_dice_controls=mo.md(""),
                )
                expected_rolls = (
                    [6, 4, 7, 7, 8, 2][:reveal] if source == "lecture" else rolls
                )
                self.assertEqual(result["revealed_rolls"], expected_rolls)
                expected = helpers["dice_update"](
                    result["dice_sides"], result["dice_priors"][prior], expected_rolls
                )
                np.testing.assert_allclose(
                    [
                        bar.get_height()
                        for bar in result["dice_predictive_axis"].patches
                    ],
                    expected["next_roll_probability"],
                )
                if not expected_rolls:
                    self.assertIn("No rolls observed yet", result["dice_note"].text)
                    self.assertNotIn("update's evidence", result["dice_note"].text)
                plt.close("all")

    def test_dice_manual_add_undo_reset_preserve_prediction_state(self):
        helpers = run_cell(6, "dice_update")
        rolls = []

        def set_rolls(update):
            nonlocal rolls
            rolls = update(rolls) if callable(update) else update

        controls = run_cell(
            6,
            "student_dice_controls",
            dice_sequence_source=control("student"),
            set_student_dice_rolls=set_rolls,
        )

        def prediction():
            return helpers["dice_update"](
                [4, 6, 8, 12, 20], [0.35, 0.35, 0.15, 0.10, 0.05], rolls
            )["next_roll_probability"]

        initial = prediction()
        controls["student_dice_add"]._on_click(None)
        self.assertEqual(rolls, [6])
        self.assertFalse(np.allclose(initial, prediction()))
        controls["student_dice_undo"]._on_click(None)
        self.assertEqual(rolls, [])
        np.testing.assert_allclose(initial, prediction())
        controls["student_dice_add"]._on_click(None)
        controls["student_dice_add"]._on_click(None)
        controls["student_dice_reset"]._on_click(None)
        self.assertEqual(rolls, [])
        np.testing.assert_allclose(initial, prediction())


if __name__ == "__main__":
    unittest.main()
