# /// script
# requires-python = ">=3.14"
# dependencies = [
#     "marimo>=0.24",
#     "matplotlib>=3.11",
#     "numpy>=2.5",
#     "pandas>=3.0",
#     "scipy>=1.18",
# ]
# ///

import marimo

__generated_with = "0.24.2"
app = marimo.App(
    width="medium",
    app_title="NHST examples",
    css_file="site/assets/notebook.css",
)


@app.cell
def chapter_header():
    from companion_style import notebook_header

    notebook_header(
        5,
        "Null-hypothesis significance testing",
        "Examples: groups, counts, ranks, and analysis of variance",
    )


@app.cell
def _():
    import marimo as mo

    return mo


@app.cell
def _():
    from itertools import combinations
    from math import comb

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from scipy import stats

    from companion_data import read_csv
    from companion_statistics import cohen_d_sample
    from companion_style import (
        COLORS,
        FIGURE_SIZE_LINKED,
        FIGURE_SIZE_SHALLOW,
        FIGURE_SIZE_STANDARD,
        compact_table,
        review_feedback,
        two_column_panel,
    )

    return (
        COLORS,
        FIGURE_SIZE_LINKED,
        FIGURE_SIZE_SHALLOW,
        FIGURE_SIZE_STANDARD,
        cohen_d_sample,
        comb,
        combinations,
        compact_table,
        np,
        pd,
        plt,
        read_csv,
        review_feedback,
        stats,
        two_column_panel,
    )


@app.cell
def _(mo, np, pd, read_csv):
    data_directory = mo.notebook_location() / "public"
    rat_data = read_csv(data_directory / "rat_data.csv", comment="#")
    peroxidase_data = read_csv(data_directory / "peroxidase.csv")
    water_data = read_csv(data_directory / "Wasserqualitaet.csv", encoding="utf-8-sig")

    assert list(rat_data.columns) == ["age", "relax"]
    assert len(rat_data) == 17
    assert set(rat_data["age"]) == {"old", "young"}
    assert rat_data.groupby("age").size().to_dict() == {"old": 9, "young": 8}
    assert np.isfinite(rat_data["relax"]).all()

    assert list(peroxidase_data.columns) == [
        "light_conditions",
        "tissue",
        "peroxidase_amount",
    ]
    assert len(peroxidase_data) == 60
    assert set(peroxidase_data["light_conditions"]) == {"light", "dark"}
    assert set(peroxidase_data["tissue"]) == {
        "root",
        "mesocotyl",
        "primary leaf",
    }
    assert set(peroxidase_data.groupby(["light_conditions", "tissue"]).size()) == {10}
    assert np.isfinite(peroxidase_data["peroxidase_amount"]).all()

    nitrate_column = "Nitratkonzentration (mg/l)"
    phosphate_column = "Phosphatkonzentration (mug/l)"
    assert {nitrate_column, phosphate_column}.issubset(water_data.columns)

    return nitrate_column, peroxidase_data, phosphate_column, rat_data, water_data


@app.cell
def _(comb, combinations, np, stats):
    def welch_mean_difference(group_1, group_2, confidence=0.95):
        sample_1 = np.asarray(group_1, dtype=float)
        sample_2 = np.asarray(group_2, dtype=float)
        n_1, n_2 = sample_1.size, sample_2.size
        variance_1 = float(sample_1.var(ddof=1))
        variance_2 = float(sample_2.var(ddof=1))
        standard_error = np.sqrt(variance_1 / n_1 + variance_2 / n_2)
        difference = float(sample_2.mean() - sample_1.mean())
        statistic = difference / standard_error
        numerator = (variance_1 / n_1 + variance_2 / n_2) ** 2
        denominator = (variance_1 / n_1) ** 2 / (n_1 - 1) + (variance_2 / n_2) ** 2 / (
            n_2 - 1
        )
        degrees_of_freedom = numerator / denominator
        alpha = 1.0 - confidence
        critical = float(stats.t.ppf(1.0 - alpha / 2.0, degrees_of_freedom))
        interval = (
            difference - critical * standard_error,
            difference + critical * standard_error,
        )
        p_value = float(2.0 * stats.t.sf(abs(statistic), degrees_of_freedom))
        return {
            "difference": difference,
            "standard_error": float(standard_error),
            "statistic": float(statistic),
            "degrees_of_freedom": float(degrees_of_freedom),
            "p_value": p_value,
            "interval": tuple(map(float, interval)),
        }

    def exact_mean_difference_permutation(values, group_2_size):
        observations = np.asarray(values, dtype=float)
        total_sum = float(observations.sum())
        group_1_size = observations.size - group_2_size
        differences = np.empty(comb(observations.size, group_2_size), dtype=float)
        for index, group_2_indices in enumerate(
            combinations(range(observations.size), group_2_size)
        ):
            group_2_sum = float(observations[list(group_2_indices)].sum())
            differences[index] = (
                group_2_sum / group_2_size - (total_sum - group_2_sum) / group_1_size
            )
        return differences

    def qq_gallery_sample(pattern, sample_size, seed=82405):
        gallery_rng = np.random.default_rng(seed)
        if pattern == "normal":
            sample = gallery_rng.normal(size=sample_size)
        elif pattern == "skewed":
            sample = gallery_rng.lognormal(mean=0.0, sigma=0.75, size=sample_size)
        elif pattern == "heavy":
            sample = gallery_rng.standard_t(df=3, size=sample_size)
        elif pattern == "bimodal":
            component = gallery_rng.integers(0, 2, size=sample_size)
            sample = gallery_rng.normal(2.8 * component - 1.4, 0.65)
        else:
            sample = gallery_rng.normal(size=sample_size)
            sample[np.argmax(sample)] = 5.5
        return np.asarray(sample, dtype=float)

    def simulate_variance_imbalance(
        small_n, large_n, small_sd, large_sd, repetitions, seed
    ):
        simulation_rng = np.random.default_rng(seed)
        group_small = simulation_rng.normal(0.0, small_sd, size=(repetitions, small_n))
        group_large = simulation_rng.normal(0.0, large_sd, size=(repetitions, large_n))
        welch_p_values = stats.ttest_ind(
            group_small, group_large, axis=1, equal_var=False
        ).pvalue
        pooled_p_values = stats.ttest_ind(
            group_small, group_large, axis=1, equal_var=True
        ).pvalue
        alpha = 0.05
        welch_rate = float(np.mean(welch_p_values <= alpha))
        pooled_rate = float(np.mean(pooled_p_values <= alpha))
        return {
            "small_n": int(small_n),
            "large_n": int(large_n),
            "small_sd": float(small_sd),
            "large_sd": float(large_sd),
            "repetitions": int(repetitions),
            "seed": int(seed),
            "welch_rate": welch_rate,
            "pooled_rate": pooled_rate,
            "welch_mcse": float(np.sqrt(welch_rate * (1.0 - welch_rate) / repetitions)),
            "pooled_mcse": float(
                np.sqrt(pooled_rate * (1.0 - pooled_rate) / repetitions)
            ),
        }

    def one_way_anova_details(frame, group_column, outcome_column, order):
        groups = [
            frame.loc[frame[group_column] == label, outcome_column].to_numpy(
                dtype=float
            )
            for label in order
        ]
        counts = np.array([group.size for group in groups], dtype=int)
        means = np.array([group.mean() for group in groups], dtype=float)
        grand_mean = float(np.concatenate(groups).mean())
        ss_between = float(np.sum(counts * (means - grand_mean) ** 2))
        ss_within = float(sum(np.sum((group - group.mean()) ** 2) for group in groups))
        ss_total = float(sum(np.sum((group - grand_mean) ** 2) for group in groups))
        df_between = len(groups) - 1
        df_within = int(counts.sum() - len(groups))
        ms_between = ss_between / df_between
        ms_within = ss_within / df_within
        statistic = ms_between / ms_within
        p_value = float(stats.f.sf(statistic, df_between, df_within))
        fitted = np.concatenate(
            [np.full(group.size, mean) for group, mean in zip(groups, means)]
        )
        residuals = np.concatenate([group - mean for group, mean in zip(groups, means)])
        assert np.isclose(ss_total, ss_between + ss_within)
        return {
            "groups": groups,
            "counts": counts,
            "means": means,
            "grand_mean": grand_mean,
            "ss_between": ss_between,
            "ss_within": ss_within,
            "ss_total": ss_total,
            "df_between": df_between,
            "df_within": df_within,
            "ms_between": ms_between,
            "ms_within": ms_within,
            "statistic": float(statistic),
            "p_value": p_value,
            "fitted": fitted,
            "residuals": residuals,
        }

    def holm_adjust(p_values):
        raw = np.asarray(p_values, dtype=float)
        order = np.argsort(raw)
        ordered = raw[order]
        scaled = (raw.size - np.arange(raw.size)) * ordered
        adjusted_ordered = np.minimum(1.0, np.maximum.accumulate(scaled))
        adjusted = np.empty_like(adjusted_ordered)
        adjusted[order] = adjusted_ordered
        return adjusted

    def two_way_balanced_details(
        frame, factor_a, factor_b, outcome, levels_a, levels_b
    ):
        values = frame[outcome].to_numpy(dtype=float)
        grand_mean = float(values.mean())
        cell_counts = frame.groupby([factor_a, factor_b]).size()
        assert cell_counts.nunique() == 1
        replicates = int(cell_counts.iloc[0])
        means_a = frame.groupby(factor_a)[outcome].mean()
        means_b = frame.groupby(factor_b)[outcome].mean()
        cell_means = frame.groupby([factor_a, factor_b])[outcome].mean()
        a_count, b_count = len(levels_a), len(levels_b)
        ss_a = float(
            b_count
            * replicates
            * sum((means_a[level] - grand_mean) ** 2 for level in levels_a)
        )
        ss_b = float(
            a_count
            * replicates
            * sum((means_b[level] - grand_mean) ** 2 for level in levels_b)
        )
        ss_interaction = float(
            replicates
            * sum(
                (
                    cell_means[level_a, level_b]
                    - means_a[level_a]
                    - means_b[level_b]
                    + grand_mean
                )
                ** 2
                for level_a in levels_a
                for level_b in levels_b
            )
        )
        ss_error = float(
            sum(
                (row[outcome] - cell_means[row[factor_a], row[factor_b]]) ** 2
                for _, row in frame.iterrows()
            )
        )
        ss_total = float(np.sum((values - grand_mean) ** 2))
        df_a = a_count - 1
        df_b = b_count - 1
        df_interaction = df_a * df_b
        df_error = a_count * b_count * (replicates - 1)
        ms_error = ss_error / df_error
        rows = []
        for source, sum_squares, degrees_of_freedom in [
            ("Illumination", ss_a, df_a),
            ("Tissue", ss_b, df_b),
            ("Interaction", ss_interaction, df_interaction),
        ]:
            mean_square = sum_squares / degrees_of_freedom
            statistic = mean_square / ms_error
            rows.append(
                {
                    "Source": source,
                    "SS": sum_squares,
                    "df": degrees_of_freedom,
                    "MS": mean_square,
                    "F": statistic,
                    "p-value": float(
                        stats.f.sf(statistic, degrees_of_freedom, df_error)
                    ),
                }
            )
        rows.extend(
            [
                {
                    "Source": "Residual",
                    "SS": ss_error,
                    "df": df_error,
                    "MS": ms_error,
                    "F": np.nan,
                    "p-value": np.nan,
                },
                {
                    "Source": "Total",
                    "SS": ss_total,
                    "df": values.size - 1,
                    "MS": np.nan,
                    "F": np.nan,
                    "p-value": np.nan,
                },
            ]
        )
        assert np.isclose(ss_total, ss_a + ss_b + ss_interaction + ss_error)
        return {
            "rows": rows,
            "cell_means": cell_means,
            "grand_mean": grand_mean,
            "residuals": frame.apply(
                lambda row: row[outcome] - cell_means[row[factor_a], row[factor_b]],
                axis=1,
            ).to_numpy(dtype=float),
        }

    return (
        exact_mean_difference_permutation,
        holm_adjust,
        one_way_anova_details,
        qq_gallery_sample,
        simulate_variance_imbalance,
        two_way_balanced_details,
        welch_mean_difference,
    )


@app.cell
def _(mo):
    mo.md(r"""
    Chapter 4 developed the logic of a null model. This chapter applies that logic
    to common biological designs while keeping the **estimated effect, uncertainty,
    raw observations, and assumptions** beside every test result.

    By the end of the chapter, you should be able to:

    1. distinguish independent, paired, correlation, count, and multi-group designs;
    2. report a Welch mean comparison with its confidence interval and measurement units;
    3. diagnose model assumptions without treating a preliminary test as a gatekeeper;
    4. explain what rank and permutation procedures do—and do not—test;
    5. decompose variation in one- and two-way ANOVA;
    6. distinguish omnibus, post-hoc, interaction, equivalence, and multiplicity claims.

    > **Two-minute preview:** Start from the observation unit and estimand. Plot the
    > data, choose a model that respects the design, quantify the effect and its
    > uncertainty, then interpret the p-value under that model. A different design
    > often changes both the calculation and the scientific claim.

    **Prerequisites:** standard errors, confidence intervals, null distributions,
    p-values, type-I error, and power from Chapters 2 and 4.
    """)


@app.cell
def _(mo):
    mo.md(r"""
    ---

    ## 1. Comparing two independent groups

    The rat dataset records maximum relaxation of urinary-bladder muscle strips
    during high-dose norepinephrine treatment. The independent units are rats; the
    groups are old and young animals. We define the estimand in one direction:

    $$\Delta=\mu_{\mathrm{young}}-\mu_{\mathrm{old}},$$

    measured in **percentage points of maximum relaxation**. Positive values mean
    greater average relaxation in young rats. Raw observations come before the test.

    **Start here:** search the table for each age group and compare the individual
    relaxation values before reading the summaries. Follow the same difference
    direction, young minus old, through the estimate, interval, and tests below.
    Sorting or searching only changes the table display; both groups remain in
    the analysis. Compare effect magnitude and uncertainty as well as p-values.
    """)


@app.cell
def _(mo, rat_data, two_column_panel):
    rat_table = mo.ui.table(
        rat_data,
        pagination=False,
        selection=None,
        show_search=True,
        show_column_summaries=True,
        show_data_types=False,
        show_download=True,
        column_widths={"age": 90, "relax": 130},
    )
    rat_provenance = mo.vstack(
        [
            mo.callout(
                mo.md(
                    "**Observed data:** 9 old and 8 young rats. `relax` is the "
                    "maximum relaxation reported as a percentage."
                ),
                kind="info",
            ),
            mo.md(
                "Sort or search the table, but remember that rows are observations—not summaries."
            ),
        ]
    )
    two_column_panel(rat_provenance, rat_table, widths=(1, 1.7))


@app.cell
def _(
    COLORS,
    FIGURE_SIZE_LINKED,
    cohen_d_sample,
    compact_table,
    mo,
    np,
    pd,
    plt,
    rat_data,
    stats,
    two_column_panel,
    welch_mean_difference,
):
    rat_old = rat_data.loc[rat_data["age"] == "old", "relax"].to_numpy(dtype=float)
    rat_young = rat_data.loc[rat_data["age"] == "young", "relax"].to_numpy(dtype=float)
    rat_welch = welch_mean_difference(rat_old, rat_young)
    rat_pooled_test = stats.ttest_ind(rat_young, rat_old, equal_var=True)
    rat_effect_size = cohen_d_sample(rat_old, rat_young)

    assert np.isclose(rat_welch["difference"], 23.545833333333327)
    assert np.isclose(rat_welch["statistic"], 3.6242456851120375)
    assert np.isclose(rat_welch["degrees_of_freedom"], 13.777967606516512)
    assert np.isclose(rat_welch["p_value"], 0.002828426914881653)
    assert np.allclose(rat_welch["interval"], (9.590585553715373, 37.50108111295128))

    rat_figure, (rat_group_axis, rat_difference_axis) = plt.subplots(
        1,
        2,
        figsize=FIGURE_SIZE_LINKED,
        gridspec_kw={"width_ratios": [1.55, 1]},
    )
    rat_jitter_rng = np.random.default_rng(82405)
    for rat_position, (rat_label, rat_values, rat_color) in enumerate(
        [
            ("Old", rat_old, COLORS["orange"]),
            ("Young", rat_young, COLORS["sky"]),
        ]
    ):
        rat_jitter = rat_jitter_rng.normal(0, 0.035, size=rat_values.size)
        rat_group_axis.scatter(
            np.full(rat_values.size, rat_position) + rat_jitter,
            rat_values,
            color=rat_color,
            edgecolor="white",
            linewidth=0.5,
            alpha=0.85,
            s=38,
            label=f"{rat_label}: raw observations",
        )
        rat_group_axis.errorbar(
            rat_position,
            rat_values.mean(),
            yerr=rat_values.std(ddof=1),
            fmt="D",
            capsize=5,
            color=COLORS["blue"],
            linewidth=1.8,
            markersize=5,
        )
    rat_group_axis.set(
        ylabel="Maximum relaxation [%]",
        xticks=[0, 1],
        xticklabels=["Old", "Young"],
        xlim=(-0.45, 1.45),
        ylim=(0, 72),
    )
    rat_group_axis.grid(axis="y", linestyle=":", alpha=0.35)

    rat_lower, rat_upper = rat_welch["interval"]
    rat_difference_axis.errorbar(
        rat_welch["difference"],
        0,
        xerr=[
            [rat_welch["difference"] - rat_lower],
            [rat_upper - rat_welch["difference"]],
        ],
        fmt="o",
        capsize=6,
        color=COLORS["blue"],
        linewidth=2.2,
        label="Estimate and 95% Welch CI",
    )
    rat_difference_axis.axvline(
        0,
        color=COLORS["gray"],
        linestyle="--",
        linewidth=1.5,
        label="Null difference",
    )
    rat_difference_axis.set(
        xlabel="Young − old [percentage points]",
        yticks=[],
        ylim=(-0.55, 0.55),
        xlim=(-8, 45),
    )
    rat_difference_axis.grid(axis="x", linestyle=":", alpha=0.35)
    rat_difference_axis.legend(frameon=False, fontsize=7, loc="upper left")
    rat_figure.tight_layout()

    rat_summary = pd.DataFrame(
        {
            "Group": ["Old", "Young"],
            "n": [rat_old.size, rat_young.size],
            "Mean [%]": [rat_old.mean(), rat_young.mean()],
            "SD [%]": [rat_old.std(ddof=1), rat_young.std(ddof=1)],
        }
    )
    rat_summary_table = compact_table(
        rat_summary,
        column_widths={"Group": 75, "n": 45, "Mean [%]": 78, "SD [%]": 70},
        format_mapping={"Mean [%]": "{:.2f}", "SD [%]": "{:.2f}"},
    )
    rat_test_table = compact_table(
        pd.DataFrame(
            {
                "Method": ["Welch (default)", "Pooled t"],
                "t": [rat_welch["statistic"], float(rat_pooled_test.statistic)],
                "df": [rat_welch["degrees_of_freedom"], float(rat_pooled_test.df)],
                "p-value": [rat_welch["p_value"], float(rat_pooled_test.pvalue)],
            }
        ),
        column_widths={"Method": 110, "t": 58, "df": 62, "p-value": 75},
        format_mapping={"t": "{:.3f}", "df": "{:.2f}", "p-value": "{:.4f}"},
    )
    rat_report = mo.callout(
        mo.md(
            f"Young rats averaged **{rat_welch['difference']:.2f} percentage points** "
            f"more relaxation than old rats (95% CI "
            f"{rat_lower:.2f} to {rat_upper:.2f}); Welch "
            f"$t({rat_welch['degrees_of_freedom']:.2f})={rat_welch['statistic']:.2f}$, "
            f"$p={rat_welch['p_value']:.4f}$. Sample Cohen's "
            f"$d={rat_effect_size:.2f}$."
        ),
        kind="info",
    )
    two_column_panel(
        mo.vstack([rat_summary_table, rat_test_table]),
        mo.vstack([rat_figure, rat_report]),
        widths=(1.15, 2.85),
    )
    return rat_old, rat_welch, rat_young


@app.cell
def _(mo):
    independent_derivation = mo.md(r"""
    ### Derivation: two standard errors for the same estimand

    For independent samples, the Welch standard error is

    $$SE_W=\sqrt{\frac{s_1^2}{n_1}+\frac{s_2^2}{n_2}},\qquad
    t_W=\frac{\bar x_2-\bar x_1}{SE_W}.$$

    Its reference t-distribution uses the Welch–Satterthwaite approximation

    $$\nu=\frac{(s_1^2/n_1+s_2^2/n_2)^2}
    {(s_1^2/n_1)^2/(n_1-1)+(s_2^2/n_2)^2/(n_2-1)}.$$

    If the scientific model instead asserts a common population variance, estimate
    it by

    $$s_p^2=\frac{(n_1-1)s_1^2+(n_2-1)s_2^2}{n_1+n_2-2}$$

    and use $SE_p=s_p\sqrt{1/n_1+1/n_2}$ with $n_1+n_2-2$ degrees of freedom.
    The pooled derivation is useful, but a variance pre-test does not justify choosing
    it after looking at the data. Welch's test is the safer general default.
    """)
    mo.accordion(
        {
            "Derivation — Welch and pooled independent-sample t tests": independent_derivation
        }
    )


@app.cell
def _(mo):
    mo.md(r"""
    ### Randomization under the null

    Under the sharp label-exchangeability null, the 17 observed values can be split
    into groups of 8 and 9 in $\binom{17}{8}=24{,}310$ distinct ways. The exact
    permutation distribution asks where the observed mean difference falls among
    those assignments. Unlike a bootstrap, it shuffles labels **without replacement**
    to model a null hypothesis rather than resampling observations to estimate
    sampling uncertainty.
    """)


@app.cell
def _(
    COLORS,
    FIGURE_SIZE_STANDARD,
    compact_table,
    exact_mean_difference_permutation,
    mo,
    np,
    pd,
    plt,
    rat_data,
    rat_welch,
    rat_young,
    two_column_panel,
):
    rat_permutation_distribution = exact_mean_difference_permutation(
        rat_data["relax"].to_numpy(dtype=float), rat_young.size
    )
    rat_observed_absolute = abs(rat_welch["difference"])
    rat_extreme_count = int(
        np.sum(np.abs(rat_permutation_distribution) >= rat_observed_absolute - 1e-12)
    )
    rat_permutation_p = rat_extreme_count / rat_permutation_distribution.size
    assert rat_permutation_distribution.size == 24_310
    assert rat_extreme_count == 84
    assert np.isclose(rat_permutation_p, 0.003455368161250514)

    permutation_figure, permutation_axis = plt.subplots(figsize=FIGURE_SIZE_STANDARD)
    permutation_axis.hist(
        rat_permutation_distribution,
        bins=48,
        density=True,
        color=COLORS["purple"],
        alpha=0.78,
        edgecolor="white",
        linewidth=0.5,
        label="All label assignments",
    )
    permutation_axis.axvspan(
        permutation_figure.axes[0].get_xlim()[0],
        -rat_observed_absolute,
        color=COLORS["vermillion"],
        alpha=0.20,
    )
    permutation_axis.axvspan(
        rat_observed_absolute,
        permutation_figure.axes[0].get_xlim()[1],
        color=COLORS["vermillion"],
        alpha=0.20,
        label="At least as extreme",
    )
    permutation_axis.axvline(
        rat_welch["difference"],
        color=COLORS["blue"],
        linestyle="--",
        linewidth=2,
        label="Observed young − old",
    )
    permutation_axis.set(
        xlabel="Permuted mean difference [percentage points]",
        ylabel="Density",
        xlim=(-45, 45),
    )
    permutation_axis.grid(axis="y", linestyle=":", alpha=0.35)
    permutation_axis.legend(frameon=False, fontsize=8)
    permutation_figure.tight_layout()

    permutation_summary = compact_table(
        pd.DataFrame(
            {
                "Quantity": ["Assignments", "Extreme assignments", "Exact p-value"],
                "Value": [
                    f"{rat_permutation_distribution.size:,}",
                    str(rat_extreme_count),
                    f"{rat_permutation_p:.6f}",
                ],
            }
        ),
        column_widths={"Quantity": 145, "Value": 110},
    )
    permutation_note = mo.callout(
        mo.md(
            f"Exactly **{rat_extreme_count} of {rat_permutation_distribution.size:,}** "
            f"assignments are at least as extreme as the observed absolute mean "
            f"difference, giving $p={rat_permutation_p:.6f}$. Exchangeability must "
            "be justified by the design; shuffling cannot repair confounding."
        ),
        kind="warn",
    )
    two_column_panel(
        permutation_summary,
        mo.vstack([permutation_figure, permutation_note]),
        widths=(1, 3),
    )


@app.cell
def _(mo):
    mo.md(r"""
    ---

    ## 2. Assumptions and graphical diagnostics

    Independence is primarily a property of how animals, samples, or experimental
    units were selected and assigned. A Q–Q plot can reveal distributional patterns,
    but it cannot diagnose independence. At small sample sizes it has little detail;
    at very large sizes, formal normality tests can flag scientifically trivial
    deviations. Inspect the data and the model discrepancy directly.

    **Try this:** select **Normal reference** with $n=20$, then compare
    **Right-skewed**, **Heavy-tailed**, **Bimodal mixture**, and **One extreme observation**. Match
    the histogram features to bends or isolated points in the Q–Q plot. Repeat
    at $n=8$ and $n=200$: a small sample can hide departures and also look
    irregular by chance. Controls update immediately with reproducible simulated
    examples. Points close to the Q–Q line support approximate distributional
    agreement; they do not establish independence.
    """)


@app.cell
def _(mo):
    qq_pattern = mo.ui.dropdown(
        {
            "Normal reference": "normal",
            "Right-skewed": "skewed",
            "Heavy-tailed": "heavy",
            "Bimodal mixture": "bimodal",
            "One extreme observation": "outlier",
        },
        value="Normal reference",
        label="Distribution pattern",
        full_width=True,
    )
    qq_sample_size = mo.ui.slider(
        8,
        200,
        value=20,
        show_value=True,
        full_width=True,
        label="Sample size n",
    )
    return qq_pattern, qq_sample_size


@app.cell
def _(
    COLORS,
    FIGURE_SIZE_LINKED,
    mo,
    np,
    plt,
    qq_gallery_sample,
    qq_pattern,
    qq_sample_size,
    stats,
    two_column_panel,
):
    qq_values = qq_gallery_sample(qq_pattern.value, int(qq_sample_size.value))
    qq_figure, (qq_hist_axis, qq_axis) = plt.subplots(1, 2, figsize=FIGURE_SIZE_LINKED)
    qq_hist_axis.hist(
        qq_values,
        bins="auto",
        color=COLORS["sky"],
        edgecolor="white",
        alpha=0.8,
    )
    qq_hist_axis.plot(
        qq_values,
        np.full(qq_values.size, -0.015),
        "|",
        color=COLORS["blue"],
        markersize=8,
    )
    qq_hist_axis.set(xlabel="Observed value", ylabel="Count")
    qq_hist_axis.grid(axis="y", linestyle=":", alpha=0.35)

    (qq_theoretical, qq_ordered), (qq_slope, qq_intercept, _qq_r) = stats.probplot(
        qq_values, dist="norm"
    )
    qq_axis.scatter(
        qq_theoretical,
        qq_ordered,
        color=COLORS["orange"],
        edgecolor="white",
        linewidth=0.4,
    )
    qq_axis.plot(
        qq_theoretical,
        qq_intercept + qq_slope * qq_theoretical,
        color=COLORS["gray"],
        linestyle="--",
        linewidth=1.5,
        label="Normal reference line",
    )
    qq_axis.set(xlabel="Theoretical normal quantile", ylabel="Ordered observation")
    qq_axis.grid(linestyle=":", alpha=0.35)
    qq_axis.legend(frameon=False, fontsize=8)
    qq_figure.tight_layout()

    qq_messages = {
        "normal": "Random normal samples do not lie perfectly on a line. Small irregularities are expected.",
        "skewed": "Systematic curvature reflects asymmetric tails rather than one isolated point.",
        "heavy": "Both ends depart from the line because the sample has heavier tails than a normal model.",
        "bimodal": "A Q–Q plot compresses clustering; inspect the raw distribution beside it.",
        "outlier": "One extreme point can influence a mean, variance, fitted model, and test statistic.",
    }
    qq_note = mo.callout(mo.md(qq_messages[qq_pattern.value]), kind="info")
    two_column_panel(
        mo.vstack([qq_pattern, qq_sample_size]),
        mo.vstack([qq_figure, qq_note]),
        widths=(1, 3),
    )


@app.cell
def _(mo):
    mo.md(r"""
    ### Interactive laboratory: variance and sample-size imbalance

    Both simulated groups have population mean zero, so every rejection is a type-I
    error. Change which group is small and variable, then rerun the studies. The
    pooled test can become liberal or conservative when unequal variances align with
    unequal sample sizes; Welch's standard error adapts to both sample variances.

    **Try this:** keep group sizes at 8 and 40, set both SDs to 1, and click
    **Run / resimulate studies**. Compare the rejection rates with the nominal
    5% reference. Raise the small-group SD to 3 and rerun; then exchange the SDs
    so the larger group has SD 3. Compare the pooled and Welch results across
    these scenarios. Both populations have equal means throughout, so these
    are false-positive rates, not power. Settings apply only after the button
    click. More simulated studies reduce Monte Carlo uncertainty but do not
    increase the sample size within either group.
    """)


@app.cell
def _(mo):
    stress_small_n = mo.ui.slider(
        5, 40, value=8, show_value=True, full_width=True, label="Small-group n"
    )
    stress_large_n = mo.ui.slider(
        10, 100, value=40, show_value=True, full_width=True, label="Large-group n"
    )
    stress_small_sd = mo.ui.slider(
        0.5,
        4.0,
        step=0.25,
        value=1.0,
        show_value=True,
        full_width=True,
        label="Small-group SD",
    )
    stress_large_sd = mo.ui.slider(
        0.5,
        4.0,
        step=0.25,
        value=1.0,
        show_value=True,
        full_width=True,
        label="Large-group SD",
    )
    stress_repetitions = mo.ui.slider(
        1_000,
        50_000,
        step=1_000,
        value=10_000,
        show_value=True,
        full_width=True,
        label="Simulated studies",
    )
    return (
        stress_large_n,
        stress_large_sd,
        stress_repetitions,
        stress_small_n,
        stress_small_sd,
    )


@app.cell
def _(mo, simulate_variance_imbalance):
    initial_stress_result = simulate_variance_imbalance(8, 40, 1.0, 1.0, 10_000, 82405)
    get_stress_result, set_stress_result = mo.state(initial_stress_result)
    return get_stress_result, set_stress_result


@app.cell
def _(
    mo,
    np,
    set_stress_result,
    simulate_variance_imbalance,
    stress_large_n,
    stress_large_sd,
    stress_repetitions,
    stress_small_n,
    stress_small_sd,
):
    def run_stress_simulation(_value):
        stress_seed = int(np.random.default_rng().integers(0, 2**32 - 1))
        set_stress_result(
            simulate_variance_imbalance(
                int(stress_small_n.value),
                int(stress_large_n.value),
                float(stress_small_sd.value),
                float(stress_large_sd.value),
                int(stress_repetitions.value),
                stress_seed,
            )
        )

    run_stress_button = mo.ui.button(
        label="Run / resimulate studies",
        kind="success",
        on_click=run_stress_simulation,
        full_width=True,
    )
    return (run_stress_button,)


@app.cell
def _(
    COLORS,
    FIGURE_SIZE_SHALLOW,
    get_stress_result,
    mo,
    np,
    plt,
    run_stress_button,
    stress_large_n,
    stress_large_sd,
    stress_repetitions,
    stress_small_n,
    stress_small_sd,
    two_column_panel,
):
    stress_result = get_stress_result()
    assert 0 <= stress_result["welch_rate"] <= 1
    assert 0 <= stress_result["pooled_rate"] <= 1

    stress_figure, stress_axis = plt.subplots(figsize=FIGURE_SIZE_SHALLOW)
    stress_methods = ["Welch", "Pooled"]
    stress_rates = [stress_result["welch_rate"], stress_result["pooled_rate"]]
    stress_errors = [
        1.96 * stress_result["welch_mcse"],
        1.96 * stress_result["pooled_mcse"],
    ]
    stress_axis.bar(
        stress_methods,
        stress_rates,
        yerr=stress_errors,
        capsize=5,
        color=[COLORS["sky"], COLORS["vermillion"]],
        alpha=0.85,
    )
    stress_axis.axhline(
        0.05,
        color=COLORS["gray"],
        linestyle="--",
        linewidth=1.5,
        label="Nominal α = 0.05",
    )
    stress_axis.set(
        ylabel="Null rejection rate",
        ylim=(0, max(0.12, max(stress_rates) * 1.25)),
    )
    stress_axis.grid(axis="y", linestyle=":", alpha=0.35)
    stress_axis.legend(frameon=False, fontsize=8)
    stress_figure.tight_layout()

    stress_note = mo.callout(
        mo.md(
            f"Last run: small group $n={stress_result['small_n']}$, "
            f"SD {stress_result['small_sd']:.2f}; large group "
            f"$n={stress_result['large_n']}$, SD {stress_result['large_sd']:.2f}; "
            f"{stress_result['repetitions']:,} null studies. Welch rejected in "
            f"**{stress_result['welch_rate']:.1%} ± "
            f"{1.96 * stress_result['welch_mcse']:.1%}** and the pooled test in "
            f"**{stress_result['pooled_rate']:.1%} ± "
            f"{1.96 * stress_result['pooled_mcse']:.1%}** (approximate 95% Monte "
            "Carlo margins)."
        ),
        kind="warn",
    )
    stress_controls = mo.vstack(
        [
            stress_small_n,
            stress_large_n,
            stress_small_sd,
            stress_large_sd,
            stress_repetitions,
            run_stress_button,
            mo.callout(
                mo.md(
                    "Controls describe the next run; the displayed result remains stable until the button is clicked."
                ),
                kind="neutral",
            ),
        ]
    )
    two_column_panel(
        stress_controls,
        mo.vstack([stress_figure, stress_note]),
        widths=(1, 2),
    )


@app.cell
def _(mo):
    mo.md(r"""
    ---

    ## 3. Paired measurements

    Pairing records two measurements from the same independent unit: before and
    after treatment, matched samples, twins, or paired technical conditions. The
    paired t-test is simply a one-sample t-test of

    $$D_i=\text{after}_i-\text{before}_i$$

    against a null mean difference of zero. Its sample size is the number of
    independent **pairs**, not twice that number. Pairing can remove stable
    between-unit variation—but only when identities are preserved.

    **Try this:** start with **Preserve subject identities** and follow each
    line connecting a subject's before and after measurements. Inspect the
    distribution of within-subject differences and its confidence interval.
    Select **Break pairs with a fixed shuffle** and compare the spread and
    paired-test result. The separate before/after distributions and overall
    mean difference stay fixed, but the differences now join unrelated subjects.
    This is a demonstration of invalid pairing, not an alternative analysis
    to choose for a preferable p-value. Switching back restores the original pairs.
    """)


@app.cell
def _(np, pd):
    paired_data = pd.DataFrame(
        {
            "subject": np.arange(1, 13),
            "before": [
                108.4,
                105.6,
                100.6,
                90.1,
                107.7,
                75.0,
                118.1,
                106.1,
                94.7,
                98.9,
                112.8,
                87.8,
            ],
            "after": [
                100.2,
                100.8,
                93.1,
                78.2,
                106.8,
                64.0,
                101.6,
                102.4,
                83.2,
                95.6,
                110.4,
                81.8,
            ],
        }
    )
    paired_shuffle_order = np.array([2, 7, 5, 9, 4, 1, 10, 0, 6, 3, 8, 11])
    assert sorted(paired_shuffle_order.tolist()) == list(range(12))
    return paired_data, paired_shuffle_order


@app.cell
def _(mo):
    pairing_mode = mo.ui.radio(
        {
            "Preserve subject identities": "correct",
            "Break pairs with a fixed shuffle": "shuffled",
        },
        value="Preserve subject identities",
        label="Pairing used in the analysis",
    )
    return (pairing_mode,)


@app.cell
def _(
    COLORS,
    FIGURE_SIZE_LINKED,
    compact_table,
    mo,
    np,
    paired_data,
    paired_shuffle_order,
    pairing_mode,
    pd,
    plt,
    stats,
    two_column_panel,
):
    paired_before = paired_data["before"].to_numpy(dtype=float)
    paired_after_original = paired_data["after"].to_numpy(dtype=float)
    paired_after = (
        paired_after_original
        if pairing_mode.value == "correct"
        else paired_after_original[paired_shuffle_order]
    )
    paired_differences = paired_after - paired_before
    paired_test = stats.ttest_1samp(paired_differences, 0.0)
    independent_test = stats.ttest_ind(
        paired_after_original, paired_before, equal_var=False
    )
    paired_standard_error = float(
        paired_differences.std(ddof=1) / np.sqrt(paired_differences.size)
    )
    paired_critical = float(stats.t.ppf(0.975, paired_differences.size - 1))
    paired_interval = (
        float(paired_differences.mean() - paired_critical * paired_standard_error),
        float(paired_differences.mean() + paired_critical * paired_standard_error),
    )
    if pairing_mode.value == "correct":
        assert np.isclose(paired_test.statistic, -5.42170902551117)
        assert np.isclose(paired_test.pvalue, 0.00020963333759590366)

    paired_figure, (paired_connection_axis, paired_difference_axis) = plt.subplots(
        1,
        2,
        figsize=(5.8, 3.8),
        gridspec_kw={"width_ratios": [1.45, 1]},
    )
    for paired_index, (before_value, after_value) in enumerate(
        zip(paired_before, paired_after)
    ):
        paired_connection_axis.plot(
            [0, 1],
            [before_value, after_value],
            color=COLORS["gray"],
            alpha=0.5,
            linewidth=1,
        )
        paired_connection_axis.scatter(
            [0, 1],
            [before_value, after_value],
            color=[COLORS["orange"], COLORS["sky"]],
            edgecolor="white",
            linewidth=0.4,
            s=30,
            zorder=3,
        )
    paired_connection_axis.set(
        ylabel="Synthetic enzyme activity [a.u.]",
        xticks=[0, 1],
        xticklabels=["Before", "After"],
        xlim=(-0.35, 1.35),
        ylim=(58, 122),
    )
    paired_connection_axis.grid(axis="y", linestyle=":", alpha=0.35)

    paired_difference_axis.scatter(
        np.zeros(paired_differences.size),
        paired_differences,
        color=COLORS["purple"],
        edgecolor="white",
        linewidth=0.4,
        alpha=0.8,
    )
    paired_difference_axis.errorbar(
        0.12,
        paired_differences.mean(),
        yerr=paired_critical * paired_standard_error,
        fmt="D",
        capsize=5,
        color=COLORS["blue"],
        linewidth=2,
        label="Mean and 95% CI",
    )
    paired_difference_axis.axhline(
        0, color=COLORS["gray"], linestyle="--", linewidth=1.4
    )
    paired_difference_axis.set(
        ylabel="After − before [a.u.]",
        xticks=[],
        xlim=(-0.2, 0.32),
        ylim=(-42, 28),
    )
    paired_difference_axis.grid(axis="y", linestyle=":", alpha=0.35)
    paired_difference_axis.legend(frameon=False, fontsize=8)
    paired_figure.tight_layout()

    paired_results_table = compact_table(
        pd.DataFrame(
            {
                "Analysis": [
                    "Displayed pairing",
                    "Ignore pairing (Welch)",
                ],
                "Estimate": [
                    paired_differences.mean(),
                    paired_after_original.mean() - paired_before.mean(),
                ],
                "Statistic": [
                    float(paired_test.statistic),
                    float(independent_test.statistic),
                ],
                "p-value": [float(paired_test.pvalue), float(independent_test.pvalue)],
            }
        ),
        column_widths={
            "Analysis": 145,
            "Estimate": 75,
            "Statistic": 75,
            "p-value": 75,
        },
        format_mapping={
            "Estimate": "{:+.2f}",
            "Statistic": "{:.3f}",
            "p-value": "{:.4f}",
        },
    )
    paired_note_kind = "info" if pairing_mode.value == "correct" else "warn"
    paired_note = mo.callout(
        mo.md(
            f"The displayed mean difference is **{paired_differences.mean():+.2f} a.u.** "
            f"(95% CI {paired_interval[0]:.2f} to {paired_interval[1]:.2f}); "
            f"$t(11)={paired_test.statistic:.2f}$, $p={paired_test.pvalue:.4f}$. "
            + (
                "Preserving identity isolates the consistent within-subject change."
                if pairing_mode.value == "correct"
                else "The group means are unchanged, but arbitrary pair differences inflate uncertainty and answer the wrong design question."
            )
        ),
        kind=paired_note_kind,
    )
    two_column_panel(
        mo.vstack([pairing_mode, paired_results_table]),
        mo.vstack([paired_figure, paired_note]),
        widths=(1, 1),
    )


@app.cell
def _(mo):
    paired_checkpoint = mo.ui.radio(
        {
            "Unmatched animals are individually randomized to treatment or control": "independent",
            "Each patient is measured before and after treatment": "paired",
            "Wells are matched only because they share a plate number": "design",
        },
        value=None,
        label="Which design directly supports a within-specimen paired comparison?",
    )
    mo.vstack(
        [
            mo.md("### Checkpoint: when is pairing part of the estimand?"),
            paired_checkpoint,
        ]
    )
    return (paired_checkpoint,)


@app.cell
def _(paired_checkpoint, review_feedback):
    review_feedback(
        paired_checkpoint.value,
        correct_value="paired",
        correct_text="**Correct.** Each patient supplies a before–after difference. A paired t analysis treats those differences as the observations and requires independence between patients, not between the two measurements within a patient.",
        incorrect_text={
            "independent": "Individually assigned animals without matching provide separate observation "
            "units, not one before–after difference per animal. Use an independent-group "
            "analysis when the design supports independence.",
            "design": "Sharing a plate can create a batch effect, but it does not define which treated well "
            "is paired with which control well. A paired comparison needs a scientifically "
            "specified unit-level match; plate effects may instead require blocking or a "
            "hierarchical model.",
        },
    )


@app.cell
def _(mo):
    mo.md(r"""
    ---

    ## 4. Testing a correlation coefficient

    Chapter 3 showed why a scatter plot and interval must accompany Pearson's
    correlation. Under a bivariate-normal linear model, the null hypothesis
    $H_0:\rho=0$ can be tested with

    $$t=r\sqrt{\frac{n-2}{1-r^2}},\qquad df=n-2.$$

    The same observed correlation produces stronger evidence as sample size grows,
    but neither a small p-value nor a narrow interval establishes causation.

    **Try this:** change the explorer correlation from its observed-data default
    to 0.30 and compare $n=10$,
    24, and 100. Follow the test statistic, interval, and p-value as information
    increases for the same association strength. Then change $r$ from 0.30 to
    −0.30: a two-sided test keeps the same p-value while the direction reverses.
    These controls update a hypothetical calculation immediately; they do not
    alter the observed nitrate–phosphate scatter plot or its reference analysis.
    """)


@app.cell
def _(mo, water_data):
    correlation_r_control = mo.ui.slider(
        -0.95,
        0.95,
        step=0.01,
        value=0.86,
        show_value=True,
        full_width=True,
        label="Explorer correlation r",
    )
    correlation_n_control = mo.ui.slider(
        4,
        200,
        value=len(water_data),
        show_value=True,
        full_width=True,
        label="Explorer sample size n",
    )
    return correlation_n_control, correlation_r_control


@app.cell
def _(
    COLORS,
    FIGURE_SIZE_LINKED,
    compact_table,
    correlation_n_control,
    correlation_r_control,
    mo,
    nitrate_column,
    np,
    pd,
    phosphate_column,
    plt,
    stats,
    two_column_panel,
    water_data,
):
    correlation_x = water_data[nitrate_column].to_numpy(dtype=float)
    correlation_y = water_data[phosphate_column].to_numpy(dtype=float)
    correlation_observed = stats.pearsonr(correlation_x, correlation_y)
    correlation_n_observed = correlation_x.size
    correlation_t_observed = float(
        correlation_observed.statistic
        * np.sqrt(
            (correlation_n_observed - 2) / (1.0 - correlation_observed.statistic**2)
        )
    )
    correlation_fisher_center = float(np.arctanh(correlation_observed.statistic))
    correlation_fisher_margin = float(
        stats.norm.ppf(0.975) / np.sqrt(correlation_n_observed - 3)
    )
    correlation_interval = tuple(
        np.tanh(
            [
                correlation_fisher_center - correlation_fisher_margin,
                correlation_fisher_center + correlation_fisher_margin,
            ]
        )
    )
    assert np.isclose(correlation_observed.statistic, 0.862420484136013)
    assert np.isclose(correlation_observed.pvalue, 6.011470439278001e-08)

    explorer_r = float(correlation_r_control.value)
    explorer_n = int(correlation_n_control.value)
    explorer_t = float(explorer_r * np.sqrt((explorer_n - 2) / (1 - explorer_r**2)))
    explorer_p = float(2 * stats.t.sf(abs(explorer_t), explorer_n - 2))

    correlation_figure, (correlation_scatter_axis, correlation_tail_axis) = (
        plt.subplots(1, 2, figsize=FIGURE_SIZE_LINKED)
    )
    correlation_scatter_axis.scatter(
        correlation_x,
        correlation_y,
        color=COLORS["sky"],
        edgecolor="white",
        linewidth=0.5,
        s=36,
        alpha=0.85,
    )
    correlation_slope, correlation_intercept = np.polyfit(
        correlation_x, correlation_y, 1
    )
    correlation_grid = np.linspace(correlation_x.min(), correlation_x.max(), 100)
    correlation_scatter_axis.plot(
        correlation_grid,
        correlation_intercept + correlation_slope * correlation_grid,
        color=COLORS["blue"],
        linewidth=2,
        label="Least-squares line",
    )
    correlation_scatter_axis.set(
        xlabel="Nitrate concentration [mg/L]",
        ylabel="Phosphate concentration [µg/L]",
    )
    correlation_scatter_axis.grid(linestyle=":", alpha=0.35)
    correlation_scatter_axis.legend(frameon=False, fontsize=8)

    correlation_t_grid = np.linspace(-8, 8, 1_000)
    correlation_t_density = stats.t.pdf(correlation_t_grid, explorer_n - 2)
    correlation_tail_axis.plot(
        correlation_t_grid,
        correlation_t_density,
        color=COLORS["purple"],
        linewidth=2,
        label=f"t reference (df = {explorer_n - 2})",
    )
    correlation_tail_axis.fill_between(
        correlation_t_grid,
        0,
        correlation_t_density,
        where=np.abs(correlation_t_grid) >= abs(explorer_t),
        color=COLORS["vermillion"],
        alpha=0.35,
        label="Two-sided p-value",
    )
    correlation_tail_axis.axvline(
        explorer_t,
        color=COLORS["blue"],
        linestyle="--",
        linewidth=1.7,
        label=f"Selected t = {explorer_t:.2f}",
    )
    correlation_tail_axis.set(
        xlabel="Correlation t statistic",
        ylabel="Density",
        xlim=(-8, 8),
        ylim=(0, 0.44),
    )
    correlation_tail_axis.grid(axis="y", linestyle=":", alpha=0.35)
    correlation_tail_axis.legend(frameon=False, fontsize=7)
    correlation_figure.tight_layout()

    correlation_table = compact_table(
        pd.DataFrame(
            {
                "Quantity": [
                    "Observed rivers",
                    "Pearson r",
                    "t (df)",
                    "p-value",
                    "95% Fisher CI",
                ],
                "Value": [
                    str(correlation_n_observed),
                    f"{correlation_observed.statistic:.3f}",
                    f"{correlation_t_observed:.3f} ({correlation_n_observed - 2})",
                    f"{correlation_observed.pvalue:.2e}",
                    f"[{correlation_interval[0]:.3f}, {correlation_interval[1]:.3f}]",
                ],
            }
        ),
        column_widths={"Quantity": 120, "Value": 125},
    )
    correlation_note = mo.callout(
        mo.md(
            f"Explorer: $r={explorer_r:.2f}$ and $n={explorer_n}$ imply "
            f"$t={explorer_t:.2f}$ and $p={explorer_p:.4g}$. The river result is "
            "an association; sampling design, measurement, nonlinear structure, and confounding still govern its scientific interpretation."
        ),
        kind="warn",
    )
    two_column_panel(
        mo.vstack([correlation_r_control, correlation_n_control, correlation_table]),
        mo.vstack([correlation_figure, correlation_note]),
        widths=(1.15, 2.85),
    )


@app.cell
def _(mo):
    mo.md(r"""
    ---

    ## 5. Counts, expected frequencies, and the chi-squared test

    Mendel observed four pea phenotypes in counts $315,108,101,32$. A dihybrid
    segregation model predicts proportions $9:3:3:1$. Expected **counts** are the
    total count multiplied by those proportions. Each category contributes

    $$\frac{(O_i-E_i)^2}{E_i}$$

    to the goodness-of-fit statistic. With four fixed category probabilities and no
    fitted parameters, $df=4-1=3$.
    """)


@app.cell
def _(
    COLORS,
    FIGURE_SIZE_STANDARD,
    compact_table,
    mo,
    np,
    pd,
    plt,
    stats,
    two_column_panel,
):
    mendel_labels = [
        "Round yellow",
        "Round green",
        "Wrinkled yellow",
        "Wrinkled green",
    ]
    mendel_observed = np.array([315, 108, 101, 32], dtype=float)
    mendel_probabilities = np.array([9, 3, 3, 1], dtype=float) / 16
    mendel_expected = mendel_observed.sum() * mendel_probabilities
    mendel_contributions = (mendel_observed - mendel_expected) ** 2 / mendel_expected
    mendel_result = stats.chisquare(mendel_observed, mendel_expected)
    assert np.isclose(mendel_result.statistic, 0.4700239808153477)
    assert np.isclose(mendel_result.pvalue, 0.925425895103616)

    mendel_table = compact_table(
        pd.DataFrame(
            {
                "Phenotype": mendel_labels,
                "Observed": mendel_observed.astype(int),
                "Expected": mendel_expected,
                "Contribution": mendel_contributions,
            }
        ),
        column_widths={
            "Phenotype": 135,
            "Observed": 70,
            "Expected": 70,
            "Contribution": 85,
        },
        format_mapping={"Expected": "{:.2f}", "Contribution": "{:.3f}"},
    )

    mendel_figure, mendel_axis = plt.subplots(figsize=FIGURE_SIZE_STANDARD)
    mendel_positions = np.arange(len(mendel_labels))
    mendel_width = 0.36
    mendel_axis.bar(
        mendel_positions - mendel_width / 2,
        mendel_observed,
        width=mendel_width,
        color=COLORS["orange"],
        label="Observed counts",
    )
    mendel_axis.bar(
        mendel_positions + mendel_width / 2,
        mendel_expected,
        width=mendel_width,
        color=COLORS["sky"],
        label="Expected under 9:3:3:1",
    )
    for mendel_position, mendel_contribution in zip(
        mendel_positions, mendel_contributions
    ):
        mendel_axis.text(
            mendel_position,
            max(mendel_observed[mendel_position], mendel_expected[mendel_position])
            + 12,
            f"χ² part {mendel_contribution:.3f}",
            ha="center",
            fontsize=7,
        )
    mendel_axis.set(
        ylabel="Pea count",
        xticks=mendel_positions,
        xticklabels=[
            "Round\nyellow",
            "Round\ngreen",
            "Wrinkled\nyellow",
            "Wrinkled\ngreen",
        ],
        ylim=(0, 365),
    )
    mendel_axis.grid(axis="y", linestyle=":", alpha=0.35)
    mendel_axis.legend(frameon=False, fontsize=8)
    mendel_figure.tight_layout()

    expected_count_note = (
        "All expected counts exceed five, so the usual chi-squared approximation is reasonable here."
        " For datasets with small expected counts, consider an exact or simulation-based calibration."
    )
    mendel_note = mo.callout(
        mo.md(
            rf"$\chi^2(3)={mendel_result.statistic:.3f}$, "
            f"$p={mendel_result.pvalue:.3f}$. {expected_count_note} Percentages alone "
            "cannot replace counts because the sampling variability depends on the total sample size."
        ),
        kind="info",
    )
    two_column_panel(
        mendel_table,
        mo.vstack([mendel_figure, mendel_note]),
        widths=(1.45, 2.55),
    )


@app.cell
def _(mo):
    mo.md(r"""
    ---

    ## 6. Rank-based and permutation alternatives

    Mann–Whitney compares the relative ordering of independent observations;
    Wilcoxon signed-rank uses within-pair differences. Their null hypotheses are not
    automatically “equal means,” and neither is assumption-free. A label-permutation
    test targets a chosen statistic under exchangeability. These methods complement,
    rather than mechanically replace, a mean comparison.

    **Try this:** move the largest young-rat observation from 65.5 towards 100
    and then 150. The upper end is deliberately unrealistic for a percentage,
    illustrating sensitivity rather than a plausible measurement. Compare the
    mean-based results with Mann–Whitney: once that observation is the largest,
    increasing its magnitude does not further increase its rank. Use the design
    question to identify the design needed for the Mann–Whitney comparison
    shown here. The feedback also explains the paired alternative; it does not switch the
    analysis of the rat data. Only this activity uses the modified observation.
    """)


@app.cell
def _(mo):
    rank_outlier_value = mo.ui.slider(
        65.5,
        150.0,
        step=0.5,
        value=65.5,
        show_value=True,
        full_width=True,
        label="Largest young-rat observation [%]",
    )
    rank_design = mo.ui.radio(
        {
            "Independent animals in two groups": "mann_whitney",
            "Before/after measurements on the same animals": "wilcoxon",
            "Only two group-average percentages, without unit-level measurements": "invalid",
        },
        value=None,
        label="Which data support the independent-group Mann–Whitney comparison shown here?",
    )
    return rank_design, rank_outlier_value


@app.cell
def _(
    COLORS,
    FIGURE_SIZE_SHALLOW,
    compact_table,
    exact_mean_difference_permutation,
    mo,
    np,
    pd,
    plt,
    rank_design,
    rank_outlier_value,
    rat_old,
    rat_young,
    review_feedback,
    stats,
    two_column_panel,
):
    rank_young_changed = rat_young.copy()
    rank_young_changed[np.argmax(rank_young_changed)] = float(rank_outlier_value.value)
    rank_welch = stats.ttest_ind(rank_young_changed, rat_old, equal_var=False)
    rank_mann_whitney = stats.mannwhitneyu(
        rank_young_changed, rat_old, alternative="two-sided"
    )
    rank_permutation_distribution = exact_mean_difference_permutation(
        np.concatenate([rat_old, rank_young_changed]), rank_young_changed.size
    )
    rank_observed_difference = float(rank_young_changed.mean() - rat_old.mean())
    rank_permutation_p = float(
        np.mean(
            np.abs(rank_permutation_distribution)
            >= abs(rank_observed_difference) - 1e-12
        )
    )

    rank_table = compact_table(
        pd.DataFrame(
            {
                "Method": [
                    "Welch means",
                    "Mann–Whitney ranks",
                    "Exact label permutation",
                ],
                "Target summary": [
                    "Mean difference",
                    "Relative ordering",
                    "Mean difference",
                ],
                "p-value": [
                    float(rank_welch.pvalue),
                    float(rank_mann_whitney.pvalue),
                    rank_permutation_p,
                ],
            }
        ),
        column_widths={"Method": 145, "Target summary": 120, "p-value": 75},
        format_mapping={"p-value": "{:.4f}"},
    )

    rank_figure, rank_axis = plt.subplots(figsize=FIGURE_SIZE_SHALLOW)
    rank_axis.scatter(
        rat_old,
        np.zeros(rat_old.size),
        color=COLORS["orange"],
        label="Old rats",
        alpha=0.8,
        edgecolor="white",
    )
    rank_axis.scatter(
        rank_young_changed,
        np.ones(rank_young_changed.size),
        color=COLORS["sky"],
        label="Young rats",
        alpha=0.8,
        edgecolor="white",
    )
    rank_axis.set(
        xlabel="Maximum relaxation [%]",
        yticks=[0, 1],
        yticklabels=["Old", "Young"],
        ylim=(-0.55, 1.55),
        xlim=(0, 158),
    )
    rank_axis.grid(axis="x", linestyle=":", alpha=0.35)
    rank_axis.legend(frameon=False, fontsize=8)
    rank_figure.tight_layout()

    rank_design_feedback = review_feedback(
        rank_design.value,
        correct_value="mann_whitney",
        correct_text="**Correct.** Mann–Whitney ranks unit-level measurements from two independent groups. It compares relative ordering; interpreting it solely as a location or median shift requires additional distributional assumptions.",
        incorrect_text={
            "wilcoxon": "Before and after measurements on the same animals are paired, so an independent-group Mann–Whitney analysis would ignore their link. Wilcoxon signed-rank works on within-animal differences and, for a location interpretation, assumes their distribution is symmetric.",
            "invalid": "Two group averages do not provide the individual observations needed to construct and compare ranks. Percentage-valued measurements can be ranked if they are recorded for independent units; the problem here is missing unit-level data, not the percentage scale itself.",
        },
        unanswered_text="Choose a design above.",
    )
    rank_note = mo.callout(
        mo.md(
            f"The modified young-group mean difference is "
            f"**{rank_observed_difference:+.2f} percentage points**. Moving one "
            "already-largest observation changes metric distances but barely changes "
            "its rank. That robustness comes with a different target of inference."
        ),
        kind="warn",
    )
    two_column_panel(
        mo.vstack([rank_outlier_value, rank_design, rank_design_feedback, rank_table]),
        mo.vstack([rank_figure, rank_note]),
        widths=(1.35, 2.65),
    )


@app.cell
def _(mo):
    equivalence_estimate = mo.ui.slider(
        -3.0,
        3.0,
        step=0.1,
        value=0.2,
        show_value=True,
        full_width=True,
        label="Estimated difference",
    )
    equivalence_se = mo.ui.slider(
        0.1,
        2.0,
        step=0.1,
        value=0.6,
        show_value=True,
        full_width=True,
        label="Standard error",
    )
    equivalence_margin = mo.ui.slider(
        0.5,
        4.0,
        step=0.25,
        value=2.0,
        show_value=True,
        full_width=True,
        label="Equivalence margin ±M",
    )
    return equivalence_estimate, equivalence_margin, equivalence_se


@app.cell
def _(
    COLORS,
    FIGURE_SIZE_SHALLOW,
    equivalence_estimate,
    equivalence_margin,
    equivalence_se,
    mo,
    plt,
    stats,
    two_column_panel,
):
    equivalence_estimate_value = float(equivalence_estimate.value)
    equivalence_se_value = float(equivalence_se.value)
    equivalence_margin_value = float(equivalence_margin.value)
    equivalence_critical = float(stats.norm.ppf(0.95))
    equivalence_interval = (
        equivalence_estimate_value - equivalence_critical * equivalence_se_value,
        equivalence_estimate_value + equivalence_critical * equivalence_se_value,
    )
    equivalence_established = bool(
        equivalence_interval[0] > -equivalence_margin_value
        and equivalence_interval[1] < equivalence_margin_value
    )
    noninferiority_established = bool(
        equivalence_interval[0] > -equivalence_margin_value
    )

    equivalence_figure, equivalence_axis = plt.subplots(figsize=FIGURE_SIZE_SHALLOW)
    equivalence_axis.axvspan(
        -equivalence_margin_value,
        equivalence_margin_value,
        color=COLORS["green"],
        alpha=0.16,
        label="Pre-specified equivalence region",
    )
    equivalence_axis.axvline(0, color=COLORS["gray"], linestyle="--", linewidth=1.3)
    equivalence_axis.axvline(
        -equivalence_margin_value,
        color=COLORS["vermillion"],
        linestyle=":",
        linewidth=1.8,
    )
    equivalence_axis.axvline(
        equivalence_margin_value,
        color=COLORS["vermillion"],
        linestyle=":",
        linewidth=1.8,
    )
    equivalence_axis.errorbar(
        equivalence_estimate_value,
        0,
        xerr=[
            [equivalence_estimate_value - equivalence_interval[0]],
            [equivalence_interval[1] - equivalence_estimate_value],
        ],
        fmt="o",
        capsize=6,
        linewidth=2.3,
        color=COLORS["blue"],
        label="Estimate and matching 90% CI",
    )
    equivalence_axis.set(
        xlabel="New − reference effect",
        yticks=[],
        ylim=(-0.55, 0.55),
        xlim=(-7, 7),
    )
    equivalence_axis.grid(axis="x", linestyle=":", alpha=0.35)
    equivalence_axis.legend(frameon=False, fontsize=8, loc="upper left")
    equivalence_figure.tight_layout()

    equivalence_conclusion = (
        "equivalence established"
        if equivalence_established
        else "non-inferiority only"
        if noninferiority_established
        else "inconclusive for both equivalence and non-inferiority"
    )
    equivalence_note = mo.callout(
        mo.md(
            f"The illustrative normal-approximation 90% interval is "
            f"**[{equivalence_interval[0]:.2f}, {equivalence_interval[1]:.2f}]**: "
            f"**{equivalence_conclusion}**. The margin must be scientifically "
            "justified before the data are examined."
        ),
        kind="info" if equivalence_established else "warn",
    )
    equivalence_panel = two_column_panel(
        mo.vstack([equivalence_estimate, equivalence_se, equivalence_margin]),
        mo.vstack([equivalence_figure, equivalence_note]),
        widths=(1, 2.5),
    )
    equivalence_content = mo.vstack(
        [
            mo.md(r"""
            A non-significant difference test does not demonstrate similarity.
            Equivalence uses two one-sided tests (TOST) against pre-specified lower
            and upper margins. At $\alpha=0.05$, both tests reject exactly when the
            matching 90% confidence interval lies entirely inside the equivalence
            region. Non-inferiority needs only the relevant one-sided bound.

            **Try this:** hold the margin at ±2 and the estimate at 0.2, then
            increase the SE from 0.6 to 1.5. The interval widens beyond the
            acceptable region, so a small estimated difference alone cannot
            establish equivalence. Restore SE to 0.6 and move the estimate
            towards the positive margin: the lower bound can still establish
            non-inferiority while the upper bound prevents equivalence.
            Here larger effects are assumed preferable, so non-inferiority
            compares the lower bound with −M. The margin control explores
            different pre-specified scientific tolerances; it is not a tool
            for moving a boundary after seeing results. Updates are immediate.
            """),
            equivalence_panel,
        ]
    )
    mo.accordion(
        {"Optional extension — equivalence and non-inferiority": equivalence_content}
    )


@app.cell
def _(mo):
    mo.md(r"""
    ---

    ## 7. One-way ANOVA as a linear model

    A one-way ANOVA asks whether at least one population mean differs across levels
    of one categorical factor. For observation $i$ in tissue group $j$,

    $$Y_{ij}=\mu+\alpha_j+\varepsilon_{ij}.$$

    The fitted value is the group mean and the residual is the observation minus
    that mean. ANOVA compares between-group variation with residual variation; an
    omnibus result does not say that every pair of means differs.

    **Explore the table:** page through the measurements and identify the two
    illumination conditions and three tissues. Then compare **Light-grown**
    and **Dark-grown seedlings** in the one-way ANOVA controls below. Each
    selection analyzes three tissues within one illumination condition and
    updates the downstream one-way diagnostics and post-hoc comparisons.
    Switch the highlighted variation between **Between groups** and **Within
    groups (residual)**: this changes the displayed deviations, not the fitted
    ANOVA or its p-value. Use the sums of squares to explain how separated group
    means and scattered observations contribute differently to the F statistic.
    Searching the raw table does not filter these calculations.
    """)


@app.cell
def _(mo, peroxidase_data, two_column_panel):
    peroxidase_table = mo.ui.table(
        peroxidase_data,
        pagination=True,
        page_size=10,
        selection=None,
        show_search=True,
        show_column_summaries=True,
        show_data_types=False,
        show_download=True,
        column_widths={
            "light_conditions": 130,
            "tissue": 125,
            "peroxidase_amount": 155,
        },
    )
    peroxidase_provenance = mo.vstack(
        [
            mo.callout(
                mo.md(
                    "**Course dataset:** peroxidase amount was measured in root, mesocotyl, and primary-leaf tissue under light and dark conditions. Each of the six cells contains 10 observations."
                ),
                kind="info",
            ),
            mo.md(
                "One-way ANOVA uses one illumination condition at a time; two-way ANOVA later uses the full balanced design."
            ),
        ]
    )
    two_column_panel(peroxidase_provenance, peroxidase_table, widths=(1, 1.7))


@app.cell
def _(mo):
    anova_illumination = mo.ui.radio(
        {"Light-grown seedlings": "light", "Dark-grown seedlings": "dark"},
        value="Light-grown seedlings",
        label="Illumination subset",
    )
    anova_source = mo.ui.radio(
        {
            "Between groups": "between",
            "Within groups (residual)": "within",
        },
        value="Between groups",
        label="Variation to highlight",
    )
    return anova_illumination, anova_source


@app.cell
def _(
    COLORS,
    FIGURE_SIZE_STANDARD,
    anova_illumination,
    anova_source,
    compact_table,
    mo,
    np,
    one_way_anova_details,
    pd,
    peroxidase_data,
    plt,
    two_column_panel,
):
    tissue_order = ["root", "mesocotyl", "primary leaf"]
    selected_peroxidase = peroxidase_data.loc[
        peroxidase_data["light_conditions"] == anova_illumination.value
    ].copy()
    selected_anova = one_way_anova_details(
        selected_peroxidase, "tissue", "peroxidase_amount", tissue_order
    )
    if anova_illumination.value == "light":
        assert np.isclose(selected_anova["statistic"], 22.438016528925633)
        assert np.isclose(selected_anova["p_value"], 1.817713653668423e-06)

    anova_figure, anova_axis = plt.subplots(figsize=(5.2, 3.8))
    anova_jitter_rng = np.random.default_rng(82405)
    for _anova_tissue_index, (tissue_label, tissue_values, tissue_mean) in enumerate(
        zip(tissue_order, selected_anova["groups"], selected_anova["means"])
    ):
        tissue_jitter = anova_jitter_rng.normal(0, 0.045, size=tissue_values.size)
        anova_axis.scatter(
            np.full(tissue_values.size, _anova_tissue_index) + tissue_jitter,
            tissue_values,
            color=[COLORS["orange"], COLORS["sky"], COLORS["purple"]][
                _anova_tissue_index
            ],
            edgecolor="white",
            linewidth=0.5,
            alpha=0.82,
            s=38,
            label=f"{tissue_label}: raw values",
            zorder=3,
        )
        anova_axis.scatter(
            _anova_tissue_index,
            tissue_mean,
            marker="D",
            color=COLORS["blue"],
            s=42,
            zorder=5,
        )
        if anova_source.value == "between":
            anova_axis.plot(
                [_anova_tissue_index, _anova_tissue_index],
                [selected_anova["grand_mean"], tissue_mean],
                color=COLORS["vermillion"],
                linewidth=3,
                alpha=0.75,
            )
        else:
            for x_value, observation in zip(
                np.full(tissue_values.size, _anova_tissue_index) + tissue_jitter,
                tissue_values,
            ):
                anova_axis.plot(
                    [x_value, _anova_tissue_index],
                    [observation, tissue_mean],
                    color=COLORS["gray"],
                    linewidth=1,
                    alpha=0.65,
                    zorder=1,
                )
    anova_axis.axhline(
        selected_anova["grand_mean"],
        color=COLORS["green"],
        linestyle="--",
        linewidth=1.6,
        label="Grand mean",
    )
    anova_axis.set(
        ylabel="Peroxidase amount [a.u.]",
        xticks=np.arange(3),
        xticklabels=["Root", "Mesocotyl", "Primary leaf"],
        xlim=(-0.45, 2.45),
        ylim=(0, 3.65),
    )
    anova_axis.grid(axis="y", linestyle=":", alpha=0.35)
    anova_axis.legend(frameon=False, fontsize=7, ncols=2, loc="upper left")
    anova_figure.tight_layout()

    anova_table_data = pd.DataFrame(
        {
            "Source": ["Between tissues", "Within tissues", "Total"],
            "SS": [
                selected_anova["ss_between"],
                selected_anova["ss_within"],
                selected_anova["ss_total"],
            ],
            "df": [
                selected_anova["df_between"],
                selected_anova["df_within"],
                selected_anova["df_between"] + selected_anova["df_within"],
            ],
            "MS": [
                selected_anova["ms_between"],
                selected_anova["ms_within"],
                np.nan,
            ],
            "F": [selected_anova["statistic"], np.nan, np.nan],
            "p-value": [selected_anova["p_value"], np.nan, np.nan],
        }
    )
    anova_table = compact_table(
        anova_table_data,
        column_widths={
            "Source": 120,
            "SS": 58,
            "df": 45,
            "MS": 58,
            "F": 58,
            "p-value": 78,
        },
        format_mapping={
            "SS": "{:.3f}",
            "MS": "{:.3f}",
            "F": "{:.3f}",
            "p-value": "{:.3g}",
        },
    )
    anova_report = mo.callout(
        mo.md(
            f"For **{anova_illumination.value}**, $F({selected_anova['df_between']},"
            f"{selected_anova['df_within']})={selected_anova['statistic']:.3f}$, "
            f"$p={selected_anova['p_value']:.3g}$. This is evidence that at least "
            "one tissue mean differs; it is not yet a list of pairwise conclusions."
        ),
        kind="info",
    )
    two_column_panel(
        mo.vstack([anova_illumination, anova_source, anova_table]),
        mo.vstack([anova_figure, anova_report]),
        widths=(1, 1),
    )
    return selected_anova, selected_peroxidase, tissue_order


@app.cell
def _(mo):
    anova_derivation = mo.md(r"""
    ### Derivation: partitioning total variation

    With group sizes $n_j$, group means $\bar y_j$, and grand mean $\bar y$,

    $$SS_T=\sum_{j,i}(y_{ij}-\bar y)^2
    =\underbrace{\sum_j n_j(\bar y_j-\bar y)^2}_{SS_B}
    +\underbrace{\sum_{j,i}(y_{ij}-\bar y_j)^2}_{SS_W}.$$

    Divide each sum of squares by its degrees of freedom:

    $$F=\frac{MS_B}{MS_W}
    =\frac{SS_B/(k-1)}{SS_W/(N-k)}.$$

    Under the common-variance null model, both mean squares estimate the same
    residual variance and $F$ tends to be near one. Separated group means increase
    the numerator. For exactly two groups under the same equal-variance model, the
    tests are algebraically identical: $F_{1,n_1+n_2-2}=t^2_{n_1+n_2-2}$.
    """)
    mo.accordion({"Derivation — ANOVA sums of squares and F = t²": anova_derivation})


@app.cell
def _(mo):
    mo.md(r"""
    ### ANOVA assumptions and residual diagnostics

    The model assumes independent experimental units, approximately normal
    residuals within groups, and a common residual variance. Independence comes from
    the design. Residual plots can expose skew, outliers, and variance patterns, but
    small datasets cannot certify that assumptions are true. Alternatives answer
    related—not always identical—questions.
    """)


@app.cell
def _(
    COLORS,
    FIGURE_SIZE_LINKED,
    compact_table,
    mo,
    np,
    pd,
    plt,
    selected_anova,
    stats,
    tissue_order,
    two_column_panel,
):
    diagnostic_figure, (diagnostic_fitted_axis, diagnostic_qq_axis) = plt.subplots(
        1, 2, figsize=FIGURE_SIZE_LINKED
    )
    diagnostic_fitted_axis.scatter(
        selected_anova["fitted"],
        selected_anova["residuals"],
        color=COLORS["sky"],
        edgecolor="white",
        linewidth=0.5,
        alpha=0.85,
    )
    diagnostic_fitted_axis.axhline(
        0, color=COLORS["gray"], linestyle="--", linewidth=1.4
    )
    diagnostic_fitted_axis.set(
        xlabel="Fitted tissue mean [a.u.]",
        ylabel="Residual [a.u.]",
    )
    diagnostic_fitted_axis.grid(linestyle=":", alpha=0.35)

    (
        (
            diagnostic_theoretical,
            diagnostic_ordered,
        ),
        (
            diagnostic_slope,
            diagnostic_intercept,
            _diagnostic_r,
        ),
    ) = stats.probplot(selected_anova["residuals"], dist="norm")
    diagnostic_qq_axis.scatter(
        diagnostic_theoretical,
        diagnostic_ordered,
        color=COLORS["orange"],
        edgecolor="white",
        linewidth=0.5,
    )
    diagnostic_qq_axis.plot(
        diagnostic_theoretical,
        diagnostic_intercept + diagnostic_slope * diagnostic_theoretical,
        color=COLORS["gray"],
        linestyle="--",
        linewidth=1.5,
    )
    diagnostic_qq_axis.set(
        xlabel="Theoretical normal quantile", ylabel="Ordered residual"
    )
    diagnostic_qq_axis.grid(linestyle=":", alpha=0.35)
    diagnostic_figure.tight_layout()

    diagnostic_classical = stats.f_oneway(*selected_anova["groups"])
    diagnostic_welch = stats.f_oneway(*selected_anova["groups"], equal_var=False)
    diagnostic_kruskal = stats.kruskal(*selected_anova["groups"])
    diagnostic_methods = compact_table(
        pd.DataFrame(
            {
                "Procedure": ["Classical ANOVA", "Welch ANOVA", "Kruskal–Wallis"],
                "Primary target": [
                    "Means, common variance",
                    "Means, unequal variances",
                    "Rank distributions",
                ],
                "Statistic": [
                    float(diagnostic_classical.statistic),
                    float(diagnostic_welch.statistic),
                    float(diagnostic_kruskal.statistic),
                ],
                "p-value": [
                    float(diagnostic_classical.pvalue),
                    float(diagnostic_welch.pvalue),
                    float(diagnostic_kruskal.pvalue),
                ],
            }
        ),
        column_widths={
            "Procedure": 115,
            "Primary target": 155,
            "Statistic": 70,
            "p-value": 75,
        },
        format_mapping={"Statistic": "{:.3f}", "p-value": "{:.3g}"},
    )
    diagnostic_note = mo.callout(
        mo.md(
            f"Displayed groups: **{', '.join(tissue_order)}**. Do not select a "
            "procedure solely because a preliminary normality or variance test crossed 0.05; use design, estimand, balance, diagnostics, and sensitivity analysis together."
        ),
        kind="warn",
    )
    two_column_panel(
        diagnostic_methods,
        mo.vstack([diagnostic_figure, diagnostic_note]),
        widths=(1.45, 2.55),
    )


@app.cell
def _(mo):
    mo.md(r"""
    ---

    ## 8. Multiple comparisons and post-hoc procedures

    A significant omnibus ANOVA says that not all means are equal. It does not
    identify the differing pairs. Testing many pairs at unadjusted $\alpha$ inflates
    the chance of at least one false positive. Bonferroni and Holm control the
    family-wise error rate; Tukey's procedure is designed for all pairwise mean
    comparisons under the ANOVA model.
    """)


@app.cell
def _(
    COLORS,
    FIGURE_SIZE_STANDARD,
    anova_illumination,
    compact_table,
    holm_adjust,
    mo,
    np,
    pd,
    selected_peroxidase,
    plt,
    stats,
    tissue_order,
    two_column_panel,
):
    posthoc_frame = selected_peroxidase
    posthoc_groups = [
        posthoc_frame.loc[
            posthoc_frame["tissue"] == tissue, "peroxidase_amount"
        ].to_numpy(dtype=float)
        for tissue in tissue_order
    ]
    posthoc_pairs = [(0, 1), (0, 2), (1, 2)]
    posthoc_raw_p = np.array(
        [
            stats.ttest_ind(
                posthoc_groups[first], posthoc_groups[second], equal_var=True
            ).pvalue
            for first, second in posthoc_pairs
        ],
        dtype=float,
    )
    posthoc_bonferroni = np.minimum(1.0, posthoc_raw_p * len(posthoc_pairs))
    posthoc_holm = holm_adjust(posthoc_raw_p)
    posthoc_tukey_result = stats.tukey_hsd(*posthoc_groups)
    posthoc_tukey = np.array(
        [posthoc_tukey_result.pvalue[first, second] for first, second in posthoc_pairs],
        dtype=float,
    )
    posthoc_differences = np.array(
        [
            posthoc_groups[second].mean() - posthoc_groups[first].mean()
            for first, second in posthoc_pairs
        ]
    )
    posthoc_labels = [
        f"{tissue_order[second]} − {tissue_order[first]}"
        for first, second in posthoc_pairs
    ]
    posthoc_table = compact_table(
        pd.DataFrame(
            {
                "Comparison": posthoc_labels,
                "Mean difference": posthoc_differences,
                "Raw p": posthoc_raw_p,
                "Bonferroni": posthoc_bonferroni,
                "Holm": posthoc_holm,
                "Tukey": posthoc_tukey,
            }
        ),
        column_widths={
            "Comparison": 175,
            "Mean difference": 95,
            "Raw p": 68,
            "Bonferroni": 82,
            "Holm": 68,
            "Tukey": 68,
        },
        format_mapping={
            "Mean difference": "{:+.2f}",
            "Raw p": "{:.4f}",
            "Bonferroni": "{:.4f}",
            "Holm": "{:.4f}",
            "Tukey": "{:.4f}",
        },
    )

    posthoc_figure, posthoc_axis = plt.subplots(figsize=FIGURE_SIZE_STANDARD)
    posthoc_y = np.arange(len(posthoc_pairs))
    posthoc_axis.axvline(0, color=COLORS["gray"], linestyle="--", linewidth=1.3)
    for row_index, ((first, second), difference) in enumerate(
        zip(posthoc_pairs, posthoc_differences)
    ):
        posthoc_low = posthoc_tukey_result.confidence_interval().low[first, second]
        posthoc_high = posthoc_tukey_result.confidence_interval().high[first, second]
        # scipy reports first - second; reverse to match the displayed second - first.
        display_low, display_high = -posthoc_high, -posthoc_low
        posthoc_axis.errorbar(
            difference,
            row_index,
            xerr=[[difference - display_low], [display_high - difference]],
            fmt="o",
            capsize=5,
            color=COLORS["blue"]
            if posthoc_tukey[row_index] <= 0.05
            else COLORS["gray"],
            linewidth=2,
        )
    posthoc_axis.set(
        title=f"{anova_illumination.value.capitalize()}-grown seedlings",
        xlabel="Pairwise mean difference [a.u.] with Tukey 95% CI",
        yticks=posthoc_y,
        yticklabels=posthoc_labels,
        ylim=(-0.7, len(posthoc_pairs) - 0.3),
    )
    posthoc_axis.grid(axis="x", linestyle=":", alpha=0.35)
    posthoc_figure.tight_layout()

    posthoc_supported = [
        label
        for label, p_value in zip(posthoc_labels, posthoc_tukey)
        if p_value <= 0.05
    ]
    posthoc_conclusion = (
        "Tukey supports differences for: " + "; ".join(posthoc_supported) + "."
        if posthoc_supported
        else "Tukey does not support any pairwise difference at the 5% family-wise level."
    )
    posthoc_note = mo.callout(
        mo.md(
            f"For {anova_illumination.value}-grown seedlings, {posthoc_conclusion} "
            "Other pairwise differences remain uncertain. The adjustment applies "
            "to the pre-defined family of all three tissue comparisons."
        ),
        kind="info",
    )
    two_column_panel(
        posthoc_table,
        mo.vstack([posthoc_figure, posthoc_note]),
        widths=(1.65, 2.35),
    )


@app.cell
def _(mo):
    mo.md(r"""
    ### Explore the chance of at least one false positive

    **Try this:** set the per-test threshold to 0.05 and compare 1, 20, and
    100 independent true-null tests. Read the probability of at least one false
    rejection, $1-(1-\alpha)^m$, rather than interpreting it as the fraction of
    tests expected to reject. Then lower the per-test threshold to 0.01.
    Both controls update the theoretical calculation immediately. This example
    assumes independent tests and does not change the tissue comparisons above;
    those comparisons need their stated multiple-testing procedure.
    """)


@app.cell
def _(mo):
    multiplicity_alpha = mo.ui.slider(
        0.01,
        0.10,
        step=0.01,
        value=0.05,
        show_value=True,
        full_width=True,
        label="Per-test threshold α",
    )
    multiplicity_tests = mo.ui.slider(
        1,
        100,
        value=20,
        show_value=True,
        full_width=True,
        label="Independent true-null tests m",
    )
    return multiplicity_alpha, multiplicity_tests


@app.cell
def _(
    COLORS,
    FIGURE_SIZE_SHALLOW,
    mo,
    multiplicity_alpha,
    multiplicity_tests,
    np,
    plt,
    two_column_panel,
):
    multiplicity_alpha_value = float(multiplicity_alpha.value)
    multiplicity_m_value = int(multiplicity_tests.value)
    multiplicity_fwer = float(
        1 - (1 - multiplicity_alpha_value) ** multiplicity_m_value
    )
    multiplicity_grid = np.arange(1, 101)
    multiplicity_curve = 1 - (1 - multiplicity_alpha_value) ** multiplicity_grid

    multiplicity_figure, multiplicity_axis = plt.subplots(figsize=FIGURE_SIZE_SHALLOW)
    multiplicity_axis.plot(
        multiplicity_grid,
        multiplicity_curve,
        color=COLORS["purple"],
        linewidth=2.2,
        label=r"$1-(1-\alpha)^m$",
    )
    multiplicity_axis.scatter(
        [multiplicity_m_value],
        [multiplicity_fwer],
        color=COLORS["vermillion"],
        s=50,
        zorder=3,
        label="Selected family",
    )
    multiplicity_axis.axhline(
        multiplicity_alpha_value,
        color=COLORS["gray"],
        linestyle="--",
        linewidth=1.3,
        label="Single-test α",
    )
    multiplicity_axis.set(
        xlabel="Number of independent true-null tests m",
        ylabel="Probability of ≥1 false positive",
        xlim=(1, 100),
        ylim=(0, 1.02),
    )
    multiplicity_axis.grid(linestyle=":", alpha=0.35)
    multiplicity_axis.legend(frameon=False, fontsize=8, loc="center right")
    multiplicity_figure.tight_layout()

    multiplicity_note = mo.callout(
        mo.md(
            f"For {multiplicity_m_value} independent true-null tests at "
            f"$\\alpha={multiplicity_alpha_value:.2f}$, the illustrative chance of "
            f"at least one false positive is **{multiplicity_fwer:.1%}**. "
            "Bonferroni and Holm control the probability of any false positive. "
            "Benjamini–Hochberg instead controls the expected false-discovery proportion among rejections, often a more useful goal in omics."
        ),
        kind="warn",
    )
    two_column_panel(
        mo.vstack([multiplicity_alpha, multiplicity_tests]),
        mo.vstack([multiplicity_figure, multiplicity_note]),
        widths=(1, 3),
    )


@app.cell
def _(mo):
    mo.md(r"""
    ---

    ## 9. Two-way ANOVA and interactions

    The full peroxidase design has two factors: illumination and tissue. A two-way
    model decomposes each observation into a grand mean, an illumination effect, a
    tissue effect, their interaction, and a residual:

    $$Y_{ijk}=\mu+\alpha_i+\beta_j+(\alpha\beta)_{ij}+\varepsilon_{ijk}.$$

    An interaction means that the illumination effect depends on tissue. Inspect
    cell means and profiles before reducing the result to isolated main-effect
    p-values.
    """)


@app.cell
def _(
    COLORS,
    FIGURE_SIZE_STANDARD,
    compact_table,
    mo,
    np,
    pd,
    peroxidase_data,
    plt,
    two_column_panel,
    two_way_balanced_details,
):
    two_way_light_levels = ["light", "dark"]
    two_way_tissue_levels = ["root", "mesocotyl", "primary leaf"]
    two_way_result = two_way_balanced_details(
        peroxidase_data,
        "light_conditions",
        "tissue",
        "peroxidase_amount",
        two_way_light_levels,
        two_way_tissue_levels,
    )
    two_way_table_data = pd.DataFrame(two_way_result["rows"])
    two_way_interaction_row = two_way_table_data.loc[
        two_way_table_data["Source"] == "Interaction"
    ].iloc[0]
    assert np.isclose(two_way_interaction_row["F"], 6.105143721633889)
    assert np.isclose(two_way_interaction_row["p-value"], 0.00407046228977902)

    two_way_table = compact_table(
        two_way_table_data,
        column_widths={
            "Source": 100,
            "SS": 62,
            "df": 45,
            "MS": 62,
            "F": 65,
            "p-value": 78,
        },
        format_mapping={
            "SS": "{:.3f}",
            "MS": "{:.3f}",
            "F": "{:.3f}",
            "p-value": "{:.3g}",
        },
    )

    two_way_figure, two_way_axis = plt.subplots(figsize=FIGURE_SIZE_STANDARD)
    two_way_positions = np.arange(len(two_way_tissue_levels))
    two_way_jitter_rng = np.random.default_rng(82405)
    for _two_way_illumination_index, (
        _two_way_illumination,
        _two_way_color,
        _two_way_marker,
    ) in enumerate(
        [
            ("light", COLORS["orange"], "o"),
            ("dark", COLORS["purple"], "s"),
        ]
    ):
        cell_means = []
        for tissue_index, tissue in enumerate(two_way_tissue_levels):
            cell_values = peroxidase_data.loc[
                (peroxidase_data["light_conditions"] == _two_way_illumination)
                & (peroxidase_data["tissue"] == tissue),
                "peroxidase_amount",
            ].to_numpy(dtype=float)
            horizontal_offset = -0.08 if _two_way_illumination == "light" else 0.08
            cell_jitter = two_way_jitter_rng.normal(0, 0.025, cell_values.size)
            two_way_axis.scatter(
                np.full(cell_values.size, tissue_index + horizontal_offset)
                + cell_jitter,
                cell_values,
                color=_two_way_color,
                marker=_two_way_marker,
                alpha=0.35,
                s=24,
            )
            cell_means.append(cell_values.mean())
        two_way_axis.plot(
            two_way_positions,
            cell_means,
            color=_two_way_color,
            marker=_two_way_marker,
            linewidth=2,
            markersize=6,
            label=f"{_two_way_illumination.capitalize()} cell means",
        )
    two_way_axis.set(
        ylabel="Peroxidase amount [a.u.]",
        xticks=two_way_positions,
        xticklabels=["Root", "Mesocotyl", "Primary leaf"],
        xlim=(-0.4, 2.4),
        ylim=(0, 3.65),
    )
    two_way_axis.grid(axis="y", linestyle=":", alpha=0.35)
    two_way_axis.legend(frameon=False, fontsize=8)
    two_way_figure.tight_layout()

    two_way_note = mo.callout(
        mo.md(
            f"The tissue × illumination interaction is "
            f"$F(2,54)={two_way_interaction_row['F']:.3f}$, "
            f"$p={two_way_interaction_row['p-value']:.4f}$. The profiles are not "
            "parallel: the light–dark difference is much larger in primary leaves than in roots. A single overall illumination effect therefore hides biologically relevant dependence on tissue."
        ),
        kind="warn",
    )
    two_column_panel(
        two_way_table,
        mo.vstack([two_way_figure, two_way_note]),
        widths=(1.35, 2.65),
    )


@app.cell
def _(mo):
    mo.md(r"""
    ### Compare interaction patterns

    **Try this:** start with **Observed peroxidase pattern**, then choose
    **Parallel profiles**. Compare the light–dark difference across tissues:
    a constant difference gives parallel profiles. Try the converging,
    diverging, and crossing examples and identify where the difference changes
    magnitude or sign. Look at the interaction result before interpreting
    averages over tissues. The hypothetical examples replace cell means while
    keeping the observed residuals and sample sizes, isolating the effect of
    the profile shape. Selecting the observed pattern restores the original
    data in this explorer; it does not modify the source dataset.
    """)


@app.cell
def _(mo):
    interaction_pattern = mo.ui.dropdown(
        {
            "Observed peroxidase pattern": "observed",
            "Parallel profiles": "parallel",
            "Converging profiles": "converging",
            "Diverging profiles": "diverging",
            "Crossing profiles": "crossing",
        },
        value="Observed peroxidase pattern",
        label="Cell-mean pattern",
        full_width=True,
    )
    return (interaction_pattern,)


@app.cell
def _(
    COLORS,
    FIGURE_SIZE_STANDARD,
    compact_table,
    interaction_pattern,
    mo,
    np,
    pd,
    peroxidase_data,
    plt,
    two_column_panel,
    two_way_balanced_details,
):
    pattern_light_levels = ["light", "dark"]
    pattern_tissue_levels = ["root", "mesocotyl", "primary leaf"]
    pattern_actual_means = peroxidase_data.groupby(["light_conditions", "tissue"])[
        "peroxidase_amount"
    ].mean()
    pattern_residuals = peroxidase_data.apply(
        lambda row: (
            row["peroxidase_amount"]
            - pattern_actual_means[row["light_conditions"], row["tissue"]]
        ),
        axis=1,
    ).to_numpy(dtype=float)
    pattern_profiles = {
        "parallel": {
            "light": [1.0, 1.5, 2.0],
            "dark": [0.5, 1.0, 1.5],
        },
        "converging": {
            "light": [1.5, 1.7, 1.9],
            "dark": [0.5, 1.0, 1.5],
        },
        "diverging": {
            "light": [0.9, 1.7, 2.5],
            "dark": [0.5, 1.0, 1.5],
        },
        "crossing": {
            "light": [0.6, 1.4, 2.2],
            "dark": [2.2, 1.4, 0.6],
        },
    }
    pattern_frame = peroxidase_data.copy()
    if interaction_pattern.value == "observed":
        pattern_frame["pattern_amount"] = peroxidase_data["peroxidase_amount"]
    else:
        selected_profiles = pattern_profiles[interaction_pattern.value]
        pattern_mean_lookup = {
            (illumination, tissue): selected_profiles[illumination][tissue_index]
            for illumination in pattern_light_levels
            for tissue_index, tissue in enumerate(pattern_tissue_levels)
        }
        pattern_frame["pattern_amount"] = (
            np.array(
                [
                    pattern_mean_lookup[row.light_conditions, row.tissue]
                    for row in pattern_frame.itertuples()
                ]
            )
            + pattern_residuals
        )
    pattern_result = two_way_balanced_details(
        pattern_frame,
        "light_conditions",
        "tissue",
        "pattern_amount",
        pattern_light_levels,
        pattern_tissue_levels,
    )
    pattern_rows = pd.DataFrame(pattern_result["rows"]).iloc[:3]
    pattern_table = compact_table(
        pattern_rows[["Source", "F", "p-value"]],
        column_widths={"Source": 105, "F": 75, "p-value": 85},
        format_mapping={"F": "{:.3f}", "p-value": "{:.3g}"},
    )

    pattern_figure, pattern_axis = plt.subplots(figsize=FIGURE_SIZE_STANDARD)
    pattern_positions = np.arange(3)
    for _pattern_illumination, _pattern_color, _pattern_marker in [
        ("light", COLORS["orange"], "o"),
        ("dark", COLORS["purple"], "s"),
    ]:
        profile_means = (
            pattern_frame.loc[
                pattern_frame["light_conditions"] == _pattern_illumination
            ]
            .groupby("tissue", sort=False)["pattern_amount"]
            .mean()
        )
        ordered_profile_means = [
            profile_means[tissue] for tissue in pattern_tissue_levels
        ]
        pattern_axis.plot(
            pattern_positions,
            ordered_profile_means,
            color=_pattern_color,
            marker=_pattern_marker,
            linewidth=2.2,
            markersize=7,
            label=_pattern_illumination.capitalize(),
        )
    pattern_axis.set(
        ylabel="Cell mean with fixed residuals [a.u.]",
        xticks=pattern_positions,
        xticklabels=["Root", "Mesocotyl", "Primary leaf"],
        xlim=(-0.3, 2.3),
        ylim=(-0.1, 3.2),
    )
    pattern_axis.grid(axis="y", linestyle=":", alpha=0.35)
    pattern_axis.legend(frameon=False, fontsize=8)
    pattern_figure.tight_layout()

    pattern_interaction = pattern_rows.loc[
        pattern_rows["Source"] == "Interaction"
    ].iloc[0]
    pattern_note = mo.callout(
        mo.md(
            f"With the residuals held fixed, the selected profile gives interaction "
            f"$F={pattern_interaction['F']:.3f}$ and "
            f"$p={pattern_interaction['p-value']:.3g}$. Parallel lines encode no "
            "interaction; non-parallel lines show that one factor's effect depends on the other."
        ),
        kind="info",
    )
    two_column_panel(
        mo.vstack([interaction_pattern, pattern_table]),
        mo.vstack([pattern_figure, pattern_note]),
        widths=(1, 3),
    )


@app.cell
def _(mo):
    mo.md(r"""
    ---

    ## 10. End-of-chapter method map

    A method follows from the scientific question, observation unit, outcome type,
    group structure, pairing, and estimand—not from a desire to obtain a particular
    p-value. This map is a starting point for an analysis plan, not a substitute for
    design knowledge and model checking.

    **Try this:** choose a scenario and name the observation unit, outcome type,
    and estimand before reading the explanation. Compare different animals on
    two diets with the same cultures measured twice. The pairing changes which
    variation supplies the standard error, even though both questions involve
    two sets of measurements. Then compare the count and factorial scenarios.
    The selector reveals an analysis-plan suggestion immediately; it does not
    run a new test or decide whether the study design is valid.
    """)


@app.cell
def _(mo):
    method_scenario = mo.ui.dropdown(
        {
            "Different animals assigned to two diets": "independent",
            "The same cultures measured before and after induction": "paired",
            "Four phenotype counts compared with a genetic ratio": "counts",
            "Tissue and illumination studied together": "factorial",
        },
        value="Different animals assigned to two diets",
        label="Analysis-plan scenario",
        full_width=True,
    )
    return (method_scenario,)


@app.cell
def _(compact_table, method_scenario, mo, pd, two_column_panel):
    method_map = compact_table(
        pd.DataFrame(
            {
                "Question/design": [
                    "Two independent group means",
                    "One within-pair mean change",
                    "Linear association",
                    "Categorical count model",
                    "Three or more group means",
                    "Two categorical factors",
                    "Pre-specified similarity margin",
                ],
                "Primary procedure": [
                    "Welch t + CI",
                    "Paired t / one-sample differences",
                    "Pearson r, Fisher CI, correlation t",
                    "Chi-squared or exact/simulation test",
                    "One-way ANOVA; Welch if needed",
                    "Two-way ANOVA with interaction",
                    "TOST / non-inferiority test",
                ],
                "Plot first": [
                    "Raw group distributions",
                    "Connected observations + differences",
                    "Scatter plot",
                    "Observed and expected counts",
                    "Raw points + group means",
                    "Cell means + interaction profiles",
                    "Estimate, interval, and margins",
                ],
            }
        ),
        column_widths={
            "Question/design": 190,
            "Primary procedure": 225,
            "Plot first": 210,
        },
        wrapped_columns=["Question/design", "Primary procedure", "Plot first"],
    )
    method_answers = {
        "independent": "Observation unit: animal. Estimate a diet mean difference; plot raw groups; report Welch t and a CI.",
        "paired": "Observation unit: culture. Analyze within-culture differences; preserve complete pairs in plots and resampling.",
        "counts": "Use observed counts and expected counts from the genetic model; check whether asymptotic calibration is adequate.",
        "factorial": "Fit both factors and their interaction; inspect the six cell distributions and interaction profiles before main effects.",
    }
    method_panel = mo.vstack(
        [
            method_scenario,
            mo.callout(mo.md(method_answers[method_scenario.value]), kind="info"),
        ]
    )
    two_column_panel(method_panel, method_map, widths=(1, 2.2))


@app.cell
def _(mo):
    mo.md(r"""
    ---

    ## 11. Review questions

    Choose one answer for each question. Feedback appears immediately and identifies
    the interpretation or design principle at stake.
    """)


@app.cell
def _(mo):
    review_two_group = mo.ui.radio(
        {
            "Estimated young-minus-old mean difference: 23.55 percentage points (95% Welch CI 9.59 to 37.50)": "estimate",
            "There is a 95% probability that the true difference lies between 9.59 and 37.50": "posterior",
            "95% of individual young rats exceed all old rats by 9.59 to 37.50 points": "individual",
        },
        value=None,
        label="Choose one answer",
    )
    review_assumptions = mo.ui.radio(
        {
            "A Q–Q plot can prove observations are independent": "plot_proves",
            "Independence must be justified from the observation units, design, and process": "design",
            "Passing a normality test guarantees the t model is correct": "normality_gate",
        },
        value=None,
        label="Choose one answer",
    )
    review_pairing = mo.ui.radio(
        {
            "n = 24 observations; df = 23": "measurements",
            "n = 12 differences; df = 11": "pairs",
            "n = 24 observations; df = 22": "twice",
        },
        value=None,
        label="Choose one answer",
    )
    review_anova = mo.ui.radio(
        {
            "Every tissue mean differs from every other tissue mean": "all_pairs",
            "There is evidence against equality of all population tissue means under the model": "omnibus",
            "The null hypothesis has probability 0.000002": "null_probability",
        },
        value=None,
        label="Choose one answer",
    )
    review_interaction = mo.ui.radio(
        {
            "Illumination has one identical effect in every tissue": "constant",
            "The illumination effect depends on tissue": "depends",
            "The tissue and illumination variables are correlated": "correlated",
        },
        value=None,
        label="Choose one answer",
    )
    return (
        review_anova,
        review_assumptions,
        review_interaction,
        review_pairing,
        review_two_group,
    )


@app.cell
def _(
    mo,
    review_anova,
    review_assumptions,
    review_feedback,
    review_interaction,
    review_pairing,
    review_two_group,
):
    review_blocks = [
        mo.vstack(
            [
                mo.md(
                    "**1. How should the rat Welch estimate and 95% confidence interval be reported?**"
                ),
                review_two_group,
                review_feedback(
                    review_two_group.value,
                    correct_value="estimate",
                    correct_text="**Correct.** State the direction (young minus old), estimate, units, and uncertainty. The 95% Welch interval concerns a population mean difference and has approximately 95% repeated-sampling coverage under its assumptions; it does not describe individual rats.",
                    incorrect_text={
                        "posterior": "The Welch interval is frequentist: its repeated-sampling procedure has "
                        "approximately 95% coverage under its assumptions. A 95% posterior probability would "
                        "require a Bayesian analysis with a stated prior.",
                        "individual": "The interval concerns a population mean difference, not all pairwise differences "
                        "between individual rats. Groups can overlap substantially even when the confidence "
                        "interval for their mean difference excludes zero.",
                    },
                ),
            ]
        ),
        mo.vstack(
            [
                mo.md("**2. Which statement about independence is correct?**"),
                review_assumptions,
                review_feedback(
                    review_assumptions.value,
                    correct_value="design",
                    correct_text="**Correct.** Independence is justified by the observation units, sampling, allocation, and data-generating process.",
                    incorrect_text={
                        "plot_proves": "A Q–Q plot assesses marginal distributional shape, not whether one observation "
                        "depends on another. Measurements can look normal while sharing an animal, cage, "
                        "or batch.",
                        "normality_gate": "Failure to reject normality is not proof of normality, especially in a small "
                        "sample. It also says nothing about independence, allocation, or whether the "
                        "chosen comparison answers the scientific question.",
                    },
                ),
            ]
        ),
        mo.vstack(
            [
                mo.md(
                    "**3. Twelve independent patients each have a before and an after measurement, with no missing values. What are n and the degrees of freedom for a paired t test?**"
                ),
                review_pairing,
                review_feedback(
                    review_pairing.value,
                    correct_value="pairs",
                    correct_text="**Correct.** The 24 raw measurements form 12 within-patient differences. Estimating their mean leaves 12 − 1 = 11 degrees of freedom for the SD used in the paired t statistic.",
                    incorrect_text={
                        "measurements": "There are 24 measurements but only 12 independent within-patient differences. "
                        "The paired t statistic uses the SD of those 12 differences and has 12 − 1 = 11 "
                        "degrees of freedom.",
                        "twice": "The 22 degrees of freedom would describe a pooled independent two-group test with 12 "
                        "observations per group. Before and after measurements on the same patients are paired, "
                        "so analyze 12 differences with 11 degrees of freedom.",
                    },
                ),
            ]
        ),
        mo.vstack(
            [
                mo.md(
                    "**4. For the light-grown subset, the one-way ANOVA gives p ≈ 0.00000182. Which conclusion is justified, assuming the model is appropriate?**"
                ),
                review_anova,
                review_feedback(
                    review_anova.value,
                    correct_value="omnibus",
                    correct_text="**Correct.** The omnibus evidence is against equality of all tissue means; post-hoc comparisons locate supported differences.",
                    incorrect_text={
                        "all_pairs": "The omnibus test provides evidence against all means being equal; it does not show "
                        "that every pair differs. Use the multiplicity-adjusted comparisons to identify "
                        "which differences are supported.",
                        "null_probability": "The p-value is a tail probability for F under equal population means and the "
                        "ANOVA assumptions. It does not assign a probability to that null hypothesis.",
                    },
                ),
            ]
        ),
        mo.vstack(
            [
                mo.md(
                    "**5. In the observed peroxidase data, what does the tissue × illumination interaction describe on the additive mean-response scale?**"
                ),
                review_interaction,
                review_feedback(
                    review_interaction.value,
                    correct_value="depends",
                    correct_text="**Correct.** The light–dark contrast changes across tissues, so a single overall illumination effect is incomplete.",
                    incorrect_text={
                        "constant": "An identical light–dark difference at every tissue would give parallel profiles and "
                        "no interaction in the population mean model. Interaction means that contrast changes "
                        "across tissues.",
                        "correlated": "Interaction concerns the response: how the light–dark mean difference varies by "
                        "tissue. It can occur in a balanced factorial design where the factor labels "
                        "themselves are independent.",
                    },
                ),
            ]
        ),
    ]
    mo.vstack(review_blocks, gap=1.0)


@app.cell
def _(mo):
    mo.md(r"""
    ---

    ## 12. Summary and bridge

    - Define the observation unit, estimand, design, and direction before choosing a test.
    - Welch's test is the general default for two independent means; report the mean
      difference, its units, confidence interval, statistic, degrees of freedom, and p-value.
    - A paired t-test is a one-sample test on complete within-unit differences.
    - Correlation testing does not replace the scatter plot, interval, or assessment of confounding.
    - Chi-squared calculations use counts and expected counts; small expected counts
      may require exact or simulation-based calibration.
    - Rank and permutation procedures have assumptions and estimands of their own.
    - One-way ANOVA compares between-group with residual variation. Its omnibus result
      requires multiplicity-aware post-hoc analysis to support pairwise claims.
    - In factorial designs, interpret cell means and interactions before isolated main effects.
    - Failure to reject a difference does not establish equivalence; equivalence needs a pre-specified margin and matching procedure.
    - Statistical significance, effect magnitude, uncertainty, scientific relevance,
      study quality, and multiplicity remain separate considerations.

    **Next:** Chapter 6 changes the inferential language by combining prior
    information with a likelihood to form a posterior distribution. The design,
    model checking, effect scale, and scientific interpretation developed here remain essential.
    """)


if __name__ == "__main__":
    app.run()
