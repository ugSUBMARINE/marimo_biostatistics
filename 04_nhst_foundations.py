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
    app_title="NHST foundations",
    css_file="site/assets/notebook.css",
)


@app.cell
def chapter_header():
    from companion_style import notebook_header

    notebook_header(
        4,
        "Null-hypothesis significance testing",
        "Foundations: null models, p-values, decisions, and power",
    )


@app.cell
def _():
    import marimo as mo

    return mo


@app.cell
def _():
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from scipy import stats

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
        compact_table,
        np,
        pd,
        plt,
        review_feedback,
        stats,
        two_column_panel,
    )


@app.cell
def _(np, stats):
    def exact_binomial_details(successes, trials, null_probability, alternative):
        outcome_values = np.arange(trials + 1)
        outcome_probabilities = stats.binom.pmf(
            outcome_values, trials, null_probability
        )
        observed_probability = float(
            stats.binom.pmf(successes, trials, null_probability)
        )
        if alternative == "greater":
            included_mask = outcome_values >= successes
        elif alternative == "less":
            included_mask = outcome_values <= successes
        else:
            included_mask = outcome_probabilities <= observed_probability + 1e-14
        exact_result = stats.binomtest(
            successes, trials, null_probability, alternative=alternative
        )
        included_probability = float(outcome_probabilities[included_mask].sum())
        assert np.isclose(included_probability, exact_result.pvalue)
        return {
            "outcomes": outcome_values,
            "probabilities": outcome_probabilities,
            "included": included_mask,
            "p_value": float(exact_result.pvalue),
        }

    def one_sample_mean_test(
        observed_mean,
        null_mean,
        scale,
        sample_size,
        test_type,
        alternative,
        alpha,
    ):
        standard_error = scale / np.sqrt(sample_size)
        statistic = (observed_mean - null_mean) / standard_error
        reference = stats.norm if test_type == "z" else stats.t(df=sample_size - 1)
        if alternative == "greater":
            p_value = float(reference.sf(statistic))
        elif alternative == "less":
            p_value = float(reference.cdf(statistic))
        else:
            p_value = float(2 * reference.sf(abs(statistic)))
        return {
            "difference": float(observed_mean - null_mean),
            "standard_error": float(standard_error),
            "statistic": float(statistic),
            "degrees_of_freedom": None if test_type == "z" else sample_size - 1,
            "p_value": min(1.0, p_value),
            "reject": bool(p_value <= alpha),
        }

    def z_test_power(effect, scale, sample_size, alpha, alternative):
        standard_error = scale / np.sqrt(sample_size)
        null_reference = stats.norm(loc=0.0, scale=standard_error)
        if alternative == "greater":
            true_change = effect
            lower_critical = None
            upper_critical = float(null_reference.ppf(1.0 - alpha))
            power = float(
                stats.norm(loc=true_change, scale=standard_error).sf(upper_critical)
            )
        elif alternative == "less":
            true_change = -effect
            lower_critical = float(null_reference.ppf(alpha))
            upper_critical = None
            power = float(
                stats.norm(loc=true_change, scale=standard_error).cdf(lower_critical)
            )
        else:
            true_change = effect
            lower_critical = float(null_reference.ppf(alpha / 2.0))
            upper_critical = float(null_reference.ppf(1.0 - alpha / 2.0))
            alternative_reference = stats.norm(loc=true_change, scale=standard_error)
            power = float(
                alternative_reference.cdf(lower_critical)
                + alternative_reference.sf(upper_critical)
            )
        return {
            "standard_error": float(standard_error),
            "true_change": float(true_change),
            "lower_critical": lower_critical,
            "upper_critical": upper_critical,
            "power": power,
            "beta": 1.0 - power,
        }

    def minimum_sample_size(effect, scale, alpha, alternative, target_power):
        for candidate_size in range(2, 2001):
            candidate_power = z_test_power(
                effect, scale, candidate_size, alpha, alternative
            )["power"]
            if candidate_power >= target_power:
                return candidate_size
        return None

    def one_sample_t_pvalues(samples):
        sample_means = samples.mean(axis=-1)
        sample_sds = samples.std(axis=-1, ddof=1)
        sample_sizes = samples.shape[-1]
        t_statistics = sample_means / (sample_sds / np.sqrt(sample_sizes))
        return 2.0 * stats.t.sf(np.abs(t_statistics), df=sample_sizes - 1)

    def simulate_pvalue_experiment(
        effect, sample_size, repetitions, strategy, alpha, seed
    ):
        simulation_rng = np.random.default_rng(seed)
        reported_pvalues = np.empty(repetitions, dtype=float)
        chunk_size = 2_000
        for chunk_start in range(0, repetitions, chunk_size):
            chunk_stop = min(chunk_start + chunk_size, repetitions)
            chunk_count = chunk_stop - chunk_start
            if strategy == "single":
                simulated_samples = simulation_rng.normal(
                    effect, 1.0, size=(chunk_count, sample_size)
                )
                chunk_pvalues = one_sample_t_pvalues(simulated_samples)
            elif strategy == "interim":
                simulated_samples = simulation_rng.normal(
                    effect, 1.0, size=(chunk_count, sample_size)
                )
                look_sizes = np.unique(
                    np.linspace(5, sample_size, num=min(5, sample_size - 4), dtype=int)
                )
                interim_pvalues = np.column_stack(
                    [
                        one_sample_t_pvalues(simulated_samples[:, :look_size])
                        for look_size in look_sizes
                    ]
                )
                chunk_pvalues = interim_pvalues.min(axis=1)
            else:
                simulated_samples = simulation_rng.normal(
                    effect, 1.0, size=(chunk_count, 5, sample_size)
                )
                outcome_pvalues = one_sample_t_pvalues(simulated_samples)
                chunk_pvalues = outcome_pvalues.min(axis=1)
            reported_pvalues[chunk_start:chunk_stop] = chunk_pvalues
        rejection_rate = float(np.mean(reported_pvalues <= alpha))
        monte_carlo_se = float(
            np.sqrt(rejection_rate * (1.0 - rejection_rate) / repetitions)
        )
        return {
            "p_values": reported_pvalues,
            "effect": float(effect),
            "sample_size": int(sample_size),
            "repetitions": int(repetitions),
            "strategy": strategy,
            "alpha": float(alpha),
            "rejection_rate": rejection_rate,
            "monte_carlo_se": monte_carlo_se,
            "seed": int(seed),
        }

    lecture_coin = exact_binomial_details(13, 20, 0.5, "greater")
    lecture_z = one_sample_mean_test(2200, 3000, 1000, 10, "z", "two-sided", 0.05)
    lecture_t = one_sample_mean_test(2200, 3000, 900, 10, "t", "two-sided", 0.05)
    lecture_t_margin = float(stats.t.ppf(0.975, 9) * 900 / np.sqrt(10))
    lecture_t_interval = (2200 - lecture_t_margin, 2200 + lecture_t_margin)
    assert np.isclose(lecture_coin["p_value"], 0.13158798217773438)
    assert np.isclose(lecture_z["statistic"], -2.5298221281)
    assert np.isclose(lecture_z["p_value"], 0.0114120364)
    assert np.isclose(lecture_t["statistic"], -2.8109134757)
    assert np.isclose(lecture_t["p_value"], 0.0203466769)
    assert np.allclose(lecture_t_interval, (1556.18, 2843.82), atol=0.01)

    return (
        exact_binomial_details,
        minimum_sample_size,
        one_sample_mean_test,
        simulate_pvalue_experiment,
        z_test_power,
    )


@app.cell
def _(mo):
    mo.md(r"""
    Null-hypothesis significance testing asks how compatible an observed statistic is
    with a specified **null model**. It does not assign probabilities to hypotheses,
    and it cannot replace careful design, effect estimates, or uncertainty intervals.

    By the end of the chapter, you should be able to:

    1. define a null hypothesis, test statistic, and p-value in conditional language;
    2. distinguish one-sided from two-sided alternatives;
    3. calculate and interpret one-sample z and t statistics;
    4. connect a matched hypothesis test with its confidence interval;
    5. distinguish type-I error, type-II error, power, and diagnostic accuracy;
    6. recognize common p-value misconceptions and data-dependent analysis choices.

    > **Two-minute preview:** A p-value is a tail probability calculated under
    > $H_0$. A decision threshold controls a long-run error rate, while power depends
    > on the true effect, variation, sample size, threshold, and alternative. Report
    > the estimated effect and its uncertainty alongside any test result.

    **Prerequisites:** probability distributions, sampling distributions, standard
    errors, t-distributions, and confidence intervals from Chapter 2.
    """)


@app.cell
def _(mo):
    mo.md(r"""
    ---

    ## 1. Null models and exact p-values

    A hypothesis test begins by specifying what random variation would look like if a
    particular null hypothesis were true. The **test statistic** compresses the data
    into a value whose null distribution can be calculated or simulated.

    For a fair coin, $H_0:p=0.5$ makes the number of heads in $n$ independent tosses a
    binomial random variable. For the directional alternative $H_1:p>0.5$, outcomes
    at least as extreme as 13 heads are 13, 14, ..., 20 heads.

    **Try this:** keep $n=20$ and $p_0=0.5$, choose **More heads**, and move the
    observed count from 10 to 13 to 18. Follow the highlighted outcomes and their
    summed probability. At 13 heads, compare the one-sided and two-sided
    alternatives. Then change $p_0$ to 0.7: the same observation is assessed
    against a different null model. Every control updates immediately; changing
    $n$ resets the observed-count control, so check it before comparing results.
    These are hypothetical comparisons for learning; a real test's direction
    must be specified before inspecting its data.
    """)


@app.cell
def _(mo):
    coin_trials = mo.ui.slider(
        5,
        100,
        value=20,
        show_value=True,
        full_width=True,
        label="Number of tosses n",
    )
    coin_null_probability = mo.ui.slider(
        0.1,
        0.9,
        step=0.05,
        value=0.5,
        show_value=True,
        full_width=True,
        label="Null probability p₀",
    )
    coin_alternative = mo.ui.radio(
        {
            "More heads (greater)": "greater",
            "Fewer heads (less)": "less",
            "Either direction (two-sided)": "two-sided",
        },
        value="More heads (greater)",
        label="Alternative hypothesis",
    )
    return coin_alternative, coin_null_probability, coin_trials


@app.cell
def _(coin_trials, mo):
    selected_coin_trials = int(coin_trials.value)
    coin_heads = mo.ui.slider(
        0,
        selected_coin_trials,
        value=min(13, selected_coin_trials),
        show_value=True,
        full_width=True,
        label="Observed heads",
    )
    return (coin_heads,)


@app.cell
def _(
    COLORS,
    FIGURE_SIZE_STANDARD,
    coin_alternative,
    coin_heads,
    coin_null_probability,
    coin_trials,
    exact_binomial_details,
    mo,
    np,
    pd,
    plt,
    compact_table,
    two_column_panel,
):
    coin_n = int(coin_trials.value)
    coin_k = int(coin_heads.value)
    coin_p0 = float(coin_null_probability.value)
    coin_direction = coin_alternative.value
    coin_result = exact_binomial_details(coin_k, coin_n, coin_p0, coin_direction)

    coin_colors = np.where(
        coin_result["included"], COLORS["vermillion"], COLORS["purple"]
    )
    coin_figure, coin_axis = plt.subplots(figsize=FIGURE_SIZE_STANDARD)
    coin_axis.bar(
        coin_result["outcomes"],
        coin_result["probabilities"],
        width=0.78,
        color=coin_colors,
        edgecolor="white",
        linewidth=0.5,
    )
    coin_axis.axvline(
        coin_k,
        color=COLORS["blue"],
        linestyle="--",
        linewidth=1.7,
        label=f"Observed: {coin_k} heads",
    )
    coin_axis.set(
        xlabel=f"Number of heads in {coin_n} tosses",
        ylabel="Probability",
        xlim=(-0.8, coin_n + 0.8),
        ylim=(0, max(coin_result["probabilities"]) * 1.13),
    )
    coin_axis.grid(axis="y", linestyle=":", alpha=0.35)
    coin_axis.legend(frameon=False, fontsize=8)
    coin_figure.tight_layout()

    included_outcomes = coin_result["outcomes"][coin_result["included"]]
    coin_summary_data = pd.DataFrame(
        {
            "Quantity": ["Null hypothesis", "Included outcomes", "Exact p-value"],
            "Value": [
                f"p = {coin_p0:.2f}",
                (
                    ", ".join(map(str, included_outcomes[:8]))
                    + (", …" if included_outcomes.size > 8 else "")
                ),
                f"{coin_result['p_value']:.5f}",
            ],
        }
    )
    coin_summary_table = compact_table(
        coin_summary_data,
        column_widths={"Quantity": 135, "Value": 170},
    )
    coin_controls = mo.vstack(
        [
            coin_trials,
            coin_heads,
            coin_null_probability,
            coin_alternative,
            coin_summary_table,
        ]
    )
    coin_note_text = (
        "The orange bars are the outcomes included in the exact tail probability. "
        "Their probabilities sum to the p-value."
        if coin_direction != "two-sided"
        else "SciPy's exact two-sided binomial test includes outcomes whose null "
        "probability is no greater than that of the observed outcome. Because the "
        "distribution is discrete, this is not generally twice a one-sided p-value."
    )
    coin_note = mo.callout(mo.md(coin_note_text), kind="info")
    two_column_panel(
        coin_controls,
        mo.vstack([coin_figure, coin_note]),
        widths=(1, 2),
    )


@app.cell
def _(mo):
    conditional_checkpoint = mo.ui.radio(
        {
            "Assuming the coin is fair, the probability of 13 or more heads in 20 tosses is about 0.132.": "conditional",
            "After observing 13 heads, the probability that the coin is fair is about 0.132.": "reversed",
            "There is a 13.2% probability that chance caused the observation.": "chance",
        },
        value=None,
        label="Choose the valid interpretation",
    )
    mo.vstack(
        [
            mo.md(
                "### Checkpoint: which sentence keeps the conditioning in the correct direction?"
            ),
            conditional_checkpoint,
        ]
    )
    return (conditional_checkpoint,)


@app.cell
def _(conditional_checkpoint, mo):
    conditional_feedback = (
        mo.callout(
            mo.md(
                "**Correct.** The assumption $H_0:p=0.5$ comes before the probability statement."
            ),
            kind="success",
        )
        if conditional_checkpoint.value == "conditional"
        else mo.callout(
            mo.md(
                r"A p-value conditions on $H_0$ and describes data or a statistic. It does not provide $P(H_0\mid\text{data})$ or assign a probability to 'chance.'"
            ),
            kind="danger",
        )
        if conditional_checkpoint.value is not None
        else mo.callout(mo.md("Select an answer above."), kind="neutral")
    )
    conditional_feedback  # noqa: B018


@app.cell
def _(mo):
    mo.md(r"""
    ---

    ## 2. One-sided and two-sided tails

    “More extreme” is defined by the test statistic and the alternative hypothesis.
    A lower-tailed alternative asks about unusually small values, an upper-tailed
    alternative about unusually large values, and a two-sided alternative about
    deviations in either direction. The direction must be chosen from the scientific
    question before inspecting the result.

    The density plot shows area in the relevant tail or tails. The linked CDF reports
    accumulated probability up to the observed statistic.

    **Try this:** use the standard normal model and statistic 2.0. Compare
    **Upper tail**, **Lower tail**, and **Two-sided**; match the shaded area to
    $1-F(2)$, $F(2)$, or the two tails combined. Move the statistic to −2 and
    check which directional tail becomes small. Switch to a t-distribution and
    reduce degrees of freedom to see the effect of heavier tails. The first
    degrees-of-freedom control affects t, chi-squared, and F models; the second
    affects F only, and neither affects the standard normal. Switching models
    resets the statistic and tail controls to suitable defaults. Changes update
    immediately, and the displayed tail convention defines what is being shaded.
    """)


@app.cell
def _(mo):
    tail_distribution = mo.ui.dropdown(
        {
            "Standard normal": "normal",
            "t-distribution": "t",
            "Chi-squared distribution": "chi2",
            "F-distribution": "f",
        },
        value="Standard normal",
        label="Null distribution",
        full_width=True,
    )
    tail_df_1 = mo.ui.slider(
        1,
        30,
        value=5,
        show_value=True,
        full_width=True,
        label="Degrees of freedom 1",
    )
    tail_df_2 = mo.ui.slider(
        1,
        40,
        value=15,
        show_value=True,
        full_width=True,
        label="Degrees of freedom 2 (F only)",
    )
    return tail_df_1, tail_df_2, tail_distribution


@app.cell
def _(mo, tail_distribution):
    selected_tail_distribution = tail_distribution.value
    if selected_tail_distribution in {"normal", "t"}:
        tail_observed = mo.ui.slider(
            -4.0,
            4.0,
            step=0.1,
            value=2.0,
            show_value=True,
            full_width=True,
            label="Observed statistic",
        )
        tail_alternative = mo.ui.radio(
            {
                "Lower tail": "less",
                "Upper tail": "greater",
                "Two-sided": "two-sided",
            },
            value="Two-sided",
            label="Alternative",
        )
    elif selected_tail_distribution == "chi2":
        tail_observed = mo.ui.slider(
            0.1,
            15.0,
            step=0.1,
            value=6.0,
            show_value=True,
            full_width=True,
            label="Observed statistic",
        )
        tail_alternative = mo.ui.radio(
            {"Lower tail": "less", "Upper tail": "greater"},
            value="Upper tail",
            label="Alternative",
        )
    else:
        tail_observed = mo.ui.slider(
            0.1,
            5.0,
            step=0.1,
            value=2.5,
            show_value=True,
            full_width=True,
            label="Observed statistic",
        )
        tail_alternative = mo.ui.radio(
            {"Lower tail": "less", "Upper tail": "greater"},
            value="Upper tail",
            label="Alternative",
        )
    return tail_alternative, tail_observed


@app.cell
def _(
    COLORS,
    FIGURE_SIZE_LINKED,
    mo,
    np,
    plt,
    stats,
    tail_alternative,
    tail_df_1,
    tail_df_2,
    tail_distribution,
    tail_observed,
    two_column_panel,
):
    tail_name = tail_distribution.value
    tail_df1_value = int(tail_df_1.value)
    tail_df2_value = int(tail_df_2.value)
    tail_observed_value = float(tail_observed.value)
    tail_direction = tail_alternative.value
    if tail_name == "normal":
        tail_reference = stats.norm()
        tail_grid = np.linspace(-4.5, 4.5, 1_000)
        tail_title = "Standard normal distribution"
    elif tail_name == "t":
        tail_reference = stats.t(df=tail_df1_value)
        tail_grid = np.linspace(-5.5, 5.5, 1_000)
        tail_title = f"t-distribution (df = {tail_df1_value})"
    elif tail_name == "chi2":
        tail_reference = stats.chi2(df=tail_df1_value)
        tail_grid = np.linspace(0, max(15, tail_reference.ppf(0.997)), 1_000)
        tail_title = rf"$\chi^2$ distribution (df = {tail_df1_value})"
    else:
        tail_reference = stats.f(dfn=tail_df1_value, dfd=tail_df2_value)
        tail_grid = np.linspace(0.01, max(5, tail_reference.ppf(0.995)), 1_000)
        tail_title = f"F distribution ({tail_df1_value}, {tail_df2_value} df)"

    tail_density = tail_reference.pdf(tail_grid)
    tail_cdf_values = tail_reference.cdf(tail_grid)
    if tail_direction == "less":
        tail_p_value = float(tail_reference.cdf(tail_observed_value))
        tail_mask = tail_grid <= tail_observed_value
    elif tail_direction == "greater":
        tail_p_value = float(tail_reference.sf(tail_observed_value))
        tail_mask = tail_grid >= tail_observed_value
    else:
        tail_p_value = float(
            min(
                1.0,
                2
                * min(
                    tail_reference.cdf(tail_observed_value),
                    tail_reference.sf(tail_observed_value),
                ),
            )
        )
        tail_mask = np.abs(tail_grid) >= abs(tail_observed_value)

    tail_figure, (tail_pdf_axis, tail_cdf_axis) = plt.subplots(
        1, 2, figsize=FIGURE_SIZE_LINKED
    )
    tail_pdf_axis.plot(tail_grid, tail_density, color=COLORS["purple"], linewidth=2.2)
    tail_pdf_axis.fill_between(
        tail_grid,
        0,
        tail_density,
        where=tail_mask,
        color=COLORS["vermillion"],
        alpha=0.5,
        label="p-value area",
    )
    tail_pdf_axis.axvline(
        tail_observed_value,
        color=COLORS["blue"],
        linestyle="--",
        linewidth=1.5,
        label="Observed statistic",
    )
    if tail_direction == "two-sided":
        tail_pdf_axis.axvline(
            -tail_observed_value,
            color=COLORS["blue"],
            linestyle="--",
            linewidth=1.5,
        )
    tail_pdf_axis.set(
        title="Probability density",
        xlabel="Test statistic",
        ylabel="Density",
        xlim=(tail_grid[0], tail_grid[-1]),
        ylim=(0, np.nanmax(tail_density) * 1.12),
    )
    tail_pdf_axis.grid(axis="y", linestyle=":", alpha=0.35)
    tail_pdf_axis.legend(frameon=False, fontsize=7)

    tail_cdf_at_observed = float(tail_reference.cdf(tail_observed_value))
    tail_cdf_axis.plot(tail_grid, tail_cdf_values, color=COLORS["green"], linewidth=2.2)
    tail_cdf_axis.plot(
        [tail_observed_value],
        [tail_cdf_at_observed],
        marker="o",
        color=COLORS["blue"],
        markersize=6,
    )
    tail_cdf_axis.axvline(
        tail_observed_value, color=COLORS["gray"], linestyle=":", linewidth=1.2
    )
    tail_cdf_axis.axhline(
        tail_cdf_at_observed, color=COLORS["gray"], linestyle=":", linewidth=1.2
    )
    tail_cdf_axis.set(
        title="Cumulative distribution",
        xlabel="Test statistic",
        ylabel="Cumulative probability",
        xlim=(tail_grid[0], tail_grid[-1]),
        ylim=(0, 1.02),
    )
    tail_cdf_axis.grid(axis="y", linestyle=":", alpha=0.35)
    tail_figure.suptitle(tail_title, fontsize=10)
    tail_figure.tight_layout()

    tail_controls_items = [
        tail_distribution,
        tail_observed,
        tail_alternative,
        tail_df_1,
    ]
    if tail_name == "f":
        tail_controls_items.append(tail_df_2)
    tail_controls_items.append(
        mo.stat(f"{tail_p_value:.4f}", label="Tail probability", bordered=True)
    )
    tail_asymmetry_note = (
        mo.callout(
            mo.md(
                "For this asymmetric distribution the explorer presents directional tails only. A two-sided test requires a test-specific definition of extremeness; simply doubling the smaller tail is not universal."
            ),
            kind="warn",
        )
        if tail_name in {"chi2", "f"}
        else mo.callout(
            mo.md(
                "For these symmetric reference distributions, the displayed two-sided p-value includes equally distant values in both tails."
            ),
            kind="info",
        )
    )
    two_column_panel(
        mo.vstack(tail_controls_items),
        mo.vstack([tail_figure, tail_asymmetry_note]),
        widths=(1, 3),
    )


@app.cell
def _(mo):
    scenario_direction = mo.ui.radio(
        {
            "Lower / greater / two-sided": "wrong_1",
            "Lower / lower / two-sided": "correct",
            "Two-sided / lower / greater": "wrong_2",
        },
        value=None,
        label="Choose the three alternatives in order",
    )
    mo.vstack(
        [
            mo.md(r"""
            ### Checkpoint: choose the alternatives

            1. Does a treatment **reduce** mean blood pressure?
            2. Does a mutation **decrease** enzyme activity?
            3. Does a culture condition **change** mean growth rate in either direction?
            """),
            scenario_direction,
        ]
    )
    return (scenario_direction,)


@app.cell
def _(mo, scenario_direction):
    scenario_result = (
        mo.callout(
            mo.md(
                "**Correct.** The first two questions specify decreases; the third explicitly allows either direction."
            ),
            kind="success",
        )
        if scenario_direction.value == "correct"
        else mo.callout(
            mo.md(
                "Translate the scientific wording before seeing the result: 'reduce' and 'decrease' are lower-tailed, while 'change in either direction' is two-sided."
            ),
            kind="danger",
        )
        if scenario_direction.value is not None
        else mo.callout(mo.md("Select an answer above."), kind="neutral")
    )
    scenario_result  # noqa: B018


@app.cell
def _(mo):
    mo.md(r"""
    ---

    ## 3. A testing workflow and one-sample mean tests

    A defensible test is more than a formula. Its order matters:

    1. define the biological target, sampling design, assumptions, and analysis;
    2. state $H_0$, $H_1$, the test statistic, and the decision threshold $\alpha$;
    3. collect the data and calculate the statistic;
    4. identify its distribution under $H_0$ and calculate the p-value;
    5. reject or fail to reject $H_0$ using the pre-specified rule;
    6. report the effect estimate, uncertainty interval, test result, and limitations.

    Independence comes from the design, not from a normality test. Looking at the
    result before choosing the alternative or threshold changes the procedure and its
    long-run error properties.

    ### Worked example: birth weight

    Ten newborns have a mean birth weight of 2200 g, compared with a standard value
    of 3000 g. If the population SD is known, standardize with $\sigma/\sqrt n$ and
    use a z reference distribution. If it is estimated from the sample, standardize
    with $s/\sqrt n$ and use a t distribution with $n-1$ degrees of freedom.

    **Try this:** begin with the t test and move the observed mean towards the
    null mean. The standardized discrepancy approaches zero and the two-sided
    p-value grows. Restore the original means, then increase $n$ or decrease the
    sample SD to see stronger evidence for the same mean difference. Compare
    z and t tests after setting their two SD controls to the same value: only
    the SD belonging to the selected test enters its calculation. Finally vary
    $\alpha$; the decision can change while the p-value stays fixed. These sliders
    describe hypothetical sample summaries rather than resampling raw birth weights.
    """)


@app.cell
def _(mo):
    mean_test_type = mo.ui.radio(
        {
            "Known population SD: z test": "z",
            "Estimated sample SD: t test": "t",
        },
        value="Estimated sample SD: t test",
        label="One-sample test",
    )
    mean_observed = mo.ui.slider(
        1500,
        3500,
        step=50,
        value=2200,
        show_value=True,
        full_width=True,
        label="Observed mean [g]",
    )
    mean_null = mo.ui.slider(
        2000,
        3500,
        step=50,
        value=3000,
        show_value=True,
        full_width=True,
        label="Null mean μ₀ [g]",
    )
    mean_population_sd = mo.ui.slider(
        300,
        1600,
        step=50,
        value=1000,
        show_value=True,
        full_width=True,
        label="Known population SD σ [g]",
    )
    mean_sample_sd = mo.ui.slider(
        300,
        1600,
        step=50,
        value=900,
        show_value=True,
        full_width=True,
        label="Estimated sample SD s [g]",
    )
    mean_sample_size = mo.ui.slider(
        2,
        100,
        value=10,
        show_value=True,
        full_width=True,
        label="Sample size n",
    )
    mean_alpha = mo.ui.slider(
        0.01,
        0.10,
        step=0.01,
        value=0.05,
        show_value=True,
        full_width=True,
        label="Decision threshold α",
    )
    mean_alternative = mo.ui.radio(
        {
            "Lower mean": "less",
            "Higher mean": "greater",
            "Different mean": "two-sided",
        },
        value="Different mean",
        label="Alternative hypothesis",
    )
    return (
        mean_alpha,
        mean_alternative,
        mean_null,
        mean_observed,
        mean_population_sd,
        mean_sample_sd,
        mean_sample_size,
        mean_test_type,
    )


@app.cell
def _(
    COLORS,
    FIGURE_SIZE_STANDARD,
    compact_table,
    mean_alpha,
    mean_alternative,
    mean_null,
    mean_observed,
    mean_population_sd,
    mean_sample_sd,
    mean_sample_size,
    mean_test_type,
    mo,
    np,
    one_sample_mean_test,
    pd,
    plt,
    stats,
    two_column_panel,
):
    anatomy_test_type = mean_test_type.value
    anatomy_mean = float(mean_observed.value)
    anatomy_null = float(mean_null.value)
    anatomy_n = int(mean_sample_size.value)
    anatomy_alpha = float(mean_alpha.value)
    anatomy_alternative = mean_alternative.value
    anatomy_scale = float(
        mean_population_sd.value if anatomy_test_type == "z" else mean_sample_sd.value
    )
    anatomy_result = one_sample_mean_test(
        anatomy_mean,
        anatomy_null,
        anatomy_scale,
        anatomy_n,
        anatomy_test_type,
        anatomy_alternative,
        anatomy_alpha,
    )
    anatomy_reference = (
        stats.norm
        if anatomy_test_type == "z"
        else stats.t(df=anatomy_result["degrees_of_freedom"])
    )
    anatomy_grid = np.linspace(-5.5, 5.5, 1_000)
    anatomy_density = anatomy_reference.pdf(anatomy_grid)
    anatomy_statistic = anatomy_result["statistic"]
    if anatomy_alternative == "less":
        anatomy_tail_mask = anatomy_grid <= anatomy_statistic
    elif anatomy_alternative == "greater":
        anatomy_tail_mask = anatomy_grid >= anatomy_statistic
    else:
        anatomy_tail_mask = np.abs(anatomy_grid) >= abs(anatomy_statistic)

    anatomy_figure, anatomy_axis = plt.subplots(figsize=FIGURE_SIZE_STANDARD)
    anatomy_axis.plot(
        anatomy_grid,
        anatomy_density,
        color=COLORS["purple"],
        linewidth=2.2,
        label=(
            "Standard normal"
            if anatomy_test_type == "z"
            else f"t-distribution (df = {anatomy_n - 1})"
        ),
    )
    anatomy_axis.fill_between(
        anatomy_grid,
        0,
        anatomy_density,
        where=anatomy_tail_mask,
        color=COLORS["vermillion"],
        alpha=0.5,
        label="p-value area",
    )
    anatomy_axis.axvline(
        anatomy_statistic,
        color=COLORS["blue"],
        linestyle="--",
        linewidth=1.7,
        label=f"Observed {anatomy_test_type} = {anatomy_statistic:.2f}",
    )
    anatomy_axis.set(
        xlabel="Standardized test statistic",
        ylabel="Probability density",
        xlim=(-5.5, 5.5),
        ylim=(0, 0.44),
    )
    anatomy_axis.grid(axis="y", linestyle=":", alpha=0.35)
    anatomy_axis.legend(frameon=False, fontsize=8)
    anatomy_figure.tight_layout()

    anatomy_rows = pd.DataFrame(
        {
            "Component": [
                "Difference",
                "Standard error",
                "Test statistic",
                "Degrees of freedom",
                "p-value",
            ],
            "Value": [
                f"{anatomy_result['difference']:+.0f} g",
                f"{anatomy_result['standard_error']:.1f} g",
                f"{anatomy_result['statistic']:.3f}",
                (
                    "not applicable"
                    if anatomy_result["degrees_of_freedom"] is None
                    else str(anatomy_result["degrees_of_freedom"])
                ),
                f"{anatomy_result['p_value']:.4f}",
            ],
        }
    )
    anatomy_table = compact_table(
        anatomy_rows,
        column_widths={"Component": 145, "Value": 125},
    )
    anatomy_decision = "reject" if anatomy_result["reject"] else "fail to reject"
    anatomy_symbol = "z" if anatomy_test_type == "z" else "t"
    anatomy_df_text = (
        "" if anatomy_test_type == "z" else f"({anatomy_result['degrees_of_freedom']})"
    )
    anatomy_report = mo.callout(
        mo.md(
            f"The estimated difference was {anatomy_result['difference']:+.0f} g; "
            f"{anatomy_symbol}{anatomy_df_text} = {anatomy_statistic:.2f}, "
            f"p = {anatomy_result['p_value']:.4f}. At α = {anatomy_alpha:.2f}, "
            f"we **{anatomy_decision} $H_0$**."
        ),
        kind="info" if anatomy_result["reject"] else "warn",
    )
    anatomy_controls = mo.vstack(
        [
            mean_test_type,
            mean_observed,
            mean_null,
            mean_population_sd if anatomy_test_type == "z" else mean_sample_sd,
            mean_sample_size,
            mean_alpha,
            mean_alternative,
            anatomy_table,
        ]
    )
    two_column_panel(
        anatomy_controls,
        mo.vstack([anatomy_figure, anatomy_report]),
        widths=(1, 2),
    )


@app.cell
def _(mo):
    test_statistic_derivation = mo.md(r"""
    ### Derivation: standardized distance from the null value

    Under $H_0:\mu=\mu_0$, the sample mean has expected value $\mu_0$ and standard
    error $\sigma/\sqrt n$. Subtract the null value and divide by that standard error:

    $$Z=\frac{\bar X-\mu_0}{\sigma/\sqrt n}.$$

    If the population SD is unknown, replace it with the sample estimate $S$:

    $$T=\frac{\bar X-\mu_0}{S/\sqrt n},\qquad df=n-1.$$

    The t reference distribution has heavier tails because the denominator is now
    estimated and varies across samples. In either case, the statistic is the observed
    difference expressed in **standard-error units**.
    """)
    mo.accordion(
        {"Derivation — one-sample z and t statistics": test_statistic_derivation}
    )


@app.cell
def _(mo):
    mo.md(r"""
    ---

    ## 4. Tests and confidence intervals

    For a matching two-sided test and interval, rejecting at level $\alpha$ is
    equivalent to the null value lying outside the $(1-\alpha)$ confidence interval.
    The model, standard error, tail convention, and confidence level must match.

    The interval communicates more than the binary decision: it shows the estimated
    direction, precision, and effect sizes still compatible with the data.

    **Try this:** at 95% confidence, slide the null mean across the interval and
    observe where the two-sided decision changes. The estimate, SD, and sample
    size are fixed here at 2200 g, 900 g, and 10; moving the null does not move
    the interval. Increase confidence to 99% and compare the wider interval with
    the stricter matching threshold $\alpha=0.01$. This panel is independent of
    the hypothetical summaries selected in the preceding activity.
    """)


@app.cell
def _(mo):
    duality_null = mo.ui.slider(
        1500,
        3500,
        step=25,
        value=3000,
        show_value=True,
        full_width=True,
        label="Null mean μ₀ [g]",
    )
    duality_confidence = mo.ui.slider(
        0.80,
        0.99,
        step=0.01,
        value=0.95,
        show_value=True,
        full_width=True,
        label="Confidence level",
    )
    return duality_confidence, duality_null


@app.cell
def _(
    COLORS,
    FIGURE_SIZE_SHALLOW,
    duality_confidence,
    duality_null,
    mo,
    np,
    one_sample_mean_test,
    plt,
    stats,
    two_column_panel,
):
    duality_mean_value = 2200.0
    duality_sd_value = 900.0
    duality_n_value = 10
    duality_null_value = float(duality_null.value)
    duality_level = float(duality_confidence.value)
    duality_alpha_value = 1.0 - duality_level
    duality_test_result = one_sample_mean_test(
        duality_mean_value,
        duality_null_value,
        duality_sd_value,
        duality_n_value,
        "t",
        "two-sided",
        duality_alpha_value,
    )
    duality_critical = float(
        stats.t.ppf(1.0 - duality_alpha_value / 2.0, df=duality_n_value - 1)
    )
    duality_margin = duality_critical * duality_sd_value / np.sqrt(duality_n_value)
    duality_interval = (
        duality_mean_value - duality_margin,
        duality_mean_value + duality_margin,
    )
    duality_null_inside = bool(
        duality_interval[0] <= duality_null_value <= duality_interval[1]
    )
    assert duality_test_result["reject"] == (not duality_null_inside)

    duality_figure, duality_axis = plt.subplots(figsize=FIGURE_SIZE_SHALLOW)
    duality_axis.errorbar(
        duality_mean_value,
        0,
        xerr=duality_margin,
        fmt="o",
        capsize=7,
        color=COLORS["blue"],
        linewidth=2.5,
        label=f"{100 * duality_level:.0f}% t confidence interval",
    )
    duality_axis.axvline(
        duality_null_value,
        color=(COLORS["green"] if duality_null_inside else COLORS["vermillion"]),
        linestyle="--",
        linewidth=2,
        label=f"Null value: {duality_null_value:.0f} g",
    )
    duality_axis.set(
        xlabel="Population mean birth weight [g]",
        yticks=[],
        ylim=(-0.6, 0.6),
        xlim=(1200, 3700),
    )
    duality_axis.grid(axis="x", linestyle=":", alpha=0.35)
    duality_axis.legend(frameon=False, fontsize=8, loc="upper left")
    duality_figure.tight_layout()

    duality_decision_text = (
        "outside the interval, and the matching test rejects"
        if not duality_null_inside
        else "inside the interval, and the matching test fails to reject"
    )
    duality_note = mo.callout(
        mo.md(
            f"The interval is **[{duality_interval[0]:.0f}, "
            f"{duality_interval[1]:.0f}] g**. The null value is "
            f"{duality_decision_text} at α = {duality_alpha_value:.2f} "
            f"(two-sided p = {duality_test_result['p_value']:.4f})."
        ),
        kind="success" if not duality_null_inside else "info",
    )
    duality_controls = mo.vstack(
        [
            mo.callout(
                mo.md("Fixed data: mean 2200 g, SD 900 g, n = 10"), kind="neutral"
            ),
            duality_null,
            duality_confidence,
            mo.stat(f"{duality_alpha_value:.2f}", label="Matching α", bordered=True),
            mo.stat(
                f"{duality_test_result['p_value']:.4f}",
                label="Two-sided p-value",
                bordered=True,
            ),
        ]
    )
    two_column_panel(
        duality_controls,
        mo.vstack([duality_figure, duality_note]),
        widths=(1, 3),
    )


@app.cell
def _(mo):
    mo.md(r"""
    ---

    ## 5. Type-I error, Type-II error, and power

    A test is a repeated-decision procedure. If $H_0$ is true, rejecting it is a
    **type-I error**; the procedure is designed so that this probability is $\alpha$
    under its assumptions. If a particular alternative is true, failing to reject is
    a **type-II error** with probability $\beta$. The probability of rejecting that
    false null is the **power**, $1-\beta$.

    Power is a design-stage property for a specified effect. It increases when the
    scientifically meaningful effect is larger, variability is smaller, sample size
    is larger, or the decision threshold is less stringent. It is not usefully
    recalculated from the observed effect after an underpowered study.

    **Try this:** keep the effect and SD at their defaults and compare $n=10$
    with $n=40$. Identify the rejection region under the null and the probability
    mass of the alternative inside it: that latter area is power. Restore
    $n=10$, halve the true change, and check the loss of power. Next raise the
    target-power control; it changes the sample-size recommendation, not the
    power of the currently selected $n$. Compare planned one- and two-sided
    alternatives, then lower $\alpha$ to see the trade-off between false
    positives and detection. All calculations update immediately.
    """)


@app.cell
def _(mo):
    power_effect = mo.ui.slider(
        100,
        1200,
        step=50,
        value=800,
        show_value=True,
        full_width=True,
        label="True change magnitude [g]",
    )
    power_scale = mo.ui.slider(
        400,
        1600,
        step=50,
        value=1000,
        show_value=True,
        full_width=True,
        label="Population SD σ [g]",
    )
    power_sample_size = mo.ui.slider(
        5,
        200,
        value=10,
        show_value=True,
        full_width=True,
        label="Sample size n",
    )
    power_alpha = mo.ui.slider(
        0.01,
        0.10,
        step=0.01,
        value=0.05,
        show_value=True,
        full_width=True,
        label="Significance threshold α",
    )
    power_alternative = mo.ui.radio(
        {
            "Detect a decrease": "less",
            "Detect an increase": "greater",
            "Detect either direction": "two-sided",
        },
        value="Detect a decrease",
        label="Planned alternative",
    )
    power_target = mo.ui.slider(
        0.70,
        0.95,
        step=0.05,
        value=0.80,
        show_value=True,
        full_width=True,
        label="Target power",
    )
    return (
        power_alpha,
        power_alternative,
        power_effect,
        power_sample_size,
        power_scale,
        power_target,
    )


@app.cell
def _(
    COLORS,
    FIGURE_SIZE_STANDARD,
    minimum_sample_size,
    mo,
    np,
    plt,
    power_alpha,
    power_alternative,
    power_effect,
    power_sample_size,
    power_scale,
    power_target,
    stats,
    two_column_panel,
    z_test_power,
):
    power_effect_value = float(power_effect.value)
    power_scale_value = float(power_scale.value)
    power_n_value = int(power_sample_size.value)
    power_alpha_value = float(power_alpha.value)
    power_direction = power_alternative.value
    power_target_value = float(power_target.value)
    power_result = z_test_power(
        power_effect_value,
        power_scale_value,
        power_n_value,
        power_alpha_value,
        power_direction,
    )
    power_required_n = minimum_sample_size(
        power_effect_value,
        power_scale_value,
        power_alpha_value,
        power_direction,
        power_target_value,
    )
    power_grid = np.linspace(-3200, 3200, 1_400)
    power_null_density = stats.norm.pdf(
        power_grid, loc=0.0, scale=power_result["standard_error"]
    )
    power_alt_density = stats.norm.pdf(
        power_grid,
        loc=power_result["true_change"],
        scale=power_result["standard_error"],
    )
    if power_direction == "greater":
        power_rejection_mask = power_grid >= power_result["upper_critical"]
        power_acceptance_mask = ~power_rejection_mask
        power_critical_values = [power_result["upper_critical"]]
        power_achieved_alpha = float(
            stats.norm(scale=power_result["standard_error"]).sf(
                power_result["upper_critical"]
            )
        )
    elif power_direction == "less":
        power_rejection_mask = power_grid <= power_result["lower_critical"]
        power_acceptance_mask = ~power_rejection_mask
        power_critical_values = [power_result["lower_critical"]]
        power_achieved_alpha = float(
            stats.norm(scale=power_result["standard_error"]).cdf(
                power_result["lower_critical"]
            )
        )
    else:
        power_rejection_mask = (power_grid <= power_result["lower_critical"]) | (
            power_grid >= power_result["upper_critical"]
        )
        power_acceptance_mask = ~power_rejection_mask
        power_critical_values = [
            power_result["lower_critical"],
            power_result["upper_critical"],
        ]
        power_null_model = stats.norm(scale=power_result["standard_error"])
        power_achieved_alpha = float(
            power_null_model.cdf(power_result["lower_critical"])
            + power_null_model.sf(power_result["upper_critical"])
        )
    assert np.isclose(power_achieved_alpha, power_alpha_value)
    assert np.isclose(power_result["beta"] + power_result["power"], 1.0)

    power_figure, power_axis = plt.subplots(figsize=FIGURE_SIZE_STANDARD)
    power_axis.plot(
        power_grid,
        power_null_density,
        color=COLORS["blue"],
        linewidth=2.2,
        label=r"Sampling distribution under $H_0$",
    )
    power_axis.plot(
        power_grid,
        power_alt_density,
        color=COLORS["purple"],
        linewidth=2.2,
        label=r"Sampling distribution under $H_1$",
    )
    power_axis.fill_between(
        power_grid,
        0,
        power_null_density,
        where=power_rejection_mask,
        color=COLORS["vermillion"],
        alpha=0.45,
        label=rf"Type-I error $\alpha={power_alpha_value:.2f}$",
    )
    power_axis.fill_between(
        power_grid,
        0,
        power_alt_density,
        where=power_acceptance_mask,
        color=COLORS["gray"],
        alpha=0.35,
        hatch="//",
        label=rf"Type-II error $\beta={power_result['beta']:.2f}$",
    )
    power_axis.fill_between(
        power_grid,
        0,
        power_alt_density,
        where=power_rejection_mask,
        color=COLORS["green"],
        alpha=0.28,
        label=f"Power = {power_result['power']:.2f}",
    )
    for power_critical_value in power_critical_values:
        power_axis.axvline(
            power_critical_value,
            color=COLORS["gray"],
            linestyle=":",
            linewidth=1.4,
        )
    power_axis.set(
        xlabel="Observed difference in sample mean [g]",
        ylabel="Probability density",
        xlim=(-3200, 3200),
        ylim=(0, max(power_null_density.max(), power_alt_density.max()) * 1.12),
    )
    power_axis.grid(axis="y", linestyle=":", alpha=0.35)
    power_axis.legend(frameon=False, fontsize=7, loc="upper right")
    power_figure.tight_layout()

    power_required_text = (
        f"{power_required_n}" if power_required_n is not None else "more than 2000"
    )
    power_controls = mo.vstack(
        [
            power_effect,
            power_scale,
            power_sample_size,
            power_alpha,
            power_alternative,
            power_target,
            mo.stat(
                f"{100 * power_result['power']:.1f}%",
                label="Power at current n",
                bordered=True,
            ),
            mo.stat(
                power_required_text,
                label=f"Minimum n for {100 * power_target_value:.0f}% power",
                bordered=True,
            ),
        ]
    )
    power_note = mo.callout(
        mo.md(
            f"With a true change of {power_result['true_change']:+.0f} g and "
            f"SE = {power_result['standard_error']:.1f} g, this design has "
            f"{100 * power_result['power']:.1f}% power and "
            f"{100 * power_result['beta']:.1f}% type-II error probability."
        ),
        kind="info",
    )
    two_column_panel(
        power_controls,
        mo.vstack([power_figure, power_note]),
        widths=(1, 2),
    )


@app.cell
def _(mo):
    mo.md(r"""
    ---

    ## 6. Diagnostic-test analogy

    A diagnostic threshold provides a useful analogy for statistical decisions. Let
    $H_0$ represent a healthy person and $H_1$ a person with porphyria. A concentration
    above the threshold is called positive.

    - false-positive rate $=\alpha$ and specificity $=1-\alpha$;
    - false-negative rate $=\beta$ and sensitivity $=1-\beta$, analogous to power.

    The analogy has limits: disease status is not literally a statistical hypothesis.
    Predictive values also depend on prevalence; those posterior probabilities are
    reserved for Chapter 6.

    **Try this:** move the positive-test threshold from 76 towards 40, then
    towards 120 µmol/L. A lower threshold captures more sick people but also
    labels more healthy people positive. Follow sensitivity and false-positive
    rate on the distribution plot and the linked ROC point. Raising the
    threshold reverses the trade-off; it does not change either group's
    underlying concentration distribution. Decide which type of error would
    matter more for a proposed use before choosing a threshold.
    """)


@app.cell
def _(mo):
    diagnostic_threshold = mo.ui.slider(
        40,
        120,
        step=1,
        value=76,
        show_value=True,
        full_width=True,
        label="Positive-test threshold [µmol/L]",
    )
    return (diagnostic_threshold,)


@app.cell
def _(
    COLORS,
    FIGURE_SIZE_LINKED,
    compact_table,
    diagnostic_threshold,
    mo,
    np,
    pd,
    plt,
    stats,
    two_column_panel,
):
    diagnostic_cutoff = float(diagnostic_threshold.value)
    healthy_reference = stats.norm(loc=60.0, scale=10.0)
    porphyria_reference = stats.norm(loc=100.0, scale=10.0)
    diagnostic_false_positive = float(healthy_reference.sf(diagnostic_cutoff))
    diagnostic_false_negative = float(porphyria_reference.cdf(diagnostic_cutoff))
    diagnostic_specificity = float(healthy_reference.cdf(diagnostic_cutoff))
    diagnostic_sensitivity = float(porphyria_reference.sf(diagnostic_cutoff))
    assert np.isclose(diagnostic_false_positive + diagnostic_specificity, 1.0)
    assert np.isclose(diagnostic_false_negative + diagnostic_sensitivity, 1.0)

    diagnostic_grid = np.linspace(20, 140, 1_000)
    healthy_density = healthy_reference.pdf(diagnostic_grid)
    porphyria_density = porphyria_reference.pdf(diagnostic_grid)
    roc_thresholds = np.linspace(20, 140, 400)
    roc_false_positive = healthy_reference.sf(roc_thresholds)
    roc_sensitivity = porphyria_reference.sf(roc_thresholds)

    diagnostic_figure, (diagnostic_axis, roc_axis) = plt.subplots(
        1, 2, figsize=FIGURE_SIZE_LINKED
    )
    diagnostic_axis.plot(
        diagnostic_grid,
        healthy_density,
        color=COLORS["sky"],
        linewidth=2.2,
        label="Healthy",
    )
    diagnostic_axis.plot(
        diagnostic_grid,
        porphyria_density,
        color=COLORS["purple"],
        linewidth=2.2,
        label="Porphyria",
    )
    diagnostic_axis.fill_between(
        diagnostic_grid,
        0,
        healthy_density,
        where=diagnostic_grid >= diagnostic_cutoff,
        color=COLORS["vermillion"],
        alpha=0.45,
        label="False positive",
    )
    diagnostic_axis.fill_between(
        diagnostic_grid,
        0,
        porphyria_density,
        where=diagnostic_grid < diagnostic_cutoff,
        color=COLORS["gray"],
        alpha=0.4,
        hatch="//",
        label="False negative",
    )
    diagnostic_axis.axvline(
        diagnostic_cutoff,
        color=COLORS["blue"],
        linestyle="--",
        linewidth=1.7,
        label="Threshold",
    )
    diagnostic_axis.set(
        title="Measurement distributions",
        xlabel="Porphyrin-precursor concentration [µmol/L]",
        ylabel="Probability density",
        xlim=(20, 140),
        ylim=(0, 0.043),
    )
    diagnostic_axis.grid(axis="y", linestyle=":", alpha=0.35)
    diagnostic_axis.legend(frameon=False, fontsize=6.8)

    roc_axis.plot(
        roc_false_positive,
        roc_sensitivity,
        color=COLORS["blue"],
        linewidth=2.2,
        label="ROC curve",
    )
    roc_axis.plot(
        [diagnostic_false_positive],
        [diagnostic_sensitivity],
        marker="o",
        color=COLORS["vermillion"],
        markersize=7,
        label="Current threshold",
    )
    roc_axis.plot(
        [0, 1],
        [0, 1],
        color=COLORS["gray"],
        linestyle=":",
        linewidth=1.2,
        label="No discrimination",
    )
    roc_axis.set(
        title="Threshold on the ROC curve",
        xlabel="False-positive rate (1 − specificity)",
        ylabel="Sensitivity (analogous to power)",
        xlim=(-0.02, 1.02),
        ylim=(-0.02, 1.02),
    )
    roc_axis.set_aspect("equal", adjustable="box")
    roc_axis.grid(linestyle=":", alpha=0.35)
    roc_axis.legend(frameon=False, fontsize=7, loc="lower right")
    diagnostic_figure.tight_layout()

    diagnostic_metrics_data = pd.DataFrame(
        {
            "Quantity": [
                "False positive (α)",
                "Specificity (1 − α)",
                "False negative (β)",
                "Sensitivity (1 − β)",
            ],
            "Value": [
                diagnostic_false_positive,
                diagnostic_specificity,
                diagnostic_false_negative,
                diagnostic_sensitivity,
            ],
        }
    )
    diagnostic_metrics = compact_table(
        diagnostic_metrics_data,
        column_widths={"Quantity": 165, "Value": 95},
        format_mapping={"Value": "{:.1%}"},
    )
    diagnostic_controls = mo.vstack([diagnostic_threshold, diagnostic_metrics])
    diagnostic_note = mo.callout(
        mo.md(
            f"Moving the threshold trades sensitivity against specificity. At "
            f"{diagnostic_cutoff:.0f} µmol/L, sensitivity is "
            f"{diagnostic_sensitivity:.1%} and specificity is "
            f"{diagnostic_specificity:.1%}. This says nothing yet about the "
            f"probability of disease after a positive result."
        ),
        kind="warn",
    )
    two_column_panel(
        diagnostic_controls,
        mo.vstack([diagnostic_figure, diagnostic_note]),
        widths=(1, 3),
    )


@app.cell
def _(mo):
    mo.md(r"""
    ---

    ## 7. What a p-value does not mean

    A p-value is not the probability that $H_0$ is true, and $1-p$ is not the
    probability that $H_1$ is true. Failure to reject does not prove equality. A
    small p-value does not establish a large, important, or unbiased effect.
    Nor does a p-value tell you the probability that 'chance caused' your result.
    Instead, random chance is a built-in assumption of the null hypothesis, and the
    p-value simply measures how well your data matches that assumption.

    These statements assume that the analysis procedure was specified and applied as
    claimed. Repeated looks, selective outcome reporting, and trying many analyses can
    increase the chance of finding at least one small reported p-value under a true
    null.
    """)


@app.cell
def _(mo):
    mo.md(r"""
    **Try the misconception clinic:** choose the statement you think is valid
    and read its feedback, then inspect the corrections for the other claims.
    For each, identify whether it confuses probability of data with probability
    of a hypothesis, uncertainty with equivalence, or evidence with importance.
    Selecting an answer immediately reveals its explanation.
    """)


@app.cell
def _(mo):
    misconception_choice = mo.ui.radio(
        {
            "p = 0.03 means a 3% probability that H₀ is true.": "hypothesis_probability",
            "1 − p is the probability that H₁ is true.": "alternative_probability",
            "Failure to reject H₀ proves that the groups are equivalent.": "equivalence",
            "A small p-value means the effect is scientifically important.": "importance",
            "Assuming H₀ and the test assumptions, p is the probability of a statistic at least as extreme as observed.": "valid",
        },
        value=None,
        label="Choose the valid interpretation",
    )
    mo.vstack(
        [
            mo.md("### Misconception clinic"),
            misconception_choice,
        ]
    )
    return (misconception_choice,)


@app.cell
def _(misconception_choice, mo):
    misconception_corrections = {
        "hypothesis_probability": (
            r"The conditional is reversed. A p-value describes data under $H_0$; it does not provide $P(H_0\mid\mathrm{data})$."
        ),
        "alternative_probability": (
            "Neither $p$ nor $1-p$ assigns a probability to a hypothesis. Bayesian posterior probabilities require an explicit prior and likelihood."
        ),
        "equivalence": (
            "Absence of sufficient evidence against equality is not evidence of equivalence. Equivalence requires a meaningful margin and a suitable analysis."
        ),
        "importance": (
            "Statistical compatibility and scientific importance are different. Inspect the effect estimate, uncertainty, design, and biological context."
        ),
    }
    if misconception_choice.value == "valid":
        misconception_feedback = mo.callout(
            mo.md(
                "**Correct.** The statement begins with the null model and refers to the observed statistic or something more extreme."
            ),
            kind="success",
        )
    elif misconception_choice.value is None:
        misconception_feedback = mo.callout(
            mo.md("Select an interpretation above."), kind="neutral"
        )
    else:
        misconception_feedback = mo.callout(
            mo.md(misconception_corrections[misconception_choice.value]),
            kind="danger",
        )
    misconception_feedback  # noqa: B018


@app.cell
def _(mo):
    mo.md(r"""
    ### Interactive laboratory: repeated testing and selective reporting

    Each simulated study draws independent standard-normal measurements and performs
    one-sample, two-sided t tests against a null mean of zero. Compare:

    - one pre-specified final test;
    - up to five interim looks, reporting the smallest p-value;
    - five outcomes, reporting the smallest p-value.

    Under a true null, p-values from the single valid test are approximately uniform,
    and about $\alpha$ of studies reject. The two data-dependent strategies answer a
    different question but are incorrectly displayed here as if only one test had
    been planned. This exposes their false-positive inflation.

    **Try this:** set the true standardized mean to 0 and use **One pre-specified
    test**, then click **Run / resimulate studies**. Compare the rejection fraction
    with $\alpha$ and inspect the p-value histogram. Run each of the other two
    strategies with the same settings: reporting the smallest p-value can raise
    the false-positive rate. Finally set the true mean to 0.5 and rerun the single
    test; its rejection fraction now estimates power for that alternative.

    Every setting is applied only when the button is pressed; the displayed
    caption records the completed run. Repeating a run shows simulation variation.
    More simulated studies estimate the rejection rate more precisely, whereas
    larger $n$ changes the evidence available within each study.
    """)


@app.cell
def _(mo):
    simulation_effect = mo.ui.slider(
        -1.0,
        1.0,
        step=0.1,
        value=0.0,
        show_value=True,
        full_width=True,
        label="True standardized mean",
    )
    simulation_sample_size = mo.ui.slider(
        5,
        80,
        value=20,
        show_value=True,
        full_width=True,
        label="Final sample size n",
    )
    simulation_alpha = mo.ui.slider(
        0.01,
        0.10,
        step=0.01,
        value=0.05,
        show_value=True,
        full_width=True,
        label="Decision threshold α",
    )
    simulation_repetitions = mo.ui.slider(
        1_000,
        50_000,
        step=1_000,
        value=10_000,
        show_value=True,
        full_width=True,
        label="Simulated studies",
    )
    simulation_strategy = mo.ui.dropdown(
        {
            "One pre-specified test": "single",
            "Up to five interim looks": "interim",
            "Smallest p among five outcomes": "outcomes",
        },
        value="One pre-specified test",
        label="Analysis strategy",
        full_width=True,
    )
    return (
        simulation_alpha,
        simulation_effect,
        simulation_repetitions,
        simulation_sample_size,
        simulation_strategy,
    )


@app.cell
def _(mo, simulate_pvalue_experiment):
    initial_simulation_result = simulate_pvalue_experiment(
        effect=0.0,
        sample_size=20,
        repetitions=10_000,
        strategy="single",
        alpha=0.05,
        seed=82404,
    )
    get_simulation_result, set_simulation_result = mo.state(initial_simulation_result)
    return get_simulation_result, set_simulation_result


@app.cell
def _(
    mo,
    np,
    set_simulation_result,
    simulate_pvalue_experiment,
    simulation_alpha,
    simulation_effect,
    simulation_repetitions,
    simulation_sample_size,
    simulation_strategy,
):
    def run_pvalue_simulation(_value):
        fresh_seed = int(np.random.default_rng().integers(0, 2**32 - 1))
        set_simulation_result(
            simulate_pvalue_experiment(
                effect=float(simulation_effect.value),
                sample_size=int(simulation_sample_size.value),
                repetitions=int(simulation_repetitions.value),
                strategy=simulation_strategy.value,
                alpha=float(simulation_alpha.value),
                seed=fresh_seed,
            )
        )

    simulation_run_button = mo.ui.button(
        label="Run / resimulate studies",
        kind="success",
        on_click=run_pvalue_simulation,
        full_width=True,
    )
    return (simulation_run_button,)


@app.cell
def _(
    COLORS,
    FIGURE_SIZE_STANDARD,
    get_simulation_result,
    mo,
    np,
    plt,
    simulation_alpha,
    simulation_effect,
    simulation_repetitions,
    simulation_run_button,
    simulation_sample_size,
    simulation_strategy,
    two_column_panel,
):
    pvalue_simulation_result = get_simulation_result()
    pvalue_array = pvalue_simulation_result["p_values"]
    assert pvalue_array.shape == (pvalue_simulation_result["repetitions"],)
    assert np.all((pvalue_array >= 0.0) & (pvalue_array <= 1.0))
    assert 0.0 <= pvalue_simulation_result["rejection_rate"] <= 1.0

    pvalue_histogram_density, _ = np.histogram(
        pvalue_array, bins=np.linspace(0, 1, 31), density=True
    )
    pvalue_figure, pvalue_axis = plt.subplots(figsize=FIGURE_SIZE_STANDARD)
    pvalue_axis.hist(
        pvalue_array,
        bins=np.linspace(0, 1, 31),
        density=True,
        color=COLORS["purple"],
        alpha=0.75,
        edgecolor="white",
        linewidth=0.5,
        label="Reported p-values",
    )
    pvalue_axis.axvspan(
        0,
        pvalue_simulation_result["alpha"],
        color=COLORS["vermillion"],
        alpha=0.18,
        label=r"Reported $p\leq\alpha$",
    )
    if (
        pvalue_simulation_result["strategy"] == "single"
        and pvalue_simulation_result["effect"] == 0.0
    ):
        pvalue_axis.axhline(
            1.0,
            color=COLORS["gray"],
            linestyle="--",
            linewidth=1.5,
            label="Uniform density under a true null",
        )
    pvalue_axis.set(
        xlabel="Reported p-value",
        ylabel="Density",
        xlim=(0, 1),
        ylim=(0, max(1.35, 1.28 * pvalue_histogram_density.max())),
    )
    pvalue_axis.grid(axis="y", linestyle=":", alpha=0.35)
    pvalue_axis.legend(frameon=False, fontsize=8, loc="upper right")
    pvalue_figure.tight_layout()

    strategy_labels = {
        "single": "one pre-specified test",
        "interim": "up to five interim looks",
        "outcomes": "the smallest result among five outcomes",
    }
    pvalue_rate = pvalue_simulation_result["rejection_rate"]
    pvalue_mcse = pvalue_simulation_result["monte_carlo_se"]
    pvalue_note_kind = (
        "info" if pvalue_simulation_result["strategy"] == "single" else "warn"
    )
    pvalue_note = mo.callout(
        mo.md(
            f"The last run used **{strategy_labels[pvalue_simulation_result['strategy']]}**, "
            f"true mean {pvalue_simulation_result['effect']:+.1f}, n = "
            f"{pvalue_simulation_result['sample_size']}, and "
            f"{pvalue_simulation_result['repetitions']:,} studies. The fraction with "
            f"reported p ≤ {pvalue_simulation_result['alpha']:.2f} was "
            f"**{pvalue_rate:.1%} ± {1.96 * pvalue_mcse:.1%}** (approximate 95% "
            f"Monte Carlo margin)."
        ),
        kind=pvalue_note_kind,
    )
    pvalue_controls = mo.vstack(
        [
            simulation_effect,
            simulation_sample_size,
            simulation_alpha,
            simulation_repetitions,
            simulation_strategy,
            simulation_run_button,
            mo.callout(
                mo.md(
                    "Controls describe the **next** run. The figure remains stable until the button is clicked."
                ),
                kind="neutral",
            ),
            mo.stat(
                f"{pvalue_rate:.1%}",
                label="Reported p ≤ α",
                bordered=True,
            ),
            mo.stat(
                f"±{1.96 * pvalue_mcse:.1%}",
                label="95% Monte Carlo margin",
                bordered=True,
            ),
        ]
    )
    two_column_panel(
        pvalue_controls,
        mo.vstack([pvalue_figure, pvalue_note]),
        widths=(1, 2),
    )


@app.cell
def _(mo):
    mo.md(r"""
    ---

    ## 8. Review questions

    Choose one answer for each question. Feedback appears immediately and explains
    the underlying reasoning.
    """)


@app.cell
def _(mo):
    review_question_1 = mo.ui.radio(
        {
            "The null hypothesis has a 3% probability of being true": "null_probability",
            "Assuming H₀ and the model, data at least this extreme have probability 0.03": "conditional",
            "Chance caused the result with probability 3%": "chance",
        },
        value=None,
        label="Choose one answer",
    )
    mo.vstack(
        [
            mo.md("**1. What does a two-sided p-value of 0.03 mean?**"),
            review_question_1,
        ]
    )
    return (review_question_1,)


@app.cell
def _(review_feedback, review_question_1):
    review_feedback(
        review_question_1.value,
        correct_value="conditional",
        correct_text="**Correct.** The p-value conditions on the null model and describes the statistic or more extreme values.",
        incorrect_text="A p-value does not assign probabilities to hypotheses or to 'chance.' It is a tail probability under the assumed null model.",
    )


@app.cell
def _(mo):
    review_question_2 = mo.ui.radio(
        {
            "A treatment is expected only to reduce the response, decided before data collection": "lower",
            "Either an increase or decrease would be scientifically important": "two_sided",
            "The observed mean happened to be lower": "posthoc",
        },
        value=None,
        label="Choose one answer",
    )
    mo.vstack(
        [
            mo.md(
                "**2. Which situation justifies a pre-specified lower-tailed alternative?**"
            ),
            review_question_2,
        ]
    )
    return (review_question_2,)


@app.cell
def _(review_feedback, review_question_2):
    review_feedback(
        review_question_2.value,
        correct_value="lower",
        correct_text="**Correct.** Direction follows the scientific question and must be specified before seeing the result.",
        incorrect_text="A two-sided question needs a two-sided analysis, and choosing direction because of the observed sign is data-dependent.",
    )


@app.cell
def _(mo):
    review_question_3 = mo.ui.radio(
        {
            "Use z because every sample mean is normal": "always_z",
            "Use t because the population SD is estimated from the sample": "use_t",
            "Either distribution gives exactly the same p-value at n = 10": "same",
        },
        value=None,
        label="Choose one answer",
    )
    mo.vstack(
        [
            mo.md(
                "**3. In the birth-weight study, the population SD is unknown and n = 10. Why use a t test?**"
            ),
            review_question_3,
        ]
    )
    return (review_question_3,)


@app.cell
def _(review_feedback, review_question_3):
    review_feedback(
        review_question_3.value,
        correct_value="use_t",
        correct_text="**Correct.** Estimating the denominator adds uncertainty represented by the t distribution with n − 1 degrees of freedom.",
        incorrect_text="The z test assumes the population SD is known. At small n, the t distribution's heavier tails materially change the tail probability.",
    )


@app.cell
def _(mo):
    review_question_4 = mo.ui.radio(
        {
            "Reject in the matching two-sided 5% test": "reject",
            "Fail to reject because the interval is wide": "fail",
            "The test decision cannot be related to the interval": "unrelated",
        },
        value=None,
        label="Choose one answer",
    )
    mo.vstack(
        [
            mo.md(
                "**4. A matching 95% confidence interval excludes the null value. What is the 5% two-sided test decision?**"
            ),
            review_question_4,
        ]
    )
    return (review_question_4,)


@app.cell
def _(review_feedback, review_question_4):
    review_feedback(
        review_question_4.value,
        correct_value="reject",
        correct_text="**Correct.** With the same model, standard error, and tail convention, exclusion from the 95% interval matches rejection at α = 0.05.",
        incorrect_text="The matched interval and two-sided test encode the same boundary. Excluding the null value corresponds to p ≤ 0.05.",
    )


@app.cell
def _(mo):
    review_question_5 = mo.ui.radio(
        {
            "Sensitivity ↔ power; specificity ↔ 1 − α": "correct_map",
            "Sensitivity ↔ α; specificity ↔ β": "reversed",
            "Sensitivity and specificity determine the posterior disease probability without prevalence": "posterior",
        },
        value=None,
        label="Choose one answer",
    )
    mo.vstack(
        [
            mo.md(
                "**5. Which mapping correctly connects diagnostic testing with hypothesis-test errors?**"
            ),
            review_question_5,
        ]
    )
    return (review_question_5,)


@app.cell
def _(review_feedback, review_question_5):
    review_feedback(
        review_question_5.value,
        correct_value="correct_map",
        correct_text="**Correct.** Sensitivity is the probability of detection when disease is present, analogous to power; specificity is the true-negative rate, 1 − α.",
        incorrect_text="False positives correspond to α and false negatives to β. Predictive probabilities additionally require prevalence or prior odds.",
    )


@app.cell
def _(mo):
    mo.md(r"""
    ---

    ## 9. Summary and bridge

    - A null hypothesis determines the reference distribution of a test statistic.
    - A p-value is $P(\text{statistic at least as extreme as observed}\mid H_0)$,
      using an extremeness rule defined by the test and alternative.
    - Choose direction, assumptions, and $\alpha$ before inspecting the result.
    - A one-sample z test uses a known population SD; a t test represents uncertainty
      from estimating that SD and has $n-1$ degrees of freedom.
    - A matched two-sided test and confidence interval give the same boundary, while
      the interval also communicates direction and precision.
    - $\alpha$ is the type-I error probability under a true null; $\beta$ is the
      type-II error probability for a specified alternative; power is $1-\beta$.
    - Statistical significance, effect magnitude, uncertainty, study quality, and
      scientific relevance are separate considerations and should be reported together.
    - Selective analyses and repeated looks change error rates unless the procedure
      accounts for them.

    **Next:** Chapter 5 applies this framework to independent groups, paired data,
    correlations, counts, rank-based alternatives, equivalence, ANOVA, interactions,
    multiple comparisons, and post-hoc procedures.
    """)


if __name__ == "__main__":
    app.run()
