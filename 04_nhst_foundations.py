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
        difference = float(observed_mean - null_mean)
        critical = reference.isf(alpha / 2 if alternative == "two-sided" else alpha)
        margin = float(critical * standard_error)
        interval = (
            -np.inf if alternative == "less" else difference - margin,
            np.inf if alternative == "greater" else difference + margin,
        )
        return {
            "difference": difference,
            "standardized_effect": difference / scale,
            "difference_interval": interval,
            "interval_width": float(interval[1] - interval[0]),
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
        sample_size, repetitions, strategy, alpha, seed, max_tests=5
    ):
        if strategy == "interim" and not 1 <= max_tests <= sample_size - 4:
            raise ValueError("Repeated checks require 1 to n − 4 distinct tests.")
        simulation_rng = np.random.default_rng(seed)
        reported_pvalues = np.empty(repetitions, dtype=float)
        test_count = 1 if strategy == "single" else max_tests
        look_sizes = np.linspace(sample_size, 5, test_count, dtype=int)
        rejection_counts = np.zeros(test_count, dtype=int)
        chunk_size = 500
        for chunk_start in range(0, repetitions, chunk_size):
            chunk_stop = min(chunk_start + chunk_size, repetitions)
            chunk_count = chunk_stop - chunk_start
            if strategy == "single":
                simulated_samples = simulation_rng.normal(
                    0.0, 1.0, size=(chunk_count, sample_size)
                )
                test_pvalues = one_sample_t_pvalues(simulated_samples)[:, None]
            elif strategy == "interim":
                simulated_samples = simulation_rng.normal(
                    0.0, 1.0, size=(chunk_count, sample_size)
                )
                test_pvalues = np.column_stack(
                    [
                        one_sample_t_pvalues(simulated_samples[:, :look_size])
                        for look_size in look_sizes
                    ]
                )
            else:
                simulated_samples = simulation_rng.normal(
                    0.0, 1.0, size=(chunk_count, test_count, sample_size)
                )
                test_pvalues = one_sample_t_pvalues(simulated_samples)
            cumulative_pvalues = np.minimum.accumulate(test_pvalues, axis=1)
            rejection_counts += (cumulative_pvalues <= alpha).sum(axis=0)
            reported_pvalues[chunk_start:chunk_stop] = cumulative_pvalues[:, -1]
        curve_rates = rejection_counts / repetitions
        curve_mcse = np.sqrt(curve_rates * (1 - curve_rates) / repetitions)
        rejection_rate = float(np.mean(reported_pvalues <= alpha))
        monte_carlo_se = float(
            np.sqrt(rejection_rate * (1.0 - rejection_rate) / repetitions)
        )
        return {
            "p_values": reported_pvalues,
            "sample_size": int(sample_size),
            "repetitions": int(repetitions),
            "strategy": strategy,
            "alpha": float(alpha),
            "rejection_rate": rejection_rate,
            "monte_carlo_se": monte_carlo_se,
            "seed": int(seed),
            "test_counts": np.arange(1, test_count + 1),
            "curve_rates": curve_rates,
            "curve_mcse": curve_mcse,
            "look_sizes": look_sizes if strategy == "interim" else None,
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
    Does a treatment change enzyme activity, or could the observed difference be
    consistent with ordinary variation between samples? **Null-hypothesis
    significance testing (NHST)** helps us assess this question. We start with a
    specific claim, often “no change”, and ask how unusual our result would be if
    that claim and the test's assumptions were correct.

    By the end of the chapter, you should be able to:

    1. explain a null hypothesis, test statistic, and p-value in your own words;
    2. choose a test for a change in one direction or in either direction;
    3. use z and t tests to compare a sample mean with a reference value;
    4. explain how a confidence interval relates to a test result;
    5. distinguish false alarms, missed effects, and the ability to detect an effect;
    6. recognize misleading interpretations of p-values and problems caused by
       choosing analyses after seeing the results.

    > **Keep the biological question in view:** A test result alone cannot tell
    > you whether an effect is large enough to matter. Report the estimated change
    > in meaningful units, such as enzyme activity or growth rate, together with a
    > confidence interval showing its uncertainty.

    **Prerequisites:** probability distributions, sampling distributions, standard
    errors, t-distributions, and confidence intervals from Chapter 2.
    """)


@app.cell
def _(mo):
    mo.md(r"""
    ---

    ## 1. Null models and exact p-values

    The **null hypothesis**, written $H_0$ (“H zero”), is the specific claim we
    test. For example, it might say that mean enzyme activity equals a reference
    value. The **alternative hypothesis**, $H_1$, describes the change we are
    looking for, such as higher activity or a difference in either direction.

    The null hypothesis together with assumptions about how measurements vary
    forms the **null model**. A **test statistic** is a number calculated from the
    data, such as a count or a mean difference. Its **null distribution** describes
    the values we would expect across repeated studies if the null model were true.

    We begin with coin tosses because their probabilities are easy to calculate.
    For a fair coin, $H_0:p=0.5$, where $p$ is the probability of heads on each toss.
    With independent tosses and a common heads probability, the count follows
    the **binomial model** from Chapter 2, Section 3. Here that model supplies a
    null distribution against which to assess the observed count.

    A **p-value** is the probability, assuming the null model is true, of obtaining
    the observed test statistic or a result more extreme in the direction(s) being
    tested. For $H_1:p>0.5$, “more extreme” means more heads. With 13 heads in 20
    tosses, we therefore add the probabilities of 13, 14, ..., 20 heads. The sum is
    about 0.132. **Exact** means we calculate directly from the binomial model
    without a large-sample approximation; the model assumptions still matter.
    Here, $p$ in $H_0:p=0.5$ is the heads probability, not the p-value of the test.

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
        "The orange bars mark the observed count and all counts further in the tested direction. "
        "Their probabilities sum to the p-value."
        if coin_direction != "two-sided"
        else "For this two-sided binomial test, the orange bars mark counts that are "
        "no more likely than the observed count under the null model. Their "
        "probabilities sum to the p-value. Counts are whole numbers, so the "
        "p-value changes in steps and need not equal twice a one-sided p-value."
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
            "Assuming 20 independent fair-coin tosses, the probability of 13 or more heads is about 0.132.": "conditional",
            "After observing 13 heads, the probability that the coin is fair is about 0.132.": "reversed",
            "There is a 13.2% probability that chance caused the observation.": "chance",
        },
        value=None,
        label="Choose the valid interpretation",
    )
    mo.vstack(
        [
            mo.md(
                "### Checkpoint: what does the p-value assume, and what does it tell us?"
            ),
            conditional_checkpoint,
        ]
    )
    return (conditional_checkpoint,)


@app.cell
def _(conditional_checkpoint, review_feedback):
    conditional_feedback = review_feedback(
        conditional_checkpoint.value,
        correct_value="conditional",
        correct_text="**Correct.** If we repeated sets of 20 independent fair-coin tosses, about 13.2% would produce 13 or more heads. We include the observed count and all higher counts, not just exactly 13 heads.",
        incorrect_text={
            "reversed": "The calculation starts by assuming the coin is fair and asks how often 13 or more heads would occur. It does not calculate the probability that the coin is fair after seeing the tosses.",
            "chance": "The 13.2% describes how often a fair coin would give 13 or more heads in 20 independent tosses. It does not give the probability that chance caused this particular result.",
        },
    )
    conditional_feedback  # noqa: B018


@app.cell
def _(mo):
    mo.md(r"""
    ---

    ## 2. One-sided and two-sided tails

    A **tail** is an end of a distribution, where unusually low or high values lie.
    A **one-sided test** asks about a change in one chosen direction: a lower-tailed
    test looks for a decrease, and an upper-tailed test looks for an increase.
    A **two-sided test** looks for either. Choose the direction from the biological
    question before inspecting the data. If either increased or decreased enzyme
    activity would answer your question, use a two-sided test even if you expect
    a decrease.

    Use the density and **cumulative distribution function (CDF)** from Chapter 2
    to read the p-value: the left plot shades the relevant area; the right gives
    $F(x)$, the probability at or below $x$. Thus $F(2)$ is the left-tail area
    and $1-F(2)$ the right-tail area.

    Compare the **standard normal** reference with the heavier-tailed
    **t-distribution** from Chapter 2, Section 5. For a one-sample t test,
    **degrees of freedom (df)** are $n-1$; larger df bring the t curve closer to
    the normal curve. The explorer also previews the asymmetric chi-squared
    ($\chi^2$) and F distributions used in Chapter 5. The F distribution has
    two df settings, one for each variation estimate being compared.

    **Try this:** use the standard normal model and statistic 2.0. Compare
    **Upper tail**, **Lower tail**, and **Two-sided**; match the shaded area to
    $1-F(2)$, $F(2)$, or the two tails combined. Move the statistic to −2 and
    check which directional tail becomes small. Switch to a t-distribution and
    reduce degrees of freedom to see the effect of heavier tails. The first
    degrees-of-freedom control affects t, chi-squared, and F models; the second
    affects F only, and neither affects the standard normal. Switching models
    resets the statistic and tail controls to suitable defaults. Changes update
    immediately; the selected tail determines which area is shaded.
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
                "This curve has different shapes on its two sides, so the explorer shows one tail at a time. Whether small values, large values, or both count against the null depends on the particular test. Doubling the smaller tail area is not a general rule for every test."
            ),
            kind="warn",
        )
        if tail_name in {"chi2", "f"}
        else mo.callout(
            mo.md(
                "These curves are symmetric around zero. The two-sided p-value adds the areas at least as far from zero as the observed statistic, on both sides."
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

            Calculate each difference as intervention minus reference, and decide the
            scientific question before examining outcomes.

            1. Does a treatment **reduce** mean blood pressure?
            2. Does a mutation **decrease** enzyme activity?
            3. Does a culture condition **change** mean growth rate in either direction?
            """),
            scenario_direction,
        ]
    )
    return (scenario_direction,)


@app.cell
def _(review_feedback, scenario_direction):
    scenario_result = review_feedback(
        scenario_direction.value,
        correct_value="correct",
        correct_text="**Correct.** For intervention minus reference, reductions in blood pressure and enzyme activity correspond to negative differences and lower-tailed alternatives. A growth-rate change in either direction requires a two-sided alternative. Reversing the subtraction would reverse the one-sided directions.",
        incorrect_text={
            "wrong_1": "The blood-pressure and growth-rate directions are right. For mutation minus reference, decreased enzyme activity gives a negative difference, so the second alternative must also be lower-tailed, not greater-tailed.",
            "wrong_2": "The enzyme-activity direction is right. The first question asks specifically for a reduction, so its alternative is lower-tailed; the third includes both an increase and a decrease, so it must be two-sided rather than greater-tailed.",
        },
    )
    scenario_result  # noqa: B018


@app.cell
def _(mo):
    mo.md(r"""
    ---

    ## 3. A testing workflow and one-sample mean tests

    Plan the test around the biological question:

    1. define what you want to measure, which population you want to learn about,
       and how you will obtain independent observations;
    2. state $H_0$ and $H_1$, choose a test suited to the measurements and design,
       and set the decision threshold $\alpha$ (“alpha”);
    3. collect the data, check the assumptions that can be assessed from them,
       and calculate the test statistic and p-value;
    4. compare the p-value with the planned threshold;
    5. report the estimated effect, confidence interval, test result, and limitations.

    The **significance level**, $\alpha$, is the threshold used to make the decision.
    This chapter calls a result **statistically significant** when $p\leq\alpha$,
    and we **reject $H_0$**. Otherwise, we **fail to reject $H_0$**: the data do not
    provide enough evidence against it with this test. This does not establish that
    $H_0$ is true. Section 5 explains how $\alpha$ relates to false alarms.

    Use the independent biological units identified in Chapter 2, Section 1,
    as the sample size; technical replicates do not increase that count.
    Independence comes from the design, not a normality check. Choosing the test
    direction or threshold after seeing the result can increase false alarms.

    ### Worked example: birth weight

    In this teaching example, ten newborns have a mean birth weight of 2200 g,
    compared with a fixed reference of 3000 g. A **one-sample test** compares the
    mean of one sample with such a reference. Let $\bar x$ be the observed sample
    mean, $\mu$ (“mu”) the population mean we want to learn about, and $\mu_0$ the
    value proposed by $H_0$. Here the estimated difference is $2200-3000=-800$ g.

    Apply Chapter 2's distinction: **SD** describes variation between newborns;
    **SE** describes sampling uncertainty in their mean. The test statistic divides
    the observed mean difference by its SE.

    With known population SD $\sigma$, use $SE=\sigma/\sqrt n$ and a **z test**.
    With SD estimated by sample $s$, use the estimated SE $s/\sqrt n$ and a
    **t test** with $df=n-1$ (here 9). These are the same two uncertainty models
    used for the mean intervals in Chapter 2, Section 5.

    **Try this:** begin with the t test and move the observed mean towards the
    null mean. The difference in standard-error units approaches zero and the
    two-sided p-value grows. Restore the original means, then increase $n$ or decrease the
    sample SD to see stronger evidence for the same mean difference. Compare
    z and t tests at the same SD: set the visible SD control, switch test type,
    then set the newly visible SD control to the same value. Each test retains
    its own SD setting, and only the selected test's SD enters its calculation. Finally vary
    $\alpha$; the decision can change while the p-value stays fixed. These sliders
    describe hypothetical sample summaries rather than resampling raw birth weights.

    **Effect versus precision:** keep both means and the selected SD fixed, then
    compare $n=10$ with $n=40$. The raw difference and standardized effect stay
    constant, while the standard error halves and the interval narrows. For a
    nonzero difference in a two-sided test, the p-value falls. This is a controlled illustration,
    not a sequence of newly sampled data.

    The standardized effect expresses the observed difference in **SD units**.
    For the t test, $d=(\bar x-\mu_0)/s$ is **Cohen's d for a one-sample comparison**:
    it uses the SD estimated from the sample. For the z test,
    $(\bar x-\mu_0)/\sigma$ uses the known population SD instead.
    The test statistic divides by **SE** rather than SD, so it changes with n
    even when the standardized effect stays fixed.
    Neither a small p-value nor a large n makes the observed effect larger or
    establishes biological importance.

    The displayed confidence interval, with confidence level $1-\alpha$ (95% when
    $\alpha=0.05$), estimates $\mu-\mu_0$ in grams, treating
    $\mu_0$ as fixed. It is two-sided for “Different mean”, a lower bound for
    “Higher mean”, and an upper bound for “Lower mean”. A one-sided interval
    gives a limit in just one direction; the other end is unbounded, so its total
    width is shown as infinite.

    **Connect the displays:** zero means no difference from the reference.
    Compare its position relative to the interval with the test decision.
    Section 4 explains the correspondence, including the exact boundary.

    These calculations assume independent observations. With small samples, the
    z and t tests used here require a normally distributed population for their
    stated error rates to be exact. With larger samples they can be useful
    approximations, but strong skew or extreme observations may still cause problems.
    Here we treat 3000 g as a fixed reference; comparing with a mean estimated from
    another sample would also require accounting for that sample's uncertainty.
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

    anatomy_interval_low, anatomy_interval_high = anatomy_result["difference_interval"]
    anatomy_interval_text = (
        f"(−∞, {anatomy_interval_high:+.1f}] g"
        if anatomy_alternative == "less"
        else f"[{anatomy_interval_low:+.1f}, ∞) g"
        if anatomy_alternative == "greater"
        else f"[{anatomy_interval_low:+.1f}, {anatomy_interval_high:+.1f}] g"
    )
    anatomy_effect_label = (
        "Effect / population σ" if anatomy_test_type == "z" else "Sample Cohen's d"
    )
    anatomy_rows = pd.DataFrame(
        {
            "Component": [
                "Difference",
                anatomy_effect_label,
                f"{100 * (1 - anatomy_alpha):.0f}% CI for μ − μ₀",
                "Interval width",
                "Standard error",
                "Test statistic",
                "Degrees of freedom",
                "p-value",
            ],
            "Value": [
                f"{anatomy_result['difference']:+.0f} g",
                f"{anatomy_result['standardized_effect']:+.3f}",
                anatomy_interval_text,
                (
                    f"{anatomy_result['interval_width']:.1f} g"
                    if anatomy_alternative == "two-sided"
                    else "∞ (one-sided)"
                ),
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
        column_widths={"Component": 180, "Value": 210},
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
        ]
    )
    two_column_panel(
        anatomy_controls,
        mo.vstack([anatomy_figure, anatomy_table, anatomy_report]),
        widths=(1, 2),
    )


@app.cell
def _(mo):
    test_statistic_derivation = mo.md(r"""
    ### Where the formulas come from: difference divided by uncertainty

    Under $H_0:\mu=\mu_0$, the sample mean has expected value $\mu_0$ and standard
    error $\sigma/\sqrt n$. Subtract the null value and divide by that standard error:

    $$Z=\frac{\bar X-\mu_0}{\sigma/\sqrt n}.$$

    Here $\bar X$ represents the sample mean. If the population SD is unknown,
    replace it with the sample SD, $S$ (written $s$ for an observed sample):

    $$T=\frac{\bar X-\mu_0}{S/\sqrt n},\qquad df=n-1.$$

    When independent observations come from a normal population, $Z$ follows the
    standard normal distribution and $T$ follows a t-distribution with $n-1$ degrees
    of freedom. The t curve allows more values far from zero because the SD in the
    denominator is itself estimated from the sample. Both statistics express the
    observed difference in **standard-error units**.
    """)
    mo.accordion(
        {"Derivation — one-sample z and t statistics": test_statistic_derivation}
    )


@app.cell
def _(mo):
    mo.md(r"""
    ---

    ## 4. Tests and confidence intervals

    Chapter 2 introduced confidence intervals through repeated-sampling coverage.
    Here we connect them to tests: which proposed population means are compatible
    with the data at the matching threshold? This panel shows an interval for
    $\mu$ itself, rather than for the difference $\mu-\mu_0$ shown above.

    For the same data and assumptions, a two-sided 95% t interval and a two-sided
    t test at $\alpha=0.05$ answer related questions. A null mean strictly outside
    the interval gives $p<0.05$; one strictly inside gives $p>0.05$. At an endpoint,
    $p=0.05$, which this notebook counts as rejection. The confidence level, test
    type, and choice of one or two sides must match for this connection to hold.

    The interval also helps answer the biological question: does the range include
    only changes large enough to matter, or both negligible and important changes?
    This information is lost when we report only “significant” or “not significant”.

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

    Even a well-designed study can lead to an incorrect decision because samples
    vary. There are two kinds of error:

    - A **type-I error (false positive or false alarm)** occurs when we reject a
      true null hypothesis. A test at $\alpha=0.05$ is designed to do this in at
      most 5% of repeated studies when $H_0$ and the assumptions hold. Some tests,
      including tests of whole-number counts, may have a lower false-alarm rate.
    - A **type-II error (false negative or missed effect)** occurs when we fail to
      reject $H_0$ even though a specified effect is present. Its probability is
      $\beta$ (“beta”). **Power**, $1-\beta$, is the probability of detecting that
      effect by rejecting $H_0$. For example, 80% power means about 80 of 100
      repeated studies would detect that particular effect under the assumptions.

    Neither rate tells us the probability that the conclusion of this one study
    is wrong. Power depends on the true effect: a design may readily detect a large
    change in enzyme activity but often miss a small change.

    Use power when planning a study. Choose a change that would matter biologically
    and a realistic SD, then work out the sample size needed. For an effect in the
    tested direction, power increases with a larger effect, less variation, more
    independent observations, or a higher $\alpha$. Raising $\alpha$ also allows
    more false alarms. After collecting data, use the effect estimate and confidence
    interval to assess what the study tells you; recalculating power from the
    observed effect adds little information.

    This explorer uses a **z test with known population SD** and independent,
    normally distributed observations. Its sample-size result applies to that
    simplified design. A study using a t test or a different design needs a
    corresponding power calculation. For a one-sided test, the explorer places
    the true change in the chosen direction.

    **Try this:** keep the effect and SD at their defaults and compare $n=10$
    with $n=40$. The **rejection region** contains the observed mean differences
    that would produce a significant result. Under the curve for a real effect,
    the area in that region is power. Restore $n=10$, halve the true change, and check the loss of power. Next raise the
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

    A laboratory test can also make false-positive and false-negative decisions.
    In this simplified illustration, a concentration above a chosen threshold is
    called positive. We compare people without porphyria with people who have
    porphyria, a disorder of haem production. The concentration distributions and
    thresholds here are invented for teaching, not clinical reference values.

    - **Sensitivity** is the fraction of people with the disease who test positive.
      It is analogous to power, $1-\beta$. The remaining fraction, $\beta$, consists
      of false negatives: people with the disease whose tests are negative.
    - **Specificity** is the fraction of people without the disease who test
      negative. It is analogous to $1-\alpha$. The remaining fraction, $\alpha$,
      consists of false positives: people without the disease whose tests are positive.

    The right-hand **ROC curve** (receiver operating characteristic curve) plots
    sensitivity against the false-positive rate across thresholds. Each point
    represents one threshold; the highlighted point matches the slider. The upper
    left corner represents perfect detection with no false positives.

    The analogy treats “no disease” like $H_0$ and “disease present” like $H_1$.
    Sensitivity is not the probability that a person with a positive result has the
    disease. That probability also depends on **prevalence**, the proportion of
    people with the disease in the population being tested. Chapter 6 returns to
    this distinction.

    **Try this:** move the positive-test threshold from 76 towards 40, then
    towards 120 µmol/L. A lower threshold detects more people with the disease but
    also labels more people without the disease positive. Follow sensitivity and
    false-positive rate on the distribution plot and the linked ROC point. Raising the
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
    small p-value does not establish a large or important effect, or rule out
    systematic errors such as measuring all treated samples in one batch and all
    controls in another.
    Nor does a p-value tell you the probability that 'chance caused' your result.
    Instead, the null model describes how observations would vary across samples. The
    p-value is the probability, under that null model, of a test statistic at
    least as extreme as the observed one in the direction(s) of the alternative.

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
    of a hypothesis, an uncertain result with evidence of similar effects, or
    statistical significance with biological importance.
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
def _(misconception_choice, review_feedback):
    misconception_corrections = {
        "hypothesis_probability": (
            "A p-value starts by assuming the null model is true and asks about possible data. It does not calculate the probability that the null hypothesis is true after seeing the data."
        ),
        "alternative_probability": (
            "Neither $p$ nor $1-p$ assigns a probability to a hypothesis. To calculate probabilities for hypotheses, we need a different approach that combines assumptions about their probabilities before the study with a model for the data (Chapter 6)."
        ),
        "equivalence": (
            "The study may simply be too imprecise to detect a difference. To establish equivalence—effects similar enough for the biological purpose—define the largest difference you would consider negligible before the study, then use an analysis that tests whether the effect lies within those limits. A difference-test p-value above 0.05 is insufficient."
        ),
        "importance": (
            "A tiny effect can have a small p-value with a large, precise sample. Inspect the effect in measurement units, its interval, and the biological context to judge importance; statistical significance alone cannot do that."
        ),
    }
    misconception_feedback = review_feedback(
        misconception_choice.value,
        correct_value="valid",
        correct_text="**Correct.** Start by assuming the null model is true. The test and the direction chosen before the study determine which results count as at least as extreme as the observed one. The p-value adds up their probabilities under that model.",
        incorrect_text=misconception_corrections,
        unanswered_text="Select an interpretation above.",
    )
    misconception_feedback  # noqa: B018


@app.cell
def _(mo):
    mo.md(r"""
    ### Interactive laboratory: more tests, more false alarms

    **There is no real effect in any of these simulated studies.** The population
    mean is always zero, but individual measurements vary randomly around it.
    A statistically significant result is therefore a **false positive**: the test
    suggests a difference when none exists.
    We generate normally distributed measurements and use two-sided t tests,
    so a change in either direction can count as significant.

    Imagine measuring the change in blood pressure after a treatment that actually
    has no effect. We can use three testing strategies:

    - **Test once at the end.** Plan to study 20 people, collect all 20 measurements,
      and test whether the mean change differs from zero. There is just one chance
      to get a significant result.
    - **Test repeatedly as data arrive.** Test after 5 people, again after 10, and
      again after all 20. Count the study as significant if any test is significant.
      The later tests include the earlier participants, so these are related checks
      of the same data, not three separate studies.
    - **Test several independent outcomes.** Imagine separate experiments measuring
      blood-pressure change, heart-rate change, and cholesterol change, each in a
      different group of people. None has a real effect. Test each outcome and report
      whichever gives the smallest p-value. This simulation treats the outcomes as
      independent: knowing one result tells us nothing about the others. Measurements
      taken from the same people would often be related instead.

    In every strategy, each test uses the same threshold α. For example, α = 0.05
    allows a 5% false-positive rate for **one planned test**. It does not guarantee
    a 5% rate for a study that tries several tests and reports any significant result.
    We make no correction for the extra tests here, so we can see what happens.

    **Try this:** run **Test once at the end** with α = 0.05. About 5 in 100 studies
    should give a false alarm. Then select **Test several independent outcomes**,
    set the number of tests to 10, and rerun. About 40 in 100 studies will give at
    least one false alarm. Try **Test repeatedly as data arrive** too: the checks share data,
    so the increase is different from that for independent outcomes.

    **Reading the figures:** the first plot shows the fraction of studies with at
    least one false alarm. Its first point always represents one test. The small
    vertical bars show uncertainty due to the limited number of simulated studies
    (approximate 95% intervals for each point). More simulated studies make these
    bars smaller; they do not remove the problem of false alarms from extra tests.
    The second plot shows the reported p-values: one per study, taking the smallest
    if several tests were tried. The shaded area marks significant results. The
    vertical axis is density: the area of a bar, rather than its height alone,
    represents the fraction of studies
    in that p-value range. With one planned test and no effect, p-values spread evenly
    from 0 to 1 in this simulation; selecting the smallest of several shifts them
    towards zero.

    Repeated checks start after at least five people, with at least one new person
    between checks. With 10 people, at most six checks are possible (at 5 through
    10 people). To try ten checks, set the final sample size to at least 14.
    The number-of-tests slider adjusts to the selected strategy and sample size.

    Click **Run / resimulate studies** to apply changed settings. Until then, the
    figures show the previous run. Running again gives slightly different results
    because new random measurements are generated.
    """)


@app.cell
def _(mo):
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
            "Test once at the end": "single",
            "Test repeatedly as data arrive": "interim",
            "Test several independent outcomes": "outcomes",
        },
        value="Test once at the end",
        label="Testing strategy",
        full_width=True,
    )
    return (
        simulation_alpha,
        simulation_repetitions,
        simulation_sample_size,
        simulation_strategy,
    )


@app.cell
def _(mo, simulation_sample_size, simulation_strategy):
    simulation_test_limit = (
        min(10, int(simulation_sample_size.value) - 4)
        if simulation_strategy.value == "interim"
        else 1
        if simulation_strategy.value == "single"
        else 10
    )
    simulation_max_tests = mo.ui.slider(
        1,
        simulation_test_limit,
        value=min(5, simulation_test_limit),
        show_value=True,
        full_width=True,
        disabled=simulation_test_limit == 1,
        label=f"Number of tests (maximum {simulation_test_limit} for these settings)",
    )
    return (simulation_max_tests,)


@app.cell
def _(mo, simulate_pvalue_experiment):
    initial_simulation_result = simulate_pvalue_experiment(
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
    simulation_repetitions,
    simulation_sample_size,
    simulation_strategy,
    simulation_max_tests,
):
    def run_pvalue_simulation(_value):
        fresh_seed = int(np.random.default_rng().integers(0, 2**32 - 1))
        set_simulation_result(
            simulate_pvalue_experiment(
                sample_size=int(simulation_sample_size.value),
                repetitions=int(simulation_repetitions.value),
                strategy=simulation_strategy.value,
                alpha=float(simulation_alpha.value),
                seed=fresh_seed,
                max_tests=int(simulation_max_tests.value),
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
    simulation_repetitions,
    simulation_run_button,
    simulation_sample_size,
    simulation_strategy,
    simulation_max_tests,
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
    if pvalue_simulation_result["strategy"] == "single":
        pvalue_axis.axhline(
            1.0,
            color=COLORS["gray"],
            linestyle="--",
            linewidth=1.5,
            label="Expected pattern with one test and no effect",
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

    pvalue_curve_figure, pvalue_curve_axis = plt.subplots(figsize=FIGURE_SIZE_STANDARD)
    pvalue_counts = pvalue_simulation_result["test_counts"]
    pvalue_rates = pvalue_simulation_result["curve_rates"]
    pvalue_margins = 1.96 * pvalue_simulation_result["curve_mcse"]
    pvalue_curve_axis.errorbar(
        pvalue_counts,
        pvalue_rates,
        yerr=np.vstack(
            [
                np.minimum(pvalue_margins, pvalue_rates),
                np.minimum(pvalue_margins, 1 - pvalue_rates),
            ]
        ),
        fmt="o-",
        capsize=4,
        color=COLORS["blue"],
        label="Simulated false alarms (95% intervals)",
    )
    pvalue_curve_axis.axhline(
        pvalue_simulation_result["alpha"],
        color=COLORS["gray"],
        linestyle=":",
        label="Expected false alarms with one test: α",
    )
    if pvalue_simulation_result["strategy"] == "outcomes":
        pvalue_curve_axis.plot(
            pvalue_counts,
            1 - (1 - pvalue_simulation_result["alpha"]) ** pvalue_counts,
            "--",
            color=COLORS["vermillion"],
            label="Expected for independent outcomes",
        )
    pvalue_curve_axis.set(
        xlabel="Number of tests included (k)",
        ylabel="Fraction of studies with a false alarm",
        xticks=pvalue_counts,
        xlim=(0.7, max(pvalue_counts) + 0.3),
        ylim=(0, 1),
    )
    pvalue_curve_axis.grid(axis="y", linestyle=":", alpha=0.35)
    pvalue_curve_axis.legend(frameon=False, fontsize=8)
    pvalue_curve_figure.tight_layout()
    pvalue_schedule_note = mo.md(
        "**How to read this curve:** start with the final test, then add earlier checks. Sample sizes, in the order added: "
        + ", ".join(str(size) for size in pvalue_simulation_result["look_sizes"])
        + ". Each point keeps all previous checks. Checks are at least one person apart and start no earlier than 5 people, so small samples allow fewer checks."
        if pvalue_simulation_result["strategy"] == "interim"
        else "**Completed run:** the first point tests one outcome; each further point adds another independent outcome."
        if pvalue_simulation_result["strategy"] == "outcomes"
        else "**Completed run:** one planned test after all measurements are collected."
    )

    strategy_labels = {
        "single": "Test once at the end",
        "interim": "Test repeatedly as data arrive",
        "outcomes": "Test several independent outcomes",
    }
    pvalue_rate = pvalue_simulation_result["rejection_rate"]
    pvalue_mcse = pvalue_simulation_result["monte_carlo_se"]
    pvalue_note_kind = (
        "info" if pvalue_simulation_result["strategy"] == "single" else "warn"
    )
    pvalue_note = mo.callout(
        mo.md(
            f"The last run used **{strategy_labels[pvalue_simulation_result['strategy']]}**, "
            f"{len(pvalue_simulation_result['test_counts'])} test(s), "
            f"no real effect, n = "
            f"{pvalue_simulation_result['sample_size']}, and "
            f"{pvalue_simulation_result['repetitions']:,} studies. The fraction with "
            f"at least one false alarm (p ≤ {pvalue_simulation_result['alpha']:.2f}) was "
            f"**{pvalue_rate:.1%} ± {1.96 * pvalue_mcse:.1%}** (approximate 95% "
            f"simulation margin)."
        ),
        kind=pvalue_note_kind,
    )
    pvalue_controls = mo.vstack(
        [
            simulation_sample_size,
            simulation_alpha,
            simulation_repetitions,
            simulation_strategy,
            simulation_max_tests,
            simulation_run_button,
            mo.callout(
                mo.md(
                    "Controls describe the **next** run. The figure remains stable until the button is clicked."
                ),
                kind="neutral",
            ),
            mo.stat(
                f"{pvalue_rate:.1%}",
                label="Studies with a false alarm",
                bordered=True,
            ),
            mo.stat(
                f"±{1.96 * pvalue_mcse:.1%}",
                label="95% simulation margin",
                bordered=True,
            ),
        ]
    )
    two_column_panel(
        pvalue_controls,
        mo.vstack(
            [pvalue_curve_figure, pvalue_schedule_note, pvalue_figure, pvalue_note]
        ),
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
        correct_text="**Correct.** If the null model were true, results at least this far from the null value, in either direction, would occur in 3% of repeated studies using this test.",
        incorrect_text={
            "null_probability": "The calculation assumes H₀ is true and obtains a "
            "probability for the observed statistic or more extreme values; it does not calculate a "
            "probability that H₀ is true.",
            "chance": "“Chance caused the result” is not an event defined by this test. The value 0.03 is the "
            "probability of a statistic at least as extreme, in either tested direction, under H₀ "
            "and the model assumptions.",
        },
    )


@app.cell
def _(mo):
    review_question_2 = mo.ui.radio(
        {
            "The pre-specified claim is a reduction; an increase would not count as evidence for that claim": "lower",
            "Either an increase or decrease would be scientifically important": "two_sided",
            "The observed mean happened to be lower": "posthoc",
        },
        value=None,
        label="Choose one answer",
    )
    mo.vstack(
        [
            mo.md(
                "**2. Which research question calls for a lower-tailed test chosen before seeing the data?**"
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
        correct_text="**Correct.** A lower-tailed test addresses the pre-specified claim of a reduction, with increases providing no evidence for that claim. Merely expecting a decrease is insufficient if either direction would answer the research question; unexpected increases can still matter scientifically.",
        incorrect_text={
            "two_sided": "If increases and decreases both answer the scientific question, the alternative "
            "should include both directions. A lower-tailed test would not test for an increase.",
            "posthoc": "Choosing the direction after seeing whether the result increased or decreased "
            "can increase false alarms. Specify the direction before examining "
            "outcomes.",
        },
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
                "**3. In the birth-weight example, assume independent observations from a normal population. The population SD is unknown and n = 10. Why use a one-sample t test?**"
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
        correct_text="**Correct.** Replacing σ with sample s adds uncertainty. Under the stated assumptions, (mean − null mean)/(s/√n) follows a t distribution with n − 1 = 9 degrees of freedom. Small n alone does not justify the normal-population assumption.",
        incorrect_text={
            "always_z": "A sample mean is not automatically normal. Even under a normal population model, "
            "replacing the unknown σ with sample s makes the standardized statistic "
            "t-distributed, with 9 degrees of freedom here.",
            "same": "At 9 degrees of freedom the t distribution has heavier tails than the standard normal. "
            "For the same nonzero statistic its two-sided p-value is larger; the distributions "
            "approach one another as degrees of freedom increase.",
        },
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
                "**4. A null mean lies strictly outside a two-sided 95% t confidence interval. What is the decision in the matching two-sided t test at α = 0.05?**"
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
        correct_text="**Correct.** Using the same data and assumptions, a null mean strictly outside the 95% t interval gives p < 0.05. At an endpoint, p = 0.05; this notebook counts equality as rejection.",
        incorrect_text={
            "fail": "Width describes precision, but the test decision depends on whether the interval "
            "includes the null value. A matching interval can be wide and still exclude that value, "
            "implying rejection.",
            "unrelated": "The interval contains the null means for which the matching two-sided t test gives p ≥ 0.05. A value "
            "strictly outside the 95% interval gives p < 0.05; at an endpoint p = 0.05.",
        },
    )


@app.cell
def _(mo):
    review_question_5 = mo.ui.radio(
        {
            "Sensitivity ↔ power; specificity ↔ 1 − α": "correct_map",
            "Sensitivity ↔ α; specificity ↔ β": "reversed",
            "Sensitivity and specificity alone tell us the probability of disease after a positive result": "posterior",
        },
        value=None,
        label="Choose one answer",
    )
    mo.vstack(
        [
            mo.md(
                "**5. Which statement correctly connects diagnostic testing with hypothesis-test errors?**"
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
        incorrect_text={
            "reversed": "Sensitivity is a true-positive rate, corresponding to power 1 − β; specificity is a "
            "true-negative rate, corresponding to 1 − α. The corresponding error rates are one minus these values.",
            "posterior": "Sensitivity and specificity describe results for people whose disease status is known. The probability of disease "
            "after a positive result also depends on prevalence: how common the disease is in the tested population.",
        },
    )


@app.cell
def _(mo):
    mo.md(r"""
    ---

    ## 9. Take-home messages

    - A null model describes what results we would expect if a specific claim,
      often “no change”, and the test's assumptions were true.
    - A p-value asks how often that model would produce a test statistic as extreme
      as the observed one or more so, in the direction(s) chosen before the study.
      It is not the probability that the null hypothesis is true.
    - Plan the direction, test, and decision threshold $\alpha$ before examining
      the results. Check that the design and measurements support the assumptions.
    - SD describes variation between observations; SE describes uncertainty in the
      sample mean. A one-sample z or t statistic expresses the difference from a
      reference mean in SE units. Use t when estimating population SD from the sample.
    - A confidence interval shows the size, direction, and precision of the
      estimated effect. Matching tests and intervals agree apart from the stated
      convention at the exact decision boundary.
    - $\alpha$ controls false alarms when the null is true. Power, $1-\beta$, is the
      chance of detecting a specified real effect; $\beta$ is the chance of missing it.
    - Statistical significance does not establish biological importance. Report
      the estimated effect, confidence interval, and study limitations together.
      A result that is not significant does not establish “no effect”.
    - Trying more tests and reporting whichever is significant increases false
      alarms unless the analysis accounts for those extra opportunities.

    **Next:** Chapter 5 applies these ideas to comparisons between groups, paired
    measurements, relationships between variables, and counts. It also covers how
    to compare several groups, account for multiple tests, and assess whether
    differences are small enough to be considered biologically negligible.
    """)


if __name__ == "__main__":
    app.run()
