# /// script
# requires-python = ">=3.14"
# dependencies = [
#     "marimo==0.23.15",
#     "matplotlib==3.11.1",
#     "numpy==2.5.1",
#     "pandas==3.0.5",
#     "scipy==1.18.0",
# ]
# ///

import marimo

__generated_with = "0.23.15"
app = marimo.App(width="medium", app_title="Bayesian statistics")


@app.cell
def _():
    from pathlib import Path

    import marimo as mo
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
        Path,
        compact_table,
        mo,
        np,
        pd,
        plt,
        review_feedback,
        stats,
        two_column_panel,
    )


@app.cell
def _(Path, np, pd):
    water_path = Path(__file__).parent / "Wasserqualitaet.csv"
    water_raw = pd.read_csv(water_path, encoding="utf-8-sig")
    flow_column = "Fließgeschwindigkeit (m/s)"
    oxygen_column = "Sauerstoffkonzentration (mg/l)"
    assert {flow_column, oxygen_column}.issubset(water_raw.columns)
    water_data = water_raw[[flow_column, oxygen_column]].rename(
        columns={flow_column: "flow_speed_m_s", oxygen_column: "oxygen_mg_l"}
    )
    assert len(water_data) == 24
    assert np.isfinite(water_data.to_numpy(dtype=float)).all()
    return (water_data,)


@app.cell
def _(np, stats):
    def diagnostic_details(prevalence, sensitivity, specificity, population):
        sick = population * prevalence
        healthy = population - sick
        true_positive = sick * sensitivity
        false_negative = sick * (1.0 - sensitivity)
        true_negative = healthy * specificity
        false_positive = healthy * (1.0 - specificity)
        positive = true_positive + false_positive
        negative = true_negative + false_negative
        return {
            "sick": float(sick),
            "healthy": float(healthy),
            "true_positive": float(true_positive),
            "false_positive": float(false_positive),
            "true_negative": float(true_negative),
            "false_negative": float(false_negative),
            "ppv": float(true_positive / positive),
            "npv": float(true_negative / negative),
        }

    def required_specificity(prevalence, sensitivity, target_ppv):
        false_positive_rate = (
            prevalence
            * sensitivity
            * (1.0 - target_ppv)
            / (target_ppv * (1.0 - prevalence))
        )
        return float(np.clip(1.0 - false_positive_rate, 0.0, 1.0))

    def dice_update(sides, prior, observations):
        sides_array = np.asarray(sides, dtype=int)
        posterior = np.asarray(prior, dtype=float)
        posterior = posterior / posterior.sum()
        last_prior = posterior.copy()
        last_likelihood = np.ones_like(posterior)
        last_unnormalized = posterior.copy()
        last_evidence = 1.0
        for observation in observations:
            last_prior = posterior.copy()
            last_likelihood = np.where(
                observation <= sides_array, 1.0 / sides_array, 0.0
            )
            last_unnormalized = last_prior * last_likelihood
            last_evidence = float(last_unnormalized.sum())
            posterior = last_unnormalized / last_evidence
        next_roll_probability = np.array(
            [
                sum(
                    posterior[index] / side
                    for index, side in enumerate(sides_array)
                    if value <= side
                )
                for value in range(1, int(sides_array.max()) + 1)
            ]
        )
        return {
            "prior": last_prior,
            "likelihood": last_likelihood,
            "unnormalized": last_unnormalized,
            "evidence": last_evidence,
            "posterior": posterior,
            "next_roll_probability": next_roll_probability,
        }

    def beta_binomial_predictive(alpha, beta, future_trials):
        outcomes = np.arange(future_trials + 1)
        probabilities = stats.betabinom.pmf(outcomes, future_trials, alpha, beta)
        return outcomes, probabilities / probabilities.sum()

    def reflect_unit_interval(values):
        folded = np.mod(values, 2.0)
        return np.where(folded <= 1.0, folded, 2.0 - folded)

    def random_walk_metropolis(
        log_target, starts, proposal_scales, draws, rng, reflect_indices=()
    ):
        initial = np.asarray(starts, dtype=float)
        scales = np.asarray(proposal_scales, dtype=float)
        if initial.ndim != 2:
            raise ValueError("starts must have shape (chains, parameters)")
        if scales.shape != (initial.shape[1],):
            raise ValueError("proposal_scales must match the parameter count")
        samples = np.empty((int(draws), *initial.shape), dtype=float)
        proposals = np.empty_like(samples)
        accepted = np.zeros((int(draws), initial.shape[0]), dtype=bool)
        current = initial.copy()
        current_log_target = np.array([log_target(theta) for theta in current])
        samples[0] = current
        proposals[0] = current
        for draw in range(1, int(draws)):
            proposed = current + rng.normal(size=current.shape) * scales
            for parameter_index in reflect_indices:
                proposed[:, parameter_index] = reflect_unit_interval(
                    proposed[:, parameter_index]
                )
            proposed_log_target = np.array([log_target(theta) for theta in proposed])
            log_acceptance = proposed_log_target - current_log_target
            accept = np.log(rng.random(initial.shape[0])) < np.minimum(
                0.0, log_acceptance
            )
            current = np.where(accept[:, None], proposed, current)
            current_log_target = np.where(
                accept, proposed_log_target, current_log_target
            )
            samples[draw] = current
            proposals[draw] = proposed
            accepted[draw] = accept
        return {
            "samples": samples,
            "proposals": proposals,
            "accepted": accepted,
            "acceptance_rates": accepted[1:].mean(axis=0),
        }

    def autocorrelation_1d(values, max_lag=200):
        series = np.asarray(values, dtype=float)
        centered = series - series.mean()
        denominator = float(np.dot(centered, centered))
        lag_limit = min(int(max_lag), series.size - 1)
        if denominator == 0.0:
            return np.ones(lag_limit + 1)
        acf = np.empty(lag_limit + 1, dtype=float)
        acf[0] = 1.0
        for lag in range(1, lag_limit + 1):
            acf[lag] = np.dot(centered[:-lag], centered[lag:]) / denominator
        return acf

    def effective_sample_size(samples):
        draws_array = np.asarray(samples, dtype=float)
        if draws_array.ndim == 2:
            draws_array = draws_array[:, :, None]
        effective = np.empty(draws_array.shape[2], dtype=float)
        for parameter in range(draws_array.shape[2]):
            total = 0.0
            for chain in range(draws_array.shape[1]):
                acf = autocorrelation_1d(draws_array[:, chain, parameter])
                positive = acf[1:]
                nonpositive = np.flatnonzero(positive <= 0.0)
                stop = int(nonpositive[0]) if nonpositive.size else positive.size
                tau = max(1.0, 1.0 + 2.0 * float(positive[:stop].sum()))
                total += draws_array.shape[0] / tau
            effective[parameter] = min(
                total, draws_array.shape[0] * draws_array.shape[1]
            )
        return effective

    def split_rhat(samples):
        draws_array = np.asarray(samples, dtype=float)
        if draws_array.ndim == 2:
            draws_array = draws_array[:, :, None]
        half = draws_array.shape[0] // 2
        split = np.concatenate([draws_array[:half], draws_array[-half:]], axis=1)
        chain_means = split.mean(axis=0)
        chain_variances = split.var(axis=0, ddof=1)
        between = half * chain_means.var(axis=0, ddof=1)
        within = chain_variances.mean(axis=0)
        variance_hat = ((half - 1) / half) * within + between / half
        return np.sqrt(variance_hat / within)

    def run_beta_mcmc(a, b, trials, successes, proposal_scale, draws, seed):
        def beta_log_target(theta):
            probability = float(theta[0])
            return float(
                stats.beta.logpdf(probability, a, b)
                + stats.binom.logpmf(successes, trials, probability)
            )

        starts = np.array([[0.05], [0.25], [0.65], [0.95]])
        result = random_walk_metropolis(
            beta_log_target,
            starts,
            np.array([proposal_scale]),
            draws,
            np.random.default_rng(seed),
            reflect_indices=(0,),
        )
        warmup = int(0.2 * draws)
        posterior = result["samples"][warmup:]
        result.update(
            {
                "a": float(a),
                "b": float(b),
                "trials": int(trials),
                "successes": int(successes),
                "proposal_scale": float(proposal_scale),
                "draws": int(draws),
                "warmup": warmup,
                "seed": int(seed),
                "ess": effective_sample_size(posterior)[0],
                "rhat": split_rhat(posterior)[0],
            }
        )
        return result

    def run_regression_mcmc(frame, seed=82406):
        x_values = frame["flow_speed_m_s"].to_numpy(dtype=float)
        y_values = frame["oxygen_mg_l"].to_numpy(dtype=float)
        x_mean = float(x_values.mean())
        x_sd = float(x_values.std(ddof=1))
        z_values = (x_values - x_mean) / x_sd
        least_squares = stats.linregress(x_values, y_values)
        centered_slope = float(least_squares.slope * x_sd)
        centered_intercept = float(y_values.mean())
        residuals = y_values - (
            least_squares.intercept + least_squares.slope * x_values
        )
        residual_sd = float(np.sqrt(np.sum(residuals**2) / (x_values.size - 2)))

        def regression_log_target(theta):
            alpha, beta_z, log_sigma = theta
            sigma = np.exp(log_sigma)
            fitted = alpha + beta_z * z_values
            prior = (
                -0.5 * ((alpha - 7.0) / 5.0) ** 2
                - 0.5 * (beta_z / 5.0) ** 2
                - 0.5 * ((log_sigma - np.log(2.0)) / 0.75) ** 2
            )
            likelihood = (
                -y_values.size * log_sigma
                - 0.5 * np.sum((y_values - fitted) ** 2) / sigma**2
            )
            return float(prior + likelihood)

        center = np.array([centered_intercept, centered_slope, np.log(residual_sd)])
        offsets = np.array(
            [
                [-0.5, -0.5, -0.20],
                [0.3, -0.2, 0.10],
                [-0.2, 0.4, 0.20],
                [0.6, 0.5, -0.10],
            ]
        )
        result = random_walk_metropolis(
            regression_log_target,
            center + offsets,
            np.array([0.20, 0.20, 0.08]),
            8_000,
            np.random.default_rng(seed),
        )
        warmup = 2_000
        posterior_standard = result["samples"][warmup:]
        flattened = posterior_standard.reshape(-1, 3)
        sigma_samples = np.exp(flattened[:, 2])
        slope_samples = flattened[:, 1] / x_sd
        intercept_samples = flattened[:, 0] - slope_samples * x_mean
        posterior_original = np.column_stack(
            [intercept_samples, slope_samples, sigma_samples]
        )
        rng = np.random.default_rng(seed + 1)
        selected_indices = np.linspace(
            0, posterior_original.shape[0] - 1, 2_000, dtype=int
        )
        selected = posterior_original[selected_indices]
        x_grid = np.linspace(0.12, 1.05, 120)
        mean_draws = selected[:, 0, None] + selected[:, 1, None] * x_grid
        predictive_draws = (
            mean_draws + rng.normal(size=mean_draws.shape) * selected[:, 2, None]
        )
        result.update(
            {
                "warmup": warmup,
                "seed": int(seed),
                "x": x_values,
                "y": y_values,
                "x_mean": x_mean,
                "x_sd": x_sd,
                "least_squares": least_squares,
                "posterior_standard": posterior_standard,
                "posterior_original": posterior_original,
                "ess": effective_sample_size(posterior_standard),
                "rhat": split_rhat(posterior_standard),
                "x_grid": x_grid,
                "mean_draws": mean_draws,
                "predictive_draws": predictive_draws,
            }
        )
        return result

    lecture_general = diagnostic_details(0.0001, 0.82, 0.963, 1_000_000)
    lecture_sibling = diagnostic_details(0.5, 0.82, 0.963, 100)
    assert np.isclose(lecture_general["ppv"], 82 / 37_078)
    assert np.isclose(lecture_sibling["ppv"], 41 / 42.85)
    lecture_dice = dice_update([4, 6, 8, 12, 20], np.ones(5) / 5, [6])
    assert np.allclose(
        lecture_dice["posterior"], [0, 0.39215686, 0.29411765, 0.19607843, 0.11764706]
    )

    return (
        autocorrelation_1d,
        beta_binomial_predictive,
        diagnostic_details,
        dice_update,
        effective_sample_size,
        required_specificity,
        run_beta_mcmc,
        run_regression_mcmc,
        split_rhat,
    )


@app.cell
def _(mo):
    mo.md(r"""
    # Bayesian statistics

    ## Updating uncertainty with data

    Bayesian inference combines a probability model for unknown quantities with a
    likelihood for observed data. The result is a posterior distribution whose
    interpretation is always conditional on the model, prior, likelihood, and data.

    By the end of this chapter, you should be able to:

    1. distinguish frequentist and Bayesian probability statements;
    2. apply Bayes' theorem using formulas and natural frequencies;
    3. update probabilities for discrete hypotheses and a binomial proportion;
    4. distinguish parameter uncertainty from posterior predictive uncertainty;
    5. explain and diagnose a simple Metropolis sampler;
    6. interpret posterior distributions from a Bayesian linear regression.

    > **Two-minute preview:** Prior uncertainty is multiplied by the likelihood and
    > normalized to form the posterior. Some posteriors are available exactly;
    > others are approximated with dependent MCMC draws. In either case, the result
    > is only as credible as the scientific model and data behind it.

    **Prerequisites:** conditional probability, binomial models, confidence
    intervals, diagnostic tests, and simple linear regression from Chapters 2–5.
    """)


@app.cell
def _(mo):
    mo.md(r"""
    ---

    ## 1. Two interpretations of probability

    In a frequentist model, parameters such as a population mean or prevalence are
    fixed, while hypothetical samples and their intervals vary. Bayesian inference
    represents uncertainty about a parameter with a probability distribution.
    Neither approach removes the need for design, assumptions, or scientific judgment.

    For the same observation—3 people with an allergy among 10—the intervals below
    may look similar while answering different questions.

    **Read, then classify:** compare the interpretation column before choosing an
    answer below. Ask what varies: the interval across repeated samples, or the
    unknown parameter in a posterior distribution conditional on this model and
    these data? The radio buttons reveal feedback; they do not recalculate either
    interval. The numerical bounds in the quiz are illustrative, not the table's
    calculated endpoints.
    """)


@app.cell
def _(compact_table, mo, pd, stats):
    comparison_k, comparison_n = 3, 10
    frequentist_interval = stats.binomtest(comparison_k, comparison_n).proportion_ci(
        confidence_level=0.95, method="exact"
    )
    bayesian_reference = stats.beta(4, 8)
    bayesian_interval = bayesian_reference.ppf([0.025, 0.975])
    interval_table = compact_table(
        pd.DataFrame(
            {
                "Framework": ["Frequentist", "Bayesian"],
                "95% interval": [
                    f"{frequentist_interval.low:.3f} to {frequentist_interval.high:.3f}",
                    f"{bayesian_interval[0]:.3f} to {bayesian_interval[1]:.3f}",
                ],
                "Interpretation": [
                    "Procedure has 95% long-run coverage",
                    "95% posterior probability for p in this interval",
                ],
            }
        ),
        column_widths={"Framework": 105, "95% interval": 125, "Interpretation": 320},
        wrapped_columns=["Interpretation"],
    )
    mo.vstack(
        [
            interval_table,
            mo.callout(
                mo.md(
                    "The Bayesian interval uses a uniform $Beta(1,1)$ prior and a "
                    "$Beta(4,8)$ posterior. Numerical resemblance does not make the "
                    "two interpretations interchangeable."
                ),
                kind="info",
            ),
        ]
    )


@app.cell
def _(mo):
    probability_statement = mo.ui.radio(
        {
            "Across repeated studies, 95% of intervals made this way cover p": "frequentist",
            "Given this model and data, P(0.1 < p < 0.6) is 0.95": "bayesian",
            "The observed data have a 95% probability of being true": "ill_posed",
            "Either of the first two, depending on the stated framework": "either",
        },
        value=None,
        label="Which option correctly classifies the first two statements?",
    )
    return (probability_statement,)


@app.cell
def _(mo, probability_statement, review_feedback):
    mo.vstack(
        [
            probability_statement,
            review_feedback(
                probability_statement.value,
                correct_value="either",
                correct_text="**Correct.** The first is a frequentist coverage statement and the second is a Bayesian posterior statement.",
                incorrect_text="The two valid statements use different probability targets; truth of the observed data is not a random event in either analysis.",
            ),
        ]
    )


@app.cell
def _(mo):
    mo.md(r"""
    ---

    ## 2. Conditional probability and Bayes' theorem

    The direction of conditioning matters: $P(A\mid B)$ is generally not
    $P(B\mid A)$. Two factorizations of the same joint event give

    $$P(A\cap B)=P(A)P(B\mid A)=P(B)P(A\mid B),$$

    and therefore

    $$P(B\mid A)=\frac{P(B)P(A\mid B)}{P(A)}.$$

    In Bayesian language, **posterior** is proportional to **prior × likelihood**.
    The denominator—evidence or marginal likelihood—adds the corresponding products
    over all mutually exclusive hypotheses so the posterior sums to one.

    A probability tree exposes the products along each branch; a contingency table
    exposes the same denominator as a row or column total. The next example uses both
    ideas as natural frequencies.
    """)


@app.cell
def _(mo):
    mo.md(r"""
    ---

    ## 3. Diagnostic tests, prevalence, and the base-rate effect

    Sensitivity is $P(+\mid sick)$ and specificity is $P(-\mid healthy)$. Patients
    usually need the reverse conditional, such as $P(sick\mid +)$. That positive
    predictive value depends strongly on disease prevalence.

    **Before moving the controls:** predict whether a positive result is more likely
    to be a true or false positive at the default prevalence of 0.01%.

    **Try this:** keep sensitivity at 82% and specificity at 96.3%, and compare
    prevalence values of 0.01%, 1%, and 10%. Follow the true-positive and
    false-positive counts into the denominator of positive predictive value (PPV).
    Then restore prevalence to 0.01% and increase specificity to 99.9%: even a small
    false-positive rate can matter when the healthy population is very large.

    All controls update immediately. Changing the **Natural-frequency cohort**
    rescales expected counts, not the conditional probabilities; fractional counts
    are expectations, not fractions of a person. **Target PPV** only calculates
    the specificity that would be required—it does not set the actual specificity
    slider or change the current test's PPV. These are hypothetical model inputs.
    """)


@app.cell
def _(mo):
    diagnostic_prevalence = mo.ui.slider(
        0.01,
        50.0,
        step=0.01,
        value=0.01,
        show_value=True,
        full_width=True,
        label="Prevalence [%]",
    )
    diagnostic_sensitivity = mo.ui.slider(
        50.0,
        100.0,
        step=0.1,
        value=82.0,
        show_value=True,
        full_width=True,
        label="Sensitivity [%]",
    )
    diagnostic_specificity = mo.ui.slider(
        90.0,
        99.999,
        step=0.001,
        value=96.3,
        show_value=True,
        full_width=True,
        label="Specificity [%]",
    )
    diagnostic_population = mo.ui.dropdown(
        {"100 people": 100, "10,000 people": 10_000, "1,000,000 people": 1_000_000},
        value="1,000,000 people",
        label="Natural-frequency cohort",
        full_width=True,
    )
    diagnostic_target_ppv = mo.ui.slider(
        50.0,
        99.0,
        step=1.0,
        value=90.0,
        show_value=True,
        full_width=True,
        label="Target PPV [%]",
    )
    return (
        diagnostic_population,
        diagnostic_prevalence,
        diagnostic_sensitivity,
        diagnostic_specificity,
        diagnostic_target_ppv,
    )


@app.cell
def _(
    COLORS,
    FIGURE_SIZE_STANDARD,
    compact_table,
    diagnostic_details,
    diagnostic_population,
    diagnostic_prevalence,
    diagnostic_sensitivity,
    diagnostic_specificity,
    diagnostic_target_ppv,
    mo,
    np,
    pd,
    plt,
    required_specificity,
    two_column_panel,
):
    prevalence_value = diagnostic_prevalence.value / 100.0
    sensitivity_value = diagnostic_sensitivity.value / 100.0
    specificity_value = diagnostic_specificity.value / 100.0
    population_value = int(diagnostic_population.value)
    diagnostic_result = diagnostic_details(
        prevalence_value, sensitivity_value, specificity_value, population_value
    )
    specificity_needed = required_specificity(
        prevalence_value, sensitivity_value, diagnostic_target_ppv.value / 100.0
    )
    frequency_table = compact_table(
        pd.DataFrame(
            {
                "Test result": ["Positive", "Negative", "Total"],
                "Sick": [
                    diagnostic_result["true_positive"],
                    diagnostic_result["false_negative"],
                    diagnostic_result["sick"],
                ],
                "Healthy": [
                    diagnostic_result["false_positive"],
                    diagnostic_result["true_negative"],
                    diagnostic_result["healthy"],
                ],
                "Total": [
                    diagnostic_result["true_positive"]
                    + diagnostic_result["false_positive"],
                    diagnostic_result["false_negative"]
                    + diagnostic_result["true_negative"],
                    population_value,
                ],
            }
        ),
        column_widths={"Test result": 100, "Sick": 90, "Healthy": 100, "Total": 105},
        format_mapping={"Sick": "{:,.1f}", "Healthy": "{:,.1f}", "Total": "{:,.1f}"},
    )
    diagnostic_figure, diagnostic_axis = plt.subplots(figsize=FIGURE_SIZE_STANDARD)
    categories = ["True positive", "False positive", "False negative", "True negative"]
    values = [
        diagnostic_result["true_positive"],
        diagnostic_result["false_positive"],
        diagnostic_result["false_negative"],
        diagnostic_result["true_negative"],
    ]
    colors = [COLORS["green"], COLORS["vermillion"], COLORS["gray"], COLORS["sky"]]
    diagnostic_axis.bar(categories, values, color=colors)
    diagnostic_axis.set(
        ylabel="Expected people [log scale]", yscale="symlog", ylim=(0, max(values) * 2)
    )
    diagnostic_axis.tick_params(axis="x", rotation=20)
    diagnostic_axis.grid(axis="y", linestyle=":", alpha=0.3)
    diagnostic_figure.tight_layout()
    note_kind = "warn" if diagnostic_result["ppv"] < 0.5 else "info"
    result_note = mo.callout(
        mo.md(
            f"Among positive results, **{diagnostic_result['ppv']:.2%}** are expected "
            f"to be true positives; NPV is **{diagnostic_result['npv']:.2%}**. At this "
            f"prevalence and sensitivity, PPV {diagnostic_target_ppv.value:.0f}% requires "
            f"specificity of about **{specificity_needed:.4%}**."
        ),
        kind=note_kind,
    )
    controls = mo.vstack(
        [
            diagnostic_prevalence,
            diagnostic_sensitivity,
            diagnostic_specificity,
            diagnostic_population,
            diagnostic_target_ppv,
            mo.stat(
                f"{diagnostic_result['ppv']:.2%}",
                label="P(sick | positive)",
                bordered=True,
            ),
        ]
    )
    two_column_panel(
        controls,
        mo.vstack([frequency_table, diagnostic_figure, result_note]),
        widths=(1, 2.2),
    )


@app.cell
def _(mo):
    mo.md(r"""
    ---

    ## 4. Sequential updating with dice

    Suppose one hidden die was selected from dice with 4, 6, 8, 12, or 20 sides.
    Each die is a hypothesis. A roll contributes likelihood $1/s$ to a die with
    $s$ sides if the result is possible, and likelihood zero otherwise.

    **Try this:** select **All dice equally likely** and move the number of
    revealed rolls from 0 to 6, one step at a time. This reveals the fixed sequence
    [6, 4, 7, 7, 8, 2], not a new random roll. Before the first 6 and the first 7,
    predict which dice become impossible. The later 8 eliminates no additional
    dice, but still changes the relative probabilities of the surviving hypotheses.

    The gray bars show the prior **before the latest revealed roll**, including
    updates from earlier rolls; the blue bars show its posterior. Hold the revealed
    sequence fixed and change the initial prior to compare how beliefs and evidence
    combine. Moving the slider backwards recomputes from the shorter sequence; it
    does not accumulate hidden observations. A high posterior probability identifies
    the most supported die among these candidates, not a certainty about its identity.

    To design your own experiment, switch to **My own rolls**, select a result from
    1 to 20, and press **Add roll**. The probability-history plot retains every
    update so you can see when a hypothesis is ruled out and how repeated small rolls
    gradually favor smaller dice. **Undo** removes the latest roll; **Reset** clears
    your sequence without changing the selected prior.
    """)


@app.cell
def _(mo):
    dice_sequence_source = mo.ui.radio(
        {
            "Lecture sequence": "lecture",
            "My own rolls": "student",
        },
        value="Lecture sequence",
        label="Roll source",
    )
    dice_prior_choice = mo.ui.radio(
        {
            "All dice equally likely": "uniform",
            "Player favors the 20-sided die": "favor_d20",
            "Player favors the two smaller dice": "favor_small",
        },
        value="All dice equally likely",
        label="Prior belief",
    )
    dice_reveal = mo.ui.slider(
        0,
        6,
        value=1,
        show_value=True,
        full_width=True,
        label="Number of revealed rolls",
    )
    return dice_prior_choice, dice_reveal, dice_sequence_source


@app.cell
def _(mo):
    get_student_dice_rolls, set_student_dice_rolls = mo.state([])
    return get_student_dice_rolls, set_student_dice_rolls


@app.cell
def _(dice_sequence_source, mo, set_student_dice_rolls):
    student_dice_roll = mo.ui.dropdown(
        {str(roll): roll for roll in range(1, 21)},
        value="6",
        label="Next roll",
        full_width=True,
    )
    student_dice_add = mo.ui.button(
        label="Add roll",
        kind="success",
        disabled=dice_sequence_source.value != "student",
        on_click=lambda _value: set_student_dice_rolls(
            lambda rolls: [*rolls, int(student_dice_roll.value)]
        ),
        full_width=True,
    )
    student_dice_undo = mo.ui.button(
        label="Undo latest",
        disabled=dice_sequence_source.value != "student",
        on_click=lambda _value: set_student_dice_rolls(lambda rolls: rolls[:-1]),
        full_width=True,
    )
    student_dice_reset = mo.ui.button(
        label="Reset my rolls",
        disabled=dice_sequence_source.value != "student",
        on_click=lambda _value: set_student_dice_rolls([]),
        full_width=True,
    )
    student_dice_controls = mo.vstack(
        [
            student_dice_roll,
            mo.hstack(
                [student_dice_add, student_dice_undo, student_dice_reset],
                widths="equal",
                wrap=True,
            ),
        ]
    )
    return (student_dice_controls,)


@app.cell
def _(
    COLORS,
    FIGURE_SIZE_LINKED,
    compact_table,
    dice_prior_choice,
    dice_reveal,
    dice_sequence_source,
    dice_update,
    get_student_dice_rolls,
    mo,
    np,
    pd,
    plt,
    student_dice_controls,
    two_column_panel,
):
    dice_sides = np.array([4, 6, 8, 12, 20])
    dice_sequence = [6, 4, 7, 7, 8, 2]
    dice_priors = {
        "uniform": np.ones(5) / 5,
        "favor_d20": np.array([0.10, 0.10, 0.10, 0.15, 0.55]),
        "favor_small": np.array([0.35, 0.35, 0.15, 0.10, 0.05]),
    }
    lecture_rolls = dice_sequence[: int(dice_reveal.value)]
    student_rolls = list(get_student_dice_rolls())
    revealed_rolls = (
        lecture_rolls
        if dice_sequence_source.value == "lecture"
        else student_rolls
    )
    dice_result = dice_update(
        dice_sides, dice_priors[dice_prior_choice.value], revealed_rolls
    )
    assert np.isclose(dice_result["posterior"].sum(), 1.0)
    if 8 in revealed_rolls:
        assert np.all(dice_result["posterior"][:2] == 0.0)
    update_table = compact_table(
        pd.DataFrame(
            {
                "Die": [f"d{side}" for side in dice_sides],
                "Prior": dice_result["prior"],
                "Likelihood": dice_result["likelihood"],
                "Prior × likelihood": dice_result["unnormalized"],
                "Posterior": dice_result["posterior"],
            }
        ),
        column_widths={
            "Die": 55,
            "Prior": 75,
            "Likelihood": 85,
            "Prior × likelihood": 135,
            "Posterior": 85,
        },
        format_mapping={
            "Prior": "{:.3f}",
            "Likelihood": "{:.3f}",
            "Prior × likelihood": "{:.4f}",
            "Posterior": "{:.3f}",
        },
    )
    initial_dice_prior = dice_priors[dice_prior_choice.value]
    posterior_history = np.vstack(
        [
            initial_dice_prior,
            *[
                dice_update(dice_sides, initial_dice_prior, revealed_rolls[:step])[
                    "posterior"
                ]
                for step in range(1, len(revealed_rolls) + 1)
            ],
        ]
    )
    assert posterior_history.shape == (len(revealed_rolls) + 1, len(dice_sides))
    assert np.allclose(posterior_history.sum(axis=1), 1.0)

    dice_figure, (dice_axis, dice_history_axis) = plt.subplots(
        1, 2, figsize=FIGURE_SIZE_LINKED
    )
    positions = np.arange(dice_sides.size)
    dice_axis.bar(
        positions - 0.18,
        dice_result["prior"],
        width=0.36,
        color=COLORS["gray"],
        alpha=0.7,
        label="Prior before latest roll",
    )
    dice_axis.bar(
        positions + 0.18,
        dice_result["posterior"],
        width=0.36,
        color=COLORS["blue"],
        label="Posterior",
    )
    dice_axis.set(
        xticks=positions,
        xticklabels=[f"d{side}" for side in dice_sides],
        xlabel="Hidden-die hypothesis",
        ylabel="Probability",
        ylim=(0, 1),
    )
    dice_axis.grid(axis="y", linestyle=":", alpha=0.3)
    dice_axis.legend(frameon=False, fontsize=8)

    history_colors = [
        COLORS["orange"],
        COLORS["sky"],
        COLORS["green"],
        COLORS["purple"],
        COLORS["blue"],
    ]
    history_steps = np.arange(len(revealed_rolls) + 1)
    for _die_index, (_die_side, _die_color) in enumerate(
        zip(dice_sides, history_colors)
    ):
        dice_history_axis.plot(
            history_steps,
            posterior_history[:, _die_index],
            color=_die_color,
            marker="o",
            markersize=4,
            linewidth=1.8,
            label=f"d{_die_side}",
        )
    dice_history_axis.set(
        title="Posterior after each roll",
        xlabel="Number of observed rolls",
        ylabel="Posterior probability",
        xlim=(-0.2, max(1, len(revealed_rolls)) + 0.2),
        ylim=(-0.02, 1.02),
        xticks=history_steps,
    )
    dice_history_axis.grid(linestyle=":", alpha=0.3)
    dice_history_axis.legend(frameon=False, fontsize=7, ncols=2)
    dice_figure.tight_layout()
    latest_text = "none" if not revealed_rolls else str(revealed_rolls[-1])
    sequence_label = (
        "Lecture sequence"
        if dice_sequence_source.value == "lecture"
        else "Your sequence"
    )
    dice_note = mo.callout(
        mo.md(
            f"{sequence_label}: **{revealed_rolls or 'none'}**. The latest roll is "
            f"**{latest_text}** and its predictive probability before observing it "
            f"was the evidence **{dice_result['evidence']:.3f}**. Impossible results "
            "give a hypothesis likelihood—and therefore posterior probability—zero."
        ),
        kind="info",
    )
    two_column_panel(
        mo.vstack(
            [
                dice_prior_choice,
                dice_sequence_source,
                dice_reveal,
                student_dice_controls,
                update_table,
            ]
        ),
        mo.vstack([dice_figure, dice_note]),
        widths=(1.35, 2),
    )


@app.cell
def _(mo):
    mo.md(r"""
    ---

    ## 5. Bayesian inference for a proportion

    Let $p$ be the prevalence of an allergy. A beta prior and binomial likelihood
    form a conjugate pair:

    $$p\sim Beta(a,b),\qquad K\mid p\sim Binomial(n,p).$$

    Multiplying their kernels collects the powers of $p$ and $1-p$:

    $$p^{a-1}(1-p)^{b-1}\;p^k(1-p)^{n-k}
      =p^{a+k-1}(1-p)^{b+n-k-1}.$$

    Thus $p\mid k,n\sim Beta(a+k,b+n-k)$. The update behaves as if successes and
    failures add information to the two prior shape parameters.

    **Try this:** with $n=10$ and $k=3$, compare $a=b=1$ with $a=b=10$.
    Both priors have mean 0.5, but the second is more concentrated: watch the
    posterior compromise between prior and likelihood. The controls labeled prior
    successes/failures are beta shape parameters, not additional observed people.
    Restore $a=b=1$, increase $n$ to 100, and then set $k=30$ to keep the observed
    proportion at 0.3. Compare posterior widths at the same observed proportion.

    All outputs update immediately. **Set $n$ before $k$:** changing $n$ rebuilds
    the successes control and resets it to 3 (or to $n$ if smaller). The probability
    threshold changes only the reported posterior tail area; it does not alter the
    posterior or its credible interval. Interpret that area conditional on the
    chosen prior and binomial model.
    """)


@app.cell
def _(mo):
    beta_prior_a = mo.ui.slider(
        0.5,
        20.0,
        step=0.5,
        value=1.0,
        show_value=True,
        full_width=True,
        label="Prior successes a",
    )
    beta_prior_b = mo.ui.slider(
        0.5,
        20.0,
        step=0.5,
        value=1.0,
        show_value=True,
        full_width=True,
        label="Prior failures b",
    )
    beta_trials = mo.ui.slider(
        1,
        100,
        value=10,
        show_value=True,
        full_width=True,
        label="Observed sample size n",
    )
    beta_threshold = mo.ui.slider(
        0.0,
        1.0,
        step=0.01,
        value=0.5,
        show_value=True,
        full_width=True,
        label="Scientific threshold p₀",
    )
    return beta_prior_a, beta_prior_b, beta_threshold, beta_trials


@app.cell
def _(beta_trials, mo):
    beta_successes = mo.ui.slider(
        0,
        int(beta_trials.value),
        value=min(3, int(beta_trials.value)),
        show_value=True,
        full_width=True,
        label="Observed successes k",
    )
    return (beta_successes,)


@app.cell
def _(
    COLORS,
    FIGURE_SIZE_STANDARD,
    beta_prior_a,
    beta_prior_b,
    beta_successes,
    beta_threshold,
    beta_trials,
    mo,
    np,
    plt,
    stats,
    two_column_panel,
):
    beta_a = float(beta_prior_a.value)
    beta_b = float(beta_prior_b.value)
    beta_n = int(beta_trials.value)
    beta_k = int(beta_successes.value)
    posterior_a = beta_a + beta_k
    posterior_b = beta_b + beta_n - beta_k
    beta_grid = np.linspace(0.001, 0.999, 600)
    prior_density = stats.beta.pdf(beta_grid, beta_a, beta_b)
    raw_likelihood = stats.binom.pmf(beta_k, beta_n, beta_grid)
    scaled_likelihood = (
        raw_likelihood
        / raw_likelihood.max()
        * max(
            prior_density.max(),
            stats.beta.pdf(beta_grid, posterior_a, posterior_b).max(),
        )
    )
    posterior_density = stats.beta.pdf(beta_grid, posterior_a, posterior_b)
    grid_posterior = prior_density * raw_likelihood
    grid_shape = grid_posterior / grid_posterior.max()
    exact_shape = posterior_density / posterior_density.max()
    assert np.allclose(grid_shape, exact_shape, rtol=1e-10, atol=1e-10)
    posterior_reference = stats.beta(posterior_a, posterior_b)
    credible_interval = posterior_reference.ppf([0.025, 0.975])
    posterior_mean = float(posterior_reference.mean())
    probability_above = float(posterior_reference.sf(beta_threshold.value))
    beta_figure, beta_axis = plt.subplots(figsize=FIGURE_SIZE_STANDARD)
    beta_axis.plot(
        beta_grid, prior_density, color=COLORS["gray"], linestyle="--", label="Prior"
    )
    beta_axis.plot(
        beta_grid,
        scaled_likelihood,
        color=COLORS["orange"],
        linestyle=":",
        label="Likelihood (scaled)",
    )
    beta_axis.plot(
        beta_grid,
        posterior_density,
        color=COLORS["blue"],
        linewidth=2.2,
        label="Exact posterior",
    )
    beta_axis.fill_between(
        beta_grid,
        0,
        posterior_density,
        where=(beta_grid >= credible_interval[0]) & (beta_grid <= credible_interval[1]),
        color=COLORS["sky"],
        alpha=0.35,
        label="95% credible interval",
    )
    beta_axis.axvline(
        beta_threshold.value,
        color=COLORS["vermillion"],
        linestyle="--",
        label="Threshold p₀",
    )
    beta_axis.set(
        xlabel="Success probability p",
        ylabel="Density",
        xlim=(0, 1),
        ylim=(0, 1.12 * max(posterior_density.max(), prior_density.max())),
    )
    beta_axis.grid(axis="y", linestyle=":", alpha=0.3)
    beta_axis.legend(frameon=False, fontsize=8)
    beta_figure.tight_layout()
    beta_controls = mo.vstack(
        [
            beta_prior_a,
            beta_prior_b,
            beta_trials,
            beta_successes,
            beta_threshold,
            mo.stat(f"{posterior_mean:.3f}", label="Posterior mean", bordered=True),
            mo.stat(
                f"{credible_interval[0]:.3f} to {credible_interval[1]:.3f}",
                label="95% credible interval",
                bordered=True,
            ),
            mo.stat(
                f"{probability_above:.1%}", label="P(p > p₀ | data)", bordered=True
            ),
        ]
    )
    beta_note = mo.callout(
        mo.md(
            f"The posterior is **Beta({posterior_a:g}, {posterior_b:g})**. Conditional "
            f"on this model and prior, there is 95% posterior probability that $p$ "
            f"lies between **{credible_interval[0]:.3f} and {credible_interval[1]:.3f}**."
        ),
        kind="info",
    )
    two_column_panel(beta_controls, mo.vstack([beta_figure, beta_note]))


@app.cell
def _(mo):
    mo.md(r"""
    ### Posterior prediction: the next experiment

    The posterior describes uncertainty about $p$. A posterior predictive
    distribution describes future observations and averages the sampling model over
    that uncertainty. For $m$ future trials it is beta-binomial.

    A plug-in prediction fixes $p$ at its posterior mean. It therefore omits
    parameter uncertainty and is usually narrower than the full posterior predictive.

    **Try this:** keep the prior and observed data in Section 5 fixed, then compare
    future sample sizes $m=1$, 20, and 100. At $m=1$ the two predictions coincide;
    for larger samples, compare their tails and the two variances. Both predict
    **counts** of future successes, so changing $m$ also changes the horizontal
    scale. This control updates immediately and does not add observations to the
    posterior. To see how more existing evidence changes prediction, hold $m=20$
    and compare $k/n=3/10$ with $30/100$ in Section 5.
    """)


@app.cell
def _(mo):
    predictive_trials = mo.ui.slider(
        1,
        100,
        value=20,
        show_value=True,
        full_width=True,
        label="Future sample size m",
    )
    return (predictive_trials,)


@app.cell
def _(
    COLORS,
    FIGURE_SIZE_STANDARD,
    beta_binomial_predictive,
    beta_prior_a,
    beta_prior_b,
    beta_successes,
    beta_trials,
    mo,
    np,
    plt,
    predictive_trials,
    stats,
    two_column_panel,
):
    predictive_a = float(beta_prior_a.value) + int(beta_successes.value)
    predictive_b = (
        float(beta_prior_b.value) + int(beta_trials.value) - int(beta_successes.value)
    )
    future_n = int(predictive_trials.value)
    future_outcomes, predictive_probabilities = beta_binomial_predictive(
        predictive_a, predictive_b, future_n
    )
    assert np.isclose(predictive_probabilities.sum(), 1.0)
    predictive_mean_p = predictive_a / (predictive_a + predictive_b)
    plugin_probabilities = stats.binom.pmf(future_outcomes, future_n, predictive_mean_p)
    predictive_variance = float(
        np.sum(
            (future_outcomes - np.sum(future_outcomes * predictive_probabilities)) ** 2
            * predictive_probabilities
        )
    )
    plugin_variance = float(stats.binom.var(future_n, predictive_mean_p))
    assert predictive_variance >= plugin_variance - 1e-12
    prediction_figure, prediction_axis = plt.subplots(figsize=FIGURE_SIZE_STANDARD)
    prediction_axis.bar(
        future_outcomes,
        predictive_probabilities,
        color=COLORS["sky"],
        alpha=0.75,
        label="Full beta-binomial prediction",
    )
    prediction_axis.plot(
        future_outcomes,
        plugin_probabilities,
        color=COLORS["vermillion"],
        marker="o",
        markersize=3.5,
        linewidth=1.5,
        label="Plug-in binomial",
    )
    prediction_axis.set(
        xlabel="Successes in future sample",
        ylabel="Probability",
        xlim=(-0.8, future_n + 0.8),
    )
    prediction_axis.grid(axis="y", linestyle=":", alpha=0.3)
    prediction_axis.legend(frameon=False, fontsize=8)
    prediction_figure.tight_layout()
    prediction_note = mo.callout(
        mo.md(
            f"The predictive variance is **{predictive_variance:.2f}**, compared "
            f"with **{plugin_variance:.2f}** after fixing $p$ at its posterior mean. "
            "The difference is uncertainty about the still-unknown prevalence."
        ),
        kind="info",
    )
    two_column_panel(
        mo.vstack(
            [
                predictive_trials,
                mo.stat(
                    f"{future_n * predictive_mean_p:.2f}",
                    label="Expected future successes",
                    bordered=True,
                ),
                mo.stat(
                    f"{predictive_variance:.2f}",
                    label="Predictive variance",
                    bordered=True,
                ),
            ]
        ),
        mo.vstack([prediction_figure, prediction_note]),
    )


@app.cell
def _(mo):
    mo.md(r"""
    ---

    ## 6. Posterior computation with MCMC

    Some posteriors cannot be normalized or sampled from directly. Markov-chain
    Monte Carlo constructs dependent draws whose long-run distribution approximates
    the posterior. Here a random-walk Metropolis step proposes a nearby value and
    accepts it with probability

    $$\min\left(1,\frac{p(\theta'\mid D)}{p(\theta\mid D)}\right).$$

    The unknown normalizing constant appears in numerator and denominator and
    cancels. We calculate the ratio in log space. This beta-binomial example is
    intentionally one we can solve exactly, so the analytic posterior provides a
    visible check on the sampler.

    **How to run:** choose the prior and observed data in Section 5, select a
    proposal standard deviation and draws per chain here, then click **Run four
    new chains**. Until you click, the plots and diagnostics retain the previous
    run, including its previous prior and data. The initial run uses $a=b=1$,
    $n=10$, and $k=3$.

    **Try this:** keep that target and 5,000 draws fixed, and run proposal SDs
    0.005, 0.1, and 0.5. Tiny proposals can be accepted frequently while exploring
    slowly; large proposals can be rejected frequently. Compare traces,
    autocorrelation, effective sample size (ESS), and agreement with the exact beta
    density, not acceptance rate alone. Repeat a setting to see Monte Carlo variation.

    **Inspect chain 1 through step** only reveals more of the existing early walk;
    it does not run a new chain or change the full-run diagnostics below. Look for
    overlapping traces without persistent drift, decaying autocorrelation, and
    split-$\hat R$ near one. These are computational checks, not proof that the
    statistical model is appropriate.
    """)


@app.cell
def _(mo):
    mcmc_proposal_scale = mo.ui.slider(
        0.005,
        0.5,
        step=0.005,
        value=0.1,
        show_value=True,
        full_width=True,
        label="Proposal standard deviation",
    )
    mcmc_draws = mo.ui.dropdown(
        {
            "500 draws": 500,
            "2,000 draws": 2_000,
            "5,000 draws": 5_000,
            "10,000 draws": 10_000,
        },
        value="5,000 draws",
        label="Draws per chain",
        full_width=True,
    )
    return mcmc_draws, mcmc_proposal_scale


@app.cell
def _(mo, run_beta_mcmc):
    initial_beta_mcmc = run_beta_mcmc(1.0, 1.0, 10, 3, 0.1, 5_000, 8240601)
    get_beta_mcmc, set_beta_mcmc = mo.state(initial_beta_mcmc)
    return get_beta_mcmc, set_beta_mcmc


@app.cell
def _(
    beta_prior_a,
    beta_prior_b,
    beta_successes,
    beta_trials,
    mcmc_draws,
    mcmc_proposal_scale,
    mo,
    np,
    run_beta_mcmc,
    set_beta_mcmc,
):
    def run_new_beta_chain(_value):
        fresh_seed = int(np.random.default_rng().integers(0, 2**32 - 1))
        set_beta_mcmc(
            run_beta_mcmc(
                float(beta_prior_a.value),
                float(beta_prior_b.value),
                int(beta_trials.value),
                int(beta_successes.value),
                float(mcmc_proposal_scale.value),
                int(mcmc_draws.value),
                fresh_seed,
            )
        )

    run_beta_mcmc_button = mo.ui.button(
        label="Run four new chains",
        kind="success",
        full_width=True,
        on_click=run_new_beta_chain,
    )
    return (run_beta_mcmc_button,)


@app.cell
def _(get_beta_mcmc, mo):
    beta_walk = get_beta_mcmc()
    mcmc_step = mo.ui.slider(
        1,
        min(200, beta_walk["draws"] - 1),
        value=min(50, beta_walk["draws"] - 1),
        show_value=True,
        full_width=True,
        label="Inspect chain 1 through step",
    )
    return (mcmc_step,)


@app.cell
def _(
    COLORS,
    FIGURE_SIZE_LINKED,
    autocorrelation_1d,
    get_beta_mcmc,
    mcmc_draws,
    mcmc_proposal_scale,
    mcmc_step,
    mo,
    np,
    plt,
    run_beta_mcmc_button,
    stats,
    two_column_panel,
):
    beta_walk_result = get_beta_mcmc()
    beta_walk_samples = beta_walk_result["samples"][:, :, 0]
    beta_walk_posterior = beta_walk_samples[beta_walk_result["warmup"] :]
    assert np.all((beta_walk_samples >= 0.0) & (beta_walk_samples <= 1.0))
    assert np.isfinite(beta_walk_result["ess"])
    assert np.isfinite(beta_walk_result["rhat"])
    exact_beta = stats.beta(
        beta_walk_result["a"] + beta_walk_result["successes"],
        beta_walk_result["b"]
        + beta_walk_result["trials"]
        - beta_walk_result["successes"],
    )
    mcmc_mean = float(beta_walk_posterior.mean())
    if beta_walk_result["seed"] == 8240601:
        assert abs(mcmc_mean - exact_beta.mean()) < 0.035
    step_limit = int(mcmc_step.value)
    walk_figure, walk_axes = plt.subplots(1, 2, figsize=FIGURE_SIZE_LINKED)
    walk_axes[0].plot(
        np.arange(step_limit + 1),
        beta_walk_samples[: step_limit + 1, 0],
        color=COLORS["blue"],
        marker="o",
        markersize=2.5,
        linewidth=1,
        label="Chain position",
    )
    rejected = ~beta_walk_result["accepted"][: step_limit + 1, 0]
    accepted = beta_walk_result["accepted"][: step_limit + 1, 0]
    walk_axes[0].scatter(
        np.arange(step_limit + 1)[accepted],
        beta_walk_result["proposals"][: step_limit + 1, 0, 0][accepted],
        color=COLORS["green"],
        s=18,
        label="Accepted proposal",
        zorder=3,
    )
    walk_axes[0].scatter(
        np.arange(step_limit + 1)[rejected],
        beta_walk_result["proposals"][: step_limit + 1, 0, 0][rejected],
        color=COLORS["vermillion"],
        marker="x",
        s=25,
        label="Rejected proposal",
        zorder=3,
    )
    walk_axes[0].set(
        title="Metropolis walk",
        xlabel="Step",
        ylabel="Proposed and retained p",
        ylim=(-0.03, 1.03),
    )
    walk_axes[0].legend(frameon=False, fontsize=7)
    density_grid = np.linspace(0, 1, 500)
    walk_axes[1].hist(
        beta_walk_posterior.ravel(),
        bins=np.linspace(0, 1, 45),
        density=True,
        color=COLORS["sky"],
        alpha=0.7,
        label="MCMC draws",
    )
    walk_axes[1].plot(
        density_grid,
        exact_beta.pdf(density_grid),
        color=COLORS["blue"],
        linewidth=2,
        label="Exact posterior",
    )
    walk_axes[1].set(
        title="Exact check",
        xlabel="Success probability p",
        ylabel="Density",
        xlim=(0, 1),
    )
    walk_axes[1].legend(frameon=False, fontsize=8)
    for walk_axis in walk_axes:
        walk_axis.grid(linestyle=":", alpha=0.25)
    walk_figure.tight_layout()
    mean_acceptance = float(beta_walk_result["acceptance_rates"].mean())
    scale_message = (
        "Very high acceptance often means tiny, sticky steps."
        if mean_acceptance > 0.75
        else "Very low acceptance means proposals are often too ambitious."
        if mean_acceptance < 0.15
        else "The chain is moving with a useful balance of accepted and rejected proposals."
    )
    stored_settings = mo.callout(
        mo.md(
            f"Displayed chains use $a={beta_walk_result['a']:g}$, $b={beta_walk_result['b']:g}$, "
            f"$k={beta_walk_result['successes']}$ of $n={beta_walk_result['trials']}$, "
            f"proposal scale **{beta_walk_result['proposal_scale']:.3f}**, and "
            f"**{beta_walk_result['draws']:,} draws per chain**. {scale_message}"
        ),
        kind="warn" if mean_acceptance < 0.15 or mean_acceptance > 0.75 else "info",
    )
    beta_mcmc_controls = mo.vstack(
        [
            mcmc_proposal_scale,
            mcmc_draws,
            run_beta_mcmc_button,
            mo.callout(
                mo.md(
                    "Controls describe the **next** run. The displayed chains remain stable until the button is pressed."
                ),
                kind="neutral",
            ),
            mcmc_step,
            mo.stat(f"{mean_acceptance:.1%}", label="Mean acceptance", bordered=True),
            mo.stat(
                f"{beta_walk_result['ess']:.0f}", label="Approximate ESS", bordered=True
            ),
            mo.stat(f"{beta_walk_result['rhat']:.3f}", label="Split R̂", bordered=True),
        ]
    )
    two_column_panel(
        beta_mcmc_controls, mo.vstack([walk_figure, stored_settings]), widths=(1, 2.4)
    )


@app.cell
def _(
    COLORS,
    FIGURE_SIZE_LINKED,
    autocorrelation_1d,
    get_beta_mcmc,
    mo,
    np,
    plt,
):
    diagnostic_walk = get_beta_mcmc()
    diagnostic_samples = diagnostic_walk["samples"][:, :, 0]
    diagnostic_posterior = diagnostic_samples[diagnostic_walk["warmup"] :]
    mcmc_diagnostic_figure, mcmc_diagnostic_axes = plt.subplots(
        1, 2, figsize=FIGURE_SIZE_LINKED
    )
    for chain_index in range(diagnostic_samples.shape[1]):
        mcmc_diagnostic_axes[0].plot(
            diagnostic_samples[:, chain_index],
            linewidth=0.65,
            alpha=0.65,
            label=f"Chain {chain_index + 1}",
        )
    mcmc_diagnostic_axes[0].axvline(
        diagnostic_walk["warmup"],
        color=COLORS["vermillion"],
        linestyle="--",
        label="Warm-up ends",
    )
    mcmc_diagnostic_axes[0].set(
        title="Trace plot", xlabel="Draw", ylabel="p", ylim=(0, 1)
    )
    for _diagnostic_chain_index in range(diagnostic_posterior.shape[1]):
        chain_acf = autocorrelation_1d(
            diagnostic_posterior[:, _diagnostic_chain_index], max_lag=100
        )
        mcmc_diagnostic_axes[1].plot(
            chain_acf,
            linewidth=1.1,
            label=f"Chain {_diagnostic_chain_index + 1}",
        )
    mcmc_diagnostic_axes[1].axhline(0, color=COLORS["gray"], linestyle=":")
    mcmc_diagnostic_axes[1].set(
        title="Autocorrelation", xlabel="Lag", ylabel="Correlation", ylim=(-0.2, 1.02)
    )
    for _mcmc_diagnostic_axis in mcmc_diagnostic_axes:
        _mcmc_diagnostic_axis.grid(linestyle=":", alpha=0.25)
        _mcmc_diagnostic_axis.legend(frameon=False, fontsize=7)
    mcmc_diagnostic_figure.tight_layout()
    mo.vstack(
        [
            mcmc_diagnostic_figure,
            mo.callout(
                mo.md(r"""
            Trust requires more than a plausible histogram. Chains should overlap and
            mix, autocorrelation should decay, effective sample size should be
            adequate, and $\hat R$ should be near one. More samples do not fix a bad
            likelihood, prior, coding error, or non-converged chain; routine thinning
            only discards information.
            """),
                kind="warn",
            ),
        ]
    )


@app.cell
def _(mo):
    mo.md(r"""
    ---

    ## 7. Bayesian linear regression with Metropolis sampling

    We return to the river data from Chapter 3 and model oxygen concentration from
    flow speed. Standardizing flow speed improves the random walk's geometry:

    $$Oxygen_i\sim N(\alpha+\beta_z z_i,\sigma),\qquad
      z_i=\frac{x_i-\bar x}{s_x}.$$

    The weakly informative priors are

    $$\alpha\sim N(7,5^2),\quad \beta_z\sim N(0,5^2),\quad
      \log\sigma\sim N(\log 2,0.75^2).$$

    Four fixed-proposal chains use 8,000 draws each and discard the first 2,000 as
    warm-up. The sampler is the same simple Metropolis mechanism as above, now in
    three dimensions. Standardized samples are transformed back to intercept and
    slope in the original scientific units.

    **Read this worked example:** this fit runs automatically with fixed data and
    settings; the Section 6 controls do not change it. First read the slope and its
    95% posterior interval in mg/L per m/s, then compare the uncertainty band for
    mean oxygen with the wider predictive band for an individual river observation.
    The latter includes residual variation as well as parameter uncertainty.
    Inspect the posterior predictive check and chain diagnostics before interpreting
    the slope. An association in these observational data is not a causal effect of
    changing flow speed.
    """)


@app.cell
def _(run_regression_mcmc, water_data):
    regression_mcmc = run_regression_mcmc(water_data, seed=82406)
    return (regression_mcmc,)


@app.cell
def _(compact_table, np, pd, regression_mcmc):
    regression_parameter_labels = [
        "Intercept β₀ [mg/L]",
        "Slope β₁ [mg/L per m/s]",
        "Residual SD σ [mg/L]",
    ]
    regression_samples = regression_mcmc["posterior_original"]
    regression_summary_data = pd.DataFrame(
        {
            "Parameter": regression_parameter_labels,
            "Posterior mean": regression_samples.mean(axis=0),
            "Posterior SD": regression_samples.std(axis=0, ddof=1),
            "2.5%": np.quantile(regression_samples, 0.025, axis=0),
            "97.5%": np.quantile(regression_samples, 0.975, axis=0),
            "ESS": regression_mcmc["ess"],
            "Split R-hat": regression_mcmc["rhat"],
        }
    )
    regression_summary_table = compact_table(
        regression_summary_data,
        column_widths={
            "Parameter": 180,
            "Posterior mean": 105,
            "Posterior SD": 95,
            "2.5%": 75,
            "97.5%": 75,
            "ESS": 70,
            "Split R-hat": 85,
        },
        format_mapping={
            "Posterior mean": "{:.3f}",
            "Posterior SD": "{:.3f}",
            "2.5%": "{:.3f}",
            "97.5%": "{:.3f}",
            "ESS": "{:.0f}",
            "Split R-hat": "{:.3f}",
        },
    )
    return (regression_summary_table,)


@app.cell
def _(
    COLORS,
    FIGURE_SIZE_LINKED,
    mo,
    np,
    plt,
    regression_mcmc,
    regression_summary_table,
):
    regression_grid = regression_mcmc["x_grid"]
    posterior_mean_lower, posterior_mean_upper = np.quantile(
        regression_mcmc["mean_draws"], [0.025, 0.975], axis=0
    )
    posterior_predictive_lower, posterior_predictive_upper = np.quantile(
        regression_mcmc["predictive_draws"], [0.025, 0.975], axis=0
    )
    assert np.all(
        posterior_predictive_upper - posterior_predictive_lower
        > posterior_mean_upper - posterior_mean_lower
    )
    original_samples = regression_mcmc["posterior_original"]
    assert np.isfinite(original_samples).all()
    assert np.all(original_samples[:, 2] > 0)
    regression_figure, regression_axes = plt.subplots(1, 2, figsize=FIGURE_SIZE_LINKED)
    regression_axes[0].scatter(
        regression_mcmc["x"],
        regression_mcmc["y"],
        color=COLORS["orange"],
        edgecolor="white",
        linewidth=0.5,
        label="Observed rivers",
        zorder=4,
    )
    regression_axes[0].fill_between(
        regression_grid,
        posterior_predictive_lower,
        posterior_predictive_upper,
        color=COLORS["purple"],
        alpha=0.15,
        label="95% posterior predictive",
    )
    regression_axes[0].fill_between(
        regression_grid,
        posterior_mean_lower,
        posterior_mean_upper,
        color=COLORS["sky"],
        alpha=0.4,
        label="95% credible band for mean",
    )
    posterior_intercept_mean, posterior_slope_mean = original_samples[:, :2].mean(
        axis=0
    )
    regression_axes[0].plot(
        regression_grid,
        posterior_intercept_mean + posterior_slope_mean * regression_grid,
        color=COLORS["blue"],
        linewidth=2,
        label="Posterior mean line",
    )
    least_squares = regression_mcmc["least_squares"]
    regression_axes[0].plot(
        regression_grid,
        least_squares.intercept + least_squares.slope * regression_grid,
        color=COLORS["green"],
        linestyle="--",
        linewidth=1.6,
        label="Least-squares line",
    )
    regression_axes[0].set(
        title="Posterior regression and prediction",
        xlabel="Flow speed [m/s]",
        ylabel="Oxygen concentration [mg/L]",
        xlim=(0.1, 1.08),
        ylim=(-1, 17),
    )
    regression_axes[0].legend(frameon=False, fontsize=6.8)
    joint_indices = np.linspace(0, original_samples.shape[0] - 1, 3_000, dtype=int)
    regression_axes[1].scatter(
        original_samples[joint_indices, 0],
        original_samples[joint_indices, 1],
        color=COLORS["purple"],
        alpha=0.18,
        s=8,
        rasterized=True,
    )
    regression_axes[1].scatter(
        [least_squares.intercept],
        [least_squares.slope],
        color=COLORS["green"],
        marker="*",
        s=95,
        label="Least-squares estimate",
        zorder=4,
    )
    regression_axes[1].set(
        title="Joint posterior",
        xlabel="Intercept β₀ [mg/L]",
        ylabel="Slope β₁ [mg/L per m/s]",
    )
    regression_axes[1].legend(frameon=False, fontsize=7)
    for regression_axis in regression_axes:
        regression_axis.grid(linestyle=":", alpha=0.25)
    regression_figure.tight_layout()
    regression_acceptance = regression_mcmc["acceptance_rates"]
    regression_note = mo.callout(
        mo.md(
            f"The mean acceptance rate is **{regression_acceptance.mean():.1%}** "
            f"(chain range {regression_acceptance.min():.1%}–{regression_acceptance.max():.1%}). "
            "The narrow band concerns the unknown mean oxygen response. The wider "
            "posterior predictive interval also includes river-to-river residual variation."
        ),
        kind="info",
    )
    mo.vstack([regression_summary_table, regression_figure, regression_note])


@app.cell
def _(
    COLORS,
    FIGURE_SIZE_LINKED,
    autocorrelation_1d,
    mo,
    np,
    plt,
    regression_mcmc,
):
    regression_trace = regression_mcmc["samples"]
    regression_post = regression_mcmc["posterior_standard"]
    regression_diagnostic_figure, regression_diagnostic_axes = plt.subplots(
        2, 3, figsize=(7.6, 5.8), sharex="row"
    )
    diagnostic_labels = ["Centered intercept α", "Standardized slope βz", "log σ"]
    for parameter_index, parameter_label in enumerate(diagnostic_labels):
        trace_axis = regression_diagnostic_axes[0, parameter_index]
        acf_axis = regression_diagnostic_axes[1, parameter_index]
        for _regression_chain_index in range(regression_trace.shape[1]):
            trace_axis.plot(
                regression_trace[:, _regression_chain_index, parameter_index],
                linewidth=0.42,
                alpha=0.52,
            )
            regression_acf = autocorrelation_1d(
                regression_post[:, _regression_chain_index, parameter_index],
                max_lag=100,
            )
            acf_axis.plot(regression_acf, linewidth=0.9, alpha=0.75)
        trace_axis.axvline(
            regression_mcmc["warmup"],
            color=COLORS["vermillion"],
            linestyle="--",
            linewidth=1,
        )
        trace_axis.set(title=parameter_label, ylabel="Trace")
        acf_axis.axhline(0, color=COLORS["gray"], linestyle=":")
        acf_axis.set(xlabel="Lag", ylabel="Autocorrelation", ylim=(-0.2, 1.02))
        trace_axis.grid(linestyle=":", alpha=0.18)
        acf_axis.grid(linestyle=":", alpha=0.18)
    regression_diagnostic_figure.tight_layout()
    mo.vstack(
        [
            mo.md("### Computational diagnostics for the regression"),
            regression_diagnostic_figure,
            mo.callout(
                mo.md(r"""
            The chains start apart, pass through warm-up, and should settle into
            overlapping stationary traces. The reported ESS and split-$\hat R$
            summarize, but do not replace, visual inspection.
            """),
                kind="warn",
            ),
        ]
    )


@app.cell
def _(mo):
    mo.md(r"""
    ---

    ## 8. Bayesian workflow and responsible interpretation

    A posterior is not the end of an analysis. A defensible workflow is:

    1. define the biological question, observation unit, and generative model;
    2. choose priors and inspect what they predict before seeing the data;
    3. fit the model and diagnose the computation;
    4. inspect posterior estimates and posterior predictive behavior;
    5. check sensitivity to scientifically reasonable alternatives;
    6. report the conditional nature and limitations of the inference.

    The regression prior below is deliberately broad. Prior predictive simulation
    translates abstract parameter scales into oxygen concentrations where implausible
    behavior is easier to recognize.

    **Inspect, then decide:** find prior mean curves that enter the shaded negative
    oxygen region and read the fraction of simulated observations outside the
    illustrative 0–25 mg/L range. The gray curves describe possible mean relations;
    the reported fraction also includes observation noise. Name one model check
    that chain convergence cannot answer, then select a workflow conclusion to
    reveal feedback. The answer buttons do not change the priors or refit the model.
    A useful follow-up question is which scientifically justified prior or likelihood
    alternatives should be compared before relying on the fitted relationship.
    """)


@app.cell
def _(mo):
    audit_choice = mo.ui.radio(
        {
            "The chains have R-hat near one, so no further checks are needed": "diagnostics_only",
            "Inspect prior predictions, computation, posterior predictions, and sensitivity": "full_workflow",
            "Use a wider prior until the posterior becomes more certain": "wider",
        },
        value=None,
        label="Which workflow conclusion is defensible?",
    )
    return (audit_choice,)


@app.cell
def _(
    COLORS,
    FIGURE_SIZE_STANDARD,
    audit_choice,
    mo,
    np,
    plt,
    regression_mcmc,
    two_column_panel,
):
    prior_rng = np.random.default_rng(8240602)
    prior_alpha = prior_rng.normal(7.0, 5.0, 250)
    prior_beta_z = prior_rng.normal(0.0, 5.0, 250)
    prior_sigma = np.exp(prior_rng.normal(np.log(2.0), 0.75, 250))
    prior_x = np.linspace(0.12, 1.05, 80)
    prior_z = (prior_x - regression_mcmc["x_mean"]) / regression_mcmc["x_sd"]
    prior_mean_lines = prior_alpha[:, None] + prior_beta_z[:, None] * prior_z
    prior_predictive = (
        prior_mean_lines
        + prior_rng.normal(size=prior_mean_lines.shape) * prior_sigma[:, None]
    )
    implausible_fraction = float(
        np.mean((prior_predictive < 0) | (prior_predictive > 25))
    )
    prior_figure, prior_axis = plt.subplots(figsize=FIGURE_SIZE_STANDARD)
    for prior_line in prior_mean_lines[:80]:
        prior_axis.plot(
            prior_x, prior_line, color=COLORS["gray"], alpha=0.08, linewidth=0.8
        )
    prior_axis.scatter(
        regression_mcmc["x"],
        regression_mcmc["y"],
        color=COLORS["orange"],
        edgecolor="white",
        linewidth=0.4,
        label="Observed rivers",
        zorder=4,
    )
    prior_axis.axhspan(
        -15, 0, color=COLORS["vermillion"], alpha=0.08, label="Negative oxygen"
    )
    prior_axis.set(
        xlabel="Flow speed [m/s]",
        ylabel="Prior mean oxygen [mg/L]",
        xlim=(0.1, 1.08),
        ylim=(-15, 30),
    )
    prior_axis.grid(linestyle=":", alpha=0.25)
    prior_axis.legend(frameon=False, fontsize=8)
    prior_figure.tight_layout()
    audit_feedback = mo.callout(
        mo.md(
            "Select an answer above."
            if audit_choice.value is None
            else "**Correct.** Diagnostics address computation; model checking and sensitivity address different failure modes."
            if audit_choice.value == "full_workflow"
            else "Computational convergence cannot validate the likelihood, prior, measurements, or scientific design."
        ),
        kind="neutral"
        if audit_choice.value is None
        else "success"
        if audit_choice.value == "full_workflow"
        else "danger",
    )
    prior_note = mo.callout(
        mo.md(
            f"Across these fixed-seed prior predictive draws, **{implausible_fraction:.1%}** "
            "of simulated oxygen values lie below 0 or above 25 mg/L. A prior called "
            "'weak' on a coefficient scale can still imply scientifically implausible observations."
        ),
        kind="warn",
    )
    two_column_panel(
        mo.vstack([audit_choice, audit_feedback]),
        mo.vstack([prior_figure, prior_note]),
        widths=(1, 2),
    )


@app.cell
def _(mo):
    mo.md(r"""
    ---

    ## 9. Review questions

    Choose one answer for each question. Feedback appears immediately and identifies
    the interpretation or modeling principle at stake.
    """)


@app.cell
def _(mo):
    review_conditioning = mo.ui.radio(
        {
            "Positive predictive value P(sick | +)": "ppv",
            "Sensitivity P(+ | sick)": "sensitivity",
            "Specificity P(− | healthy)": "specificity",
        },
        value=None,
        label="Choose one answer",
    )
    review_sequential = mo.ui.radio(
        {
            "The d4 and d6 hypotheses receive posterior probability zero": "impossible_dice",
            "All dice become equally probable again": "uniform_again",
            "Only the d20 hypothesis remains possible": "only_d20",
        },
        value=None,
        label="Choose one answer",
    )
    review_credible = mo.ui.radio(
        {
            "95% of individual people have prevalence values in the interval": "individual",
            "Given the model, prior, and data, p has 95% posterior probability in the interval": "posterior",
            "The procedure must cover p in this particular dataset": "coverage",
        },
        value=None,
        label="Choose one answer",
    )
    review_mcmc = mo.ui.radio(
        {
            "Four chains remain in separated regions": "separated",
            "The posterior histogram is smooth": "smooth",
            "The acceptance rate is not exactly 50%": "acceptance",
        },
        value=None,
        label="Choose one answer",
    )
    review_prediction = mo.ui.radio(
        {
            "It includes residual variation for a new observation": "residual",
            "It uses fewer posterior samples": "fewer",
            "It removes parameter uncertainty": "removes",
        },
        value=None,
        label="Choose one answer",
    )
    return (
        review_conditioning,
        review_credible,
        review_mcmc,
        review_prediction,
        review_sequential,
    )


@app.cell
def _(
    mo,
    review_conditioning,
    review_credible,
    review_feedback,
    review_mcmc,
    review_prediction,
    review_sequential,
):
    review_blocks = [
        mo.vstack(
            [
                mo.md(
                    "**1. For a rare disease, which conditional probability answers whether a positive result represents disease and therefore changes with prevalence?**"
                ),
                review_conditioning,
                review_feedback(
                    review_conditioning.value,
                    correct_value="ppv",
                    correct_text="**Correct.** PPV is P(sick | positive); its denominator mixes true and false positives in proportions set partly by prevalence.",
                    incorrect_text="Sensitivity and specificity condition on disease status. PPV reverses the conditioning and depends on the population mix.",
                ),
            ]
        ),
        mo.vstack(
            [
                mo.md(
                    "**2. What happens to the dice hypotheses immediately after a revealed roll of 8?**"
                ),
                review_sequential,
                review_feedback(
                    review_sequential.value,
                    correct_value="impossible_dice",
                    correct_text="**Correct.** A roll of 8 has zero likelihood under d4 and d6, so their posterior probabilities become zero.",
                    incorrect_text="Sequential updating eliminates only hypotheses under which the observation is impossible; d8, d12, and d20 remain possible.",
                ),
            ]
        ),
        mo.vstack(
            [
                mo.md(
                    "**3. What does a 95% Bayesian credible interval for prevalence mean?**"
                ),
                review_credible,
                review_feedback(
                    review_credible.value,
                    correct_value="posterior",
                    correct_text="**Correct.** The probability statement is conditional on the specified model, prior, and observed data.",
                    incorrect_text="A parameter interval does not describe individual people, and frequentist coverage is a different statement.",
                ),
            ]
        ),
        mo.vstack(
            [
                mo.md(
                    "**4. Which trace behavior most clearly warns against trusting the combined MCMC histogram?**"
                ),
                review_mcmc,
                review_feedback(
                    review_mcmc.value,
                    correct_value="separated",
                    correct_text="**Correct.** Persistent between-chain disagreement indicates that the chains have not explored the same stationary distribution.",
                    incorrect_text="A smooth histogram or an acceptance rate different from one arbitrary target does not establish convergence failure by itself.",
                ),
            ]
        ),
        mo.vstack(
            [
                mo.md(
                    "**5. Why is a posterior predictive interval wider than a credible band for the mean line?**"
                ),
                review_prediction,
                review_feedback(
                    review_prediction.value,
                    correct_value="residual",
                    correct_text="**Correct.** Prediction combines uncertainty in the mean response with observation-to-observation residual scatter.",
                    incorrect_text="The width reflects additional residual variability, not fewer samples or removal of parameter uncertainty.",
                ),
            ]
        ),
    ]
    mo.vstack(review_blocks, gap=1.0)


@app.cell
def _(mo):
    mo.md(r"""
    ---

    ## 10. Summary and bridge

    - Bayesian probabilities quantify uncertainty conditional on a model and prior;
      frequentist probabilities describe repeated-sampling behavior.
    - Bayes' theorem reverses a conditional by combining a prior with a likelihood
      and normalizing over all hypotheses.
    - Predictive values depend on prevalence even when sensitivity and specificity
      are held fixed.
    - Sequential updating passes each posterior forward as the next prior.
    - The beta-binomial model provides an exact posterior and predictive distribution,
      making it an ideal reference for learning MCMC.
    - Metropolis draws are dependent. Trace plots, autocorrelation, ESS, multiple
      chains, and split-$\hat R$ assess computation, not scientific model validity.
    - Bayesian regression produces joint parameter distributions, credible bands for
      mean responses, and wider predictive intervals for future observations.
    - Every probability statement remains conditional on data quality, likelihood,
      prior assumptions, and study design.

    **Bridge:** This final chapter changes the language of inference but preserves the
    course's central habit: begin with a scientific question and design, show the data,
    quantify uncertainty, check the model, and state what the analysis cannot establish.
    """)


if __name__ == "__main__":
    app.run()
