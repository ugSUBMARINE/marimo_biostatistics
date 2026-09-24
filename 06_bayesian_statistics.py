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
    app_title="Bayesian statistics",
    css_file="site/assets/notebook.css",
)


@app.cell
def chapter_header():
    from companion_style import notebook_header

    notebook_header(
        6,
        "Bayesian statistics",
        "Updating uncertainty with data",
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

    from companion_data import read_csv
    from companion_style import (
        COLORS,
        FIGURE_SIZE_LINKED,
        FIGURE_SIZE_STANDARD,
        compact_table,
        responsive_row,
        review_feedback,
        two_column_panel,
    )

    return (
        COLORS,
        FIGURE_SIZE_LINKED,
        FIGURE_SIZE_STANDARD,
        compact_table,
        np,
        pd,
        plt,
        read_csv,
        responsive_row,
        review_feedback,
        stats,
        two_column_panel,
    )


@app.cell
def _(mo, np, pd, read_csv):
    water_path = mo.notebook_location() / "public" / "Wasserqualitaet.csv"
    water_raw = read_csv(water_path, encoding="utf-8-sig")
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
        next_roll_outcomes = np.arange(1, sides_array.max() + 1)
        next_roll_probability = (
            np.where(
                next_roll_outcomes[:, None] <= sides_array,
                1.0 / sides_array,
                0.0,
            )
            @ posterior
        )
        return {
            "prior": last_prior,
            "likelihood": last_likelihood,
            "unnormalized": last_unnormalized,
            "evidence": last_evidence,
            "posterior": posterior,
            "next_roll_outcomes": next_roll_outcomes,
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
    How common is an allergy? What fraction of bacterial isolates are resistant to
    an antibiotic? Bayesian statistics helps us update what we know about an
    unknown quantity when new data arrive. Instead of reporting only one estimate,
    we describe a range of possible values and how strongly each is supported.

    Three terms will recur throughout this chapter: the **prior** describes our
    uncertainty before using the current data; the **likelihood** describes how
    well each possible value explains those data; and the **posterior** describes
    our updated uncertainty. These conclusions depend on the assumptions we make
    about the experiment and the population it represents.

    By the end of this chapter, you should be able to:

    1. distinguish frequentist and Bayesian probability statements;
    2. apply Bayes' theorem using formulas and expected counts of people;
    3. update probabilities for competing explanations and for a population proportion;
    4. distinguish uncertainty about a population from variation in a future sample;
    5. explain how computer simulation estimates a posterior and check its reliability;
    6. interpret a Bayesian regression relating river flow to oxygen concentration.

    > **Main idea:** Start with plausible possibilities, give more weight to those
    > that better explain the observations, and report the uncertainty that remains.
    > We first do this with simple calculations, then with computer simulation.

    **Prerequisites:** conditional probability, binomial models, confidence
    intervals, diagnostic tests, and simple linear regression from Chapters 2–5.
    """)


@app.cell
def _(mo):
    mo.md(r"""
    ---

    ## 1. Two interpretations of probability

    Our first unknown **parameter** is allergy **prevalence**, the population
    fraction with an allergy. As in Chapter 2, a sample provides information
    about a fixed but unknown population quantity.

    A **frequentist** analysis evaluates how a method would perform if we repeatedly
    collected new samples in the same way. A **Bayesian** analysis assigns
    probabilities to possible parameter values to express our uncertainty given
    the available information. The population fraction need not physically change:
    it is our knowledge of it that changes. Both approaches require an appropriate
    study design and assumptions.

    For the same observation—3 people with an allergy among 10—the intervals below
    may look similar while answering different questions.

    A **95% confidence interval** retains Chapter 2's repeated-sampling meaning.
    The exact binomial method used here has coverage of **at least** 95%, rather
    than exactly 95%, because counts are discrete.
    A **95% credible interval** contains 95% of the Bayesian posterior probability
    for that fraction, given the prior, model, and observed sample.

    **Read, then classify:** decide whether each statement below concerns repeated
    sampling or our uncertainty after observing this sample. The answer buttons
    reveal feedback; they do not recalculate the intervals.
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
                    "At least 95% of intervals from repeated samples contain the true fraction p",
                    "Given the prior, model, and data, 95% probability that p is in this interval",
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
                    "For the Bayesian interval, we initially give equal probability to "
                    "equal-length ranges of $p$ between 0 and 1. This is a uniform "
                    "$\\operatorname{Beta}(1,1)$ prior; the data update it to "
                    "$\\operatorname{Beta}(4,8)$, explained in Section 6. The confidence "
                    "interval method used here includes the true fraction in at least "
                    "95% of repeated samples. This percentage is called its **coverage**. "
                    "Similar endpoints do not make the two interpretations interchangeable."
                ),
                kind="info",
            ),
        ]
    )


@app.cell
def _(mo):
    probability_statement = mo.ui.radio(
        {
            "A is frequentist; B is Bayesian": "frameworks",
            "A is Bayesian; B is frequentist": "reversed",
            "Both are frequentist": "both_frequentist",
            "Both are Bayesian": "both_bayesian",
        },
        value=None,
        label="Which classification is correct?",
    )
    return (probability_statement,)


@app.cell
def _(mo, probability_statement, review_feedback):
    mo.vstack(
        [
            mo.md(
                "**A.** If we repeatedly collected samples in the same way, at least "
                "95% of the intervals calculated by this method would contain the "
                "true population fraction p.\n\n"
                "**B.** Given the model, prior, and observed data, 95% of the posterior "
                "probability for p lies inside the reported credible interval."
            ),
            probability_statement,
            review_feedback(
                probability_statement.value,
                correct_value="frameworks",
                correct_text="**Correct.** In A, each new sample gives a new interval, while the true population fraction stays fixed. In B, the probability describes our uncertainty about that fraction after using this sample, prior, and model.",
                incorrect_text={
                    "reversed": "The classifications are reversed. A asks how often intervals from new samples "
                    "would contain the true fraction. B describes uncertainty about that fraction "
                    "given the observed data, prior, and model.",
                    "both_frequentist": "A is frequentist, but B assigns probability to the parameter after observing "
                    "the data. That is a Bayesian posterior statement, requiring a model and "
                    "prior.",
                    "both_bayesian": "B is Bayesian, but A concerns repeated use of an interval procedure with the "
                    "parameter fixed. That is a frequentist coverage statement, not posterior "
                    "probability for one interval.",
                },
            ),
        ]
    )


@app.cell
def _(mo):
    mo.md(r"""
    ---

    ## 2. Conditional probability and Bayes' theorem

    **Conditional probability** means probability given some information.
    $P(A\mid B)$ reads “the probability of A, given B”; the vertical bar means
    “given.” For example, the chance of a positive test in someone with a disease
    is different from the chance of disease in someone with a positive test.

    $P(A\cap B)$ means the probability that both A and B occur. We can calculate it
    in either order:

    $$P(A\cap B)=P(A)P(B\mid A)=P(B)P(A\mid B),$$

    and therefore

    $$P(B\mid A)=\frac{P(B)P(A\mid B)}{P(A)}.$$

    Let B be a possible explanation and A the observation. The **prior**, $P(B)$,
    is its probability before the observation. The **likelihood**, $P(A\mid B)$,
    asks how probable that observation would be if the explanation were true.
    The **posterior**, $P(B\mid A)$, is its probability after the observation.
    A likelihood is not itself the probability that the explanation is true.

    To update, multiply each explanation's prior by its likelihood, then divide
    by the total of these products. This last step, called **normalizing**, makes
    the updated probabilities add to 1. For a complete set of explanations that
    cannot both be true, the total is $P(A)$, the overall probability of the
    observation. It is also called the **evidence**.

    The next example makes this calculation concrete with **natural frequencies**:
    expected counts of people, such as “82 true positives among 1,000,000 tested.”
    """)


@app.cell
def _(mo):
    mo.md(r"""
    ---

    ## 3. Diagnostic tests, prevalence, and the base-rate effect

    Chapter 4 varied a diagnostic threshold to explore sensitivity and specificity.
    Here we first hold test performance fixed and ask how prevalence changes the meaning
    of a result. In this hypothetical infection screen, “sick” means the infection
    is present and “healthy” means it is absent; these labels do not describe a
    person's overall health. The symbols + and − mean positive and negative tests.

    - **Sensitivity:** among people with the infection, the fraction who test
      positive, $P(+\mid\text{sick})$.
    - **Specificity:** among people without it, the fraction who test negative,
      $P(-\mid\text{healthy})$.
    - **Positive predictive value (PPV):** among people who test positive, the
      fraction who actually have the infection, $P(\text{sick}\mid +)$.
    - **Negative predictive value (NPV):** among people who test negative, the
      fraction who do not have it, $P(\text{healthy}\mid -)$.

    **Prevalence**, also called the base rate, is the fraction of the tested
    population with the infection before the result is known. Even a test with
    high sensitivity and specificity can produce mostly false positives when
    the infection is very rare: there are so many more uninfected people to test.

    Let $p$ be prevalence, $\mathrm{Se}$ sensitivity, and $\mathrm{Sp}$ specificity,
    all expressed as proportions between 0 and 1. Bayes' theorem gives

    $$\mathrm{PPV}=\frac{\mathrm{Se}\,p}
    {\mathrm{Se}\,p+(1-\mathrm{Sp})(1-p)}.$$

    $$\mathrm{NPV}=\frac{\mathrm{Sp}(1-p)}
    {\mathrm{Sp}(1-p)+(1-\mathrm{Se})p}.$$

    The PPV denominator is the probability of **any positive result**: true
    positives plus false positives. The NPV denominator is the probability of
    **any negative result**: true negatives plus false negatives. In the
    natural-frequency table below, the same ratios are
    $\mathrm{PPV}=\mathrm{TP}/(\mathrm{TP}+\mathrm{FP})$ and
    $\mathrm{NPV}=\mathrm{TN}/(\mathrm{TN}+\mathrm{FN})$, where TP, FP, TN, and FN
    mean true positives, false positives, true negatives, and false negatives.
    “False” means the test result disagrees with the person's actual infection status.

    **Before moving the controls:** predict whether a positive result is more likely
    to be a true or false positive at the default prevalence of 0.01%.

    **Try this:** keep sensitivity at 82% and specificity at 96.3%, and compare
    prevalence values of 0.01%, 1%, and 10%. Follow the true-positive and
    false-positive counts into the denominator of PPV.
    Then restore prevalence to 0.01% and increase specificity to 99.9%: even a small
    false-positive rate can matter when the healthy population is very large.

    All controls update immediately. Changing the **Natural-frequency cohort**
    rescales expected counts, not the conditional probabilities; fractional counts
    are expectations, not fractions of a person. **Target PPV** only calculates
    the specificity that would be required—it does not set the actual specificity
    slider or change the current test's PPV. Prevalence, sensitivity, specificity,
    cohort size, and target PPV are hypothetical inputs for this teaching model.
    The bar chart compresses large counts using a logarithmic scale away from zero;
    use the table to compare exact counts.
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
        ylabel="Expected people [symmetric-log scale]",
        yscale="symlog",
        ylim=(0, max(values) * 2),
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

    ## 4. What does a significant test tell us?

    A null-hypothesis significance test (NHST) can be treated like the diagnostic
    test above. The two possibilities are $H_0$ (no effect) and $H_1$ (an effect
    under a specified alternative model). A “positive” result is now the event
    $S=\{p\leq\alpha\}$: the test rejects $H_0$ at a threshold chosen in advance.

    | Diagnostic test | Hypothesis test |
    | :--- | :--- |
    | Disease prevalence | Prior probability $\pi=P(H_1)$ |
    | Sensitivity | Power $P(S\mid H_1)=1-\beta$ |
    | False-positive rate, $1-\text{specificity}$ | $P(S\mid H_0)=\alpha$ |
    | Positive predictive value | $P(H_1\mid S)$ |

    Bayes' theorem gives

    $$P(H_1\mid S)=\frac{\pi(1-\beta)}{\pi(1-\beta)+(1-\pi)\alpha},
    \qquad
    P(H_0\mid S)=\frac{(1-\pi)\alpha}{\pi(1-\beta)+(1-\pi)\alpha}.$$

    With a 50% prior probability, 80% power, and $\alpha=0.05$, imagine 1,000
    studies: 400 of the 500 studies with an effect and 25 of the 500 without an
    effect yield significance. Among the 425 significant results, the expected
    fraction with an effect is $400/425=94.1\%$. The remaining **5.9%** is the
    posterior probability of $H_0$ under these assumptions.

    **Try this:** lower the prior probability to 10%, then 1%, keeping power at
    80% and $\alpha$ at 5%. Predict whether true or false positives will dominate.
    The table shows expected counts, which can be fractional; the curve shows
    how the posterior changes with the prior at the selected power and threshold.

    This example follows the post-test probability calculation in
    [GraphPad QuickCalcs](https://www.graphpad.com/quickcalcs/interpretpvalue1/).
    """)


@app.cell
def _(mo):
    nhst_prior = mo.ui.slider(
        1,
        99,
        value=50,
        step=1,
        show_value=True,
        full_width=True,
        label="Prior probability of H₁ [%]",
    )
    nhst_power = mo.ui.slider(
        10,
        99,
        value=80,
        step=1,
        show_value=True,
        full_width=True,
        label="Assumed power [%]",
    )
    nhst_alpha = mo.ui.dropdown(
        {"0.001": 0.001, "0.01": 0.01, "0.05": 0.05, "0.10": 0.10},
        value="0.05",
        full_width=True,
        label="Significance threshold α",
    )
    return nhst_alpha, nhst_power, nhst_prior


@app.cell
def _(
    COLORS,
    FIGURE_SIZE_STANDARD,
    compact_table,
    diagnostic_details,
    mo,
    nhst_alpha,
    nhst_power,
    nhst_prior,
    np,
    pd,
    plt,
    two_column_panel,
):
    nhst_result = diagnostic_details(
        nhst_prior.value / 100, nhst_power.value / 100, 1 - nhst_alpha.value, 1000
    )
    nhst_table = compact_table(
        pd.DataFrame(
            {
                "Result": ["Significant", "Not significant", "Total"],
                "H₁ true": [
                    nhst_result["true_positive"],
                    nhst_result["false_negative"],
                    nhst_result["sick"],
                ],
                "H₀ true": [
                    nhst_result["false_positive"],
                    nhst_result["true_negative"],
                    nhst_result["healthy"],
                ],
            }
        ),
        column_widths={"Result": 140, "H₁ true": 100, "H₀ true": 100},
        format_mapping={"H₁ true": "{:,.1f}", "H₀ true": "{:,.1f}"},
    )
    nhst_prior_grid = np.linspace(0, 1, 201)
    nhst_posterior_curve = (
        nhst_prior_grid
        * nhst_power.value
        / 100
        / (
            nhst_prior_grid * nhst_power.value / 100
            + (1 - nhst_prior_grid) * nhst_alpha.value
        )
    )
    nhst_figure, nhst_axis = plt.subplots(figsize=FIGURE_SIZE_STANDARD)
    nhst_axis.plot(
        nhst_prior_grid * 100,
        nhst_posterior_curve * 100,
        color=COLORS["blue"],
        label="After significance",
    )
    nhst_axis.plot(
        [0, 100], [0, 100], linestyle=":", color=COLORS["gray"], label="Prior unchanged"
    )
    nhst_axis.scatter(
        [nhst_prior.value],
        [nhst_result["ppv"] * 100],
        color=COLORS["vermillion"],
        zorder=3,
    )
    nhst_axis.set(
        xlabel="Prior probability of H₁ [%]",
        ylabel="P(H₁ | significant) [%]",
        xlim=(0, 100),
        ylim=(0, 100),
    )
    nhst_axis.legend(frameon=False, fontsize=8)
    nhst_axis.grid(linestyle=":", alpha=0.3)
    nhst_figure.tight_layout()
    two_column_panel(
        mo.vstack(
            [
                nhst_prior,
                nhst_power,
                nhst_alpha,
                mo.stat(
                    f"{nhst_result['ppv']:.1%}",
                    label="P(H₁ | significant)",
                    bordered=True,
                ),
                mo.stat(
                    f"{1 - nhst_result['ppv']:.1%}",
                    label="P(H₀ | significant)",
                    bordered=True,
                ),
            ]
        ),
        mo.vstack(
            [mo.md("**Expected counts among 1,000 studies**"), nhst_table, nhst_figure]
        ),
        widths=(1, 2.2),
    )


@app.cell
def _(mo):
    mo.md(r"""
    **What this calculation assumes.** The prior assigns probabilities to the
    two competing hypotheses before observing the current data. Power depends
    on effect size, sample size, variability, and the test. For a composite $H_1$,
    use power averaged over an explicit prior distribution of alternative effect
    sizes. The controls explore assumed operating characteristics: changing
    $\alpha$ in an actual fixed study also changes its power.

    Here $P(S\mid H_0)=\alpha$ assumes a correctly calibrated test. A conservative
    or discrete test may have a smaller actual false-positive rate. Unaccounted
    multiple testing, optional stopping, or selective reporting can invalidate
    the assumed rates or require a model of selection.

    **What the posterior means.** We have conditioned only on significance.
    Results with $p=0.049$ and $p=0.00001$ receive the same update here at
    $\alpha=0.05$. Neither the observed p-value nor $\alpha$ is $P(H_0\mid S)$.
    A Bayesian analysis using the full data needs likelihoods and priors under
    both hypotheses and can distinguish those results. Do not substitute the
    observed p-value for $\alpha$ while leaving power unchanged.

    Likewise, a non-significant result does not establish $H_0$: its posterior
    depends on the prior and on how often the test misses effects under $H_1$.
    """)


@app.cell
def _(mo):
    mo.md(r"""
    ---

    ## 5. Sequential updating with dice

    Suppose one hidden die was selected from dice with 4, 6, 8, 12, or 20 sides.
    Each die is a **hypothesis**, or possible explanation of the rolls. We assume
    each die is fair, its faces are numbered 1 to its number of sides, and rolls
    are independent once the die is chosen. For a die with $s$ sides, each possible
    result has probability $1/s$; an impossible result has probability zero.
    These probabilities are the likelihoods used in the update. “d6” means a
    six-sided die.

    After each roll, the posterior becomes the prior for the next roll. For
    example, a 6 rules out d4 and favors d6 over d20 because a 6 is more probable
    on d6. The same updating principle applies when successive experiments give
    new information about an unknown biological quantity.

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
    your sequence without changing the selected prior. The lecture-only reveal
    slider is disabled while you enter your own rolls.

    The predictive chart shows the **next roll of the same hidden die**, averaging
    over all dice that remain possible. Reveal the first 7 and compare the plots:
    d4 and d6 are ruled out, but rolling a 1 through 6 is still possible on the
    larger dice. Before any rolls are revealed, this prediction uses the initial
    prior; after observations, it is a **posterior predictive distribution**:
    probabilities for future results that include our uncertainty about which
    die generated the data.
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
    return dice_prior_choice, dice_sequence_source


@app.cell
def _(dice_sequence_source, mo):
    dice_reveal = mo.ui.slider(
        0,
        6,
        value=0,
        show_value=True,
        full_width=True,
        label="Number of revealed rolls",
        disabled=dice_sequence_source.value != "lecture",
    )
    return (dice_reveal,)


@app.cell
def _(mo):
    get_student_dice_rolls, set_student_dice_rolls = mo.state([])
    return get_student_dice_rolls, set_student_dice_rolls


@app.cell
def _(dice_sequence_source, mo, responsive_row, set_student_dice_rolls):
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
            responsive_row(
                [student_dice_add, student_dice_undo, student_dice_reset],
                min_width=9,
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
        lecture_rolls if dice_sequence_source.value == "lecture" else student_rolls
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

    dice_plot_size = (FIGURE_SIZE_LINKED[0] / 2, FIGURE_SIZE_LINKED[1])
    dice_figure, dice_axis = plt.subplots(figsize=dice_plot_size)
    positions = np.arange(dice_sides.size)
    dice_axis.bar(
        positions - 0.18,
        dice_result["prior"],
        width=0.36,
        color=COLORS["gray"],
        alpha=0.7,
        label="Prior before\nlatest roll",
    )
    dice_axis.bar(
        positions + 0.18,
        dice_result["posterior"],
        width=0.36,
        color=COLORS["blue"],
        label="Posterior",
    )
    dice_axis.set(
        title="Which die is likely?",
        xticks=positions,
        xticklabels=[f"d{side}" for side in dice_sides],
        xlabel="Hidden-die hypothesis",
        ylabel="Probability",
        ylim=(0, 1),
    )
    dice_axis.grid(axis="y", linestyle=":", alpha=0.3)
    dice_axis.legend(frameon=False, fontsize=8)
    dice_figure.tight_layout()

    dice_history_figure, dice_history_axis = plt.subplots(figsize=dice_plot_size)
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
    dice_history_figure.tight_layout()

    next_roll_outcomes = dice_result["next_roll_outcomes"]
    next_roll_probability = dice_result["next_roll_probability"]
    assert np.isclose(next_roll_probability.sum(), 1.0)
    dice_predictive_figure, dice_predictive_axis = plt.subplots(figsize=dice_plot_size)
    dice_predictive_axis.bar(
        next_roll_outcomes,
        next_roll_probability,
        width=0.8,
        color=COLORS["green"],
    )
    dice_predictive_axis.set(
        title="What might the next roll be?",
        xlabel="Next roll of the same die",
        ylabel="Predictive probability",
        xticks=[1, 4, 6, 8, 12, 16, 20],
        xlim=(0.3, 20.7),
        ylim=(0, 0.27),
    )
    dice_predictive_axis.grid(axis="y", linestyle=":", alpha=0.3)
    dice_predictive_figure.tight_layout()
    next_roll_above_six = float(next_roll_probability[next_roll_outcomes > 6].sum())
    sequence_label = (
        "Lecture sequence"
        if dice_sequence_source.value == "lecture"
        else "Your sequence"
    )
    latest_roll_note = (
        f"Before observing the latest **{revealed_rolls[-1]}**, its predictive "
        f"probability was **{dice_result['evidence']:.3f}** (the update's evidence). "
        "The predictive chart uses the updated posterior to predict the next roll."
        if revealed_rolls
        else "No rolls observed yet: the predictive chart uses the initial prior."
    )
    dice_note = mo.callout(
        mo.md(
            f"""
            **From beliefs to predictions**

            The first two plots describe **which die is hidden**. The third plot
            predicts **its next roll**, averaging each die's roll probabilities
            using its current posterior weight.

            Chance of a roll above 6: **{next_roll_above_six:.1%}**.
            Ruling out a small die does not rule out small rolls on larger dice.

            **{sequence_label}:** {revealed_rolls or "none"}.

            {latest_roll_note}
            """
        ),
        kind="info",
    )
    dice_controls = mo.vstack(
        [dice_prior_choice, dice_sequence_source, dice_reveal, student_dice_controls]
    )
    mo.Html(
        f'<div class="dice-explorer-controls">'
        f"<div>{dice_controls.text}</div>"
        f"<div>{mo.as_html(update_table).text}</div>"
        f"</div>"
        f'<div class="dice-explorer-grid">'
        f"<div>{mo.as_html(dice_figure).text}</div>"
        f"<div>{mo.as_html(dice_history_figure).text}</div>"
        f"<div>{mo.as_html(dice_predictive_figure).text}</div>"
        f'<div class="dice-explorer-note">{dice_note.text}</div>'
        f"</div>"
    )


@app.cell
def _(mo):
    mo.md(r"""
    ---

    ## 6. Bayesian inference for a proportion

    Let $p$ be the fraction of a population with an allergy. We observe $k$ people
    with the allergy in a sample of $n$. The same calculation could describe
    resistant isolates among independently sampled bacteria, or successful
    cultures among independent attempts under the same conditions.

    Use Chapter 2's **binomial model**: $n$ independent binary observations with
    a common probability $p$, counting allergy as “success.” In Chapter 4 we tested
    a specified value of $p$; here we update uncertainty across its possible values.
    Assume allergy status is measured correctly; this example does not model the
    test errors from Section 3.

    A **beta distribution** describes uncertainty about a proportion between 0 and 1.
    Its two settings, $a$ and $b$, control its shape. The prior mean is $a/(a+b)$;
    increasing $a+b$ while keeping this mean fixed concentrates the prior more
    tightly around it. With $a=b=1$, the prior is flat: equal-length ranges of $p$
    have equal probability.

    We write the prior and observation model as:

    $$p\sim \operatorname{Beta}(a,b),\qquad K\mid p\sim \operatorname{Binomial}(n,p).$$

    The capital $K$ represents the count before it is observed; $k$ is the actual
    count. Multiplying the prior by the likelihood gives a particularly simple
    update:

    $$p\mid k,n\sim \operatorname{Beta}(a+k,b+n-k).$$

    Add the $k$ counted outcomes to $a$ and the $n-k$ other outcomes to $b$.
    For example, a Beta(1,1) prior and 3 people with an allergy out of 10 give
    Beta(4,8). This convenient pairing is called **conjugate**: the posterior stays
    in the same distribution family as the prior. The prior settings are not
    extra people observed in the study.

    **Reading the plot:** the prior and posterior curves show **probability
    density**. Probability is the area under a curve over a range, not its height
    at one point; a density can exceed 1. The shaded area contains 95% of the
    posterior probability, leaving 2.5% on either side. The likelihood curve has
    been rescaled to compare its shape; its height is not a probability for $p$.

    **Try this:** with $n=10$ and $k=3$, compare $a=b=1$ with $a=b=10$.
    Both priors have mean 0.5, but the second is more concentrated: watch the
    posterior move between values favored by the prior and those favored by the data.
    Restore $a=b=1$, increase $n$ to 100, and then set $k=30$ to keep the observed
    proportion at 0.3. Compare posterior widths at the same observed proportion.

    All outputs update immediately. **Set $n$ before $k$:** changing $n$ rebuilds
    the count control and resets it to 3 (or to $n$ if smaller). The threshold $p_0$
    changes only the reported probability that $p$ exceeds that value,
    $P(p>p_0\mid\text{data})$. For example, $p_0=0.2$ asks how probable it is that
    more than 20% of the population has the allergy. This probability depends on
    the chosen prior and binomial model; moving the threshold does not change them.
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
        label="Prior shape a",
    )
    beta_prior_b = mo.ui.slider(
        0.5,
        20.0,
        step=0.5,
        value=1.0,
        show_value=True,
        full_width=True,
        label="Prior shape b",
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
        label="Observed count k (with allergy)",
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

    “What fraction of the population has the allergy?” and “How many people in my
    next sample will have it?” are different questions. The **posterior** answers
    the first. The **posterior predictive distribution** answers the second,
    allowing for both uncertainty about $p$ and chance variation in a new sample
    from the same population. It combines predictions from possible values of $p$,
    weighted by their posterior probabilities. For $m$ future observations, this
    count distribution is called **beta-binomial**.

    The comparison curve uses only the posterior mean of $p$, treating that value
    as known. This shortcut is called a **plug-in prediction**. It includes chance
    variation in the new sample but leaves out uncertainty about $p$.

    **Try this:** keep the prior and observed data in Section 6 fixed, then compare
    future sample sizes $m=1$, 20, and 100. At $m=1$ the two predictions coincide;
    for larger samples, compare the probabilities of unusually low or high counts.
    Compare the reported **variances** as well as the tail probabilities.
    Both predict **counts** of future successes, so changing $m$ also changes
    the horizontal scale. This control updates immediately and does not add observations to the
    posterior. To see how more existing evidence changes prediction, hold $m=20$
    and compare $k/n=3/10$ with $30/100$ in Section 6.
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

    ## 7. Posterior computation with MCMC

    For the allergy example, we can calculate the posterior exactly. More complex
    models often need a computer approximation. **Markov chain Monte Carlo (MCMC)**
    explores possible parameter values by taking a sequence of random steps. Each
    recorded value is a **draw**; the sequence is a **chain**. With a suitable
    algorithm and enough exploration, the fraction of draws in a range estimates
    that range's posterior probability. These draws are simulated parameter values,
    not new experimental observations.

    Here we use the **Metropolis algorithm**. Propose a random move from the current
    value $\theta$ to a new value $\theta'$. Accept a move to a higher posterior
    density; accept a move to a lower density with probability given by the ratio
    below. If rejected, record the current value again. Accepting some moves to
    lower density lets the chain explore uncertainty rather than simply find a peak.
    For the symmetric proposals used here, the acceptance probability is

    $$\min\left(1,\frac{p(\theta'\mid D)}{p(\theta\mid D)}\right).$$

    Here $D$ denotes the observed data and $\theta$ is the allergy prevalence $p$.
    The scaling factor that makes the posterior's total area equal to 1 cancels
    in the ratio. The code uses logarithms to avoid numerical problems with very
    small probabilities. We start with the exactly solvable allergy example so
    that we can compare the simulated histogram with the true posterior.

    **Reading the checks below:**

    - A **trace plot** shows the recorded values in order. Chains started at
      different values should explore similar ranges without a lasting upward or
      downward drift. This exploration is often called **mixing**.
    - **Autocorrelation** measures how strongly values within a chain resemble
      earlier ones. **Lag** is their separation in steps. Correlation that stays
      high over many steps means the chain is exploring slowly.
    - **Effective sample size (ESS)** estimates how many independent draws would
      provide comparable precision. Thousands of similar consecutive draws can
      contain much less information than thousands of independent draws.
    - **Split R-hat** compares variation within and between chain halves. A value
      near 1 is reassuring; a clearly larger value warns that the chains disagree
      or are still drifting. Agreement is not proof of adequate exploration.

    The ESS and R-hat calculations here are simplified teaching diagnostics. Use
    them together with the plots, rather than treating one number as a pass/fail test.

    **How to run:** choose the prior and observed data in Section 6, select a
    proposal standard deviation (SD, the scale of attempted moves) and draws per
    chain here, then click **Run four
    new chains**. Until you click, the plots and diagnostics retain the previous
    run, including its previous prior and data. The initial run uses $a=b=1$,
    $n=10$, and $k=3$.

    **Try this:** keep that target and 5,000 draws fixed, and run proposal SDs
    0.005, 0.1, and 0.5. Tiny proposals can be accepted frequently while exploring
    slowly; large proposals can be rejected frequently. Compare traces,
    autocorrelation, effective sample size (ESS), and agreement with the exact beta
    density, not acceptance rate alone. Repeat a setting to see how the computer's
    random draws change the approximation even though the data have not changed.

    **Inspect chain 1 through step** only reveals more of the existing early walk;
    it does not run a new chain or change the full-run diagnostics below. A new
    run resets this inspection slider to step 50. We discard the first 20% of
    beta-model draws as **warmup**, an initial period excluded to reduce the
    influence of starting values; the regression example uses 2,000 of 8,000 draws
    (25%). These are teaching choices; the trace and convergence checks are
    needed to assess whether warmup was sufficient. Look for
    overlapping traces without persistent drift, decaying autocorrelation, and
    split R-hat near one. These are computational checks, not proof that the
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
        "Very high acceptance can occur when moves are too small to explore efficiently."
        if mean_acceptance > 0.75
        else "Very low acceptance can occur when attempted moves are too large."
        if mean_acceptance < 0.15
        else "Both accepted and rejected moves occur; check the traces and ESS to assess exploration."
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
            mo.stat(
                f"{beta_walk_result['rhat']:.3f}", label="Split R-hat", bordered=True
            ),
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
            A smooth histogram alone is not enough. Check whether the chains explore
            similar ranges, whether autocorrelation falls toward zero, and whether
            ESS is large enough for the precision you need. Split R-hat should be
            near 1. A longer run may help slow exploration, but it cannot repair
            inappropriate model assumptions or coding errors. Keeping only every
            tenth draw, for example, discards values without improving the walk itself.
            """),
                kind="warn",
            ),
        ]
    )


@app.cell
def _(mo):
    mo.md(r"""
    ---

    ## 8. Bayesian linear regression with Metropolis sampling

    We return to the 24 river observations from Chapter 3. How does average oxygen
    concentration differ with flow speed, and how uncertain is this relationship?
    Our model uses a straight line for the average and allows individual
    observations to fall above or below it.

    To make the computer's exploration easier, we subtract mean flow speed
    $\bar x$ and divide by its standard deviation $s_x$. This **standardized**
    flow speed, $z$, is zero at the sample mean and increases by 1 for each
    standard deviation of flow speed:

    $$\text{Oxygen}_i\sim N(\alpha+\beta_z z_i,\sigma^2),\qquad
      z_i=\frac{x_i-\bar x}{s_x}.$$

    Here $i$ labels a river; as before, $N(\mu,\sigma^2)$ uses mean and variance.
    **$\alpha$** is mean oxygen at average flow speed, not the significance level
    from Chapters 4–5. **$\beta_z$** is the change in mean oxygen for a
    one-standard-deviation increase in flow speed.
    **Residual SD $\sigma$** describes variation around the line that flow speed
    does not explain, including other environmental differences and measurement
    variation. We assume independent observations and the same residual SD at
    every flow speed.

    We use the following broad priors, whose biological plausibility we inspect
    in Section 9:

    $$\alpha\sim N(7,5^2),\quad \beta_z\sim N(0,5^2),\quad
      \log\sigma\sim N(\log 2,0.75^2).$$

    These priors center average-flow oxygen at 7 mg/L and allow either a positive
    or negative slope. Modeling $\log\sigma$ (the natural logarithm of $\sigma$)
    gives $\sigma$ a **log-normal** prior, using the distribution introduced in
    Chapter 1. It stays positive, with a prior median of 2 mg/L.

    Four chains use 8,000 draws each and discard the first 2,000 as warmup.
    Each Metropolis step now proposes values for all three unknown quantities.
    Results are converted back to the original units: **slope $\beta_1$** is the
    change in mean oxygen in mg/L per 1 m/s increase in flow speed, and
    **intercept $\beta_0$** is the modeled mean at zero flow. Zero flow is outside
    the observed range, so the intercept needs care as a biological interpretation.

    In the table, **posterior mean** is the average of the retained draws and
    **posterior SD** measures their spread, describing uncertainty about each
    parameter. The 2.5% and 97.5% columns bound its 95% credible interval.
    ESS and R-hat refer to the quantities used by the sampler ($\alpha$, $\beta_z$,
    and $\log\sigma$), which are also shown in the diagnostic plots.

    **Read this worked example:** this fit runs automatically with fixed data and
    settings; the Section 7 controls do not change it. First read the slope and its
    95% posterior interval in mg/L per m/s, then compare the uncertainty band for
    mean oxygen with the wider predictive band for an individual river observation.
    As in Chapter 3, predicting an individual adds variation around the mean;
    here both bands use posterior probabilities. They give 95% intervals at each
    flow speed, not a 95% probability for an entire curve to stay inside a band.
    In the **joint posterior** plot, each dot pairs an intercept and slope from
    the same draw. Its shape shows how uncertainty in these two quantities is linked.

    Inspect the chain diagnostics before interpreting the slope. The prediction
    band illustrates uncertainty; it is not by itself a check that the model
    describes the data adequately. Such a check would compare simulated datasets
    with observed patterns, as discussed next. These observational data establish
    an association, not the effect of experimentally changing flow speed.
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
            "prediction band also includes variation among individual river observations "
            "that flow speed does not explain."
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
            Each column tracks one quantity used by the sampler: mean oxygen at
            average flow, the slope for standardized flow, or the logarithm of
            residual SD. After warmup, the chains should explore similar ranges
            without lasting drift. Check these plots alongside ESS and split R-hat.
            """),
                kind="warn",
            ),
        ]
    )


@app.cell
def _(mo):
    mo.md(r"""
    ---

    ## 9. Bayesian workflow and responsible interpretation

    A precise answer can still be misleading if the model is inappropriate. A
    useful workflow connects the calculations back to the experiment:

    1. **Define the question and what counts as an independent observation.**
       Separate cultures may be independent; repeated readings from one culture
       generally are not. Describe how the model represents variation in the data.
    2. **Check the priors.** Simulate observations using only the priors and the
       observation model. This **prior predictive check** asks whether the model
       allows biologically reasonable results before it learns from the data.
    3. **Fit the model and check the computation.** Use the chain plots, ESS, and
       R-hat to assess whether the computer has explored the posterior adequately.
    4. **Compare predictions with observations.** A **posterior predictive check**
       simulates new datasets using the fitted model. For the river example, ask
       whether they reproduce the observed spread and relationship with flow speed.
       Systematic differences suggest something is missing from the model.
    5. **Try reasonable alternative assumptions.** This is a **sensitivity analysis**:
       check whether conclusions change with scientifically plausible priors or
       models. Here “sensitivity” does not mean diagnostic-test sensitivity.
    6. **Report estimates, uncertainty, and limitations.** Explain the assumptions,
       how observations were collected, and which scientific claims the study supports.

    The regression prior below is deliberately broad. Prior predictive simulation
    translates its settings into oxygen concentrations, making implausible
    predictions easier to recognize. Negative concentrations are impossible;
    25 mg/L is an illustrative upper reference here, not a universal biological limit.

    **Inspect, then decide:** find prior mean curves that enter the shaded negative
    oxygen region and read the fraction of simulated observations outside the
    illustrative 0–25 mg/L range. The gray curves describe possible mean relations;
    the reported fraction also includes variation around those means. These are
    prior predictions; the observed points are shown only for comparison and do
    not update the curves. Name one biological assumption that agreement between
    chains cannot verify, then select a workflow conclusion to
    reveal feedback. The answer buttons do not change the priors or refit the model.
    A useful follow-up question is which scientifically justified prior or likelihood
    alternatives should be compared before relying on the fitted relationship.
    """)


@app.cell
def _(mo):
    audit_choice = mo.ui.radio(
        {
            "The chains have split R-hat near one, so no further checks are needed": "diagnostics_only",
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
    review_feedback,
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
    audit_feedback = review_feedback(
        audit_choice.value,
        correct_value="full_workflow",
        correct_text="**Correct.** Check what the model predicts before fitting, whether the chains explore adequately, whether predictions after fitting resemble the data, and whether reasonable alternative assumptions change the conclusions. Each check answers a different question.",
        incorrect_text={
            "diagnostics_only": "Split R-hat near one is reassuring about agreement between chains. They can still agree when the model makes biologically unrealistic assumptions. Check predictions, study design, and the effect of alternative assumptions as well.",
            "wider": "A wider prior allows more parameter values; it does not guarantee a more certain or more credible posterior. Inspect prior predictions and justify alternatives scientifically rather than choosing a prior to force precision.",
        },
    )
    prior_note = mo.callout(
        mo.md(
            f"In this reproducible simulation before fitting, **{implausible_fraction:.1%}** "
            "of simulated oxygen values lie below 0 or above 25 mg/L. Allowing a "
            "broad range of intercepts and slopes can produce biologically implausible predictions."
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

    ## 10. Review questions

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
            "Any 95% credible-interval procedure also has exactly 95% repeated-sampling coverage": "coverage",
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
            "It includes variation of a new observation around the mean line": "residual",
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
                    "**1. Holding sensitivity and specificity fixed, which probability answers ‘Given a positive test, how likely is disease?’ and changes with prevalence?**"
                ),
                review_conditioning,
                review_feedback(
                    review_conditioning.value,
                    correct_value="ppv",
                    correct_text="**Correct.** PPV is P(sick | positive): true positives divided by all positive results. When disease is rare, false positives from the much larger group without disease can outnumber true positives.",
                    incorrect_text={
                        "sensitivity": "Sensitivity asks how often the test is positive among people with disease. The "
                        "question starts with a positive result and asks about disease status, so it "
                        "requires PPV and the prevalence as well as test performance.",
                        "specificity": "Specificity asks how often the test is negative among healthy people. It helps "
                        "determine false positives, but the probability of disease after a positive result "
                        "is PPV.",
                    },
                ),
            ]
        ),
        mo.vstack(
            [
                mo.md(
                    "**2. Start with positive prior probabilities for the fair d4, d6, d8, d12, and d20. After a first roll of 8, which hypotheses receive zero posterior probability?**"
                ),
                review_sequential,
                review_feedback(
                    review_sequential.value,
                    correct_value="impossible_dice",
                    correct_text="**Correct.** A roll of 8 has zero likelihood under d4 and d6, so their posterior probabilities become zero.",
                    incorrect_text={
                        "uniform_again": "The new likelihood reweights the current probabilities; it does not reset the "
                        "prior. A roll of 8 has zero probability under d4 and d6, and likelihoods 1/8, "
                        "1/12, and 1/20 under the surviving dice.",
                        "only_d20": "The d8 and d12 can also produce an 8. With positive prior probabilities, all of d8, "
                        "d12, and d20 survive; only d4 and d6 assign zero likelihood to this roll.",
                    },
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
                    incorrect_text={
                        "individual": "Prevalence p is a population proportion, not a different numerical value carried "
                        "by each person. The interval expresses uncertainty about that proportion, not "
                        "variation among individual disease outcomes.",
                        "coverage": "Repeated-sampling coverage and posterior probability are different properties. A 95% "
                        "credible interval contains 95% of the posterior probability given this model, prior, and "
                        "data; it does not automatically have 95% frequentist coverage.",
                    },
                ),
            ]
        ),
        mo.vstack(
            [
                mo.md(
                    "**4. Which finding after warmup most clearly warns against trusting the combined MCMC histogram?**"
                ),
                review_mcmc,
                review_feedback(
                    review_mcmc.value,
                    correct_value="separated",
                    correct_text="**Correct.** If chains stay in different ranges after warmup, they may not have explored the posterior adequately. Combining their draws into one histogram can hide this problem. Inspect the separate traces, split R-hat, and ESS before using the results.",
                    incorrect_text={
                        "smooth": "Histogram smoothness depends partly on binning and draw count. A smooth pooled "
                        "histogram can still hide chains trapped in different regions; compare their traces, "
                        "split R-hat, and effective sample sizes.",
                        "acceptance": "There is no universal requirement of exactly 50% acceptance. Tiny steps may be "
                        "accepted often yet explore slowly, and suitable acceptance rates depend on "
                        "the attempted moves and posterior shape. Assess exploration and ESS together.",
                    },
                ),
            ]
        ),
        mo.vstack(
            [
                mo.md(
                    "**5. At a given flow speed, why is the 95% prediction interval for a new river observation wider than the 95% credible interval for mean oxygen concentration?**"
                ),
                review_prediction,
                review_feedback(
                    review_prediction.value,
                    correct_value="residual",
                    correct_text="**Correct.** Even if we knew mean oxygen concentration exactly, individual observations would vary around it. Predicting a new observation includes this variation as well as uncertainty about the mean.",
                    incorrect_text={
                        "fewer": "Using fewer computer draws makes the estimated interval less precise. "
                        "The reason for the wider prediction interval is different: it includes "
                        "variation among new observations as well as uncertainty about the mean.",
                        "removes": "Posterior prediction retains uncertainty in the fitted mean and adds variation of a "
                        "new observation around that mean. Removing parameter uncertainty would omit a source "
                        "of variation rather than explain the wider interval.",
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

    ## 11. Summary and bridge

    - **Prior → data → posterior:** update uncertainty by giving more weight to
      explanations that better account for the observations. Each posterior can
      become the prior when new data arrive.
    - **Ask which probability you need.** The chance of a positive test given
      disease differs from the chance of disease given a positive test. The latter
      depends on how common the disease is in the tested population.
    - **Keep the interval interpretations distinct.** A credible interval describes
      uncertainty given this model, prior, and data. A confidence-interval method
      is evaluated by how often its intervals contain the truth across new samples.
    - **Interpret significance in context.** The probability of an effect after
      significance depends on its prior probability, power, and the false-positive
      rate. This update uses only the significant/non-significant outcome.
    - **Separate estimation from prediction.** Estimating an allergy prevalence or
      mean oxygen concentration differs from predicting a new sample or observation.
      Prediction also includes variation among observations.
    - **Check the computer's approximation.** MCMC draws are linked through the
      steps of a chain. Use trace plots, autocorrelation, ESS, and split R-hat
      together to assess whether exploration is adequate.
    - **Check the biology too.** Reliable computation cannot rescue inappropriate
      assumptions. Examine predictions, compare reasonable alternatives, and
      consider study design and data quality before drawing scientific conclusions.

    **Bridge:** This final chapter changes the language of inference but preserves the
    course's central habit: begin with a scientific question and design, show the data,
    quantify uncertainty, check the model, and state what the analysis cannot establish.
    """)


if __name__ == "__main__":
    app.run()
