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
    app_title="Inferential statistics",
    css_file="site/assets/notebook.css",
)


@app.cell
def chapter_header():
    from companion_style import notebook_header

    notebook_header(
        2,
        "Inferential statistics",
        "Samples, errors, and confidence",
    )


@app.cell
def _():
    import marimo as mo

    return mo


@app.cell
def _():
    from itertools import product

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from scipy import stats

    from companion_data import read_csv
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
        product,
        read_csv,
        review_feedback,
        stats,
        two_column_panel,
    )


@app.cell
def _(mo):
    mo.md(r"""
    Descriptive statistics summarize the measurements we have. **Inferential
    statistics** uses those measurements to learn about a larger group or about
    future repetitions of an experiment, while describing how uncertain our
    conclusions are. For example, how well does the mean enzyme activity in six
    independent cultures estimate the mean activity under those growth conditions?

    By the end of this chapter, you should be able to:

    1. identify the population you want to study and the sample you measured;
    2. read probabilities from graphs for counts and continuous measurements;
    3. distinguish variation among individual measurements from variation among
       averages calculated from repeated samples;
    4. explain standard errors and confidence intervals in words;
    5. use resampling and simulation to explore uncertainty in calculated results.

    The simulations use artificial data. A **random seed** is a starting setting
    that makes a simulation reproducible. Buttons marked **Draw again** or
    **Run a new simulation** generate new random results; unrelated controls do
    not change the observations.

    > **Preview:** Repeating an experiment usually gives a different estimate.
    > A standard error describes how much estimates vary across repetitions.
    > A confidence interval expresses uncertainty using a method designed to
    > include the true value in a stated proportion of repeated experiments.
    """)


@app.cell
def _(mo, np, pd, read_csv, stats):
    height_data_path = mo.notebook_location() / "public" / "height_data.csv"
    lecture_height_data = read_csv(height_data_path)
    assert set(lecture_height_data.columns) == {"gender", "height"}
    assert len(lecture_height_data) == 97
    assert lecture_height_data["height"].notna().all()

    def make_height_population(seed=824):
        rng = np.random.default_rng(seed)
        population_size = 20_000
        groups = rng.choice(
            np.array(["female", "male"]),
            size=population_size,
            p=[0.55, 0.45],
        )
        heights = np.where(
            groups == "female",
            rng.normal(163.0, 7.3, population_size),
            rng.normal(178.0, 7.7, population_size),
        )
        return pd.DataFrame({"group": groups, "height": heights})

    def draw_height_sample(population, sample_size, sampling_scheme, rng):
        available = (
            population
            if sampling_scheme == "Representative random sample"
            else population.loc[population["group"] == "male"]
        )
        chosen = rng.choice(len(available), size=sample_size, replace=False)
        sample = available.iloc[chosen].reset_index(drop=True)
        return {
            "sample": sample,
            "sample_size": sample_size,
            "scheme": sampling_scheme,
            "estimate": float(sample["height"].mean()),
        }

    def calculate_sample_statistic(samples, statistic_name):
        if statistic_name == "Mean":
            return np.mean(samples, axis=1)
        if statistic_name == "Median":
            return np.median(samples, axis=1)
        if statistic_name == "Standard deviation":
            return np.std(samples, axis=1, ddof=1)
        lower, upper = np.percentile(samples, [25, 75], axis=1)
        return upper - lower

    def draw_clt_values(population_shape, size, rng):
        if population_shape == "Normal":
            return rng.normal(0.0, 1.0, size)
        if population_shape == "Right-skewed":
            return (rng.gamma(shape=2.0, scale=1.0, size=size) - 2.0) / np.sqrt(2)
        if population_shape == "Bounded uniform":
            return rng.uniform(-np.sqrt(3), np.sqrt(3), size)
        if population_shape == "Bimodal":
            component = rng.integers(0, 2, size=size)
            return rng.normal(np.where(component == 0, -2.0, 2.0), 0.65) / np.sqrt(
                4.0 + 0.65**2
            )
        return rng.standard_t(df=3, size=size) / np.sqrt(3)

    def simulate_clt(population_shape, sample_size, repetitions, statistic_name, rng):
        population = draw_clt_values(population_shape, 40_000, rng)
        one_sample = draw_clt_values(population_shape, sample_size, rng)
        samples = draw_clt_values(population_shape, (repetitions, sample_size), rng)
        statistics = calculate_sample_statistic(samples, statistic_name)
        return {
            "population": population,
            "one_sample": one_sample,
            "statistics": statistics,
            "population_shape": population_shape,
            "sample_size": sample_size,
            "repetitions": repetitions,
            "statistic_name": statistic_name,
        }

    def simulate_coverage(
        population_shape, sample_size, repetitions, confidence_level, method, rng
    ):
        samples = draw_clt_values(population_shape, (repetitions, sample_size), rng)
        means = samples.mean(axis=1)
        sample_sds = samples.std(axis=1, ddof=1)
        alpha = 1.0 - confidence_level
        if method == "Known SD (z)":
            population_sd = 1.0
            critical = stats.norm.ppf(1.0 - alpha / 2.0)
            margins = critical * population_sd / np.sqrt(sample_size)
        else:
            critical = stats.t.ppf(1.0 - alpha / 2.0, df=sample_size - 1)
            margins = critical * sample_sds / np.sqrt(sample_size)
        lowers = means - margins
        uppers = means + margins
        contains = (lowers <= 0.0) & (uppers >= 0.0)
        return {
            "means": means,
            "lowers": lowers,
            "uppers": uppers,
            "contains": contains,
            "population_shape": population_shape,
            "sample_size": sample_size,
            "repetitions": repetitions,
            "confidence_level": confidence_level,
            "method": method,
        }

    def bootstrap_height_sample(values, repetitions, statistic_name, rng):
        n_observations = len(values)
        one_indices = rng.integers(0, n_observations, size=n_observations)
        bootstrap_indices = rng.integers(
            0, n_observations, size=(repetitions, n_observations)
        )
        bootstrap_samples = values[bootstrap_indices]
        statistics = calculate_sample_statistic(bootstrap_samples, statistic_name)
        original_statistic = float(
            calculate_sample_statistic(values.reshape(1, -1), statistic_name)[0]
        )
        return {
            "one_indices": one_indices,
            "statistics": statistics,
            "original_statistic": original_statistic,
            "repetitions": repetitions,
            "statistic_name": statistic_name,
        }

    height_population = make_height_population()
    return (
        bootstrap_height_sample,
        draw_height_sample,
        height_population,
        lecture_height_data,
        simulate_clt,
        simulate_coverage,
    )


@app.cell
def _(mo):
    mo.md(r"""
    ---

    ## 1. From a population to an estimate

    A **population** is the full group we want to learn about. An **observation
    unit** is the entity being measured, such as a person or an independent culture.
    The population may be finite, such as all students enrolled in a course, or conceptual, such as all
    repetitions of a particular assay under the same conditions.

    A **sample** is the subset we actually measure. A **parameter** is a number
    describing the population, such as its mean enzyme activity. We estimate it
    from the sample. Three terms distinguish the target, the calculation, and its
    result: the **estimand** is the quantity we want to know; the **estimator** is
    the calculation we use, such as taking an average; and the **estimate** is the
    number obtained, such as 12 U/mL. Here, our target is the population mean.

    The explorer below uses a synthetic target population of 20,000 people. Each
    person is assigned to the female group with probability 0.55 or the male group
    with probability 0.45. Heights are then generated from the same normal models as
    in Chapter 1: mean 163.0 cm and SD 7.3 cm for the female group, and mean 178.0 cm
    and SD 7.7 cm for the male group. The population is generated once with a fixed
    seed, so it remains unchanged while you compare sampling designs.

    The option **Representative random sample** gives every person among the
    20,000 an equal chance of selection. An individual sample may still differ
    from the population by chance. The deliberately **biased sample** draws only from the male subgroup while its sample
    mean is still compared with the mean of the complete mixed population. It
    therefore illustrates systematic exclusion of one subgroup, not a claim that
    male participants are intrinsically biased observations.

    **Sampling variability** means chance differences between samples.
    **Sampling bias** means a systematic tendency to miss the target. Repeating a representative
    design produces varying estimates around the target; systematically excluding a
    subgroup can move the estimates away from the target.
    """)


@app.cell
def _(draw_height_sample, height_population, mo, np):
    initial_height_sample = draw_height_sample(
        height_population,
        sample_size=30,
        sampling_scheme="Representative random sample",
        rng=np.random.default_rng(82401),
    )
    get_height_sample, set_height_sample = mo.state(initial_height_sample)
    return get_height_sample, set_height_sample


@app.cell
def _(mo):
    mo.md(r"""
    **Try this:** use the representative design with 30 observations and click
    **Draw again** several times. Compare the sample mean with the population
    reference: individual draws vary on either side. Then choose **Biased: male
    subgroup only**, draw again, then increase the sample size to 100 and draw
    again. Greater precision within that subgroup does not repair its mismatch
    with the target
    population. The design and sample-size controls take effect on the next click;
    the result caption describes the sample currently shown.
    """)


@app.cell
def _(draw_height_sample, height_population, mo, np, set_height_sample):
    sampling_scheme = mo.ui.radio(
        ["Representative random sample", "Biased: male subgroup only"],
        value="Representative random sample",
        label="Sampling design",
    )
    sampling_size = mo.ui.slider(
        5,
        100,
        step=5,
        value=30,
        show_value=True,
        full_width=True,
        label="Sample size",
    )

    def draw_new_height_sample(_value):
        set_height_sample(
            draw_height_sample(
                height_population,
                sample_size=int(sampling_size.value),
                sampling_scheme=sampling_scheme.value,
                rng=np.random.default_rng(),
            )
        )

    draw_height_button = mo.ui.button(
        label="Draw again",
        kind="success",
        full_width=True,
        on_click=draw_new_height_sample,
    )
    return draw_height_button, sampling_scheme, sampling_size


@app.cell
def _(
    COLORS,
    FIGURE_SIZE_STANDARD,
    draw_height_button,
    get_height_sample,
    height_population,
    mo,
    np,
    plt,
    sampling_scheme,
    sampling_size,
    two_column_panel,
):
    height_sample_result = get_height_sample()
    sampled_heights = height_sample_result["sample"]["height"].to_numpy()
    population_mean_height = float(height_population["height"].mean())
    sampling_bias = height_sample_result["estimate"] - population_mean_height

    height_figure, height_axis = plt.subplots(figsize=FIGURE_SIZE_STANDARD)
    height_axis.hist(
        height_population["height"],
        bins=np.arange(135, 205, 2),
        density=True,
        color=COLORS["gray"],
        alpha=0.28,
        label="Synthetic target population",
    )
    jitter = np.random.default_rng(120).uniform(0.001, 0.012, len(sampled_heights))
    height_axis.scatter(
        sampled_heights,
        jitter,
        color=COLORS["orange"],
        edgecolor="white",
        linewidth=0.4,
        s=32,
        label="Current sample",
        zorder=3,
    )
    height_axis.axvline(
        population_mean_height,
        color=COLORS["green"],
        linestyle="--",
        linewidth=2,
        label="Population mean",
    )
    height_axis.axvline(
        height_sample_result["estimate"],
        color=COLORS["blue"],
        linewidth=2,
        label="Sample estimate",
    )
    height_axis.set(xlim=(135, 205), xlabel="Height [cm]", ylabel="Density")
    height_axis.grid(axis="y", linestyle=":", alpha=0.35)
    height_axis.legend(frameon=False, fontsize=8)
    height_figure.tight_layout()

    sampling_controls = mo.vstack(
        [
            sampling_scheme,
            sampling_size,
            draw_height_button,
            mo.stat(
                f"{height_sample_result['estimate']:.1f} cm",
                label="Sample mean",
                bordered=True,
            ),
            mo.stat(
                f"{sampling_bias:+.1f} cm",
                label="Estimate − target",
                bordered=True,
            ),
        ]
    )
    sampling_interpretation = mo.callout(
        mo.md(
            rf"""
            The displayed sample used **{height_sample_result["scheme"].lower()}**
            with $n={height_sample_result["sample_size"]}$. A larger random sample
            usually reduces variability, but increasing $n$ does not repair a design
            that systematically excludes part of the target population.
            """
        ),
        kind="warn" if "Biased" in height_sample_result["scheme"] else "info",
    )
    two_column_panel(
        sampling_controls,
        mo.vstack([height_figure, sampling_interpretation]),
        widths=(1, 3),
    )


@app.cell
def _(mo):
    mo.md(r"""
    ### Replication and the target of inference

    Biological replicates are independently sampled biological units. Technical
    replicates repeat a measurement on the same unit and primarily quantify
    measurement variation. Repeated measurements retain a link to the same unit;
    treating them as independent biological replicates is **pseudoreplication**.

    For example, measuring enzyme activity three times in each of four independent
    cultures gives four biological replicates, not twelve. The repeat readings help
    assess measurement precision, but do not replace additional cultures.

    Before calculating anything, state the target population, the observation unit,
    and how units entered the sample. Precision formulas cannot rescue a mismatch
    between the sampling design and the scientific claim.
    """)


@app.cell
def _(mo):
    mo.md(r"""
    ---

    ## 2. Reading probabilities

    The **sample space**, written $\Omega$, is the set of all possible outcomes.
    An **event** is an outcome or group of outcomes of interest, such as a cell
    surviving treatment. Events are **mutually exclusive** if they cannot happen
    together. They are **independent** if knowing that one happened does not change
    the probability of the other. These are different ideas: survival and death
    of the same cell are mutually exclusive, but not independent.

    A **random variable** is a numerical outcome that can vary, such as the number
    of surviving cells. For **discrete** outcomes, such as whole-number counts, the
    **probability mass function (PMF)** gives the probability of each value:
    $P(X=x)$. The **cumulative distribution function (CDF)** adds up probabilities
    up to a chosen value: $F_X(x)=P(X\leq x)$. Thus $F_X(5)$ is the probability of
    observing five or fewer survivors.

    The **expected value** $E(X)$ is the average we would approach over many
    repetitions. The **variance** $\operatorname{Var}(X)$ measures spread using
    squared distances from that average. Its square root is the **standard
    deviation (SD)**, expressed in the original measurement units:

    $$E(X)=\sum_x xP(X=x), \qquad
      \operatorname{Var}(X)=\sum_x[x-E(X)]^2P(X=x).$$

    Here $\sum_x$ means “add over all possible values of $x$.” Each value contributes
    according to its probability.

    Two fair dice make these ideas easy to see. Each ordered pair is equally
    likely, but a sum of 7 can occur in more ways than a sum of 2. Orange bars show
    exact probabilities calculated from all possible pairs. Blue markers show the
    fraction of simulated rolls giving each sum. More rolls usually bring those
    fractions closer to the exact probabilities.
    """)


@app.cell
def _(mo):
    def valid_dice_interval(interval, number_of_sides):
        maximum_sum = 2 * int(number_of_sides)
        lower_bound, upper_bound = map(int, interval)
        if upper_bound < 1 or upper_bound > maximum_sum:
            upper_bound = maximum_sum
        if lower_bound < 1 or lower_bound >= maximum_sum:
            lower_bound = maximum_sum - 2
        if lower_bound > upper_bound:
            lower_bound = max(1, upper_bound - 2)
        return lower_bound, upper_bound

    assert valid_dice_interval((6, 9), 4) == (6, 8)
    assert valid_dice_interval((12, 12), 4) == (6, 8)

    get_dice_interval, set_dice_interval = mo.state((6, 9))
    return get_dice_interval, set_dice_interval, valid_dice_interval


@app.cell
def _(mo):
    mo.md(r"""
    **Try this:** with six-sided dice, select the interval $(6,9]$. It includes
    sums 7, 8, and 9, but excludes 6. Compare the highlighted probability masses
    with the CDF difference $F(9)-F(6)$: the probability of 9 or less minus the
    probability of 6 or less. Increase simulated rolls from 100 to
    10,000 and compare the observed fractions with the exact probabilities;
    simulation precision improves, while the theoretical model stays fixed.
    Then change the number of sides: this changes the possible sums and their
    probabilities. All controls update immediately, and the same random seed
    makes a return to the same settings reproducible.
    """)


@app.cell
def _(get_dice_interval, mo, set_dice_interval, valid_dice_interval):
    def synchronize_dice_interval(number_of_sides):
        set_dice_interval(valid_dice_interval(get_dice_interval(), number_of_sides))

    dice_sides = mo.ui.slider(
        4,
        12,
        value=6,
        show_value=True,
        full_width=True,
        label="Sides per die",
        on_change=synchronize_dice_interval,
    )
    dice_trials = mo.ui.slider(
        100,
        10_000,
        step=100,
        value=5_000,
        show_value=True,
        full_width=True,
        label="Simulated rolls (fixed comparison seed)",
    )
    return dice_sides, dice_trials


@app.cell
def _(
    dice_sides,
    get_dice_interval,
    mo,
    set_dice_interval,
    valid_dice_interval,
):
    maximum_dice_sum = 2 * int(dice_sides.value)
    displayed_dice_interval = valid_dice_interval(get_dice_interval(), dice_sides.value)

    def store_dice_interval(interval):
        set_dice_interval(tuple(map(int, interval)))

    dice_interval = mo.ui.range_slider(
        1,
        maximum_dice_sum,
        step=1,
        value=displayed_dice_interval,
        show_value=True,
        full_width=True,
        label="Interval (a, b]",
        on_change=store_dice_interval,
    )
    return (dice_interval,)


@app.cell
def _(
    COLORS,
    FIGURE_SIZE_LINKED,
    dice_interval,
    dice_sides,
    dice_trials,
    mo,
    np,
    plt,
    product,
    two_column_panel,
):
    sides = int(dice_sides.value)
    all_outcomes = np.array(
        [sum(outcome) for outcome in product(range(1, sides + 1), repeat=2)]
    )
    dice_values, dice_counts = np.unique(all_outcomes, return_counts=True)
    dice_pmf = dice_counts / dice_counts.sum()
    dice_cdf = np.cumsum(dice_pmf)
    extended_dice_cdf_x = np.concatenate(([1], dice_values, [dice_values[-1] + 1]))
    extended_dice_cdf_y = np.concatenate(([0.0], dice_cdf, [1.0]))
    interval_a, interval_b = map(int, dice_interval.value)
    exact_interval_probability = float(
        dice_pmf[(dice_values > interval_a) & (dice_values <= interval_b)].sum()
    )
    values_at_or_below_b = dice_values <= interval_b
    values_at_or_below_a = dice_values <= interval_a
    cdf_at_b = (
        float(dice_cdf[values_at_or_below_b][-1])
        if np.any(values_at_or_below_b)
        else 0.0
    )
    cdf_at_a = (
        float(dice_cdf[values_at_or_below_a][-1])
        if np.any(values_at_or_below_a)
        else 0.0
    )
    assert np.isclose(dice_pmf.sum(), 1.0)
    assert np.isclose(exact_interval_probability, cdf_at_b - cdf_at_a)

    dice_rng = np.random.default_rng(82402)
    simulated_sums = dice_rng.integers(
        1, sides + 1, size=(int(dice_trials.value), 2)
    ).sum(axis=1)
    simulated_dice_pmf = np.array(
        [(simulated_sums == value).mean() for value in dice_values]
    )

    dice_figure, dice_axes = plt.subplots(1, 2, figsize=FIGURE_SIZE_LINKED)
    dice_axes[0].bar(
        dice_values,
        dice_pmf,
        color=COLORS["orange"],
        edgecolor="white",
        label="Exact PMF",
    )
    dice_axes[0].plot(
        dice_values,
        simulated_dice_pmf,
        color=COLORS["blue"],
        marker="o",
        markersize=3,
        linewidth=1.3,
        label="Simulated\nfrequency",
    )
    dice_axes[0].set(
        xlabel="Sum of two dice",
        ylabel="Probability",
        xlim=(0.7, 2 * sides + 0.7),
    )
    dice_axes[0].grid(axis="y", linestyle=":", alpha=0.35)
    dice_axes[0].legend(frameon=False, fontsize=8)

    dice_axes[1].step(
        extended_dice_cdf_x,
        extended_dice_cdf_y,
        where="post",
        color=COLORS["purple"],
        linewidth=2,
        label="Exact CDF",
    )
    dice_axes[1].vlines(
        [interval_a, interval_b],
        [0, 0],
        [cdf_at_a, cdf_at_b],
        colors=[COLORS["gray"], COLORS["vermillion"]],
        linestyles=[":", "--"],
        linewidth=1.5,
    )
    dice_axes[1].scatter(
        [interval_a, interval_b],
        [cdf_at_a, cdf_at_b],
        color=[COLORS["gray"], COLORS["vermillion"]],
        zorder=3,
    )
    dice_axes[1].set(
        xlabel="Sum x",
        ylabel=r"$F_X(x)=P(X\leq x)$",
        xlim=(0.7, 2 * sides + 1.3),
        ylim=(-0.03, 1.05),
    )
    dice_axes[1].grid(axis="y", linestyle=":", alpha=0.35)
    dice_axes[1].legend(frameon=False, fontsize=8, loc="upper left")
    dice_figure.tight_layout()

    dice_controls = mo.vstack(
        [
            dice_sides,
            dice_trials,
            dice_interval,
            mo.stat(
                f"{exact_interval_probability:.3f}",
                label=f"P({interval_a} < X ≤ {interval_b})",
                bordered=True,
            ),
        ]
    )
    dice_explanation = mo.callout(
        mo.md(
            rf"""
            $F({interval_b})-F({interval_a})={cdf_at_b:.3f}-{cdf_at_a:.3f}
            ={exact_interval_probability:.3f}$. The blue simulation approaches the
            exact orange PMF as the number of rolls increases, but its small mismatch
            is random variation from using a finite number of simulated rolls.
            Simulation using random draws is called **Monte Carlo simulation**.
            """
        ),
        kind="info",
    )
    two_column_panel(
        dice_controls,
        mo.vstack([dice_figure, dice_explanation]),
        widths=(1, 3),
    )


@app.cell
def _(mo):
    mo.md(r"""
    ---

    ## 3. Discrete and continuous probability models

    ### Binomial counts

    The **binomial model** describes how many of a fixed number of trials have a
    particular outcome. For example, among $n$ cells, how many survive treatment?
    “Success” simply names the outcome being counted. If trials are independent
    and each has the same survival probability $p$, then

    $$P(X=k)=\binom{n}{k}p^k(1-p)^{n-k}, \qquad
      E(X)=np, \qquad \operatorname{Var}(X)=np(1-p).$$

    Here $k$ is the number surviving, and $\binom{n}{k}$ counts the ways to choose
    those $k$ survivors from $n$ cells.

    The assumptions matter: the number of trials is fixed, each outcome is binary
    (one of two possibilities, here surviving or not surviving),
    $p$ is constant, and trials are independent. Cells sharing a culture environment,
    for example, may respond together rather than independently.
    """)


@app.cell
def _(mo):
    mo.md(r"""
    **Try this:** keep $n=20$ and move survival probability from 0.1 through 0.5
    to 0.9. Follow the distribution's center and asymmetry. At $p=0.5$, increase
    $n$ and compare the mean and SD: the number surviving becomes more variable
    in absolute terms but more concentrated as a fraction of $n$. Each bar is
    the probability of an exact count, not an individual simulated experiment.
    """)


@app.cell
def _(mo):
    binomial_n = mo.ui.slider(
        1,
        100,
        value=20,
        show_value=True,
        full_width=True,
        label="Number of cells n",
    )
    binomial_p = mo.ui.slider(
        0.01,
        0.99,
        step=0.01,
        value=0.70,
        show_value=True,
        full_width=True,
        label="Survival probability p",
    )
    return binomial_n, binomial_p


@app.cell
def _(
    COLORS,
    FIGURE_SIZE_STANDARD,
    binomial_n,
    binomial_p,
    mo,
    np,
    plt,
    stats,
    two_column_panel,
):
    n_trials = int(binomial_n.value)
    success_probability = float(binomial_p.value)
    binomial_x = np.arange(n_trials + 1)
    binomial_probabilities = stats.binom.pmf(binomial_x, n_trials, success_probability)
    binomial_mean = n_trials * success_probability
    binomial_variance = n_trials * success_probability * (1 - success_probability)
    assert np.isclose(binomial_probabilities.sum(), 1.0)
    assert np.isclose(stats.binom.mean(n_trials, success_probability), binomial_mean)

    binomial_figure, binomial_axis = plt.subplots(figsize=FIGURE_SIZE_STANDARD)
    binomial_axis.bar(
        binomial_x,
        binomial_probabilities,
        color=COLORS["sky"],
        edgecolor="white",
        label="Binomial probability",
    )
    binomial_axis.axvline(
        binomial_mean,
        color=COLORS["green"],
        linestyle="--",
        linewidth=2,
        label=rf"Mean $np={binomial_mean:.1f}$",
    )
    binomial_axis.set(
        xlabel="Number of surviving cells k",
        ylabel="Probability",
        xlim=(-1, n_trials + 1),
        ylim=(0, min(1.02, max(0.22, 1.15 * binomial_probabilities.max()))),
    )
    binomial_axis.grid(axis="y", linestyle=":", alpha=0.35)
    binomial_axis.legend(frameon=False, fontsize=8)
    binomial_figure.tight_layout()

    binomial_controls = mo.vstack(
        [
            binomial_n,
            binomial_p,
            mo.stat(f"{binomial_mean:.2f}", label="Expected count np", bordered=True),
            mo.stat(
                f"{np.sqrt(binomial_variance):.2f}",
                label="Standard deviation",
                bordered=True,
            ),
        ]
    )
    two_column_panel(
        binomial_controls,
        mo.vstack(
            [
                binomial_figure,
                mo.callout(
                    mo.md(
                        "The bars describe the count of survivors, not the survival "
                        "probability of an individual cell. Independence and a common "
                        "probability are scientific assumptions, not consequences of "
                        "the formula."
                    ),
                    kind="warn",
                ),
            ]
        ),
    )


@app.cell
def _(mo):
    mo.md(r"""
    ### Poisson event rates

    A **Poisson model** can describe the number of events in a fixed time interval
    when events occur independently at a constant average rate. Its parameter
    $\lambda$ is the **expected count in that interval**, not the rate itself:
    $\lambda=\text{rate}\times\text{time}$. For example, 2 events per hour observed
    for 3 hours gives $\lambda=6$. The probability of exactly $k$ events is

    $$P(X=k)=e^{-\lambda}\frac{\lambda^k}{k!}, \qquad
      E(X)=\operatorname{Var}(X)=\lambda.$$

    Here $k!$ means $k\times(k-1)\times\cdots\times1$, with $0!=1$.
    A rate is not a probability and can exceed 1 event per hour. Doubling the observation interval doubles the
    expected count, provided the underlying event rate remains stable. A sample
    variance much larger than its mean signals **overdispersion** and suggests that
    rates differ among units or events are dependent.

    The simulation treats each observation interval as an independent repetition.
    For the selected rate and interval length, it calculates
    $\lambda=\text{rate}\times\text{time}$ and draws 5,000 counts from the resulting
    Poisson distribution. The tile compares the simulated mean and sample variance;
    both should be close to $\lambda$, although finite-simulation variation keeps them
    from being exactly equal. A fixed seed makes the comparison stable while you
    change the controls.
    """)


@app.cell
def _(mo):
    mo.md(r"""
    **Try this:** compare 2 events/hour observed for 2 hours with 4 events/hour
    observed for 1 hour. Both give the same expected count $\lambda=4$ and hence
    the same Poisson distribution. Now increase only the observation window:
    the expected count and variance increase together. Read the bars as counts
    within the selected window; the rate itself is expressed per hour.
    """)


@app.cell
def _(mo):
    poisson_rate = mo.ui.slider(
        0.5,
        8.0,
        step=0.5,
        value=2.0,
        show_value=True,
        full_width=True,
        label="Colony events per hour",
    )
    poisson_window = mo.ui.slider(
        0.5,
        4.0,
        step=0.5,
        value=2.0,
        show_value=True,
        full_width=True,
        label="Observation window [hours]",
    )
    return poisson_rate, poisson_window


@app.cell
def _(
    COLORS,
    FIGURE_SIZE_STANDARD,
    mo,
    np,
    plt,
    poisson_rate,
    poisson_window,
    stats,
    two_column_panel,
):
    event_rate = float(poisson_rate.value)
    observation_window = float(poisson_window.value)
    poisson_lambda = event_rate * observation_window
    poisson_x = np.arange(0, 80)
    poisson_probabilities = stats.poisson.pmf(poisson_x, poisson_lambda)
    assert np.isclose(poisson_probabilities.sum(), 1.0, atol=1e-8)
    assert np.isclose(stats.poisson.var(poisson_lambda), poisson_lambda)

    poisson_rng = np.random.default_rng(82403)
    poisson_counts = poisson_rng.poisson(poisson_lambda, size=5_000)
    realized_mean = float(poisson_counts.mean())
    realized_variance = float(poisson_counts.var(ddof=1))

    poisson_figure, poisson_axis = plt.subplots(figsize=FIGURE_SIZE_STANDARD)
    visible = poisson_probabilities > 1e-5
    poisson_axis.bar(
        poisson_x[visible],
        poisson_probabilities[visible],
        color=COLORS["purple"],
        edgecolor="white",
        label="Poisson PMF",
    )
    poisson_axis.axvline(
        poisson_lambda,
        color=COLORS["green"],
        linestyle="--",
        linewidth=2,
        label=rf"$\lambda={poisson_lambda:.1f}$",
    )
    poisson_axis.set(
        xlabel=f"Events in {observation_window:g} hours",
        ylabel="Probability",
        xlim=(-1, max(12, poisson_lambda + 5 * np.sqrt(poisson_lambda))),
        ylim=(0, min(1.02, max(0.22, 1.15 * poisson_probabilities.max()))),
    )
    poisson_axis.grid(axis="y", linestyle=":", alpha=0.35)
    poisson_axis.legend(frameon=False, fontsize=8)
    poisson_figure.tight_layout()

    poisson_controls = mo.vstack(
        [
            poisson_rate,
            poisson_window,
            mo.stat(f"{poisson_lambda:.1f}", label="Expected count λ", bordered=True),
            mo.stat(
                f"{realized_mean:.2f} / {realized_variance:.2f}",
                label="Simulated mean / variance",
                caption="5,000 fixed-seed intervals",
                bordered=True,
            ),
        ]
    )
    two_column_panel(
        poisson_controls,
        mo.vstack(
            [
                poisson_figure,
                mo.callout(
                    mo.md(
                        "The simulated mean and variance are close but not identical: "
                        "the equality is a property of the theoretical model, while "
                        "finite samples fluctuate."
                    ),
                    kind="info",
                ),
            ]
        ),
    )


@app.cell
def _(mo):
    mo.md(r"""
    ### Continuous measurements and the normal model

    A **continuous** model describes measurements such as height or concentration
    on a scale without gaps. Its **probability density** is a curve whose area over
    an interval gives the probability of a value in that interval. The height of
    the curve is not itself a probability. The total area is 1, even when a narrow
    curve has a peak above 1. Try an SD of 0.2 in the explorer to see this.

    In a continuous model, one exact value has probability zero because it has no
    width. Actual instruments record rounded values: a reading of 1.20 represents
    a small interval, not an infinitely precise point.

    The **normal distribution** is the familiar symmetric bell-shaped model.
    $X\sim N(\mu,\sigma^2)$ means that $X$ follows this model with mean $\mu$ and
    SD $\sigma$ (so $\sigma^2$ is its variance). **Standardizing** subtracts the
    mean and divides by the SD:

    $$Z=\frac{X-\mu}{\sigma}\sim N(0,1), \qquad
      P(a<X\leq b)=F_X(b)-F_X(a).$$

    Thus $Z=2$ means “two SDs above the mean.” The CDF difference gives the area
    between $a$ and $b$. Biological measurements need not follow a normal model;
    concentrations, for example, can have a long tail towards high values.
    """)


@app.cell
def _(mo):
    mo.md(r"""
    **Try this:** start with $\mu=0$, $\sigma=1$, and interval $(-1,1]$.
    Compare the shaded density area with the CDF difference. Keep the interval
    fixed and shift the mean to 2; then restore the mean and increase the SD
    to 2. Both changes can reduce the probability within this fixed interval
    for different reasons. Move both interval handles to select another region.
    The probability is an area, not the height of the density curve.
    Finally reduce the SD to 0.2: the density peak exceeds one, while its total
    area remains one.
    """)


@app.cell
def _(mo):
    normal_mean = mo.ui.slider(
        -2.0,
        2.0,
        step=0.1,
        value=0.0,
        show_value=True,
        full_width=True,
        label="Location μ",
    )
    normal_sd = mo.ui.slider(
        0.2,
        2.0,
        step=0.1,
        value=1.0,
        show_value=True,
        full_width=True,
        label="Scale σ",
    )
    normal_interval = mo.ui.range_slider(
        -6.0,
        6.0,
        step=0.1,
        value=[-1.0, 1.0],
        show_value=True,
        full_width=True,
        label="Interval (a, b]",
    )
    return normal_interval, normal_mean, normal_sd


@app.cell
def _(
    COLORS,
    FIGURE_SIZE_LINKED,
    mo,
    normal_interval,
    normal_mean,
    normal_sd,
    np,
    plt,
    stats,
    two_column_panel,
):
    selected_mean = float(normal_mean.value)
    selected_sd = float(normal_sd.value)
    normal_a, normal_b = map(float, normal_interval.value)
    normal_distribution = stats.norm(selected_mean, selected_sd)
    normal_x = np.linspace(-8, 8, 800)
    normal_pdf = normal_distribution.pdf(normal_x)
    normal_cdf = normal_distribution.cdf(normal_x)
    normal_probability = float(
        normal_distribution.cdf(normal_b) - normal_distribution.cdf(normal_a)
    )
    normal_z_a = (normal_a - selected_mean) / selected_sd
    normal_z_b = (normal_b - selected_mean) / selected_sd

    normal_figure, normal_axes = plt.subplots(1, 2, figsize=FIGURE_SIZE_LINKED)
    normal_axes[0].plot(normal_x, normal_pdf, color=COLORS["blue"], linewidth=2)
    shade = (normal_x > normal_a) & (normal_x <= normal_b)
    normal_axes[0].fill_between(
        normal_x[shade],
        normal_pdf[shade],
        color=COLORS["orange"],
        alpha=0.55,
        label="Interval area",
    )
    normal_axes[0].set(
        xlabel="Value x",
        ylabel="Probability density",
        xlim=(-8, 8),
        ylim=(0, max(0.9, normal_pdf.max() * 1.1)),
    )
    normal_axes[0].grid(axis="y", linestyle=":", alpha=0.35)
    normal_axes[0].legend(frameon=False, fontsize=8)

    normal_axes[1].plot(normal_x, normal_cdf, color=COLORS["purple"], linewidth=2)
    cdf_a = normal_distribution.cdf(normal_a)
    cdf_b = normal_distribution.cdf(normal_b)
    normal_axes[1].vlines(
        [normal_a, normal_b],
        [0, 0],
        [cdf_a, cdf_b],
        colors=[COLORS["gray"], COLORS["vermillion"]],
        linestyles=[":", "--"],
    )
    normal_axes[1].scatter(
        [normal_a, normal_b],
        [cdf_a, cdf_b],
        color=[COLORS["gray"], COLORS["vermillion"]],
        zorder=3,
    )
    normal_axes[1].set(
        xlabel="Value x",
        ylabel=r"$F_X(x)$",
        xlim=(-8, 8),
        ylim=(-0.03, 1.03),
    )
    normal_axes[1].grid(axis="y", linestyle=":", alpha=0.35)
    normal_figure.tight_layout()

    normal_controls = mo.vstack(
        [
            normal_mean,
            normal_sd,
            normal_interval,
            mo.stat(
                f"{normal_probability:.3f}",
                label=f"P({normal_a:g} < X ≤ {normal_b:g})",
                bordered=True,
            ),
            mo.stat(
                f"{normal_z_a:.2f} to {normal_z_b:.2f}",
                label="Standardized limits",
                bordered=True,
            ),
        ]
    )
    density_note = (
        "The density peak exceeds one, which is valid: probability is area, and the "
        "total area remains one."
        if normal_pdf.max() > 1
        else "The orange PDF area and the difference between the two CDF heights "
        "represent the same interval probability."
    )
    two_column_panel(
        normal_controls,
        mo.vstack([normal_figure, mo.callout(mo.md(density_note), kind="info")]),
    )


@app.cell
def _(mo):
    mo.md(r"""
    ---

    ## 4. Estimators, standard errors, and the Central Limit Theorem

    Imagine repeating a culture experiment many times, each time using a new set
    of independent cultures and calculating their mean enzyme activity. The means
    will differ. Their distribution is the **sampling distribution of the mean**.
    Its standard deviation is the **standard error (SE)** of the mean.

    **SD describes differences among individual observations; SE describes
    differences among estimates from repeated samples.** SE does not tell us how
    far this particular estimate is from the true value.

    Below, $n$ is the number of independent observations, $\bar X$ is their mean,
    and $\sigma$ is the population SD. Usually $\sigma$ is unknown, so we estimate
    the mean's SE using the sample SD $s$: $\widehat{\mathrm{SE}}=s/\sqrt n$.

    For independent observations with population variance $\sigma^2$,

    $$\operatorname{Var}(\bar X)
      =\operatorname{Var}\left(\frac{1}{n}\sum_{i=1}^n X_i\right)
      =\frac{1}{n^2}\sum_{i=1}^n\sigma^2
      =\frac{\sigma^2}{n},
      \qquad \operatorname{SE}(\bar X)=\frac{\sigma}{\sqrt n}.$$

    This $1/\sqrt n$ relationship assumes independent observations. Doubling the
    sample size does not halve the standard error; a fourfold increase does.
    """)


@app.cell
def _(mo):
    mo.md(r"""
    **Try this:** compare $n=2$, 4, and 30. Each histogram summarizes variance
    estimates from 5,000 samples, computed using either $n$ or $n-1$ in the
    denominator. Compare their average estimates with the true variance of 1.
    The correction matters most at small $n$. Switch to the right-skewed
    population (one with a long tail towards high values): an **unbiased** method
    is correct on average across repeated samples. This does not guarantee that
    every corrected estimate is close to 1. Settings update
    immediately using a fixed simulation seed.
    """)


@app.cell
def _(mo):
    variance_sample_size = mo.ui.slider(
        2,
        30,
        value=4,
        show_value=True,
        full_width=True,
        label="Sample size n",
    )
    variance_population_shape = mo.ui.dropdown(
        ["Normal", "Right-skewed"],
        value="Normal",
        full_width=True,
        label="Population shape",
    )
    return variance_population_shape, variance_sample_size


@app.cell
def _(
    COLORS,
    FIGURE_SIZE_LINKED,
    mo,
    np,
    plt,
    two_column_panel,
    variance_population_shape,
    variance_sample_size,
):
    variance_n = int(variance_sample_size.value)
    variance_rng = np.random.default_rng(82404)
    if variance_population_shape.value == "Normal":
        variance_samples = variance_rng.normal(size=(5_000, variance_n))
    else:
        variance_samples = (
            variance_rng.gamma(shape=2.0, scale=1.0, size=(5_000, variance_n)) - 2.0
        ) / np.sqrt(2)
    variance_with_n = variance_samples.var(axis=1, ddof=0)
    variance_with_n_minus_1 = variance_samples.var(axis=1, ddof=1)
    average_variance_n = float(variance_with_n.mean())
    average_variance_n_minus_1 = float(variance_with_n_minus_1.mean())

    variance_figure, variance_axes = plt.subplots(
        1, 2, figsize=FIGURE_SIZE_LINKED, sharex=True, sharey=True
    )
    upper_variance_limit = float(
        np.percentile(np.concatenate([variance_with_n, variance_with_n_minus_1]), 99.5)
    )
    variance_bins = np.linspace(0, upper_variance_limit, 40)
    variance_axes[0].hist(
        variance_with_n,
        bins=variance_bins,
        density=True,
        color=COLORS["sky"],
        edgecolor="white",
    )
    variance_axes[1].hist(
        variance_with_n_minus_1,
        bins=variance_bins,
        density=True,
        color=COLORS["orange"],
        edgecolor="white",
    )
    for variance_axis, mean_estimate, title in zip(
        variance_axes,
        [average_variance_n, average_variance_n_minus_1],
        ["Divide by n", "Divide by n − 1"],
    ):
        variance_axis.axvline(
            1.0,
            color=COLORS["green"],
            linestyle="--",
            linewidth=2,
            label="Population variance = 1",
        )
        variance_axis.axvline(
            mean_estimate,
            color=COLORS["blue"],
            linewidth=2,
            label=f"Mean estimate = {mean_estimate:.2f}",
        )
        variance_axis.set(title=title, xlabel="Estimated variance", ylabel="Density")
        variance_axis.grid(axis="y", linestyle=":", alpha=0.35)
        variance_axis.legend(frameon=False, fontsize=7)
    variance_figure.tight_layout()

    variance_panel = two_column_panel(
        mo.vstack(
            [
                variance_population_shape,
                variance_sample_size,
                mo.stat(
                    f"{average_variance_n:.3f}",
                    label="Average with n",
                    bordered=True,
                ),
                mo.stat(
                    f"{average_variance_n_minus_1:.3f}",
                    label="Average with n − 1",
                    bordered=True,
                ),
            ]
        ),
        mo.vstack(
            [
                variance_figure,
                mo.callout(
                    mo.md(
                        "Unbiased means correct on average over repeated samples; "
                        "it does not mean every estimate is close to the target."
                    ),
                    kind="info",
                ),
            ]
        ),
        widths=(1, 3),
    )
    variance_derivation = mo.vstack(
        [
            mo.md(r"""
            ### Derivation: why sample variance uses $n-1$

            Once the sample mean is estimated, the deviations satisfy
            $\sum_i(x_i-\bar x)=0$. If $n-1$ deviations are known, the final one is
            fixed by that constraint: only $n-1$ deviations are free to vary.

            Repeated sampling makes the practical consequence visible. Dividing the
            squared deviations by $n$ systematically underestimates population
            variance because the same data selected the center $\bar x$. Dividing by
            $n-1$ corrects that long-run bias.
            """),
            variance_panel,
        ]
    )
    mo.accordion({"Deep dive — bias and the n − 1 correction": variance_derivation})


@app.cell
def _(mo):
    mo.md(r"""
    ### Three distributions that must not be confused

    1. The **population distribution** describes individual values in the target.
    2. The **sample distribution** describes the observed values in one sample.
    3. The **sampling distribution** describes one statistic across repeated samples.

    A **statistic** is a number calculated from a sample, such as a mean or median.
    The **Central Limit Theorem (CLT)** concerns the third distribution: for
    independent observations drawn from the same population with finite, nonzero
    variance, the distribution of sample means becomes approximately bell-shaped
    as $n$ grows. Individual measurements need not become bell-shaped.
    There is no universal sample-size cutoff: strongly asymmetric populations or
    frequent extreme values can require much larger samples.

    In the controls, **right-skewed** means a long tail towards high values;
    **bounded uniform** means values are equally likely across a limited range;
    **bimodal** means two peaks; and **heavy-tailed** means extreme values occur
    more often than under a normal model.

    **Try this:** choose **Right-skewed**, **Mean**, and $n=2$, then click
    **Run a new simulation**. Read the three panels from left to right:
    population, one sample, then a statistic across repeated samples. Increase
    $n$ to 50 and run again. The means should be more tightly concentrated and
    more nearly normal even though the underlying population remains skewed.
    Next try **Median**: this changes which statistic is being collected, so the
    mean's standard-error formula no longer applies.

    All four controls describe the next run and need a button click. Increasing
    repetitions improves the simulation's precision; it does not increase $n$
    in an individual sample. Repeated clicks at unchanged settings produce new
    simulations under the same model.
    """)


@app.cell
def _(mo, np, simulate_clt):
    initial_clt_result = simulate_clt(
        "Right-skewed",
        sample_size=2,
        repetitions=5_000,
        statistic_name="Mean",
        rng=np.random.default_rng(82405),
    )
    get_clt_result, set_clt_result = mo.state(initial_clt_result)
    return get_clt_result, set_clt_result


@app.cell
def _(mo, np, set_clt_result, simulate_clt):
    clt_population_shape = mo.ui.dropdown(
        ["Normal", "Right-skewed", "Bounded uniform", "Bimodal", "Heavy-tailed t(3)"],
        value="Right-skewed",
        full_width=True,
        label="Population shape",
    )
    clt_sample_size = mo.ui.slider(
        2,
        100,
        value=2,
        show_value=True,
        full_width=True,
        label="Sample size n",
    )
    clt_repetitions = mo.ui.slider(
        200,
        10_000,
        step=200,
        value=5_000,
        show_value=True,
        full_width=True,
        label="Repetitions",
    )
    clt_statistic = mo.ui.dropdown(
        ["Mean", "Median", "Standard deviation", "IQR"],
        value="Mean",
        full_width=True,
        label="Statistic",
    )

    def run_clt_simulation(_value):
        set_clt_result(
            simulate_clt(
                clt_population_shape.value,
                int(clt_sample_size.value),
                int(clt_repetitions.value),
                clt_statistic.value,
                np.random.default_rng(),
            )
        )

    run_clt_button = mo.ui.button(
        label="Run a new simulation",
        kind="success",
        full_width=True,
        on_click=run_clt_simulation,
    )
    return (
        clt_population_shape,
        clt_repetitions,
        clt_sample_size,
        clt_statistic,
        run_clt_button,
    )


@app.cell
def _(
    COLORS,
    FIGURE_SIZE_LINKED,
    clt_population_shape,
    clt_repetitions,
    clt_sample_size,
    clt_statistic,
    get_clt_result,
    mo,
    np,
    plt,
    run_clt_button,
    stats,
    two_column_panel,
):
    clt_result = get_clt_result()
    clt_population = clt_result["population"]
    clt_one_sample = clt_result["one_sample"]
    clt_statistics = clt_result["statistics"]
    empirical_standard_error = float(np.std(clt_statistics, ddof=1))
    theoretical_mean_se = float(
        np.std(clt_population, ddof=1) / np.sqrt(clt_result["sample_size"])
    )
    assert np.isclose(
        1 / np.sqrt(4 * clt_result["sample_size"]),
        (1 / np.sqrt(clt_result["sample_size"])) / 2,
    )

    clt_figure, clt_axes = plt.subplots(1, 3, figsize=FIGURE_SIZE_LINKED)
    population_limits = np.percentile(clt_population, [0.5, 99.5])
    clt_axes[0].hist(
        clt_population,
        bins=40,
        range=tuple(population_limits),
        density=True,
        color=COLORS["gray"],
        edgecolor="white",
    )
    clt_axes[0].set(title="Population", xlabel="Individual value", ylabel="Density")

    sample_jitter = np.random.default_rng(405).uniform(-0.04, 0.04, len(clt_one_sample))
    clt_axes[1].scatter(
        clt_one_sample,
        sample_jitter,
        color=COLORS["orange"],
        edgecolor="white",
        linewidth=0.4,
        s=34,
    )
    clt_axes[1].axvline(
        np.mean(clt_one_sample),
        color=COLORS["blue"],
        linewidth=2,
        label="Sample\nmean",
    )
    clt_axes[1].set(
        title=f"One sample (n={clt_result['sample_size']})",
        xlabel="Observed value",
        yticks=[],
        ylim=(-0.12, 0.12),
    )
    clt_axes[1].legend(frameon=False, fontsize=7, loc="upper right")

    statistic_limits = np.percentile(clt_statistics, [0.5, 99.5])
    clt_axes[2].hist(
        clt_statistics,
        bins=40,
        range=tuple(statistic_limits),
        density=True,
        color=COLORS["sky"],
        edgecolor="white",
        label="Simulated\nstatistics",
    )
    if clt_result["statistic_name"] == "Mean":
        clt_normal_grid = np.linspace(statistic_limits[0], statistic_limits[1], 300)
        fitted_normal = stats.norm(
            np.mean(clt_statistics), np.std(clt_statistics, ddof=1)
        )
        clt_axes[2].plot(
            clt_normal_grid,
            fitted_normal.pdf(clt_normal_grid),
            color=COLORS["blue"],
            linewidth=2,
            label="Matched\nnormal\ndistribution",
        )
    clt_axes[2].set(
        title="Sampling distribution",
        xlabel=clt_result["statistic_name"],
        ylabel="Density",
    )
    clt_axes[2].legend(frameon=False, fontsize=7, loc="upper right")
    for clt_axis in clt_axes:
        clt_axis.grid(axis="y", linestyle=":", alpha=0.3)
    clt_figure.tight_layout()

    clt_controls = mo.vstack(
        [
            clt_population_shape,
            clt_sample_size,
            clt_repetitions,
            clt_statistic,
            run_clt_button,
            mo.stat(
                f"{empirical_standard_error:.3f}",
                label="Simulated standard error",
                bordered=True,
            ),
        ]
    )
    clt_note = (
        f"For the mean, the simulated SE is {empirical_standard_error:.3f}; "
        f"the population SD/√n prediction is {theoretical_mean_se:.3f}. The normal "
        "curve concerns the sampling distribution, not the raw population."
        if clt_result["statistic_name"] == "Mean"
        else "The statistic selector changes the sampling distribution. The simple "
        "σ/√n formula and the classical CLT statement here apply specifically to "
        "the sample mean."
    )
    two_column_panel(
        clt_controls,
        mo.vstack([clt_figure, mo.callout(mo.md(clt_note), kind="warn")]),
        widths=(1, 3),
    )


@app.cell
def _(mo):
    mo.md(r"""
    ### Planning for precision

    **Try this:** keep $\sigma=10$, compare $n=25$ with $n=100$, and watch
    SE fall from 2 to 1. The extra marker shows what four times the current $n$
    would achieve. Halve the target SE to see the required sample size quadruple
    (apart from integer rounding). All controls update immediately.

    Choose a target **SE** or a two-sided confidence interval's **margin of
    error** (half its width), in the same units as the measurements. With
    $m$ denoting that margin and $z$ a multiplier chosen for the confidence level
    ($z\approx1.96$ for 95%), the planning rules are

    $$n_{\rm SE}=\left\lceil(\sigma/\mathrm{SE}_{\rm target})^2\right\rceil,
    \qquad n_{\rm margin}=\left\lceil(z\sigma/m)^2\right\rceil.$$

    The brackets $\lceil\ \rceil$ mean “round up to a whole number.”

    These rules assume independent observations from a population with a fixed,
    finite SD. Supply a plausible population SD from prior information; an
    uncertain planning SD makes the required $n$ uncertain too. The margin rule
    is exact for independent normal observations with known population SD. For non-normal populations it uses
    a normal approximation; when SD must be estimated, a t interval has a random
    width and this z-based plan is only an approximation. This plans how precisely
    we estimate a mean. It does not calculate **statistical power** (see chapter 4), the chance of
    detecting a specified effect with a statistical test. More observations also
    cannot correct biased sampling or make related measurements independent.
    """)


@app.cell
def _(mo):
    precision_sd = mo.ui.slider(
        0.5,
        20,
        step=0.5,
        value=10,
        show_value=True,
        full_width=True,
        label="Planning population SD σ",
    )
    precision_n = mo.ui.slider(
        5,
        500,
        step=5,
        value=25,
        show_value=True,
        full_width=True,
        label="Current sample size n",
    )
    precision_mode = mo.ui.dropdown(
        ["Target SE", "Target margin of error"],
        value="Target SE",
        full_width=True,
        label="Precision target",
    )
    precision_target = mo.ui.slider(
        0.1,
        5,
        step=0.1,
        value=1,
        show_value=True,
        full_width=True,
        label="Target size (measurement units)",
    )
    precision_confidence = mo.ui.slider(
        0.80,
        0.99,
        step=0.01,
        value=0.95,
        show_value=True,
        full_width=True,
        label="Confidence (margin target only)",
    )
    return (
        precision_confidence,
        precision_mode,
        precision_n,
        precision_sd,
        precision_target,
    )


@app.cell
def _(
    np,
    precision_confidence,
    precision_mode,
    precision_n,
    precision_sd,
    precision_target,
    stats,
):
    planning_sd = float(precision_sd.value)
    planning_n = int(precision_n.value)
    planning_critical = (
        float(stats.norm.ppf((1 + precision_confidence.value) / 2))
        if precision_mode.value == "Target margin of error"
        else 1.0
    )
    planning_target_se = float(precision_target.value) / planning_critical
    planning_required_n = max(1, int(np.ceil((planning_sd / planning_target_se) ** 2)))
    planning_current_se = planning_sd / np.sqrt(planning_n)
    planning_achieved_se = planning_sd / np.sqrt(planning_required_n)
    return (
        planning_achieved_se,
        planning_critical,
        planning_current_se,
        planning_n,
        planning_required_n,
        planning_sd,
        planning_target_se,
    )


@app.cell
def _(
    COLORS,
    FIGURE_SIZE_SHALLOW,
    mo,
    np,
    plt,
    two_column_panel,
    precision_confidence,
    precision_mode,
    precision_n,
    precision_sd,
    precision_target,
    planning_achieved_se,
    planning_critical,
    planning_current_se,
    planning_n,
    planning_required_n,
    planning_sd,
    planning_target_se,
):
    import matplotlib.ticker as _ticker

    scaling_highlight_n = [planning_n, 4 * planning_n, planning_required_n]
    scaling_sample_sizes = np.geomspace(
        max(1, min(scaling_highlight_n) / 2), max(scaling_highlight_n) * 2, 400
    )
    scaling_figure, scaling_axis = plt.subplots(figsize=FIGURE_SIZE_SHALLOW)
    scaling_axis.plot(
        scaling_sample_sizes,
        planning_sd / np.sqrt(scaling_sample_sizes),
        color=COLORS["blue"],
        linewidth=2,
        label=r"$\sigma/\sqrt{n}$",
    )
    for scaling_size, scaling_color, scaling_marker, scaling_label in zip(
        scaling_highlight_n,
        [COLORS["blue"], COLORS["purple"], COLORS["vermillion"]],
        ["o", "s", "X"],
        [
            f"Current n = {planning_n:,}",
            f"Fourfold n = {4 * planning_n:,}",
            f"Required n = {planning_required_n:,}",
        ],
    ):
        scaling_axis.scatter(
            scaling_size,
            planning_sd / np.sqrt(scaling_size),
            color=scaling_color,
            marker=scaling_marker,
            s=65,
            zorder=3,
            label=scaling_label,
        )
    scaling_axis.axhline(
        planning_target_se,
        color=COLORS["vermillion"],
        linestyle=":",
        label=f"Target SE = {planning_target_se:.3f}",
    )
    scaling_axis.set(
        xscale="log",
        xlabel="Sample size n (log scale)",
        ylabel="Standard error of the mean",
        ylim=(0, None),
    )
    scaling_axis.xaxis.set_major_locator(
        _ticker.LogLocator(base=10, subs=(1, 2, 5), numticks=8)
    )
    scaling_axis.xaxis.set_major_formatter(_ticker.StrMethodFormatter("{x:,.0f}"))
    scaling_axis.xaxis.set_minor_formatter(_ticker.NullFormatter())
    scaling_axis.grid(linestyle=":", alpha=0.35)
    scaling_axis.legend(frameon=False, fontsize=8)
    scaling_figure.tight_layout()
    precision_status = (
        "The current sample size meets the target."
        if planning_n >= planning_required_n
        else f"The target requires {planning_required_n - planning_n:,} more independent observations."
    )
    precision_margin_note = (
        f"At {precision_confidence.value:.0%} confidence, the planned margin is "
        f"{planning_critical * planning_current_se:.3f} now and "
        f"{planning_critical * planning_achieved_se:.3f} at the required n. "
        if precision_mode.value == "Target margin of error"
        else ""
    )
    two_column_panel(
        mo.vstack(
            [
                precision_sd,
                precision_n,
                precision_mode,
                precision_target,
                precision_confidence,
                mo.stat(
                    f"{planning_required_n:,}",
                    label="Required integer n",
                    bordered=True,
                ),
            ]
        ),
        mo.vstack(
            [
                scaling_figure,
                mo.callout(
                    mo.md(
                        f"**Current SE: {planning_current_se:.3f}; at fourfold n: "
                        f"{planning_current_se / 2:.3f}.** The population SD stays "
                        f"{planning_sd:g}. At the required n, SE is {planning_achieved_se:.3f}. "
                        f"{precision_margin_note}{precision_status} "
                        "The horizontal axis uses a log scale to keep all three positions visible."
                    ),
                    kind="info",
                ),
            ]
        ),
    )


@app.cell
def _(mo):
    mo.md(r"""
    ---

    ## 5. Confidence intervals for a mean

    A **confidence interval** gives a range around an estimate to express sampling
    uncertainty. For a mean, we calculate **estimate ± multiplier × standard
    error**. The multiplier, also called a **critical value**, sets the confidence
    level. The next section explains what that level means across repeated studies.

    With independent normal observations and known population SD $\sigma$, use
    a multiplier from the standard normal distribution:

    $$\bar x\pm z_{1-\alpha/2}\frac{\sigma}{\sqrt n}.$$

    Here $\bar x$ is the observed sample mean. For 95% confidence, $\alpha=0.05$;
    the subscript $1-\alpha/2=0.975$ selects a cutoff with 2.5% of the normal
    distribution above it. This gives $z\approx1.96$.

    Usually we estimate the SD from the same sample. Using the sample SD $s$ adds
    uncertainty, so we use a larger multiplier from the **t-distribution**:

    $$\bar x\pm t_{n-1,\,1-\alpha/2}\frac{s}{\sqrt n}.$$

    Both formulas have the stated coverage for independent normal observations
    under their respective SD assumptions. For other populations, they are
    approximations that may be poor with small samples or strong asymmetry.

    At fixed sample size and SD, higher confidence requires a wider interval.
    At fixed SD and confidence level, larger independent samples narrow it.
    Changing only the observed mean moves the interval without changing its width.
    """)


@app.cell
def _(mo):
    mo.md(r"""
    ### Why the t-distribution has heavier tails

    When $\sigma$ is unknown, the standardized mean uses the sample standard
    deviation $S$:

    $$T=\frac{\bar X-\mu}{S/\sqrt n}.$$

    The numerator varies because the sample mean varies, and the denominator now
    varies as well. In some samples, $S$ underestimates $\sigma$, making the absolute
    t statistic unusually large. Under normal sampling, the t-distribution
    describes this extra variation.
    It has **heavier tails**: values far from zero are more likely than under
    the standard normal distribution.

    **Degrees of freedom** count how many deviations from the sample mean can
    vary freely. The $n$ deviations must sum to zero, so knowing $n-1$ of them
    determines the last. Thus $df=n-1$; no observation is discarded.
    Small $df$ means that $S$ is estimated imprecisely, so the tails are
    especially heavy. As $df$ increases, $S$ becomes more stable and the
    t-distribution approaches the standard normal distribution.
    """)


@app.cell
def _(mo):
    mo.md(r"""
    **Try this:** compare 1, 4, and 25 degrees of freedom. Look beyond the center
    of the curves to the tails and critical values. At low degrees of freedom,
    the t model allows more extreme standardized estimates; as degrees of freedom
    increase it approaches the normal reference. Moving the slider changes a
    theoretical distribution, not a newly drawn sample.
    """)


@app.cell
def _(mo):
    t_degrees_of_freedom = mo.ui.slider(
        1,
        25,
        value=4,
        show_value=True,
        full_width=True,
        label="Degrees of freedom",
    )
    return (t_degrees_of_freedom,)


@app.cell
def _(
    COLORS,
    FIGURE_SIZE_STANDARD,
    mo,
    np,
    plt,
    stats,
    t_degrees_of_freedom,
    two_column_panel,
):
    selected_degrees_of_freedom = int(t_degrees_of_freedom.value)
    t_comparison_x = np.linspace(-6, 6, 1_200)
    t_comparison_density = stats.t.pdf(t_comparison_x, df=selected_degrees_of_freedom)
    normal_comparison_density = stats.norm.pdf(t_comparison_x)
    normal_critical_95 = float(stats.norm.ppf(0.975))
    t_critical_95 = float(stats.t.ppf(0.975, df=selected_degrees_of_freedom))
    t_probability_beyond_normal_critical = float(
        2 * stats.t.sf(normal_critical_95, df=selected_degrees_of_freedom)
    )
    assert np.isclose(2 * stats.norm.sf(normal_critical_95), 0.05)
    assert t_critical_95 > normal_critical_95

    t_comparison_figure, t_comparison_axis = plt.subplots(figsize=FIGURE_SIZE_STANDARD)
    t_comparison_axis.plot(
        t_comparison_x,
        t_comparison_density,
        color=COLORS["purple"],
        linewidth=2.3,
        label=f"t-distribution (df = {selected_degrees_of_freedom})",
    )
    t_comparison_axis.plot(
        t_comparison_x,
        normal_comparison_density,
        color=COLORS["blue"],
        linestyle="--",
        linewidth=2,
        label="Standard normal distribution",
    )
    t_tail_mask = np.abs(t_comparison_x) >= normal_critical_95
    t_comparison_axis.fill_between(
        t_comparison_x,
        t_comparison_density,
        where=t_tail_mask,
        color=COLORS["purple"],
        alpha=0.18,
        label=r"t area beyond $\pm1.96$",
    )
    t_comparison_axis.axvline(
        -normal_critical_95,
        color=COLORS["gray"],
        linestyle=":",
        linewidth=1.3,
    )
    t_comparison_axis.axvline(
        normal_critical_95,
        color=COLORS["gray"],
        linestyle=":",
        linewidth=1.3,
    )
    t_comparison_axis.set(
        xlabel="Standardized value",
        ylabel="Probability density",
        xlim=(-6, 6),
        ylim=(0, 0.43),
    )
    t_comparison_axis.grid(axis="y", linestyle=":", alpha=0.35)
    t_comparison_axis.legend(frameon=False, fontsize=8)
    t_comparison_figure.tight_layout()

    t_comparison_controls = mo.vstack(
        [
            t_degrees_of_freedom,
            mo.stat(
                f"{t_critical_95:.3f}",
                label="Central 95% critical value t*",
                bordered=True,
            ),
            mo.stat(
                f"{100 * t_probability_beyond_normal_critical:.1f}%",
                label="P(|T| > 1.96)",
                bordered=True,
            ),
        ]
    )
    t_comparison_note = mo.callout(
        mo.md(
            rf"""
            With $df={selected_degrees_of_freedom}$, the t-distribution places
            {100 * t_probability_beyond_normal_critical:.1f}% of its probability
            beyond $\pm1.96$, compared with 5.0% for the standard normal. Its central
            95% critical value is $\pm{t_critical_95:.3f}$ rather than
            $\pm{normal_critical_95:.3f}$. This larger critical value produces wider
            confidence intervals when the sample SD must estimate $\sigma$.
            """
        ),
        kind="info",
    )
    two_column_panel(
        t_comparison_controls,
        mo.vstack([t_comparison_figure, t_comparison_note]),
        widths=(1, 3),
    )


@app.cell
def _(mo):
    mo.md(r"""
    **Try this:** hold the observed mean and SD fixed and compare $n=10$ with
    $n=40$. Both intervals narrow as the standard error decreases; the t interval
    also reflects the change in degrees of freedom. Next raise confidence from
    0.80 to 0.99 and see the intervals widen. Finally move only the mean: the
    intervals shift without changing width. For the z interval, the SD control
    is treated as a known population SD; for the t interval, it is treated as an
    estimated sample SD. These controls update the calculation immediately.
    """)


@app.cell
def _(mo):
    ci_mean = mo.ui.slider(
        165.0,
        180.0,
        step=0.1,
        value=171.8,
        show_value=True,
        full_width=True,
        label="Observed mean [cm]",
    )
    ci_sd = mo.ui.slider(
        5.0,
        20.0,
        step=0.1,
        value=11.4,
        show_value=True,
        full_width=True,
        label="SD: known σ (z) / estimated s (t) [cm]",
    )
    ci_sample_size = mo.ui.slider(
        2,
        200,
        value=10,
        show_value=True,
        full_width=True,
        label="Sample size n",
    )
    ci_confidence = mo.ui.slider(
        0.80,
        0.99,
        step=0.01,
        value=0.95,
        show_value=True,
        full_width=True,
        label="Confidence level",
    )
    return ci_confidence, ci_mean, ci_sample_size, ci_sd


@app.cell
def _(
    COLORS,
    FIGURE_SIZE_STANDARD,
    ci_confidence,
    ci_mean,
    ci_sample_size,
    ci_sd,
    mo,
    np,
    plt,
    stats,
    two_column_panel,
):
    observed_mean = float(ci_mean.value)
    observed_sd = float(ci_sd.value)
    interval_n = int(ci_sample_size.value)
    confidence_level = float(ci_confidence.value)
    interval_alpha = 1.0 - confidence_level
    standard_error = observed_sd / np.sqrt(interval_n)
    z_critical = float(stats.norm.ppf(1.0 - interval_alpha / 2.0))
    t_critical = float(stats.t.ppf(1.0 - interval_alpha / 2.0, df=interval_n - 1))
    z_margin = z_critical * standard_error
    t_margin = t_critical * standard_error
    z_interval = (observed_mean - z_margin, observed_mean + z_margin)
    t_interval = (observed_mean - t_margin, observed_mean + t_margin)
    assert np.isclose(observed_mean - z_interval[0], z_interval[1] - observed_mean)
    assert np.isclose(observed_mean - t_interval[0], t_interval[1] - observed_mean)
    ci_plot_half_width = max(27.5, 1.15 * t_margin)

    ci_figure, ci_axis = plt.subplots(figsize=FIGURE_SIZE_STANDARD)
    ci_axis.errorbar(
        observed_mean,
        1,
        xerr=z_margin,
        fmt="o",
        color=COLORS["blue"],
        capsize=5,
        linewidth=2,
        label="Known SD: z interval",
    )
    ci_axis.errorbar(
        observed_mean,
        0,
        xerr=t_margin,
        fmt="s",
        color=COLORS["vermillion"],
        capsize=5,
        linewidth=2,
        label="Unknown SD: t interval",
    )
    ci_axis.axvline(observed_mean, color=COLORS["gray"], linestyle=":", linewidth=1.5)
    ci_axis.set(
        xlabel="Population mean compatible with the interval [cm]",
        yticks=[0, 1],
        yticklabels=["t", "z"],
        ylim=(-0.7, 1.7),
        xlim=(
            observed_mean - ci_plot_half_width,
            observed_mean + ci_plot_half_width,
        ),
    )
    ci_axis.grid(axis="x", linestyle=":", alpha=0.35)
    ci_axis.legend(frameon=False, fontsize=8)
    ci_figure.tight_layout()

    ci_controls = mo.vstack(
        [
            ci_mean,
            ci_sd,
            ci_sample_size,
            ci_confidence,
            mo.stat(f"{standard_error:.2f} cm", label="Standard error", bordered=True),
            mo.stat(
                f"{t_interval[0]:.1f} to {t_interval[1]:.1f} cm",
                label="t interval",
                bordered=True,
            ),
        ]
    )
    ci_note = mo.callout(
        mo.md(
            f"""
            With {interval_n - 1} degrees of freedom, $t^*={t_critical:.3f}$ versus
            $z^*={z_critical:.3f}$. Estimating the SD therefore adds
            {t_margin - z_margin:.2f} cm to each side of the interval at these
            settings.
            """
        ),
        kind="info",
    )
    two_column_panel(ci_controls, mo.vstack([ci_figure, ci_note]))


@app.cell
def _(mo):
    known_sd_derivation = mo.md(r"""
    ### Derivation: from probability to an interval

    For independent normal observations with known population SD, start with the
    probability that the sample mean lies within $z^*$ standard errors of $\mu$:

    $$P\left(-z^*\leq\frac{\bar X-\mu}{\sigma/\sqrt n}\leq z^*\right)=1-\alpha.$$

    Multiply by the positive standard error and rearrange the inequality:

    $$P\left(\bar X-z^*\frac{\sigma}{\sqrt n}\leq\mu\leq
             \bar X+z^*\frac{\sigma}{\sqrt n}\right)=1-\alpha.$$

    Before sampling, the endpoints are random because $\bar X$ is random. After the
    data are observed, the interval is fixed: it either contains the fixed $\mu$ or
    it does not.
    """)
    mo.accordion({"Derivation — known population SD": known_sd_derivation})


@app.cell
def _(mo):
    mo.md(r"""
    ---

    ## 6. How often do confidence intervals include the true mean?

    Imagine repeating a study with new independent cultures each time and making
    a 95% confidence interval for mean enzyme activity. If the method's assumptions
    hold, about 95% of the intervals would include the true population mean over
    many repetitions. This proportion is called **coverage**. This repeated-study
    interpretation is what **frequentist** confidence means.

    In a real study we usually have just one interval and do not know whether it
    includes the true mean. The 95% describes the method's success rate; it does
    not mean that 95% of individual enzyme activities lie in the interval, nor
    does it assign a 95% probability to the true mean being in this particular
    interval. A probability statement about the unknown mean uses a different
    approach, Bayesian inference, introduced in Chapter 6.
    """)


@app.cell
def _(mo, np, simulate_coverage):
    initial_coverage_result = simulate_coverage(
        "Normal",
        sample_size=10,
        repetitions=5_000,
        confidence_level=0.95,
        method="Unknown SD (t)",
        rng=np.random.default_rng(82406),
    )
    get_coverage_result, set_coverage_result = mo.state(initial_coverage_result)
    return get_coverage_result, set_coverage_result


@app.cell
def _(mo):
    mo.md(r"""
    **Try this:** use a normal population, $n=10$, and 95% confidence, then click
    **Run a new simulation** several times. Each horizontal interval comes from
    a separate sample; the vertical reference marks the fixed true mean. Count
    a few misses, then compare the overall coverage with 95%. The displayed
    Monte Carlo uncertainty describes variation due to the finite number of
    simulated studies. Only a subset of intervals is drawn, but the coverage tile uses
    every repetition. Switch to a right-skewed population with $n=2$, run again,
    then increase $n$ to 50 to explore how model approximation affects coverage.
    All settings, including interval method, apply on the next button click.
    """)


@app.cell
def _(mo, np, set_coverage_result, simulate_coverage):
    coverage_population_shape = mo.ui.dropdown(
        ["Normal", "Right-skewed", "Bounded uniform", "Bimodal", "Heavy-tailed t(3)"],
        value="Normal",
        full_width=True,
        label="Population shape",
    )
    coverage_sample_size = mo.ui.slider(
        2,
        100,
        value=10,
        show_value=True,
        full_width=True,
        label="Sample size n",
    )
    coverage_confidence = mo.ui.slider(
        0.80,
        0.99,
        step=0.01,
        value=0.95,
        show_value=True,
        full_width=True,
        label="Nominal confidence",
    )
    coverage_repetitions = mo.ui.slider(
        200,
        10_000,
        step=200,
        value=5_000,
        show_value=True,
        full_width=True,
        label="Intervals generated",
    )
    coverage_method = mo.ui.radio(
        ["Unknown SD (t)", "Known SD (z)"],
        value="Unknown SD (t)",
        label="Interval method",
    )

    def run_coverage_simulation(_value):
        set_coverage_result(
            simulate_coverage(
                coverage_population_shape.value,
                int(coverage_sample_size.value),
                int(coverage_repetitions.value),
                float(coverage_confidence.value),
                coverage_method.value,
                np.random.default_rng(),
            )
        )

    run_coverage_button = mo.ui.button(
        label="Run a new simulation",
        kind="success",
        full_width=True,
        on_click=run_coverage_simulation,
    )
    return (
        coverage_confidence,
        coverage_method,
        coverage_population_shape,
        coverage_repetitions,
        coverage_sample_size,
        run_coverage_button,
    )


@app.cell
def _(
    COLORS,
    FIGURE_SIZE_STANDARD,
    coverage_confidence,
    coverage_method,
    coverage_population_shape,
    coverage_repetitions,
    coverage_sample_size,
    get_coverage_result,
    mo,
    np,
    plt,
    run_coverage_button,
    two_column_panel,
):
    coverage_result = get_coverage_result()
    coverage_rate = float(np.mean(coverage_result["contains"]))
    coverage_mcse = float(
        np.sqrt(coverage_rate * (1.0 - coverage_rate) / coverage_result["repetitions"])
    )
    assert 0.0 <= coverage_rate <= 1.0

    displayed_intervals = min(80, coverage_result["repetitions"])
    interval_order = np.arange(displayed_intervals)
    interval_colors = np.where(
        coverage_result["contains"][:displayed_intervals],
        COLORS["gray"],
        COLORS["vermillion"],
    )
    coverage_figure, coverage_axis = plt.subplots(figsize=FIGURE_SIZE_STANDARD)
    for interval_index in interval_order:
        coverage_axis.plot(
            [
                coverage_result["lowers"][interval_index],
                coverage_result["uppers"][interval_index],
            ],
            [interval_index, interval_index],
            color=interval_colors[interval_index],
            linewidth=1.5,
        )
        coverage_axis.scatter(
            coverage_result["means"][interval_index],
            interval_index,
            color=interval_colors[interval_index],
            s=8,
        )
    coverage_axis.axvline(
        0.0,
        color=COLORS["green"],
        linestyle="--",
        linewidth=2,
        label="True population mean",
    )
    coverage_axis.set(
        xlabel="Confidence interval for the mean",
        ylabel="Generated interval",
        ylim=(-2, displayed_intervals + 1),
        xlim=(-4, 4),
    )
    coverage_axis.grid(axis="x", linestyle=":", alpha=0.35)
    coverage_axis.legend(frameon=False, fontsize=8)
    coverage_figure.tight_layout()

    coverage_controls = mo.vstack(
        [
            coverage_population_shape,
            coverage_sample_size,
            coverage_confidence,
            coverage_repetitions,
            coverage_method,
            run_coverage_button,
            mo.stat(
                f"{100 * coverage_rate:.1f}%",
                label="Observed coverage",
                caption=f"Monte Carlo SE ≈ {100 * coverage_mcse:.2f} percentage points",
                bordered=True,
            ),
        ]
    )
    coverage_gap = coverage_rate - coverage_result["confidence_level"]
    coverage_note = mo.callout(
        mo.md(
            f"""
            The procedure targeted {100 * coverage_result["confidence_level"]:.0f}%
            and achieved {100 * coverage_rate:.1f}% in
            {coverage_result["repetitions"]:,} repetitions
            ({100 * coverage_gap:+.1f} percentage points). Red intervals miss the
            fixed mean. With small, strongly non-normal samples, requested and actual
            coverage can differ because the interval model is only approximate.
            """
        ),
        kind="warn",
    )
    two_column_panel(
        coverage_controls,
        mo.vstack([coverage_figure, coverage_note]),
        widths=(1, 3),
    )


@app.cell
def _(mo):
    mo.callout(
        mo.md(
            r"""
            **Correct interpretation:** “If we repeated the sampling procedure many
            times and constructed an interval in the same way, about 95% of those
            intervals would contain the true mean.” It is not correct to attach 95%
            probability to the fixed parameter after this interval has been observed.
            """
        ),
        kind="info",
    )


@app.cell
def _(mo):
    mo.md(r"""
    ---

    ## 7. Bootstrapping: resampling the observations we have

    We rarely have the resources to repeat an experiment thousands of times.
    **Bootstrapping** approximates variation between samples by repeatedly drawing
    from the observations we already have. Here we use the course's artificial
    height dataset as a stand-in for the unknown population.

    Each bootstrap sample contains $n$ draws from the original $n$ observations
    **with replacement**: after selecting a record, it remains available to be
    selected again. Some records appear several times and others are omitted.
    Calculate the chosen statistic for every resample. The SD of these results
    estimates its standard error.

    A **percentile interval** uses cutoffs in the resampled results. For a 95%
    interval, take the 2.5th and 97.5th percentiles, leaving 2.5% on each side.
    Written generally, for $B$ resampled statistics $T_1^*,\ldots,T_B^*$:

    $$\left[Q_{\alpha/2}(T^*),\;Q_{1-\alpha/2}(T^*)\right].$$

    $Q_p$ is the cutoff with fraction $p$ of the results at or below it, and
    $1-\alpha$ is the requested confidence level. Actual coverage is approximate;
    small or unrepresentative samples can give misleading intervals.

    This version assumes independent observations from the same population.
    Resampling individual assay readings from shared cultures would ignore their
    dependence; resampling must respect the independent biological units.
    Bootstrapping cannot recover parts of a population missing from the sample.
    It is useful when a direct SE formula is difficult, for example for a median
    or the **interquartile range (IQR)**, the width of the middle 50% of values.
    """)


@app.cell
def _(mo):
    mo.md(r"""
    **Try this:** select **Mean** and click **Draw and run bootstrap**. The table
    highlights observations omitted or repeated in one resample; each resample
    still contains the original number of rows. The histogram collects the
    statistic over all resamples. Change **Confidence level** from 95% to 99%:
    both the percentile interval and the mean's formula-based t interval widen, using
    the same observed sample and the same bootstrap draws. Confidence updates
    immediately; statistic and repetition settings apply only when you click
    the button. Run again at the same settings, then compare
    **Median** and **Standard deviation**, clicking after each choice. Their
    distributions answer different estimation questions. More repetitions make
    the bootstrap interval more stable numerically; they add no new biological
    observations. This activity always starts from the lecture height dataset,
    independently of the sampling-design activity above.
    """)


@app.cell
def _(bootstrap_height_sample, lecture_height_data, mo, np):
    original_height_values = lecture_height_data["height"].to_numpy(dtype=float)
    initial_bootstrap_result = bootstrap_height_sample(
        original_height_values,
        repetitions=5_000,
        statistic_name="Mean",
        rng=np.random.default_rng(82407),
    )
    get_bootstrap_result, set_bootstrap_result = mo.state(initial_bootstrap_result)
    return get_bootstrap_result, original_height_values, set_bootstrap_result


@app.cell
def _(
    bootstrap_height_sample,
    mo,
    np,
    original_height_values,
    set_bootstrap_result,
):
    bootstrap_statistic = mo.ui.dropdown(
        ["Mean", "Median", "Standard deviation", "IQR"],
        value="Mean",
        full_width=True,
        label="Statistic",
    )
    bootstrap_repetitions = mo.ui.slider(
        200,
        10_000,
        step=200,
        value=5_000,
        show_value=True,
        full_width=True,
        label="Bootstrap repetitions B",
    )

    def run_bootstrap(_value):
        set_bootstrap_result(
            bootstrap_height_sample(
                original_height_values,
                int(bootstrap_repetitions.value),
                bootstrap_statistic.value,
                np.random.default_rng(),
            )
        )

    run_bootstrap_button = mo.ui.button(
        label="Draw and run bootstrap",
        kind="success",
        full_width=True,
        on_click=run_bootstrap,
    )
    return bootstrap_repetitions, bootstrap_statistic, run_bootstrap_button


@app.cell
def _(mo):
    bootstrap_confidence = mo.ui.slider(
        0.80,
        0.99,
        step=0.01,
        value=0.95,
        show_value=True,
        full_width=True,
        label="Confidence level (updates immediately)",
    )
    return (bootstrap_confidence,)


@app.cell
def _(bootstrap_confidence, get_bootstrap_result, np, original_height_values, stats):
    bootstrap_result = get_bootstrap_result()
    bootstrap_statistics = bootstrap_result["statistics"]
    bootstrap_level = float(bootstrap_confidence.value)
    bootstrap_alpha = 1 - bootstrap_level
    bootstrap_lower, bootstrap_upper = np.quantile(
        bootstrap_statistics, [bootstrap_alpha / 2, 1 - bootstrap_alpha / 2]
    )
    bootstrap_t_interval = None
    if bootstrap_result["statistic_name"] == "Mean":
        bootstrap_t_margin = stats.t.ppf(
            1 - bootstrap_alpha / 2, df=len(original_height_values) - 1
        ) * stats.sem(original_height_values)
        bootstrap_t_interval = (
            float(np.mean(original_height_values) - bootstrap_t_margin),
            float(np.mean(original_height_values) + bootstrap_t_margin),
        )
    return (
        bootstrap_level,
        bootstrap_lower,
        bootstrap_result,
        bootstrap_statistics,
        bootstrap_t_interval,
        bootstrap_upper,
    )


@app.cell
def _(
    COLORS,
    FIGURE_SIZE_STANDARD,
    bootstrap_confidence,
    bootstrap_level,
    bootstrap_lower,
    bootstrap_repetitions,
    bootstrap_result,
    bootstrap_statistic,
    bootstrap_statistics,
    bootstrap_t_interval,
    bootstrap_upper,
    compact_table,
    mo,
    np,
    original_height_values,
    pd,
    plt,
    run_bootstrap_button,
    two_column_panel,
):
    bootstrap_standard_error = float(np.std(bootstrap_statistics, ddof=1))
    bootstrap_counts = np.bincount(
        bootstrap_result["one_indices"], minlength=len(original_height_values)
    )
    assert bootstrap_counts.sum() == len(original_height_values)
    omitted_count = int(np.sum(bootstrap_counts == 0))
    duplicated_count = int(np.sum(bootstrap_counts > 1))

    multiplicity_data = pd.DataFrame(
        {
            "Observation": np.arange(1, len(original_height_values) + 1),
            "Height [cm]": original_height_values,
            "Times drawn": bootstrap_counts,
        }
    )
    notable_multiplicities = pd.concat(
        [
            multiplicity_data.loc[multiplicity_data["Times drawn"] == 0].head(5),
            multiplicity_data.loc[multiplicity_data["Times drawn"] > 1]
            .sort_values("Times drawn", ascending=False)
            .head(7),
        ],
        ignore_index=True,
    )
    multiplicity_table = compact_table(
        notable_multiplicities,
        column_widths={"Observation": 90, "Height [cm]": 100, "Times drawn": 90},
        format_mapping={"Height [cm]": "{:.1f}"},
    )

    statistic_unit = "cm"
    bootstrap_figure, bootstrap_axis = plt.subplots(figsize=FIGURE_SIZE_STANDARD)
    bootstrap_axis.hist(
        bootstrap_statistics,
        bins=38,
        density=True,
        color=COLORS["sky"],
        edgecolor="white",
        label=f"{bootstrap_result['repetitions']:,} bootstrap statistics",
    )
    bootstrap_axis.axvline(
        bootstrap_result["original_statistic"],
        color=COLORS["green"],
        linestyle="--",
        linewidth=2,
        label="Original-sample statistic",
    )
    bootstrap_axis.axvspan(
        bootstrap_lower,
        bootstrap_upper,
        color=COLORS["orange"],
        alpha=0.2,
        label=f"{bootstrap_level:.0%} percentile interval",
    )
    if bootstrap_t_interval is not None:
        bootstrap_axis.axvline(
            bootstrap_t_interval[0],
            color=COLORS["purple"],
            linewidth=1.5,
            label=f"{bootstrap_level:.0%} formula-based t interval limits",
        )
        bootstrap_axis.axvline(
            bootstrap_t_interval[1],
            color=COLORS["purple"],
            linewidth=1.5,
        )
    bootstrap_axis.set(
        xlabel=f"Bootstrap {bootstrap_result['statistic_name'].lower()} [{statistic_unit}]",
        ylabel="Density",
    )
    bootstrap_axis.grid(axis="y", linestyle=":", alpha=0.35)
    bootstrap_axis.legend(frameon=False, fontsize=8)
    bootstrap_figure.tight_layout()

    bootstrap_controls = mo.vstack(
        [
            bootstrap_statistic,
            bootstrap_repetitions,
            run_bootstrap_button,
            bootstrap_confidence,
            mo.stat(
                f"{omitted_count} / {duplicated_count}",
                label="Omitted / repeated observations",
                caption="One displayed bootstrap sample",
                bordered=True,
            ),
            multiplicity_table,
        ]
    )
    bootstrap_note = mo.callout(
        mo.md(
            f"""
            The original {bootstrap_result["statistic_name"].lower()} is
            {bootstrap_result["original_statistic"]:.2f} {statistic_unit}. The
            bootstrap SE is {bootstrap_standard_error:.2f} {statistic_unit}, and the
            {bootstrap_level:.0%} percentile interval is {bootstrap_lower:.2f} to {bootstrap_upper:.2f}
            {statistic_unit}. Results come from {bootstrap_result["repetitions"]:,}
            resamples and vary slightly when rerun.
            """
        ),
        kind="info",
    )
    bootstrap_comparison = mo.md(
        "The formula-based t interval is shown only for the mean; it does not estimate "
        "uncertainty in the median, SD, or IQR."
    )
    if bootstrap_t_interval is not None:
        bootstrap_comparison = mo.md(
            f"**{bootstrap_level:.0%} formula-based t interval for the mean: "
            f"{bootstrap_t_interval[0]:.2f} to {bootstrap_t_interval[1]:.2f} cm.** "
            "Both intervals use the same original sample. The t interval uses "
            "the sample mean ± a t critical value × s/√n; it is exact for "
            "independent normal observations and approximate otherwise. The "
            "percentile interval uses percentile cutoffs from the resampled results and can "
            "be asymmetric. They often agree when the mean's sampling distribution "
            "is nearly normal; skewness, bias, small samples, and finite bootstrap "
            "repetitions can cause differences. Neither method is universally "
            "better, and both rely on an appropriate independent sampling design."
        )
    bootstrap_pending = (
        bootstrap_statistic.value != bootstrap_result["statistic_name"]
        or int(bootstrap_repetitions.value) != bootstrap_result["repetitions"]
    )
    bootstrap_run_note = mo.md(
        "**Settings changed:** click **Draw and run bootstrap** to apply the "
        "statistic and repetition settings. The display still describes the last run."
        if bootstrap_pending
        else "Confidence changes reuse these draws; click **Draw and run bootstrap** "
        "for new resamples from the same observed data."
    )
    two_column_panel(
        bootstrap_controls,
        mo.vstack(
            [bootstrap_figure, bootstrap_note, bootstrap_comparison, bootstrap_run_note]
        ),
        widths=(1.35, 2.65),
    )


@app.cell
def _(mo):
    mo.md(r"""
    ---

    ## 8. Propagating uncertainty

    A calculated result inherits uncertainty from its inputs. For example,
    uncertainty in a measured concentration carries through to a calculated amount
    of substance. This is called **uncertainty propagation** (or error propagation);
    “error” here does not mean a mistake in the arithmetic.

    If $z=f(x,y)$, the symbols $s_x$ and $s_y$ below describe the input uncertainties
    as standard deviations (or standard errors when the inputs are estimates).
    The derivatives $\partial f/\partial x$ and $\partial f/\partial y$ measure how
    strongly the result responds to a small change in each input. Using these
    local slopes is called a **first-order Taylor approximation**:

    $$s_z^2\approx
      \left(\frac{\partial f}{\partial x}\right)^2s_x^2+
      \left(\frac{\partial f}{\partial y}\right)^2s_y^2+
      2\frac{\partial f}{\partial x}\frac{\partial f}{\partial y}
      \operatorname{Cov}(x,y).$$

    **Covariance** describes whether the inputs tend to deviate together. Positive
    covariance means both tend to be high or low together; negative covariance
    means one tends to be high when the other is low. For example, a shared
    calibration error can move two measurements in the same direction.
    Independent inputs have zero covariance, but zero covariance alone does not
    prove independence. **Correlation** expresses covariance on a scale from
    $-1$ to $1$ by dividing it by the two input SDs.

    The approximation works best when the calculation is nearly a straight line
    over the likely input range. Strong curvature can make it inaccurate.
    """)


@app.cell
def _(mo):
    mo.md(r"""
    ### Worked examples: shifting, scaling, and squaring

    Begin with a simulated sampling distribution $X\sim N(10,1)$: its estimates have
    mean $\mu_X=10$ and standard error $s_X=1$. We apply a transformation $Z=f(X)$ to
    every possible estimate. The first-order Taylor approximation replaces the curve
    by a straight line with the same slope at $\mu_X$ and predicts

    $$\mu_Z\approx f(\mu_X),\qquad
      s_Z\approx \left|f'(\mu_X)\right|s_X.$$

    Use the dropdown to select the transformation. The same
    20,000 fixed-seed draws are used for every choice, so differences come only from
    the transformation.

    **Try this:** compare adding 10 with multiplying by 2. The first shifts the
    center without changing spread; the second doubles the standard error and
    quadruples the variance. Choose **Scale and shift** to combine these effects.
    Finally choose **Square** and compare the transformed histogram, Taylor
    approximation, and Monte Carlo summaries: curvature makes a local linear
    approximation imperfect. Read the axes afresh because transformations change
    the output scale.
    """)


@app.cell
def _(mo):
    simple_transformation = mo.ui.dropdown(
        [
            "Add a constant: Z = X + 10",
            "Multiply by a constant: Z = 2X",
            "Scale and shift: Z = 2X + 10",
            "Square: Z = X²",
        ],
        value="Add a constant: Z = X + 10",
        full_width=True,
        label="Transformation",
    )
    return (simple_transformation,)


@app.cell
def _(
    COLORS,
    FIGURE_SIZE_LINKED,
    compact_table,
    mo,
    np,
    pd,
    plt,
    simple_transformation,
    stats,
    two_column_panel,
):
    simple_input_mean = 10.0
    simple_input_se = 1.0
    simple_draws = np.random.default_rng(18).normal(
        simple_input_mean, simple_input_se, 20_000
    )
    simple_draw_se = np.std(simple_draws, ddof=0)
    assert np.isclose(np.std(simple_draws + 10.0, ddof=0), simple_draw_se)
    assert np.isclose(np.std(2.0 * simple_draws, ddof=0), 2.0 * simple_draw_se)

    if simple_transformation.value == "Add a constant: Z = X + 10":
        transformed_draws = simple_draws + 10.0
        taylor_mean = simple_input_mean + 10.0
        taylor_se = simple_input_se
        combined_xlim = (5.0, 25.0)
        transformed_xlim = (16.0, 24.0)
        transformation_formula = r"$f(x)=x+10$"
        simple_note_kind = "info"
        taylor_explanation = r"""
        Here $f'(x)=1$, so Taylor propagation predicts
        $s_Z\approx 1\cdot s_X=1$. Adding the same constant to every estimate shifts the
        center from 10 to 20 but leaves every deviation from the center unchanged.
        The prediction is exact because the transformation is linear.
        """
    elif simple_transformation.value == "Multiply by a constant: Z = 2X":
        transformed_draws = 2.0 * simple_draws
        taylor_mean = 2.0 * simple_input_mean
        taylor_se = 2.0 * simple_input_se
        combined_xlim = (5.0, 30.0)
        transformed_xlim = (12.0, 28.0)
        transformation_formula = r"$f(x)=2x$"
        simple_note_kind = "info"
        taylor_explanation = r"""
        Here $f'(x)=2$, so Taylor propagation predicts
        $s_Z\approx 2\cdot s_X=2$. Multiplication doubles every deviation from the mean;
        therefore the standard error doubles and the variance is multiplied by four.
        This prediction is also exact for a linear transformation.
        """
    elif simple_transformation.value == "Scale and shift: Z = 2X + 10":
        transformed_draws = 2.0 * simple_draws + 10.0
        taylor_mean = 2.0 * simple_input_mean + 10.0
        taylor_se = 2.0 * simple_input_se
        combined_xlim = (5.0, 40.0)
        transformed_xlim = (22.0, 38.0)
        transformation_formula = r"$f(x)=2x+10$"
        simple_note_kind = "info"
        taylor_explanation = r"""
        The derivative is still $f'(x)=2$. Taylor propagation therefore predicts
        $s_Z\approx 2\cdot s_X=2$: multiplication changes the width, while the subsequent
        addition changes only the center. Because the complete transformation is
        linear, the Taylor result is exact.
        """
    else:
        transformed_draws = simple_draws**2
        taylor_mean = simple_input_mean**2
        taylor_se = 2.0 * simple_input_mean * simple_input_se
        combined_xlim = (0.0, 200.0)
        transformed_xlim = (40.0, 170.0)
        transformation_formula = r"$f(x)=x^2$"
        simple_note_kind = "warn"
        taylor_explanation = r"""
        At the input mean, $f'(10)=20$, so the first-order prediction is
        $\mu_Z\approx100$ and $s_Z\approx20$. Squaring is nonlinear, and the simulated
        distribution is slightly shifted and right-skewed. A tangent-line
        approximation cannot reproduce that shift or asymmetry. The differences
        between the Taylor row and the Monte Carlo row, and between the normal curve
        and the histogram, show the resulting approximation error.
        """

    transformed_mc_mean = float(np.mean(transformed_draws))
    transformed_mc_se = float(np.std(transformed_draws, ddof=1))
    simple_summary = pd.DataFrame(
        {
            "Method": ["Taylor approximation", "Monte Carlo"],
            "Mean": [taylor_mean, transformed_mc_mean],
            "SE / SD": [taylor_se, transformed_mc_se],
        }
    )
    simple_summary_table = compact_table(
        simple_summary,
        column_widths={"Method": 150, "Mean": 90, "SE / SD": 90},
        format_mapping={"Mean": "{:.2f}", "SE / SD": "{:.2f}"},
    )

    simple_figure, simple_axes = plt.subplots(1, 2, figsize=FIGURE_SIZE_LINKED)
    simple_axes[0].hist(
        simple_draws,
        bins=50,
        range=combined_xlim,
        density=True,
        color=COLORS["green"],
        alpha=0.6,
        edgecolor="white",
        label="Original X (SD 1)",
    )
    simple_axes[0].hist(
        transformed_draws,
        bins=50,
        range=combined_xlim,
        density=True,
        color=COLORS["purple"],
        alpha=0.6,
        edgecolor="white",
        label=f"Transformed Z (SD {transformed_mc_se:.2f})",
    )
    simple_axes[0].set(
        title="Original and transformed on one scale",
        xlabel="Estimate",
        ylabel="Density",
        xlim=combined_xlim,
    )

    transformed_grid = np.linspace(*transformed_xlim, 400)
    simple_axes[1].hist(
        transformed_draws,
        bins=50,
        range=transformed_xlim,
        density=True,
        color=COLORS["purple"],
        alpha=0.6,
        edgecolor="white",
        label="Transformed\nestimates",
    )
    simple_axes[1].plot(
        transformed_grid,
        stats.norm.pdf(transformed_grid, taylor_mean, taylor_se),
        color=COLORS["blue"],
        linestyle="solid",
        linewidth=2.2,
        label="Taylor normal\napproximation",
    )
    simple_axes[1].set(
        title=f"Transformed: {transformation_formula}",
        xlabel="Transformed estimate z",
        ylabel="Density",
        xlim=transformed_xlim,
    )
    for simple_axis in simple_axes:
        simple_axis.grid(axis="y", linestyle=":", alpha=0.3)
        simple_axis.legend(frameon=False, fontsize=7, loc="upper right")
    simple_figure.tight_layout()

    simple_controls = mo.vstack(
        [
            simple_transformation,
            mo.stat(
                "Mean 10; SE 1",
                label="Original sampling distribution",
                bordered=True,
            ),
        ]
    )
    simple_note = mo.callout(mo.md(taylor_explanation), kind=simple_note_kind)
    two_column_panel(
        simple_controls,
        mo.vstack([simple_figure, simple_summary_table, simple_note]),
        widths=(1, 3),
    )


@app.cell
def _(mo):
    mo.md(r"""
    ### Worked example: isothermal titration calorimetry

    **Isothermal titration calorimetry (ITC)** measures heat released or absorbed
    during binding at a fixed temperature. Here $\Delta H$ is the enthalpy change,
    $K_a$ the association constant, and $K_d$ its reciprocal, the dissociation
    constant. $\Delta G$ is the Gibbs free-energy change, $\Delta S$ the entropy
    change, $T$ the absolute temperature, and $R$ the gas constant.

    At $T=278.15$ K, the example uses $\Delta H=52.70\pm0.14$ kJ/mol and
    $K_a=851000\pm19500$ L/mol. For this simulation, we treat the values after
    $\pm$ as one standard error, not as confidence-interval limits. We calculate

    $$K_d=K_a^{-1},\qquad \Delta G=-RT\ln K_a,\qquad
      T\Delta S=\Delta H-\Delta G.$$

    The Monte Carlo (MC) simulation repeats a virtual ITC measurement 20,000 times.
    For each repetition, the simulation draws a correlated pair of plausible values
    for $\Delta H$ and $K_a$, then recalculates $K_d$, $\Delta G$, and $T\Delta S$.
    The resulting histograms show the distributions of those derived quantities, and
    their standard deviations are reported as **MC SD**. More precisely, the simulation
    starts from two correlated normal random values: $\Delta H$ is normal, whereas
    $K_a$ is **lognormal**, meaning its logarithm is normal. This keeps $K_a$
    positive. The correlation control sets the correlation between $\Delta H$
    and $\ln K_a$; the resulting correlation with $K_a$ itself can differ slightly.

    To make the effect of correlation visible, the displayed **Taylor SE** uses the
    independent-input rule and deliberately omits the covariance term. It therefore
    stays fixed when the correlation slider changes, while the MC SD includes the
    selected correlation.

    **Try this:** keep the uncertainty multiplier at 1 and compare correlations
    of −0.9, 0, and 0.9. Watch the uncertainty of $T\Delta S$, which depends on
    both inputs. The uncertainties of $K_d$ and $\Delta G$ considered separately should remain
    similar apart from simulation variation because they depend only on $K_a$.
    Restore correlation to 0 and increase the multiplier to 10 to amplify
    nonlinearity. Compare each MC SD with the independent-input Taylor SE; any
    discrepancy may reflect correlation, curvature, or both. The calculation
    updates immediately with fixed random numbers for repeatable comparisons.
    """)


@app.cell
def _(mo):
    itc_uncertainty_multiplier = mo.ui.slider(
        0.5,
        10.0,
        step=0.5,
        value=1.0,
        show_value=True,
        full_width=True,
        label="Uncertainty multiplier",
    )
    itc_correlation = mo.ui.slider(
        -0.9,
        0.9,
        step=0.1,
        value=0.0,
        show_value=True,
        full_width=True,
        label="Correlation between ΔH and ln Ka",
    )
    return itc_correlation, itc_uncertainty_multiplier


@app.cell
def _(itc_correlation, np):
    gas_constant = 8.314
    temperature = 278.15
    enthalpy = 52.70
    enthalpy_se_orig = 0.14
    association_constant = 851_000.0
    association_se_orig = 19_500.0

    input_correlation = float(itc_correlation.value)
    itc_rng = np.random.default_rng(82408)
    correlation_matrix = np.array([[1.0, input_correlation], [input_correlation, 1.0]])
    latent_errors = itc_rng.multivariate_normal(
        [0.0, 0.0], correlation_matrix, size=20_000
    )
    return (
        association_constant,
        association_se_orig,
        enthalpy,
        enthalpy_se_orig,
        gas_constant,
        input_correlation,
        latent_errors,
        temperature,
    )


@app.cell
def _(
    COLORS,
    FIGURE_SIZE_LINKED,
    association_constant,
    association_se_orig,
    compact_table,
    enthalpy,
    enthalpy_se_orig,
    gas_constant,
    input_correlation,
    itc_correlation,
    itc_uncertainty_multiplier,
    latent_errors,
    mo,
    np,
    pd,
    plt,
    stats,
    temperature,
    two_column_panel,
):
    multiplier = float(itc_uncertainty_multiplier.value)
    enthalpy_se = enthalpy_se_orig * multiplier
    association_se = association_se_orig * multiplier

    kd_estimate = 1e6 / association_constant
    kd_analytic_se = 1e6 * association_se / association_constant**2
    delta_g_estimate = (
        -gas_constant * temperature * np.log(association_constant) / 1_000
    )
    delta_g_analytic_se = (
        gas_constant * temperature * association_se / association_constant / 1_000
    )
    t_delta_s_estimate = enthalpy - delta_g_estimate

    t_delta_s_analytic_variance = enthalpy_se**2 + delta_g_analytic_se**2
    t_delta_s_analytic_se = np.sqrt(t_delta_s_analytic_variance)

    enthalpy_simulated = enthalpy + enthalpy_se * latent_errors[:, 0]
    relative_ka_variance = (association_se / association_constant) ** 2
    log_ka_sd = np.sqrt(np.log1p(relative_ka_variance))
    log_ka_mean = np.log(association_constant) - 0.5 * log_ka_sd**2
    association_simulated = np.exp(log_ka_mean + log_ka_sd * latent_errors[:, 1])
    kd_simulated = 1e6 / association_simulated
    delta_g_simulated = (
        -gas_constant * temperature * np.log(association_simulated) / 1_000
    )
    t_delta_s_simulated = enthalpy_simulated - delta_g_simulated

    analytic_estimates = np.array([kd_estimate, delta_g_estimate, t_delta_s_estimate])
    analytic_ses = np.array(
        [kd_analytic_se, delta_g_analytic_se, t_delta_s_analytic_se]
    )
    simulation_arrays = [kd_simulated, delta_g_simulated, t_delta_s_simulated]
    simulation_means = np.array([values.mean() for values in simulation_arrays])
    simulation_sds = np.array([values.std(ddof=1) for values in simulation_arrays])
    relative_se_differences = np.abs(simulation_sds / analytic_ses - 1.0)

    if np.isclose(multiplier, 1.0) and np.isclose(input_correlation, 0.0):
        assert np.isclose(kd_estimate, 1.18, atol=0.01)
        assert np.isclose(kd_analytic_se, 0.03, atol=0.01)
        assert np.isclose(delta_g_estimate, -31.58, atol=0.05)
        assert np.isclose(delta_g_analytic_se, 0.05, atol=0.01)
        assert np.isclose(t_delta_s_estimate, 84.28, atol=0.05)
        assert np.isclose(t_delta_s_analytic_se, 0.15, atol=0.01)

    itc_summary = pd.DataFrame(
        {
            "Quantity": ["Kd", "ΔG", "TΔS"],
            "Unit": ["µM", "kJ/mol", "kJ/mol"],
            "Estimate": analytic_estimates,
            "Taylor SE (no cov.)": analytic_ses,
            "MC mean": simulation_means,
            "MC SD": simulation_sds,
        }
    )
    itc_table = compact_table(
        itc_summary,
        column_widths={
            "Quantity": 70,
            "Unit": 80,
            "Estimate": 90,
            "Taylor SE (no cov.)": 125,
            "MC mean": 90,
            "MC SD": 80,
        },
        format_mapping={
            "Estimate": "{:.3f}",
            "Taylor SE (no cov.)": "{:.3f}",
            "MC mean": "{:.3f}",
            "MC SD": "{:.3f}",
        },
    )

    itc_figure, itc_axes = plt.subplots(1, 3, figsize=FIGURE_SIZE_LINKED)
    quantity_labels = [r"$K_d$ [µM]", r"$\Delta G$ [kJ/mol]", r"$T\Delta S$ [kJ/mol]"]
    quantity_colors = [COLORS["sky"], COLORS["purple"], COLORS["orange"]]
    for itc_axis, values, estimate, analytic_se, label, color in zip(
        itc_axes,
        simulation_arrays,
        analytic_estimates,
        analytic_ses,
        quantity_labels,
        quantity_colors,
    ):
        lower_limit, upper_limit = np.percentile(values, [0.2, 99.8])
        itc_grid = np.linspace(lower_limit, upper_limit, 300)
        itc_axis.hist(
            values,
            bins=40,
            range=(lower_limit, upper_limit),
            density=True,
            color=color,
            alpha=0.6,
            edgecolor="white",
            label="MC",
        )
        itc_axis.plot(
            itc_grid,
            stats.norm.pdf(itc_grid, estimate, analytic_se),
            color=COLORS["blue"],
            linewidth=2,
            label="Approx.\n(no cov.)",
        )
        itc_axis.set(xlabel=label, ylabel="Density")
        itc_axis.grid(axis="y", linestyle=":", alpha=0.3)
        itc_axis.legend(frameon=False, fontsize=6, loc="upper right")
    itc_figure.tight_layout()

    itc_controls = mo.vstack(
        [
            itc_uncertainty_multiplier,
            itc_correlation,
            mo.stat(
                f"{100 * relative_se_differences.max():.1f}%",
                label="Largest Taylor–MC SD difference",
                caption="20,000 fixed-seed draws",
                bordered=True,
            ),
        ]
    )
    itc_kind = "warn" if relative_se_differences.max() > 0.05 else "info"
    itc_note = mo.callout(
        mo.md(
            "The Taylor curves use first-order propagation without covariance. "
            "Increasing the input uncertainties exposes skew and curvature. Changing "
            "the correlation affects the MC distribution of $T\\Delta S$, but not its "
            "displayed Taylor SE; the resulting gap shows what is lost when covariance "
            "is omitted. Depending on its sign, that omission can overstate or "
            "understate uncertainty."
        ),
        kind=itc_kind,
    )
    two_column_panel(
        itc_controls,
        mo.vstack([itc_table, itc_figure, itc_note]),
        widths=(1, 3),
    )


@app.cell
def _(mo):
    propagation_derivation = mo.md(r"""
    ### Derivation: familiar rules from the general formula

    For independent inputs, covariance is zero. With $k$ an exact constant, the
    slope-based formula gives the rules below. The rules for scaling, addition,
    and subtraction are exact; those for multiplication and division are
    first-order approximations:

    $$z=kx:\quad s_z^2=k^2s_x^2,$$
    $$z=x\pm y:\quad s_z^2=s_x^2+s_y^2,$$
    $$z=xy:\quad s_z^2=y^2s_x^2+x^2s_y^2,$$
    $$z=x/y:\quad s_z^2=s_x^2/y^2+x^2s_y^2/y^4.$$

    If inputs are correlated, restore the covariance term with the signs of the two
    derivatives. The sign determines whether covariance raises or lowers the output
    variance.
    """)
    mo.accordion({"Deep dive — derivatives and covariance": propagation_derivation})


@app.cell
def _(mo):
    mo.md(r"""
    ---

    ## 9. Review questions

    Choose one answer for each question. Feedback explains the underlying idea rather
    than only marking the response.
    """)


@app.cell
def _(mo):
    review_question_1 = mo.ui.radio(
        {
            "They are always independent": "independent",
            "They are not independent when both have positive probability": "not_independent",
            "They must have equal probabilities": "equal",
            "Their intersection has positive probability": "intersection",
        },
        value=None,
        label="Choose one answer",
    )
    mo.vstack(
        [
            mo.md(
                "### 1. Events\n\nTwo events are mutually exclusive and each has "
                "positive probability. What follows?"
            ),
            review_question_1,
        ]
    )
    return (review_question_1,)


@app.cell
def _(review_feedback, review_question_1):
    review_feedback(
        review_question_1.value,
        correct_value="not_independent",
        correct_text=(
            "**Correct.** If one occurs, the other becomes impossible, so learning "
            "about one changes the probability of the other."
        ),
        incorrect_text=(
            {
                "independent": "Mutual exclusivity means P(A ∩ B) = 0. Independence would require P(A ∩ B) = "
                "P(A)P(B), which is positive here. Observing A rules out B, so they are dependent.",
                "equal": "Mutual exclusivity restricts overlap, not the separate probabilities. On a fair die, "
                "“roll 1” and “roll 2 or 3” are mutually exclusive but have probabilities 1/6 and 2/6.",
                "intersection": "Mutually exclusive events cannot occur together, so their intersection is empty "
                "and has probability zero. Their individual probabilities can both be positive.",
            }
        ),
    )


@app.cell
def _(mo):
    review_question_2 = mo.ui.radio(
        {
            "Binomial: events per hour; Poisson: successes in fixed trials": "reversed",
            "Binomial: successes in fixed independent trials with common p; Poisson: events in a fixed exposure interval": "correct_models",
            "Both require exactly two observed counts": "two_counts",
            "Both guarantee independence in real data": "guarantee",
        },
        value=None,
        label="Choose one answer",
    )
    mo.vstack(
        [
            mo.md(
                "### 2. Count models\n\nWhich description correctly distinguishes "
                "the binomial and Poisson models?"
            ),
            review_question_2,
        ]
    )
    return (review_question_2,)


@app.cell
def _(review_feedback, review_question_2):
    review_feedback(
        review_question_2.value,
        correct_value="correct_models",
        correct_text=(
            "**Correct.** Binomial counts successes among n independent binary trials "
            "with a common p. In the Poisson-process model, events occur independently "
            "at a constant rate, and the expected count is rate × exposure. "
            "These assumptions need scientific justification."
        ),
        incorrect_text=(
            {
                "reversed": "The descriptions are reversed. Binomial counts successes among a fixed number of "
                "independent trials with common success probability; Poisson models an event count in "
                "a specified exposure interval.",
                "two_counts": "A binomial trial has two possible outcomes, but its total count can range from 0 "
                "to n. A Poisson count can be any nonnegative integer; neither model requires "
                "exactly two observed counts.",
                "guarantee": "Independence is an assumption to justify from the design and process. Choosing a "
                "binomial or Poisson formula cannot make dependent observations independent.",
            }
        ),
    )


@app.cell
def _(mo):
    review_question_3 = mo.ui.radio(
        {
            "The raw population becomes normal": "population_normal",
            "Every individual sample becomes symmetric": "sample_normal",
            "The distribution of sample means, expressed in SE units around the true mean": "sampling_normal",
            "The sample standard deviation becomes zero": "sd_zero",
        },
        value=None,
        label="Choose one answer",
    )
    mo.vstack(
        [
            mo.md(
                "### 3. Central Limit Theorem\n\nWhat tends to become approximately "
                "normal as sample size increases for independent observations "
                "from the same population with finite, nonzero variance?"
            ),
            review_question_3,
        ]
    )
    return (review_question_3,)


@app.cell
def _(review_feedback, review_question_3):
    review_feedback(
        review_question_3.value,
        correct_value="sampling_normal",
        correct_text=(
            "**Correct.** The theorem concerns the distribution of sample means "
            "across repeated samples, not the distribution of individual values. "
            "After centering at μ and scaling by σ/√n, the mean approaches a "
            "standard normal distribution. The unscaled mean becomes more concentrated."
        ),
        incorrect_text=(
            {
                "population_normal": "Taking larger samples does not change the population distribution. For "
                "example, an exponential population stays skewed while its repeated-sample "
                "means become approximately normal.",
                "sample_normal": "The CLT concerns a statistic across repeated samples, not the shape of one "
                "sample. A large raw sample can display the population’s skewness more clearly.",
                "sd_zero": "The sample SD estimates the spread of individual observations and need not shrink "
                "toward zero. The mean’s SE shrinks as σ/√n for independent observations; its "
                "standardized sampling distribution is what the CLT describes.",
            }
        ),
    )


@app.cell
def _(mo):
    review_question_4 = mo.ui.radio(
        {
            "There is a 95% probability that the true mean μ is in this particular interval": "posterior",
            "95% of future observations must lie inside": "future",
            "The procedure captures μ in 95% of repeated samples": "coverage",
            "The interval is guaranteed to contain μ": "guarantee",
        },
        value=None,
        label="Choose one answer",
    )
    mo.vstack(
        [
            mo.md(
                "### 4. Confidence intervals\n\nWhat does a 95% frequentist confidence "
                "level describe?"
            ),
            review_question_4,
        ]
    )
    return (review_question_4,)


@app.cell
def _(review_feedback, review_question_4):
    review_feedback(
        review_question_4.value,
        correct_value="coverage",
        correct_text=(
            "**Correct.** The confidence level is the long-run success rate of the "
            "interval-generating procedure under its assumptions."
        ),
        incorrect_text=(
            {
                "posterior": "Assigning a probability to the unknown mean requires a Bayesian model, including "
                "a prior (a description of uncertainty before these data). Frequentist 95% coverage "
                "refers to intervals varying across repeated samples while the population mean stays "
                "fixed.",
                "future": "A confidence interval targets the population mean, not the spread of new observations. "
                "Predicting a new observation requires a prediction interval, which includes individual "
                "variability.",
                "guarantee": "Even a correctly calibrated 95% procedure misses the fixed parameter in about 5% of "
                "repetitions. The confidence level cannot guarantee that this particular interval "
                "contains it.",
            }
        ),
    )


@app.cell
def _(mo):
    review_question_5 = mo.ui.radio(
        {
            "Draw all 20 records without replacement": "without_replacement",
            "Draw 20 records with replacement, allowing repeats and omissions": "replacement",
            "Draw 200 records so each resample is more precise": "larger",
            "Resampling guarantees that selection bias disappears": "repairs",
        },
        value=None,
        label="Choose one answer",
    )
    mo.vstack(
        [
            mo.md(
                "### 5. Bootstrap resampling\n\nYou observed 20 independent units "
                "and want to estimate the SE of their sample mean with an ordinary "
                "bootstrap that draws directly from the observed records. How should each resample be drawn?"
            ),
            review_question_5,
        ]
    )
    return (review_question_5,)


@app.cell
def _(review_feedback, review_question_5):
    review_feedback(
        review_question_5.value,
        correct_value="replacement",
        correct_text=(
            "**Correct.** Each resample contains 20 draws from the observed records "
            "with replacement. A record can appear more than once or not at all. "
            "Calculate a mean for each resample; the SD of these means estimates "
            "the original mean’s SE under the bootstrap assumptions."
        ),
        incorrect_text=(
            {
                "without_replacement": "Drawing all 20 records without replacement would only reorder the "
                "original sample. Replacement is what lets the bootstrap generate "
                "different samples and hence a distribution of the statistic.",
                "larger": "The ordinary bootstrap used here uses the original sample size, here 20. Drawing "
                "more records in each resample changes the sampling problem and usually makes the "
                "resulting spread too small for the original estimator.",
                "repairs": "The bootstrap resamples the observed records; it has no information about "
                "systematically missing parts of the population. It estimates sampling uncertainty "
                "under its assumptions, not a correction for selection bias.",
            }
        ),
    )


@app.cell
def _(mo):
    review_question_6 = mo.ui.radio(
        {
            "Positive covariance increases the variance of both": "both_increase",
            "It increases the variance of X + Y and decreases that of X − Y": "opposite",
            "Covariance changes the means, but not the variances": "means_only",
            "It can be ignored whenever X and Y each have a normal distribution": "normal",
        },
        value=None,
        label="Choose one answer",
    )
    mo.vstack(
        [
            mo.md(
                "### 6. Covariance in propagation\n\nHold Var(X) and Var(Y) fixed. "
                "Relative to zero covariance, how does positive Cov(X, Y) affect "
                "the variances of X + Y and X − Y?"
            ),
            review_question_6,
        ]
    )
    return (review_question_6,)


@app.cell
def _(review_feedback, review_question_6):
    review_feedback(
        review_question_6.value,
        correct_value="opposite",
        correct_text="**Correct.** Var(X + Y) = Var(X) + Var(Y) + 2Cov(X, Y), whereas Var(X − Y) = Var(X) + Var(Y) − 2Cov(X, Y). Shared positive fluctuations reinforce a sum but cancel partly in a difference. These identities are exact for finite variances.",
        incorrect_text={
            "both_increase": "The covariance term changes sign for a difference. When X and Y move together, their sum fluctuates more, but subtracting Y partly cancels their shared fluctuation.",
            "means_only": "Linearity gives E(X ± Y) = E(X) ± E(Y), regardless of covariance. Covariance enters the variance through the cross term, so it affects uncertainty even when the means are unchanged.",
            "normal": "X and Y can each follow a normal distribution without being independent or having zero covariance. Correlated normal variables still require the covariance term when calculating the variance of a sum or difference.",
        },
    )


@app.cell
def _(mo):
    review_question_7 = mo.ui.radio(
        {
            "Both population SD and the mean’s SE are halved": "both",
            "Population SD stays 10 cm; the mean’s SE falls from 2 cm to 1 cm": "se_only",
            "Population SD stays 10 cm; the mean’s SE falls from 2 cm to 0.5 cm": "quarter",
            "Both remain unchanged because the population is unchanged": "neither",
        },
        value=None,
        label="Choose one answer",
    )
    mo.vstack(
        [
            mo.md(
                "### 7. SD versus SE\n\nHeights have population SD σ = 10 cm. "
                "If the number of independent observations increases from 25 to 100 "
                "from the same population, what happens to the population SD and the sample mean’s SE?"
            ),
            review_question_7,
        ]
    )
    return (review_question_7,)


@app.cell
def _(review_feedback, review_question_7):
    review_feedback(
        review_question_7.value,
        correct_value="se_only",
        correct_text="**Correct.** SD describes variation among individual heights and stays at 10 cm. SE describes variation of sample means: 10/√25 = 2 cm and 10/√100 = 1 cm. Four times as many independent observations halves the SE, not the population spread.",
        incorrect_text={
            "both": "Larger samples estimate the mean more precisely; they do not make individual heights less variable. The population SD remains 10 cm while the mean’s SE decreases.",
            "quarter": "The mean’s variance falls by a factor of four, but SE is its square root. Thus SE is divided by √4 = 2, giving 1 cm rather than 0.5 cm.",
            "neither": "The population spread stays fixed, but the statistic averages more independent observations. Its sampling spread is σ/√n, so the mean’s SE falls from 2 cm to 1 cm.",
        },
    )


@app.cell
def _(mo):
    review_question_8 = mo.ui.radio(
        {
            "10, because all ten deviations vary freely": "ten",
            "9, because deviations from the fitted sample mean sum to zero": "nine",
            "8, because estimating a mean and an SD always costs two degrees of freedom": "eight",
            "9, because one observation is removed before calculating SD": "removed",
        },
        value=None,
        label="Choose one answer",
    )
    mo.vstack(
        [
            mo.md(
                "### 8. Degrees of freedom\n\nFor ten independent observations "
                "from a normal population with unknown mean and variance, how many "
                "degrees of freedom does the one-sample t statistic have, and why?"
            ),
            review_question_8,
        ]
    )
    return (review_question_8,)


@app.cell
def _(review_feedback, review_question_8):
    review_feedback(
        review_question_8.value,
        correct_value="nine",
        correct_text="**Correct.** Estimating the mean imposes one constraint: the ten deviations sum to zero, so the last is determined by the other nine. All ten observations still contribute. Under normal sampling, this yields the t distribution with n − 1 = 9 degrees of freedom.",
        incorrect_text={
            "ten": "The deviations are measured from an estimated mean, so they must sum to zero. They cannot all vary freely: one linear constraint leaves n − 1 = 9 degrees of freedom.",
            "eight": "For the one-sample t statistic, the relevant residual sum of squares has n − 1 degrees of freedom after estimating one mean. Estimating SD from those residuals does not impose a second independent constraint.",
            "removed": "The value 9 is right, but no observation is discarded. The loss of one degree of freedom comes from the sum-to-zero constraint on deviations after estimating the mean.",
        },
    )


@app.cell
def _(mo):
    mo.md(r"""
    ---

    ## 10. Key points and next chapter

    - Start with the biological question: which population do you want to learn
      about, and which independent units did you sample? More measurements cannot
      repair biased sampling or turn technical repeats into biological replicates.
    - For counts, graph heights give probabilities of individual values. For
      continuous measurements, probabilities are areas under a density curve.
      A CDF gives the probability at or below a chosen value.
    - **SD** describes variation among observations; **SE** describes variation
      among estimates across repeated samples. For the mean, four times as many
      independent observations halves the SE.
    - The **Central Limit Theorem** explains why averages can be approximately
      normal even when individual measurements are not.
    - A **95% confidence method** includes the true value in about 95% of repeated
      studies when its assumptions hold. The t method allows for estimating the
      population SD from the sample.
    - **Bootstrapping** resamples existing observations with replacement to
      estimate sampling uncertainty. It adds no new biological information.
    - **Uncertainty propagation** tracks how uncertain inputs affect a calculated
      result. A slope-based approximation works well for small uncertainties;
      simulation helps reveal the effects of curvature and correlated inputs.

    **Next:** Chapter 3 explores relationships between two variables using
    correlation and regression. We will ask how strongly measurements vary
    together, how to describe that relationship, and how uncertain our conclusions
    are.
    """)


if __name__ == "__main__":
    app.run()
