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
    app_title="Descriptive statistics",
    css_file="site/assets/notebook.css",
)


@app.cell
def chapter_header():
    from companion_style import notebook_header

    notebook_header(
        1,
        "Descriptive statistics",
        "Learning from a dataset of student heights",
    )


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from scipy import stats

    from companion_data import read_csv
    from companion_statistics import cohen_d_sample
    from companion_style import (
        COLORS,
        FIGURE_SIZE_COMPACT,
        FIGURE_SIZE_LINKED,
        FIGURE_SIZE_SHALLOW,
        FIGURE_SIZE_STANDARD,
        compact_table,
        dataset_action_buttons,
        review_feedback,
        two_column_panel,
    )

    return (
        COLORS,
        FIGURE_SIZE_COMPACT,
        FIGURE_SIZE_LINKED,
        FIGURE_SIZE_SHALLOW,
        FIGURE_SIZE_STANDARD,
        cohen_d_sample,
        compact_table,
        dataset_action_buttons,
        np,
        pd,
        plt,
        read_csv,
        review_feedback,
        stats,
        two_column_panel,
    )


@app.cell
def _(mo):
    mo.md(r"""
    A dataset is usually too detailed to understand by inspection alone. Descriptive
    statistics help us answer three complementary questions:

    1. **Where are the observations located?** — center and quantiles
    2. **How much do they vary?** — range, variance, standard deviation, and IQR
    3. **What does their distribution look like?** — tables and graphical displays

    In this chapter you will work with the artificial height dataset used in the
    lecture. The notebook is meant to be read from top to bottom, but every control
    is live: make a prediction, change a control, and inspect what follows.

    **Start here:** inspect the lecture dataset below, then click **Generate a new
    dataset** and compare its size and group proportions with the original. Use
    **Reset to lecture dataset** to return to the reference values. These buttons
    replace the data used by the later height activities. Sorting or searching the
    table affects its display only. Sliders and selectors update their outputs
    immediately; change one at a time to see which feature it controls.

    > **Guiding principle:** A numerical summary is useful only when we know what
    > information it preserves and what information it hides.
    """)


@app.cell
def _(mo, np, pd, read_csv, stats):
    data_path = mo.notebook_location() / "public" / "height_data.csv"
    original_height_data = read_csv(data_path).sample(frac=1).reset_index(drop=True)

    expected_columns = {"gender", "height"}
    assert set(original_height_data.columns) == expected_columns
    assert len(original_height_data) == 97
    assert original_height_data["height"].notna().all()

    # The parameters of these normal distributions are the BRFSS-inspired
    # values used in the lecture's Jupyter notebooks.
    female_height_mean, female_height_sd = 163.0, 7.3
    male_height_mean, male_height_sd = 178.0, 7.7

    # With this parameterization, E[p] = 0.55 and approximately 95% of the
    # beta distribution lies between 0.45 and 0.65.
    female_beta_alpha = 51.48906087834132
    female_beta_beta = 42.12741344591562
    female_beta_interval_probability = stats.beta.cdf(
        0.65, female_beta_alpha, female_beta_beta
    ) - stats.beta.cdf(0.45, female_beta_alpha, female_beta_beta)

    def original_height_dataset():
        _data = original_height_data.copy()
        return {
            "data": _data,
            "source": "Original lecture dataset",
            "sampled_female_fraction": None,
            "realized_female_fraction": float((_data["gender"] == "female").mean()),
        }

    def generate_height_dataset():
        _rng = np.random.default_rng()
        _sampled_female_fraction = float(_rng.beta(female_beta_alpha, female_beta_beta))
        _n_students = int(_rng.choice(np.arange(91, 110, 2)))
        _n_female = int(np.rint(_sampled_female_fraction * _n_students))
        _n_female = int(np.clip(_n_female, 1, _n_students - 1))
        _n_male = _n_students - _n_female

        _data = pd.DataFrame(
            {
                "gender": np.concatenate(
                    [np.repeat("female", _n_female), np.repeat("male", _n_male)]
                ),
                "height": np.concatenate(
                    [
                        _rng.normal(female_height_mean, female_height_sd, _n_female),
                        _rng.normal(male_height_mean, male_height_sd, _n_male),
                    ]
                ).round(1),
            }
        )
        _data = _data.iloc[_rng.permutation(len(_data))].reset_index(drop=True)
        return {
            "data": _data,
            "source": "New randomly generated dataset",
            "sampled_female_fraction": _sampled_female_fraction,
            "realized_female_fraction": _n_female / _n_students,
        }

    get_height_dataset, set_height_dataset = mo.state(original_height_dataset())
    return (
        data_path,
        female_beta_alpha,
        female_beta_beta,
        female_beta_interval_probability,
        female_height_mean,
        female_height_sd,
        generate_height_dataset,
        get_height_dataset,
        male_height_mean,
        male_height_sd,
        original_height_dataset,
        set_height_dataset,
    )


@app.cell
def _(get_height_dataset):
    height_dataset = get_height_dataset()
    height_data = height_dataset["data"]
    return height_data, height_dataset


@app.cell
def _(
    dataset_action_buttons,
    generate_height_dataset,
    original_height_dataset,
    set_height_dataset,
):
    def _generate_dataset(_value):
        set_height_dataset(generate_height_dataset())

    def _reset_dataset(_value):
        set_height_dataset(original_height_dataset())

    generate_dataset_button, reset_dataset_button = dataset_action_buttons(
        on_generate=_generate_dataset,
        on_reset=_reset_dataset,
    )
    return generate_dataset_button, reset_dataset_button


@app.cell
def _(
    data_path,
    female_beta_alpha,
    female_beta_beta,
    female_beta_interval_probability,
    female_height_mean,
    female_height_sd,
    generate_dataset_button,
    height_data,
    height_dataset,
    male_height_mean,
    male_height_sd,
    mo,
    reset_dataset_button,
    two_column_panel,
):
    dataset_table = mo.ui.table(
        height_data,
        selection=None,
        page_size=10,
        show_column_summaries=True,
        show_data_types=False,
        label="Artificial height data",
    )

    _n_female = int((height_data["gender"] == "female").sum())
    _n_male = int((height_data["gender"] == "male").sum())
    if height_dataset["sampled_female_fraction"] is None:
        _source_details = (
            f"The notebook uses all values from `{data_path.name}` but displays the "
            "rows in a random order."
        )
    else:
        _source_details = (
            "The generator sampled a target female proportion of "
            f"**{height_dataset['sampled_female_fraction']:.1%}**; after conversion "
            "to whole students, the realized proportion is "
            f"**{height_dataset['realized_female_fraction']:.1%}**."
        )

    _controls = mo.vstack(
        [
            mo.md("### Choose the dataset"),
            generate_dataset_button,
            reset_dataset_button,
            mo.callout(
                mo.md(
                    f"""
                    **{height_dataset["source"]}.** The current data contain
                    **{len(height_data)}** observations: **{_n_female}** labeled female
                    and **{_n_male}** labeled male. {_source_details}

                    Each new dataset uses an odd sample size from 91 to 109. Its target
                    female proportion is drawn from
                    $\\operatorname{{Beta}}({female_beta_alpha:.1f},
                    {female_beta_beta:.1f})$; this puts
                    **{female_beta_interval_probability:.1%}** of the probability between
                    0.45 and 0.65. Heights are then drawn from the BRFSS-inspired models
                    $N({female_height_mean:.0f}, {female_height_sd:.1f}^2)$ cm for women
                    and $N({male_height_mean:.0f}, {male_height_sd:.1f}^2)$ cm for men,
                    and rounded to 0.1 cm.
                    """
                ),
                kind="info",
            ),
        ],
        gap=1,
    )
    two_column_panel(
        _controls,
        dataset_table,
        widths=[1, 1.7],
        gap=1.5,
    )


@app.cell
def _(mo):
    mo.md(r"""
    ---

    ## 1. What kind of variable do we have?

    A **categorical** variable records membership in categories. A **numerical**
    variable records a quantity for which arithmetic may be meaningful.

    Numerical variables can be **discrete** (counts such as 0, 1, 2, ...) or
    **continuous** (measurements that can, in principle, take any value in an
    interval). The recorded height values look discrete because they were rounded
    to 0.1 cm, but the underlying quantity is continuous.

    Measurement scales add another distinction:

    - **Nominal:** categories without an order
    - **Ordinal:** ordered categories, but differences need not be equal
    - **Interval:** equal differences are meaningful, but zero is arbitrary
    - **Ratio:** equal differences and ratios are meaningful; zero represents absence

    **Try this:** classify height, disease stage, and bacterial species
    in your own words before consulting the selector. Select **Height in
    centimeters**, **Disease stage I–IV**, then **Bacterial species** to compare
    the explanations. For each, ask whether ordering, differences, and ratios are
    meaningful. The selector reveals the classification; it does not submit a
    scored answer.
    """)


@app.cell
def _(mo):
    variable_examples = {
        "Height in centimeters": {
            "kind": "Numerical, continuous",
            "scale": "Ratio",
            "reason": "Differences and ratios are meaningful, and 0 cm represents zero length.",
        },
        "Bacterial species": {
            "kind": "Categorical",
            "scale": "Nominal",
            "reason": "Species labels have no intrinsic numerical ordering.",
        },
        "Disease stage I–IV": {
            "kind": "Categorical",
            "scale": "Ordinal",
            "reason": "The stages are ordered, but the step from I to II need not equal the step from III to IV.",
        },
        "Temperature in degrees Celsius": {
            "kind": "Numerical, continuous",
            "scale": "Interval",
            "reason": "Temperature differences are meaningful, but 0 °C is not an absence of temperature.",
        },
        "Number of colonies on a plate": {
            "kind": "Numerical, discrete",
            "scale": "Ratio",
            "reason": "Colonies are counted, and zero means that none were observed.",
        },
        "Agreement: poor / fair / good / excellent": {
            "kind": "Categorical",
            "scale": "Ordinal",
            "reason": "The labels are ordered, but the gaps between them are not defined numerically.",
        },
    }
    variable_selector = mo.ui.dropdown(
        list(variable_examples),
        value="Height in centimeters",
        label="Classify this variable",
        full_width=True,
    )
    return variable_examples, variable_selector


@app.cell
def _(mo, two_column_panel, variable_examples, variable_selector):
    variable_answer = variable_examples[variable_selector.value]
    two_column_panel(
        variable_selector,
        mo.callout(
            mo.md(
                f"""
                **{variable_answer["kind"]} — {variable_answer["scale"]} scale**

                {variable_answer["reason"]}
                """
            ),
            kind="success",
        ),
        widths=[1, 2],
    )


@app.cell
def _(mo):
    mo.md(r"""
    ---

    ## 2. From raw observations to a distribution

    A histogram groups observations into intervals called **bins**. Its appearance
    depends on the number and location of those bins. This makes a histogram useful,
    but not uniquely determined by the data.

    On a **count scale**, bar height is the number of observations in a bin. On a
    **density scale**, the *area* of a bar represents relative frequency. A density
    is not itself a probability; probability corresponds to area.

    **Try this:** keep **All students** selected and compare 5, 15, and 30 bins.
    Watch the shape change while the summary statistics stay fixed. Switch to
    **Use density scale** and read the vertical axis again: area now represents
    relative frequency. Toggle the frequency polygon and fitted normal curve to
    distinguish summaries of the bins from a smooth model of the observations.
    Finally compare **Female** and **Male**: this changes which data enter the
    histogram and summary table. A model overlay is a comparison, not evidence
    that the population must be normal.
    """)


@app.cell
def _(mo):
    group_selector = mo.ui.dropdown(
        ["All students", "Female", "Male"],
        value="All students",
        label="Displayed group",
        full_width=True,
    )
    bin_count = mo.ui.slider(
        5,
        30,
        value=15,
        step=1,
        show_value=True,
        label="Number of bins",
        full_width=True,
    )
    density_scale = mo.ui.checkbox(value=False, label="Use density scale")
    polygon_overlay = mo.ui.checkbox(value=True, label="Show frequency polygon")
    normal_overlay = mo.ui.checkbox(value=False, label="Overlay fitted normal model")
    return (
        bin_count,
        density_scale,
        group_selector,
        normal_overlay,
        polygon_overlay,
    )


@app.cell
def _(group_selector, height_data):
    selected_group_label = group_selector.value
    if selected_group_label == "Female":
        height_subset = height_data.loc[height_data["gender"] == "female", "height"]
    elif selected_group_label == "Male":
        height_subset = height_data.loc[height_data["gender"] == "male", "height"]
    else:
        height_subset = height_data["height"]
    return height_subset, selected_group_label


@app.cell
def _(height_subset, np, pd):
    q1_summary, q3_summary = np.quantile(height_subset, [0.25, 0.75])
    summary_table = pd.DataFrame(
        {
            "Summary": [
                "Number of observations",
                "Minimum",
                "First quartile (Q1)",
                "Arithmetic mean",
                "Median (Q2)",
                "Third quartile (Q3)",
                "Maximum",
                "Range",
                "Population SD (divide by n)",
                "Sample SD (divide by n − 1)",
                "Interquartile range",
                "Coefficient of variation",
            ],
            "Value": [
                f"{len(height_subset)}",
                f"{height_subset.min():.1f} cm",
                f"{q1_summary:.1f} cm",
                f"{height_subset.mean():.1f} cm",
                f"{height_subset.median():.1f} cm",
                f"{q3_summary:.1f} cm",
                f"{height_subset.max():.1f} cm",
                f"{height_subset.max() - height_subset.min():.1f} cm",
                f"{height_subset.std(ddof=0):.1f} cm",
                f"{height_subset.std(ddof=1):.1f} cm",
                f"{q3_summary - q1_summary:.1f} cm",
                f"{height_subset.std(ddof=0) / height_subset.mean() * 100:.1f}%",
            ],
        }
    )
    return (summary_table,)


@app.cell
def _(
    COLORS,
    FIGURE_SIZE_STANDARD,
    bin_count,
    compact_table,
    density_scale,
    group_selector,
    height_subset,
    mo,
    normal_overlay,
    np,
    plt,
    polygon_overlay,
    selected_group_label,
    stats,
    summary_table,
    two_column_panel,
):
    histogram_figure, histogram_axis = plt.subplots(figsize=FIGURE_SIZE_STANDARD)
    histogram_values, histogram_edges, _ = histogram_axis.hist(
        height_subset,
        bins=bin_count.value,
        range=(145, 200),
        density=density_scale.value,
        color=COLORS["orange"],
        edgecolor="white",
        linewidth=1.2,
        alpha=0.85,
    )
    histogram_centers = (histogram_edges[:-1] + histogram_edges[1:]) / 2

    if polygon_overlay.value:
        polygon_x = np.concatenate(
            ([histogram_edges[0]], histogram_centers, [histogram_edges[-1]])
        )
        polygon_y = np.concatenate(([0], histogram_values, [0]))
        histogram_axis.plot(
            polygon_x,
            polygon_y,
            marker="o",
            markersize=4,
            linewidth=2,
            color=COLORS["blue"],
            label="Frequency polygon",
        )

    if normal_overlay.value:
        normal_x = np.linspace(140, 205, 400)
        fitted_normal = stats.norm(
            loc=height_subset.mean(), scale=height_subset.std(ddof=0)
        )
        normal_y = fitted_normal.pdf(normal_x)
        if not density_scale.value:
            normal_y = normal_y * len(height_subset) * np.diff(histogram_edges).mean()
        histogram_axis.plot(
            normal_x,
            normal_y,
            linewidth=2.5,
            color=COLORS["vermillion"],
            label="Normal model with same mean and SD",
        )

    histogram_axis.set(
        title=f"Distribution of heights — {selected_group_label}",
        xlabel="Height [cm]",
        ylabel="Density" if density_scale.value else "Number of observations",
        xlim=(140, 205),
    )
    histogram_axis.grid(axis="y", linestyle=":", alpha=0.4)
    if polygon_overlay.value or normal_overlay.value:
        histogram_axis.legend(frameon=False)
    histogram_figure.tight_layout()

    descriptive_statistics_table = compact_table(
        summary_table,
        column_widths={"Summary": 205, "Value": 85},
        wrapped_columns=["Summary"],
    )
    histogram_controls = mo.vstack(
        [
            group_selector,
            bin_count,
            density_scale,
            polygon_overlay,
            normal_overlay,
            mo.md(f"#### Descriptive statistics: {selected_group_label}"),
            descriptive_statistics_table,
        ]
    )
    histogram_results = mo.vstack(
        [
            histogram_figure,
            mo.callout(
                mo.md(
                    r"""
                    **What should you notice?** The observations and all numerical
                    summaries remain unchanged when you alter the bin count. Only the
                    visual grouping changes. Also compare the combined sample with its
                    two subgroups: mixing subpopulations can create a shape that no
                    individual subgroup has.
                    """
                ),
                kind="warn",
            ),
        ]
    )
    two_column_panel(
        histogram_controls,
        histogram_results,
        widths=[1, 2],
    )


@app.cell
def _(mo):
    mo.md(r"""
    ---

    ## 3. The arithmetic mean

    For observations $x_1, x_2, \ldots, x_n$, the arithmetic mean is

    \[
    \bar{x}=\frac{x_1+x_2+\cdots+x_n}{n}
           =\frac{1}{n}\sum_{i=1}^{n}x_i.
    \]

    This is more than a recipe. The mean is the unique center that makes the sum of
    signed deviations equal to zero. It is also the center that minimizes the sum of
    squared deviations. The second property gives us a useful derivation.

    **Try this:** keep six example observations and move the candidate center
    below and above their displayed mean. Follow the signed deviations and their
    sum, then locate the minimum of the sum-of-squares curve. Both point to the
    mean, although the 0.5 cm slider steps may prevent an exact match. Change the
    number of example observations to repeat the exercise with another subset of
    the selected group. These controls affect this worked example, not the full
    height dataset.
    """)


@app.cell
def _(mo):
    mean_example_size = mo.ui.slider(
        3,
        10,
        value=6,
        step=1,
        show_value=True,
        label="Number of example observations",
    )
    candidate_center = mo.ui.slider(
        145,
        200,
        value=172,
        step=0.5,
        show_value=True,
        label="Candidate center c [cm]",
    )
    return candidate_center, mean_example_size


@app.cell
def _(height_subset, mean_example_size, np):
    mean_example_indices = np.linspace(
        0, len(height_subset) - 1, mean_example_size.value, dtype=int
    )
    mean_example_values = height_subset.iloc[mean_example_indices].to_numpy()
    return (mean_example_values,)


@app.cell
def _(mean_example_values, mo):
    displayed_terms = " + ".join(f"{value:.1f}" for value in mean_example_values)
    displayed_mean = mean_example_values.mean()
    mo.md(
        rf"""
        For the **{len(mean_example_values)} displayed example values**,

        \[
        \bar{{x}}=\frac{{{displayed_terms}}}{{{len(mean_example_values)}}}
        ={displayed_mean:.2f}\text{{ cm}}.
        \]
        """
    )


@app.cell
def _(mo):
    mo.md(r"""
    ### Derivation: why does this formula define the least-squares center?

    Let $c$ be any proposed center and define its total squared discrepancy:

    \[
    S(c)=\sum_{i=1}^{n}(x_i-c)^2.
    \]

    At a minimum, the derivative with respect to $c$ must be zero:

    \[
    \begin{aligned}
    \frac{dS}{dc}
      &= -2\sum_{i=1}^{n}(x_i-c)=0\\
    \sum_{i=1}^{n}x_i-nc&=0\\
    c&=\frac{1}{n}\sum_{i=1}^{n}x_i=\bar{x}.
    \end{aligned}
    \]

    Because $S(c)$ is an upward-opening quadratic, this stationary point is its
    unique minimum. Move the candidate center below to see the geometry.
    """)


@app.cell
def _(
    COLORS,
    FIGURE_SIZE_LINKED,
    candidate_center,
    mean_example_size,
    mean_example_values,
    mo,
    np,
    plt,
    two_column_panel,
):
    candidate_value = candidate_center.value
    example_mean = mean_example_values.mean()
    example_residuals = mean_example_values - candidate_value
    example_sse = np.sum(example_residuals**2)

    center_grid = np.linspace(
        mean_example_values.min() - 8, mean_example_values.max() + 8, 400
    )
    sse_curve = np.sum(
        (mean_example_values[:, np.newaxis] - center_grid[np.newaxis, :]) ** 2,
        axis=0,
    )

    mean_figure, (residual_axis, sse_axis) = plt.subplots(
        1, 2, figsize=FIGURE_SIZE_LINKED, gridspec_kw={"width_ratios": [1.2, 1]}
    )
    row_positions = np.arange(len(mean_example_values))
    for row, value in zip(row_positions, mean_example_values):
        residual_axis.plot(
            [candidate_value, value],
            [row, row],
            color=COLORS["gray"],
            linewidth=1.5,
        )
    residual_axis.scatter(
        mean_example_values,
        row_positions,
        s=45,
        color=COLORS["blue"],
        zorder=3,
        label="observations",
    )
    residual_axis.axvline(
        candidate_value,
        color=COLORS["vermillion"],
        linewidth=2.5,
        label="candidate center",
    )
    residual_axis.axvline(
        example_mean,
        color=COLORS["green"],
        linestyle="--",
        linewidth=2,
        label="mean",
    )
    residual_axis.set(
        title="Signed deviations from a candidate center",
        xlabel="Height [cm]",
        ylabel="Example observation",
        yticks=row_positions,
        yticklabels=[f"x{i + 1}" for i in row_positions],
    )
    residual_axis.legend(frameon=False, fontsize=8)

    sse_axis.plot(center_grid, sse_curve, color=COLORS["blue"], linewidth=2.5)
    sse_axis.scatter(
        [candidate_value], [example_sse], color=COLORS["vermillion"], s=55, zorder=3
    )
    sse_axis.axvline(example_mean, color=COLORS["green"], linestyle="--")
    sse_axis.set(
        title="The mean minimizes squared deviations",
        xlabel="Candidate center c [cm]",
        ylabel=r"$S(c)=\sum (x_i-c)^2$",
    )
    sse_axis.grid(linestyle=":", alpha=0.35)
    mean_figure.tight_layout()

    mean_controls = mo.vstack(
        [
            mean_example_size,
            candidate_center,
            mo.stat(
                f"{example_residuals.sum():.1f} cm",
                label="Sum of signed deviations",
                bordered=True,
            ),
            mo.stat(
                f"{example_sse:.1f} cm²",
                label="Sum of squared deviations",
                bordered=True,
            ),
            mo.stat(
                f"{example_mean:.2f} cm",
                label="Least-squares center",
                bordered=True,
            ),
        ]
    )
    two_column_panel(
        mean_controls,
        mean_figure,
        widths=[1, 3],
    )


@app.cell
def _(mo):
    mo.md(r"""
    ### Mean or median?

    The mean uses the magnitude of every value. The median uses only the ordering:
    half of the observations lie on either side. That makes the mean efficient for
    many symmetric distributions, but also sensitive to extreme observations.

    In a roughly symmetric distribution, mean and median are often close. A long
    right tail tends to pull the mean above the median; a long left tail tends to
    pull it below the median. This is a useful pattern, not a definition of skewness.

    The next control adds one hypothetical measurement to the currently selected
    group. **Try this:** start at 170 cm, then move the additional height to 220
    and 500 cm. Compare the before/after columns for mean, median, SD, and range.
    The deliberately unrealistic end of the slider makes sensitivity to extremes
    visible. The horizontal axis expands to include the added value, so compare
    the numerical summaries as well as the apparent histogram shape. Only this
    activity includes the hypothetical observation.
    """)


@app.cell
def _(mo):
    hypothetical_height = mo.ui.slider(
        130,
        500,
        value=220,
        step=1,
        show_value=True,
        label="Additional hypothetical height [cm]",
        full_width=True,
    )
    return (hypothetical_height,)


@app.cell
def _(
    COLORS,
    FIGURE_SIZE_COMPACT,
    compact_table,
    height_subset,
    hypothetical_height,
    mo,
    np,
    pd,
    plt,
    two_column_panel,
):
    original_values = height_subset.to_numpy()
    augmented_values = np.append(original_values, hypothetical_height.value)

    outlier_comparison = pd.DataFrame(
        {
            "Statistic": ["Mean", "Median", "Population SD"],
            "Original": [
                original_values.mean(),
                np.median(original_values),
                original_values.std(ddof=0),
            ],
            "With added value": [
                augmented_values.mean(),
                np.median(augmented_values),
                augmented_values.std(ddof=0),
            ],
        }
    )
    outlier_comparison["Change"] = (
        outlier_comparison["With added value"] - outlier_comparison["Original"]
    )
    outlier_axis_max = max(235, hypothetical_height.value + 15)

    outlier_figure, outlier_axis = plt.subplots(figsize=FIGURE_SIZE_COMPACT)
    outlier_axis.hist(
        original_values,
        bins=18,
        range=(130, 235),
        color=COLORS["sky"],
        alpha=0.7,
        edgecolor="white",
    )
    outlier_axis.axvline(
        original_values.mean(),
        color=COLORS["blue"],
        linewidth=2,
        label="original mean",
    )
    outlier_axis.axvline(
        augmented_values.mean(),
        color=COLORS["vermillion"],
        linewidth=2,
        linestyle="--",
        label="mean after addition",
    )
    outlier_axis.scatter(
        [hypothetical_height.value],
        [0],
        marker="D",
        s=65,
        color=COLORS["vermillion"],
        clip_on=False,
        label="added observation",
    )
    outlier_axis.set(
        xlabel="Height [cm]",
        ylabel="Count",
        xlim=(130, outlier_axis_max),
        title="Influence of one additional observation",
    )
    outlier_axis.legend(frameon=False, ncol=1, fontsize=8, loc="upper right")
    outlier_figure.tight_layout()

    outlier_table = compact_table(
        outlier_comparison,
        column_widths={
            "Statistic": 105,
            "Original": 70,
            "With added value": 125,
            "Change": 65,
        },
        format_mapping={
            "Original": "{:.2f}",
            "With added value": "{:.2f}",
            "Change": "{:+.2f}",
        },
    )
    outlier_controls = mo.vstack(
        [
            hypothetical_height,
            outlier_table,
        ]
    )
    two_column_panel(
        outlier_controls,
        outlier_figure,
        widths=[1, 1.6],
    )


@app.cell
def _(height_subset, mo):
    recorded_counts = height_subset.value_counts()
    highest_recorded_frequency = int(recorded_counts.max())
    recorded_modes = sorted(
        recorded_counts[recorded_counts == highest_recorded_frequency].index
    )
    recorded_mode_text = ", ".join(f"{value:.1f}" for value in recorded_modes)

    mo.callout(
        mo.md(
            f"""
            ### What about the mode?

            The **mode** is the most frequent recorded value. For the selected group,
            the maximum frequency is **{highest_recorded_frequency}**, and the tied
            recorded modes are **{recorded_mode_text} cm**.

            This example shows a limitation: height is continuous, but it was rounded
            to 0.1 cm, producing several ties. An exact sample mode is therefore not a
            stable description of the distribution's peak. A histogram's modal bin can
            be more interpretable, but—as the binning laboratory showed—it depends on
            the chosen intervals.
            """
        ),
        kind="info",
    )


@app.cell
def _(mo):
    mo.md(r"""
    ---

    ## 4. Alternative definitions of the mean

    The arithmetic mean is the appropriate center when observations contribute
    equally and combine **additively**. It is not, however, the only meaningful
    average. The scientific question and the process that generated the data determine
    which definition is useful.

    ### The weighted arithmetic mean

    A weighted mean allows observations or group summaries to make unequal, but
    explicitly justified, contributions:

    \[
    \bar{x}_w=\frac{\sum_{i=1}^{n}w_i x_i}{\sum_{i=1}^{n}w_i},
    \qquad w_i\geq 0.
    \]

    Only the **relative** weights matter: multiplying all weights by the same constant
    leaves the result unchanged. If all weights are equal, the formula reduces to the
    ordinary arithmetic mean.

    Common biological and statistical examples include:

    - **Combining subgroup means.** If two experimental groups contain different
      numbers of observations, their means must be weighted by their sample sizes to
      recover the mean of all observations:

      \[
      \bar{x}=\frac{n_1\bar{x}_1+n_2\bar{x}_2}{n_1+n_2}.
      \]

    - **Combining estimates with different precision.** In a meta-analysis, an
      estimate with a smaller variance can receive more weight, often proportional to
      its inverse variance, $w_i=1/s_i^2$. The assumptions behind such weights belong
      to inferential statistics and must be checked.
    - **Unequal contributions to a pooled sample.** Mean concentrations from aliquots
      of different volumes can be weighted by volume when the target is the
      concentration of the combined material.

    Weights should follow from the design or the estimand—not from whether a result is
    convenient or statistically significant. A weighted mean cannot repair biased or
    incomparable measurements.
    """)


@app.cell
def _(mo):
    mo.md(r"""
    ### The geometric mean

    For strictly positive observations, the geometric mean is

    \[
    \bar{x}_{\mathrm{geom}}
      =\left(\prod_{i=1}^{n}x_i\right)^{1/n}
      =\exp\left(\frac{1}{n}\sum_{i=1}^{n}\log x_i\right).
    \]

    Thus, it is the arithmetic mean of the **log-transformed values**, transformed
    back to the original scale. It describes a multiplicative center: equal fold
    changes above and below it balance one another. For example, the geometric mean
    of 5 and 20 is 10 because one value is half of 10 and the other is twice 10. Their
    arithmetic mean, 12.5, answers a different, additive question.

    This distinction is especially useful when positive measurements are approximately
    **log-normally distributed**:

    \[
    \log(X)\sim N(\mu,\sigma^2).
    \]

    On the original scale such data are right-skewed and may span several orders of
    magnitude; on the log scale they are approximately symmetric. The population
    geometric mean is $\exp(\mu)$, which is also the median of an exactly log-normal
    distribution. Its arithmetic mean is larger,
    $\exp(\mu+\sigma^2/2)$, and represents the expected amount on the original scale.
    Neither is universally better: they summarize different scientific targets.
    """)


@app.cell
def _(mo):
    mo.vstack(
        [
            mo.md(
                r"""
                #### Biological measurements for which a geometric mean may be useful

                - **Binding and inhibition constants** such as $K_d$, $K_i$, $IC_{50}$,
                  and $EC_{50}$. These positive concentration-like quantities often
                  span orders of magnitude, and experimental variation is frequently
                  more naturally expressed as a fold difference. Averaging their
                  logarithms—and reporting the back-transformed mean—can therefore be
                  more interpretable than averaging the raw values.
                - **Gene-expression fold changes and other response ratios.** A
                  two-fold increase and a two-fold decrease are symmetric on a log
                  scale; their geometric mean fold change is 1, indicating no typical
                  multiplicative change.
                - **Microbial counts, viral loads, antibody titres, metabolite or
                  hormone concentrations, fluorescence intensities, and enzyme
                  activities** when their empirical distribution is approximately
                  log-normal and the measurement is strictly positive.
                - **Growth factors or repeated proportional changes.** If a quantity
                  changes by successive multiplicative factors, the geometric mean of
                  those factors describes the typical factor per step.

                Consider three $K_d$ estimates: 1, 10, and 100 nM. They are evenly
                spaced on a logarithmic scale. Their geometric mean is 10 nM, whereas
                their arithmetic mean is 37 nM. The geometric mean captures the middle
                *order of magnitude*; the arithmetic mean captures the mean amount in
                nM.
                """
            ),
            mo.callout(
                mo.md(
                    r"""
                    **Before using a geometric mean, ask:** Are all values positive? Are
                    ratios and fold changes scientifically meaningful? Is variation
                    approximately multiplicative, or do the logged values form a
                    reasonably symmetric distribution?

                    Zeros, negative values, and measurements reported as “below the
                    detection limit” cannot simply be logged. Adding an arbitrary
                    pseudocount changes the result and may introduce bias; censored-data
                    methods or a measurement-specific model may be needed. Skewness by
                    itself is not a sufficient reason to use a geometric mean.
                    """
                ),
                kind="warn",
            ),
        ]
    )


@app.cell
def _(mo):
    mo.md(r"""
    ---

    ## 5. Quantiles and measures of spread

    An α-quantile $Q_{\alpha}$ is a value at or below which approximately a
    proportion α of the observations lies. The median is $Q_{0.5}$; the first and
    third quartiles are $Q_{0.25}$ and $Q_{0.75}$.

    The **interquartile range** is

    \[
    \mathrm{IQR}=Q_{0.75}-Q_{0.25}.
    \]

    It describes the width of the middle half of the data and is more resistant to
    extremes than the full range.

    **Try this:** set the quantile level to 0.25, 0.50, and 0.75 to find the lower
    quartile, median, and upper quartile. Read each height from the marker and
    compare its position in the ordered data. Then select 0.90 and complete the
    sentence “About 90% of these heights are at or below …”. Interpolation and
    tied values mean the sample fraction need not equal the requested level
    exactly. The selected group from the histogram activity is used here too.
    """)


@app.cell
def _(mo):
    quantile_level = mo.ui.slider(
        0.05,
        0.95,
        value=0.25,
        step=0.05,
        show_value=True,
        label="Quantile level α",
        full_width=True,
    )
    return (quantile_level,)


@app.cell
def _(
    COLORS,
    FIGURE_SIZE_SHALLOW,
    height_subset,
    mo,
    np,
    plt,
    quantile_level,
    two_column_panel,
):
    selected_quantile = np.quantile(height_subset, quantile_level.value)
    empirical_share = np.mean(height_subset <= selected_quantile)

    quantile_figure, quantile_axis = plt.subplots(figsize=FIGURE_SIZE_SHALLOW)
    quantile_axis.hist(
        height_subset,
        bins=18,
        range=(140, 205),
        density=True,
        color=COLORS["orange"],
        edgecolor="white",
        alpha=0.7,
    )
    quantile_axis.axvspan(
        140,
        selected_quantile,
        color=COLORS["sky"],
        alpha=0.35,
        label=f"values ≤ Q{quantile_level.value:.2f}",
    )
    quantile_axis.axvline(
        selected_quantile,
        color=COLORS["blue"],
        linewidth=2.5,
    )
    quantile_axis.set(
        xlabel="Height [cm]",
        ylabel="Density",
        xlim=(140, 205),
        title="A quantile divides the ordered observations",
    )
    quantile_axis.legend(frameon=False)
    quantile_figure.tight_layout()

    quantile_controls = mo.vstack(
        [
            quantile_level,
            mo.stat(
                f"{selected_quantile:.1f} cm",
                label=f"Q{quantile_level.value:.2f}",
                bordered=True,
            ),
            mo.stat(
                f"{empirical_share * 100:.1f}%",
                label="Observed share at or below",
                bordered=True,
            ),
        ]
    )
    quantile_results = mo.vstack(
        [
            quantile_figure,
            mo.md(
                "For finite samples, ties and interpolation conventions can make the "
                "observed percentage differ slightly from exactly 100α%."
            ),
        ]
    )
    two_column_panel(
        quantile_controls,
        quantile_results,
        widths=[1, 3],
    )


@app.cell
def _(mo):
    mo.md(r"""
    ### Variance and standard deviation

    The variance treats every deviation from the mean as a contribution to spread:

    \[
    \sigma^2=\frac{1}{n}\sum_{i=1}^{n}(x_i-\bar{x})^2,
    \qquad
    \sigma=\sqrt{\sigma^2}.
    \]

    Squaring prevents positive and negative deviations from cancelling. Variance is
    measured in squared units (cm²); the square root returns the standard deviation
    to the original unit (cm).
    """)


@app.cell
def _(compact_table, mean_example_values, mo, pd):
    deviation_center = mean_example_values.mean()
    deviation_table = pd.DataFrame(
        {
            "Observation xᵢ [cm]": mean_example_values,
            "Deviation xᵢ − x̄ [cm]": mean_example_values - deviation_center,
            "Squared deviation [cm²]": (mean_example_values - deviation_center) ** 2,
        }
    )
    population_variance_example = deviation_table["Squared deviation [cm²]"].mean()

    mo.vstack(
        [
            compact_table(
                deviation_table,
                format_mapping={
                    "Observation xᵢ [cm]": "{:.1f}",
                    "Deviation xᵢ − x̄ [cm]": "{:+.2f}",
                    "Squared deviation [cm²]": "{:.2f}",
                },
            ),
            mo.hstack(
                [
                    mo.stat(
                        f"{deviation_table['Deviation xᵢ − x̄ [cm]'].sum():.2e} cm",
                        label="Sum of deviations",
                        bordered=True,
                    ),
                    mo.stat(
                        f"{population_variance_example:.2f} cm²",
                        label="Variance (divide by n)",
                        bordered=True,
                    ),
                    mo.stat(
                        f"{population_variance_example**0.5:.2f} cm",
                        label="Standard deviation",
                        bordered=True,
                    ),
                ],
                widths="equal",
            ),
            mo.callout(
                mo.md(
                    "Here we are **describing these displayed values**, so the divisor is "
                    "n. When using a sample to estimate a population variance, the usual "
                    "unbiased estimator divides by n − 1. We derive that adjustment in "
                    "the inferential-statistics chapter."
                ),
                kind="info",
            ),
        ]
    )


@app.cell
def _(mo):
    mo.md(r"""
    ---

    ## 6. Showing distributions honestly

    A **box plot** displays the median, quartiles, IQR, and whiskers. Points beyond
    $1.5\,\mathrm{IQR}$ are often flagged, but “flagged” does not mean erroneous.

    A **violin plot** adds a smoothed estimate of distribution shape. This can be
    helpful, but with small samples its bumps may reflect smoothing choices rather
    than genuine population structure. Raw observations remain valuable.

    **Try this:** begin with **Raw points**, switch to **Raw points + box plot**,
    then to **Raw points + violin plot**. The same observations remain visible
    in every view. Locate the center and spread of each group, then ask which
    detail the added summary
    makes easier to see and which it conceals. A box contains the middle half of
    the observations; a violin's width represents estimated density, not an
    uncertainty interval.
    """)


@app.cell
def _(mo):
    distribution_view = mo.ui.radio(
        ["Raw points", "Raw points + box plot", "Raw points + violin plot"],
        value="Raw points + box plot",
        label="Choose a display",
        inline=False,
    )
    return (distribution_view,)


@app.cell
def _(
    COLORS,
    FIGURE_SIZE_STANDARD,
    distribution_view,
    height_data,
    np,
    plt,
    two_column_panel,
):
    display_labels = ["All", "Female", "Male"]
    display_groups = [
        height_data["height"].to_numpy(),
        height_data.loc[height_data["gender"] == "female", "height"].to_numpy(),
        height_data.loc[height_data["gender"] == "male", "height"].to_numpy(),
    ]
    display_positions = np.arange(1, 4)
    display_figure, display_axis = plt.subplots(figsize=FIGURE_SIZE_STANDARD)

    if distribution_view.value == "Raw points + box plot":
        box_parts = display_axis.boxplot(
            display_groups,
            positions=display_positions,
            widths=0.45,
            patch_artist=True,
            showmeans=True,
            tick_labels=display_labels,
        )
        for box, color in zip(
            box_parts["boxes"],
            [COLORS["orange"], COLORS["purple"], COLORS["green"]],
        ):
            box.set_facecolor(color)
            box.set_alpha(0.35)

    if distribution_view.value == "Raw points + violin plot":
        violin_parts = display_axis.violinplot(
            display_groups,
            positions=display_positions,
            widths=0.8,
            showmeans=False,
            showmedians=True,
            showextrema=False,
        )
        for body, color in zip(
            violin_parts["bodies"],
            [COLORS["orange"], COLORS["purple"], COLORS["green"]],
        ):
            body.set_facecolor(color)
            body.set_edgecolor(color)
            body.set_alpha(0.28)
        violin_parts["cmedians"].set_color(COLORS["blue"])

    jitter_generator = np.random.default_rng(824)
    point_colors = [COLORS["orange"], COLORS["purple"], COLORS["green"]]
    for position, values, color in zip(display_positions, display_groups, point_colors):
        jitter = jitter_generator.normal(0, 0.045, size=len(values))
        display_axis.scatter(
            np.full(len(values), position) + jitter,
            values,
            s=18,
            alpha=0.65,
            color=color,
            edgecolor="white",
            linewidth=0.3,
            zorder=3,
        )

    display_axis.set(
        title=distribution_view.value,
        ylabel="Height [cm]",
        xlim=(0.4, 3.6),
        ylim=(140, 202),
        xticks=display_positions,
        xticklabels=display_labels,
    )
    display_axis.grid(axis="y", linestyle=":", alpha=0.35)
    display_figure.tight_layout()
    two_column_panel(
        distribution_view,
        display_figure,
        widths=[1, 3],
    )


@app.cell
def _(mo):
    mo.md(r"""
    ---

    ## 7. Comparing two distributions: Cohen's d

    Differences can be reported in the original unit, as a percentage, or relative
    to the distributions' spread. Cohen's $d$ is a standardized mean difference:

    \[
    d=\frac{\mu_2-\mu_1}{s_{\mathrm{pooled}}}.
    \]

    For two samples, a common pooled standard deviation is

    \[
    s_{\mathrm{pooled}}
      =\sqrt{\frac{(n_1-1)s_1^2+(n_2-1)s_2^2}{n_1+n_2-2}}.
    \]

    Because numerator and denominator have the same unit, $d$ is **unitless**. It is
    still important to report the original-unit difference, uncertainty, and
    biological relevance.

    **Try the model explorer below:** set both SDs to 7.5 and compare mean
    differences of 0, 15, and 30. Watch standardized separation, overlap, and
    probability of superiority together. Restore the difference to 15, then
    increase both SDs to 15: the same absolute difference becomes less distinct
    relative to variation. Finally change just one SD to explore unequal spreads.
    These sliders alter two theoretical normal distributions; they do not change
    the empirical height comparison reported above the controls.
    """)


@app.cell
def _(cohen_d_sample, height_data, mo):
    female_heights = height_data.loc[
        height_data["gender"] == "female", "height"
    ].to_numpy()
    male_heights = height_data.loc[height_data["gender"] == "male", "height"].to_numpy()
    observed_difference = male_heights.mean() - female_heights.mean()
    observed_d = cohen_d_sample(female_heights, male_heights)

    mo.callout(
        mo.md(
            f"""
            **In the artificial height data:** the mean difference (male − female) is
            **{observed_difference:.1f} cm**, and the sample Cohen's d is
            **{observed_d:.2f}**.

            This describes the standardized separation in this artificial dataset. It
            does not quantify uncertainty and is not, by itself, a significance test.
            """
        ),
        kind="info",
    )


@app.cell
def _(mo):
    delta_mu = mo.ui.slider(
        0,
        30,
        value=15,
        step=0.5,
        show_value=True,
        label="Difference in means Δμ",
    )
    sigma_1 = mo.ui.slider(
        2,
        20,
        value=7.5,
        step=0.5,
        show_value=True,
        label="SD of distribution 1",
    )
    sigma_2 = mo.ui.slider(
        2,
        20,
        value=7.5,
        step=0.5,
        show_value=True,
        label="SD of distribution 2",
    )
    return delta_mu, sigma_1, sigma_2


@app.cell
def _(
    COLORS,
    FIGURE_SIZE_STANDARD,
    delta_mu,
    mo,
    np,
    plt,
    sigma_1,
    sigma_2,
    stats,
    two_column_panel,
):
    mean_1 = 0.0
    mean_2 = delta_mu.value
    spread_1 = sigma_1.value
    spread_2 = sigma_2.value
    rms_pooled_spread = np.sqrt((spread_1**2 + spread_2**2) / 2)
    theoretical_d = (mean_2 - mean_1) / rms_pooled_spread

    effect_x = np.linspace(
        min(mean_1 - 4.5 * spread_1, mean_2 - 4.5 * spread_2),
        max(mean_1 + 4.5 * spread_1, mean_2 + 4.5 * spread_2),
        900,
    )
    density_1 = stats.norm.pdf(effect_x, loc=mean_1, scale=spread_1)
    density_2 = stats.norm.pdf(effect_x, loc=mean_2, scale=spread_2)
    overlap_density = np.minimum(density_1, density_2)
    overlap_area = np.trapezoid(overlap_density, effect_x)
    probability_2_higher = stats.norm.cdf(
        (mean_2 - mean_1) / np.sqrt(spread_1**2 + spread_2**2)
    )

    effect_figure, effect_axis = plt.subplots(figsize=FIGURE_SIZE_STANDARD)
    effect_axis.fill_between(
        effect_x,
        density_1,
        color=COLORS["orange"],
        alpha=0.5,
        label=f"μ = {mean_1:.1f}, σ = {spread_1:.1f}",
    )
    effect_axis.fill_between(
        effect_x,
        density_2,
        color=COLORS["purple"],
        alpha=0.5,
        label=f"μ = {mean_2:.1f}, σ = {spread_2:.1f}",
    )
    effect_axis.fill_between(
        effect_x,
        overlap_density,
        color=COLORS["gray"],
        alpha=0.45,
        hatch="///",
        label="Overlap",
    )
    effect_axis.axvline(mean_1, color=COLORS["orange"], linestyle="--")
    effect_axis.axvline(mean_2, color=COLORS["purple"], linestyle="--")
    effect_axis.set(
        title="Standardized separation of two normal distributions",
        xlabel="Measured quantity",
        ylabel="Probability density",
        yticks=[],
    )
    effect_axis.legend(frameon=False, fontsize=8, loc="upper left")
    effect_figure.tight_layout()

    effect_controls = mo.vstack(
        [
            delta_mu,
            sigma_1,
            sigma_2,
            mo.stat(
                f"{theoretical_d:.2f}",
                label="Cohen's d",
                bordered=True,
            ),
            mo.stat(
                f"{overlap_area * 100:.1f}%",
                label="Distribution overlap",
                bordered=True,
            ),
            mo.stat(
                f"{probability_2_higher * 100:.1f}%",
                label="P(random value 2 > random value 1)",
                bordered=True,
            ),
        ]
    )
    effect_results = mo.vstack(
        [
            effect_figure,
            mo.md(
                r"""
                For this theoretical explorer, the pooled spread is the equal-weight
                root-mean-square of the two population SDs. The overlap is calculated
                numerically for the displayed normal densities. If the SDs differ, the
                same value of $d$ need not imply the same overlap or classification
                behavior.
                """
            ),
        ]
    )
    two_column_panel(
        effect_controls,
        effect_results,
        widths=[1, 3],
    )


@app.cell
def _(mo):
    mo.callout(
        mo.md(
            r"""
            **Interpret with care.** Cohen's $d$ standardizes a difference; it does not
            decide whether the difference matters biologically. A 2 cm shift could be
            crucial for one measurement and irrelevant for another. It also does not
            express how precisely $d$ has been estimated. Those questions lead to
            confidence intervals and inferential statistics in the next chapter.
            """
        ),
        kind="warn",
    )


@app.cell
def _(mo):
    review_question_1 = mo.ui.radio(
        {
            "The measured heights changed": "data",
            "The population changed": "population",
            "Only the graphical grouping changed": "display",
            "The arithmetic mean must have changed": "mean",
        },
        value=None,
        label="Choose one answer",
    )
    mo.vstack(
        [
            mo.md(
                r"""
                ---

                ## Review questions

                Use these questions to check whether you can interpret the numerical
                and graphical summaries introduced in this chapter. Feedback appears
                as soon as you select an answer.

                ### 1. Histograms and binning

                You increase the histogram from 10 bins to 25 bins and its shape
                changes. What has changed?
                """
            ),
            review_question_1,
        ]
    )
    return (review_question_1,)


@app.cell
def _(review_feedback, review_question_1):
    review_feedback(
        review_question_1.value,
        correct_value="display",
        correct_text=(
            "**Correct.** The same observations are assigned to a different set of "
            "intervals. Numerical summaries such as mean and median are unchanged."
        ),
        incorrect_text=(
            "Not quite. Changing bin boundaries changes the representation, not the "
            "observations, target population, or arithmetic mean."
        ),
    )


@app.cell
def _(mo):
    review_question_2 = mo.ui.radio(
        {
            "Mean and standard deviation": "mean_sd",
            "Median and interquartile range": "median_iqr",
            "Median and standard deviation": "median_sd",
            "Mean and interquartile range": "mean_iqr",
        },
        value=None,
        label="Choose one answer",
    )
    mo.vstack(
        [
            mo.md(
                r"""
                ### 2. An extreme observation

                One implausibly large height is accidentally added to a dataset. Which
                pair of summaries will generally change the most?
                """
            ),
            review_question_2,
        ]
    )
    return (review_question_2,)


@app.cell
def _(review_feedback, review_question_2):
    review_feedback(
        review_question_2.value,
        correct_value="mean_sd",
        correct_text=(
            "**Correct.** The mean uses every observed magnitude, and the standard "
            "deviation gives especially large weight to distant values by squaring "
            "their deviations. The median and IQR depend mainly on ranks and are "
            "therefore more resistant."
        ),
        incorrect_text=(
            "Not quite. An extreme observation pulls the mean toward it and can "
            "substantially increase the standard deviation. Median and IQR are "
            "usually less affected."
        ),
    )


@app.cell
def _(mo):
    review_question_3 = mo.ui.radio(
        {
            "SD: cm; variance: cm": "both_cm",
            "SD: cm²; variance: cm": "reversed",
            "SD: cm; variance: cm²": "correct_units",
            "Both are unitless": "unitless",
        },
        value=None,
        label="Choose one answer",
    )
    mo.vstack(
        [
            mo.md(
                r"""
                ### 3. Units of spread

                Heights are measured in centimetres. What are the units of their
                standard deviation and variance?
                """
            ),
            review_question_3,
        ]
    )
    return (review_question_3,)


@app.cell
def _(review_feedback, review_question_3):
    review_feedback(
        review_question_3.value,
        correct_value="correct_units",
        correct_text=(
            "**Correct.** Standard deviation is expressed on the original scale "
            "(cm). Variance averages squared deviations, so its unit is cm²."
        ),
        incorrect_text=(
            "Not quite. Deviations have units of centimetres; squaring them gives "
            "variance in cm². Taking the square root returns the standard deviation "
            "to centimetres."
        ),
    )


@app.cell
def _(mo):
    review_question_4 = mo.ui.radio(
        {
            "It is doubled": "doubled",
            "It is halved": "halved",
            "It is unchanged": "unchanged",
            "It necessarily becomes zero": "zero",
        },
        value=None,
        label="Choose one answer",
    )
    mo.vstack(
        [
            mo.md(
                r"""
                ### 4. Standardized effect size

                Two groups are compared using Cohen's $d$. You convert every
                measurement from centimetres to a unit whose numerical values are
                twice as large. Both group means and both standard deviations are
                therefore doubled. What happens to Cohen's $d$?
                """
            ),
            review_question_4,
        ]
    )
    return (review_question_4,)


@app.cell
def _(review_feedback, review_question_4):
    review_feedback(
        review_question_4.value,
        correct_value="unchanged",
        correct_text=(
            "**Correct.** The difference between the means and the pooled standard "
            "deviation are multiplied by the same factor, which cancels in their "
            "ratio. Cohen's $d$ is unitless."
        ),
        incorrect_text=(
            "Not quite. Cohen's $d$ divides a mean difference by a pooled standard "
            "deviation. Multiplying every measurement by the same factor multiplies "
            "both numerator and denominator by that factor, leaving $d$ unchanged."
        ),
    )


@app.cell
def _(mo):
    mo.md(r"""
    ---

    ## Summary

    - Start with the **type and scale** of each variable; they determine meaningful
      summaries and displays.
    - Inspect raw observations and distributions before compressing them into a few
      numbers.
    - The arithmetic mean is both a balance point and the center that minimizes
      squared deviations.
    - Use a weighted mean when contributions differ for a defensible reason, and a
      geometric mean when positive data and the scientific question are naturally
      multiplicative.
    - Mean and standard deviation use every magnitude and are sensitive to extremes;
      median and IQR are more resistant.
    - Histograms, box plots, and violin plots preserve different aspects of a
      distribution. None is a complete description by itself.
    - Cohen's $d$ expresses mean separation relative to spread, but scientific
      meaning and uncertainty still have to be considered.

    **Next:** We will treat a dataset as a sample from a wider population and ask how
    accurately its summaries estimate unknown population quantities.
    """)


if __name__ == "__main__":
    app.run()
