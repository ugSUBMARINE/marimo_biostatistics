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
        responsive_row,
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
        responsive_row,
        review_feedback,
        stats,
        two_column_panel,
    )


@app.cell
def _(mo):
    mo.md(r"""
    Whether you measure enzyme activity, count microbial colonies, or compare
    protein concentrations, a table of results can be difficult to interpret.
    **Descriptive statistics** are numbers and plots that summarize the data you have.
    An **observation unit** is the entity measured, such as a person or culture;
    an **observation** records its measured values. A **variable** is what you record,
    such as height or enzyme activity. We will ask three questions:

    1. **What is a typical value?** — describe the center of the data
    2. **How much do the values differ?** — describe their spread
    3. **Which values are common or rare?** — examine the distribution, meaning
       the pattern of values and how often they occur

    In this chapter you will work with the artificial height dataset used in the
    lecture. The same ideas apply to laboratory measurements. A **sample** is the
    set of observations collected; the **population** is the wider group we want to
    understand, such as all cultures grown under specified conditions. Here the
    heights are simulated, not measurements from real students.

    The notebook is meant to be read from top to bottom, but every control
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
            "to whole students, the actual proportion is "
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

                    Each new dataset contains 91–109 simulated students (always an
                    odd number). The target percentage labeled female varies randomly,
                    usually between 45% and 65%. Heights follow bell-shaped models,
                    centered at **{female_height_mean:.0f} cm** for the female group
                    and **{male_height_mean:.0f} cm** for the male group. Their standard
                    deviations, which describe spread, are **{female_height_sd:.1f} cm**
                    and **{male_height_sd:.1f} cm**, respectively. We explain this
                    measure below. Heights are rounded to 0.1 cm. These two labels
                    define the groups in this simplified teaching dataset.
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
    variable records a measured or counted quantity. A numerical code used as a
    label, such as strain 1 or strain 2, is still categorical: averaging the codes
    would not describe a biological quantity.

    Numerical variables can be **discrete** (counts such as 0, 1, 2, ...) or
    **continuous** (measurements that can, in principle, take any value in an
    interval). The recorded height values look discrete because they were rounded
    to 0.1 cm, but the underlying quantity is continuous.

    Measurement scales add another distinction:

    - **Nominal:** categories without an order, such as bacterial species.
    - **Ordinal:** ordered categories, such as disease stages. A step from stage I
      to II need not represent the same change as a step from III to IV.
    - **Interval:** equal differences have the same meaning, but zero is a reference
      point. A rise from 10 to 20 °C equals a rise from 20 to 30 °C; 20 °C is
      not “twice as hot” as 10 °C.
    - **Ratio:** both differences and ratios have meaning. A concentration of
      20 mmol/L is twice 10 mmol/L, and zero means none of the substance is present.

    **Try this:** classify height, disease stage, and bacterial species
    in your own words before consulting the selector. Select **Height in
    centimeters**, **Disease stage I–IV**, then **Bacterial species** to compare
    the explanations. For each, ask whether ordering, differences, and ratios are
    meaningful. Then classify temperature, colony counts, and the agreement scale
    to check the remaining examples. The selector reveals the classification; it does not submit a
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

    A **histogram** groups observations into intervals called **bins**. Its appearance
    depends on the number and location of those bins, so the data alone do not
    uniquely determine its appearance.

    On a **count scale**, bar height is the number of observations in a bin. On a
    **density scale**, bar area (height × width) is the fraction of observations in
    that bin. For example, an area of 0.20 means 20% of the observations. All bar
    areas together add to 1. Bar height alone is not a percentage.

    A **frequency polygon** joins the heights at the centers of the bins with lines.
    The **normal curve** is a smooth, symmetric bell shape. Here it is fitted by
    matching its mean (average) and standard deviation (SD, a measure of spread)
    to the displayed data.

    **Try this:** keep **All students** selected and compare 5, 15, and 30 bins.
    Watch the shape change while the summary statistics stay fixed. Switch to
    **Use density scale** and read the vertical axis again: area now represents
    the fraction of observations. Toggle the frequency polygon and fitted normal
    curve to compare the binned data with a smooth model of the observations.
    Finally compare **Female** and **Male**: this changes which data enter the
    histogram and summary table. A model overlay is a comparison, not evidence
    that the population must be normal.

    The table previews terms explained in the next sections: the **median** is the
    middle value after sorting; **quartiles** mark roughly 25%, 50%, and 75% of the
    ordered data; the **range** is maximum minus minimum; the **interquartile range
    (IQR)** is the width of the middle half of the data.

    The **coefficient of variation (CV)** is 100 × SD / mean. For example, a mean
    enzyme activity of 100 U/L and SD of 10 U/L give a CV of 10%. It describes
    spread relative to the mean and is useful for positive measurements with a
    meaningful zero. It is unsuitable for Celsius temperatures and unstable when
    the mean is close to zero. Here CV uses the SD calculated by dividing by the
    number of observations, $n$. The table calls this “population SD”; this formula
    describes the displayed values and does not mean they represent everyone.
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
    normal_overlay = mo.ui.checkbox(value=False, label="Overlay fitted normal curve")
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
        range=(min(145, height_subset.min()), max(200, height_subset.max())),
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
        normal_x = np.linspace(histogram_edges[0] - 5, histogram_edges[-1] + 5, 400)
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
        xlim=(histogram_edges[0] - 5, histogram_edges[-1] + 5),
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
                    two subgroups: combining groups with different typical values
                    can produce a shape unlike either group alone. The same can happen
                    when measurements from different culture conditions are combined.
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

    The **arithmetic mean**, usually called the mean or average, is the sum of all
    values divided by how many there are. We write each value as $x_i$, the number
    of values as $n$, and the mean as $\bar{x}$ (“x bar”). The symbol $\sum$ means
    “add up”:

    \[
    \bar{x}=\frac{x_1+x_2+\cdots+x_n}{n}
           =\frac{1}{n}\sum_{i=1}^{n}x_i.
    \]

    A **deviation** is a value minus a chosen center. It is positive above the
    center and negative below it. At the mean, these positive and negative
    differences add to zero: the mean acts as a balance point.

    The mean also gives the smallest possible sum of **squared deviations**:
    subtract the center from each value, square each difference, then add them.
    Squaring makes all contributions non-negative and gives more weight to large
    differences. This way of finding a center is called **least squares**.

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
    ### Optional mathematical detail: why does the mean give the smallest sum?

    You can follow the interactive example without using calculus. Let $c$ be any
    proposed center. Add up the squared differences from that center:

    \[
    S(c)=\sum_{i=1}^{n}(x_i-c)^2.
    \]

    The derivative describes how $S(c)$ changes when we move $c$. At the bottom
    of this smooth U-shaped curve, its slope is zero:

    \[
    \begin{aligned}
    \frac{dS}{dc}
      &= -2\sum_{i=1}^{n}(x_i-c)=0\\
    \sum_{i=1}^{n}x_i-nc&=0\\
    c&=\frac{1}{n}\sum_{i=1}^{n}x_i=\bar{x}.
    \end{aligned}
    \]

    The curve opens upward, so this point is its single minimum. The best center
    by this rule is therefore the arithmetic mean. Move the candidate center below
    and find the bottom of the curve.
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

    The mean depends on the size of every value. To find the **median**, sort the
    values from smallest to largest and take the middle one. With an even number
    of values, average the two middle ones. At least half the values are at or
    below the median, and at least half are at or above it.

    The mean and median are often close when the distribution is **symmetric**,
    with similar shapes on both sides of its center. A **tail** is the part that
    extends toward unusually low or high values. A long tail toward high values
    (a **right-skewed** distribution) often pulls the mean above the median.
    A long tail toward low values often pulls it below. For example, a few cultures
    with very high protein yields can raise the mean substantially while barely
    changing the median. Choose the summary that answers your scientific question:
    the mean describes the amount per observation when sharing the total equally;
    the median describes the middle observation.

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
        value=170,
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
            "Statistic": ["Mean", "Median", "Population SD", "Range"],
            "Original": [
                original_values.mean(),
                np.median(original_values),
                original_values.std(ddof=0),
                np.ptp(original_values),
            ],
            "With added value": [
                augmented_values.mean(),
                np.median(augmented_values),
                augmented_values.std(ddof=0),
                np.ptp(augmented_values),
            ],
        }
    )
    outlier_comparison["Change"] = (
        outlier_comparison["With added value"] - outlier_comparison["Original"]
    )
    outlier_axis_min = min(130, original_values.min() - 5)
    outlier_axis_max = max(
        235, original_values.max() + 5, hypothetical_height.value + 15
    )

    outlier_figure, outlier_axis = plt.subplots(figsize=FIGURE_SIZE_COMPACT)
    outlier_axis.hist(
        original_values,
        bins=18,
        range=(outlier_axis_min, max(235, original_values.max() + 5)),
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
        xlim=(outlier_axis_min, outlier_axis_max),
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
            most frequent value or values occur **{highest_recorded_frequency}**
            time(s): **{recorded_mode_text} cm**.

            Rounding heights to 0.1 cm can make different measurements have the same
            recorded value. Changing the rounding can therefore change the mode.
            The tallest histogram bar identifies the most frequent interval (the
            **modal bin**), but this also depends on the chosen bin boundaries.
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
    equally and combine **additively**. Other averages are
    useful for different questions and different ways of combining measurements.

    ### The weighted arithmetic mean

    A weighted mean allows observations or group summaries to make unequal, but
    explicitly justified, contributions. A **weight**, $w_i$, sets how much value
    $x_i$ contributes. Multiply each value by its weight, add these products, then
    divide by the sum of the weights. At least one weight must be positive:

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

    - **Combining results from several studies.** A meta-analysis combines estimates
      from separate studies. More precise (less uncertain) estimates may
      receive more weight. Choosing these weights requires information about each
      estimate's uncertainty, which we address in later chapters.
    - **Unequal contributions to a pooled sample.** Mean concentrations from aliquots
      of different volumes can be weighted by volume when the target is the
      concentration of the combined material, provided volumes add and the substance
      is not lost or produced on mixing. For example, mixing 1 mL at 2 mmol/L with
      3 mL at 6 mmol/L gives $(1 \times 2 + 3 \times 6)/4 = 5$ mmol/L.

    Choose weights based on how the experiment was designed and what you want to
    measure. Weighting cannot correct a systematic measurement error or make
    measurements from incompatible conditions comparable.
    """)


@app.cell
def _(mo):
    mo.md(r"""
    ### The geometric mean

    The **geometric mean** summarizes values that combine by multiplication, such
    as successive growth factors. For positive values,
    multiply the $n$ values and take the $n$th root. The symbol $\prod$ means
    “multiply together”:

    \[
    \bar{x}_{\mathrm{geom}}
      =\left(\prod_{i=1}^{n}x_i\right)^{1/n}
      =\exp\left(\frac{1}{n}\sum_{i=1}^{n}\log x_i\right).
    \]

    An equivalent calculation takes the natural logarithm ($\log$ or $\ln$) of
    each value, averages these logarithms, and applies $\exp$ to return to the
    original scale. Logarithms turn multiplication into addition; $\exp$ reverses
    this transformation. A **fold change** is a ratio: two-fold means twice as large.
    The geometric mean balances equal fold changes above and below it. For example, the geometric mean
    of 5 and 20 is 10 because one value is half of 10 and the other is twice 10. Their
    arithmetic mean, 12.5, answers a different, additive question.

    This distinction is especially useful when positive measurements are approximately
    **log-normally distributed**: their logarithms follow a symmetric, bell-shaped
    normal distribution. We write this as:

    \[
    \log(X)\sim N(\mu,\sigma^2).
    \]

    Here $X$ is a positive measurement, $\sim$ means “follows this distribution”,
    and $N(\mu,\sigma^2)$ denotes a normal distribution with mean $\mu$ (“mu”)
    and variance $\sigma^2$, so its standard deviation is $\sigma$ (“sigma”),
    **on the logarithmic scale**.

    On the original scale, these data have a long right tail and may span several
    **orders of magnitude**. For an exactly log-normal
    population, the geometric mean is $\exp(\mu)$ and equals the median. The
    arithmetic mean is larger, $\exp(\mu+\sigma^2/2)$, because high values in the
    tail contribute strongly. These formulas describe the theoretical model below;
    real measurements need not follow it exactly.
    """)


@app.cell
def _(mo):
    mo.md(r"""
    #### Explore multiplicative variation

    These curves describe a **theoretical population**: an idealized model of all
    possible measurements, rather than a finite set of observations. Express $X$
    in a fixed reference unit before taking its logarithm.

    **Try this:** hold μ fixed and increase σ. The geometric mean stays fixed while
    the arithmetic mean rises as the right tail grows. Then increase μ: both means
    are multiplied by the same factor. The axes rescale with the controls.
    """)


@app.cell
def _(mo):
    lognormal_mu = mo.ui.slider(
        -1, 3, step=0.1, value=1, show_value=True, label="Log-scale mean μ"
    )
    lognormal_sigma = mo.ui.slider(
        0.1, 1.5, step=0.1, value=0.6, show_value=True, label="Log-scale SD σ"
    )
    return lognormal_mu, lognormal_sigma


@app.cell
def _(lognormal_mu, lognormal_sigma, np, stats):
    lognormal_geometric = float(np.exp(lognormal_mu.value))
    lognormal_arithmetic = float(
        np.exp(lognormal_mu.value + lognormal_sigma.value**2 / 2)
    )
    lognormal_ratio = lognormal_arithmetic / lognormal_geometric
    lognormal_population = stats.lognorm(
        s=lognormal_sigma.value, scale=lognormal_geometric
    )
    return (
        lognormal_arithmetic,
        lognormal_geometric,
        lognormal_population,
        lognormal_ratio,
    )


@app.cell
def _(
    COLORS,
    FIGURE_SIZE_COMPACT,
    lognormal_arithmetic,
    lognormal_geometric,
    lognormal_mu,
    lognormal_population,
    lognormal_ratio,
    lognormal_sigma,
    mo,
    np,
    plt,
    responsive_row,
    stats,
):
    lognormal_raw_figure, lognormal_raw_axis = plt.subplots(figsize=FIGURE_SIZE_COMPACT)
    # A log-spaced grid resolves the peak while the displayed raw axis stays linear.
    lognormal_raw_x = np.geomspace(
        lognormal_population.ppf(0.0001), lognormal_population.ppf(0.99), 1200
    )
    lognormal_raw_axis.fill_between(
        lognormal_raw_x,
        lognormal_population.pdf(lognormal_raw_x),
        color=COLORS["sky"],
        alpha=0.5,
    )
    lognormal_raw_axis.axvline(
        lognormal_geometric, color=COLORS["green"], label="Geometric mean / median"
    )
    lognormal_raw_axis.axvline(
        lognormal_arithmetic,
        color=COLORS["vermillion"],
        linestyle="--",
        label="Arithmetic mean",
    )
    lognormal_raw_axis.set(
        title="Original scale: X",
        xlabel="Positive measurement (reference units)",
        ylabel="Density of X",
        xlim=(0, lognormal_raw_x[-1]),
        ylim=(0, None),
    )
    lognormal_raw_axis.legend(frameon=False, fontsize=8)
    lognormal_raw_figure.tight_layout()

    lognormal_log_figure, lognormal_log_axis = plt.subplots(figsize=FIGURE_SIZE_COMPACT)
    lognormal_log_x = np.linspace(
        lognormal_mu.value - 3.5 * lognormal_sigma.value,
        lognormal_mu.value + 3.5 * lognormal_sigma.value,
        600,
    )
    lognormal_log_axis.fill_between(
        lognormal_log_x,
        stats.norm.pdf(
            lognormal_log_x, loc=lognormal_mu.value, scale=lognormal_sigma.value
        ),
        color=COLORS["sky"],
        alpha=0.5,
    )
    lognormal_log_axis.axvline(
        lognormal_mu.value, color=COLORS["green"], label="Mean of ln(X) = μ"
    )
    lognormal_log_axis.set(
        title="Transformed scale: ln(X)",
        xlabel="Natural logarithm of measurement",
        ylabel="Density of ln(X)",
        xlim=(lognormal_log_x[0], lognormal_log_x[-1]),
        ylim=(0, None),
    )
    lognormal_log_axis.legend(frameon=False, fontsize=8)
    lognormal_log_figure.tight_layout()

    mo.vstack(
        [
            responsive_row([lognormal_mu, lognormal_sigma]),
            responsive_row([lognormal_raw_figure, lognormal_log_figure], min_width=20),
            responsive_row(
                [
                    mo.stat(
                        f"{lognormal_geometric:.2f}",
                        label="Population geometric mean",
                        bordered=True,
                    ),
                    mo.stat(
                        f"{lognormal_arithmetic:.2f}",
                        label="Population arithmetic mean",
                        bordered=True,
                    ),
                    mo.stat(
                        f"{lognormal_ratio:.2f}×",
                        label="Arithmetic / geometric mean",
                        bordered=True,
                    ),
                ]
            ),
            mo.md(
                rf"""
                The mean of $\ln(X)$ is **{lognormal_mu.value:.1f}**;
                back-transforming gives $\exp(\mu)$ = **{lognormal_geometric:.2f}**.
                The arithmetic mean is $\exp(\mu+\sigma^2/2)$, so the ratio is
                $\exp(\sigma^2/2)$ = **{lognormal_ratio:.2f}**.
                One log-scale SD corresponds to multiplying or dividing the geometric
                mean by $\exp(\sigma)$ = **{np.exp(lognormal_sigma.value):.2f}**.
                This is multiplicative variation, as in positive concentrations or
                fold changes.

                Both plots show the same population. Their density heights differ
                because the horizontal units differ. The raw view ends at the 99th
                percentile (the rightmost 1% continues beyond it); the means use
                the **entire** distribution. As σ approaches zero, the two means
                approach one another.
                """
            ),
        ]
    )


@app.cell
def _(mo):
    mo.vstack(
        [
            mo.md(
                r"""
                #### Biological measurements for which a geometric mean may be useful

                - **Repeated measurements of binding or response concentrations.**
                  Examples include the dissociation constant $K_d$, inhibition
                  constant $K_i$, and concentrations giving 50% inhibition or half
                  the maximal response ($IC_{50}$ and $EC_{50}$). When repeated
                  estimates under comparable conditions vary mainly by fold
                  differences, a geometric mean can be a useful summary.
                - **Gene-expression fold changes and other response ratios.** A
                  two-fold increase and a two-fold decrease are symmetric on a log
                  scale; their geometric mean fold change is 1, indicating no typical
                  multiplicative change.
                - **Microbial counts, viral loads, antibody titres, metabolite or
                  hormone concentrations, fluorescence intensities, and enzyme
                  activities** when all values are positive and their logarithms
                  have an approximately bell-shaped distribution.
                - **Growth factors or repeated proportional changes.** If a quantity
                  changes by successive multiplicative factors, the geometric mean of
                  those factors describes the typical factor per step.

                Consider three $K_d$ estimates: 1, 10, and 100 nM. They are evenly
                spaced on a logarithmic scale. Their geometric mean is 10 nM, whereas
                their arithmetic mean is 37 nM. The geometric mean captures the middle
                value on the logarithmic scale; the arithmetic mean gives more
                weight to the largest estimate. Neither calculation alone tells us
                how accurately the binding constant has been measured.
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
                    constant (sometimes called a **pseudocount**) changes the result.
                    A value below the detection limit means its exact value is unknown,
                    not that it is zero. Such data need methods that account for what
                    the assay can detect. A long right tail alone is not a sufficient
                    reason to use a geometric mean.
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

    A **quantile** is a cutoff in the ordered data. For example, the 0.90 quantile
    is a value at or below which roughly 90% of observations lie. It is also called
    the **90th percentile**: percentiles express the same idea using percentages.
    In $Q_{\alpha}$, $\alpha$ (“alpha”) specifies the fraction, such as 0.90.

    The **quartiles** divide the ordered data into four roughly equal parts. The
    first quartile, Q1 or $Q_{0.25}$, marks 25%; the median, Q2 or $Q_{0.5}$,
    marks 50%; and the third quartile, Q3 or $Q_{0.75}$, marks 75%.

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
    exactly. **Interpolation** means estimating a cutoff between two neighboring
    recorded values; **ties** are equal recorded values. Different calculation
    conventions can give slightly different quantiles for small datasets.
    The selected group from the histogram activity is used here too.
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
    quantile_bounds = (
        min(140, height_subset.min() - 5),
        max(205, height_subset.max() + 5),
    )

    quantile_figure, quantile_axis = plt.subplots(figsize=FIGURE_SIZE_SHALLOW)
    quantile_axis.hist(
        height_subset,
        bins=18,
        range=quantile_bounds,
        density=True,
        color=COLORS["orange"],
        edgecolor="white",
        alpha=0.7,
    )
    quantile_axis.axvspan(
        quantile_bounds[0],
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
        xlim=quantile_bounds,
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
                "The percentage shown counts the actual values at or below the cutoff. "
                "It may differ from the selected percentage because values can be tied "
                "and the cutoff can fall between recorded values."
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

    **Variance** describes spread by averaging the squared differences from the
    mean. For the displayed set of $n$ values, we divide by $n$. The **standard
    deviation (SD)** is the square root of this variance. Here $\sigma$ describes
    the displayed values as a complete finite population; in Chapter 2 it denotes
    the SD of the wider population we want to estimate:

    \[
    \sigma^2=\frac{1}{n}\sum_{i=1}^{n}(x_i-\bar{x})^2,
    \qquad
    \sigma=\sqrt{\sigma^2}.
    \]

    Squaring prevents positive and negative deviations from cancelling. Variance is
    measured in squared units (cm²); the square root returns the standard deviation
    to the original unit (cm). SD describes the size of the differences from the
    mean, but it is not the simple average of their absolute sizes. A small SD means
    values cluster closely around the mean; a large SD means they are more spread
    out. In laboratory data this spread can reflect both biological differences
    and measurement variation. SD alone cannot separate these sources.

    This worked table reuses the observations selected in Section 3. Changing
    that section's example sample size or the selected height group updates it.
    """)


@app.cell
def _(compact_table, mean_example_values, mo, pd, responsive_row):
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
            responsive_row(
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
            ),
            mo.callout(
                mo.md(
                    "Here we are **describing these displayed values**, so the divisor is "
                    "n. When using an independent random sample to estimate variance in a "
                    "wider population, we usually divide by n − 1 instead. This corrects "
                    "the tendency to underestimate variance when the same data provide "
                    "both the mean and the deviations. The square root gives the sample "
                    "SD shown earlier. We explain this adjustment in the next chapter."
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

    A **box plot** shows the middle half of the values in a box from Q1 to Q3,
    with a line at the median. In this plot, the **whiskers** extend to the most
    extreme observed values still within $Q1 - 1.5\,\mathrm{IQR}$ and
    $Q3 + 1.5\,\mathrm{IQR}$. Values beyond these limits are marked separately as
    possible **outliers**, unusually distant observations. They may reflect errors
    or real biological variation; the rule alone is not a reason to remove them.

    A **violin plot** shows a smoothed distribution: wider regions indicate where
    values are more concentrated. With few observations, apparent bumps can depend
    strongly on the smoothing settings. Keep the individual points visible to
    check what the summary is based on. Here points are shifted slightly sideways
    so they do not all overlap; only their vertical positions represent heights.

    **Try this:** begin with **Raw points**, switch to **Raw points + box plot**,
    then to **Raw points + violin plot**. The same observations remain visible
    in every view. Locate the center and spread of each group, then ask which
    detail the added summary
    makes easier to see and which it conceals. A box contains the middle half of
    the observations; a violin's width shows where values are concentrated. It does
    not show how precisely the population mean is known. The box plot also marks
    the mean with a triangle; the violin plot marks the median with a horizontal line.
    """)


@app.cell
def _(mo):
    distribution_view = mo.ui.radio(
        ["Raw points", "Raw points + box plot", "Raw points + violin plot"],
        value="Raw points",
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

    A difference between group means is easier to interpret when we also know how
    much the observations vary within each group. **Cohen's $d$** expresses the
    difference in units of a shared standard deviation. For two samples:

    \[
    d=\frac{\bar{x}_2-\bar{x}_1}{s_{\mathrm{pooled}}}.
    \]

    A value of $d=1$ means the means are one shared SD apart; $d=0$ means they are
    equal, although the groups can still differ in spread or shape. The sign
    indicates which group has the higher mean and reverses if we swap group order.

    The **pooled SD** combines the variation *within* the two groups; it is not
    the SD of all observations mixed together. With group sizes $n_1$ and $n_2$
    and sample SDs $s_1$ and $s_2$, a common formula is

    \[
    s_{\mathrm{pooled}}
      =\sqrt{\frac{(n_1-1)s_1^2+(n_2-1)s_2^2}{n_1+n_2-2}}.
    \]

    The formula combines the squared SDs, giving larger samples more weight, then
    takes the square root. A shared SD is most straightforward to interpret when
    the two groups have similar spreads. Because the mean difference and SD have
    the same unit, their units cancel: $d$ is **unitless**. Still report the
    difference in the original units, such as U/L of enzyme activity, so readers
    can judge its biological importance.

    **Try the model explorer below:** set both SDs to 7.5 and compare mean
    differences of 0, 15, and 30. Watch $d$ and the shaded overlap between the
    curves. The third readout gives the probability that an independently drawn
    value from group 2 exceeds one from group 1. It compares individual values,
    not the probability that one group mean is greater than the other. Restore the
    difference to 15, then increase both SDs to 15: the same absolute difference becomes less distinct
    relative to variation. Finally change just one SD to explore unequal spreads.
    These sliders alter two theoretical normal distributions; they do not change
    the comparison calculated from the height data above the controls. In the plot
    legend, $\mu$ denotes a model's mean and $\sigma$ its SD.
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

            The means are **{abs(observed_d):.2f} pooled SDs** apart. This describes
            the current dataset; it does not tell us how much the estimate would
            change if we collected another sample.
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
                These curves are models with no sample sizes. Here the shared spread
                is $\sqrt{(\sigma_1^2+\sigma_2^2)/2}$: square the two SDs, average
                them with equal weight, then take the square root. The shaded overlap
                is the area under the lower of the two curves, expressed as a
                percentage. If the SDs differ, the same $d$ can correspond to different
                amounts of overlap.
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
            **Interpret with care.** Cohen's $d$ describes the size of a mean
            difference relative to spread. It does not decide whether that difference
            matters biologically: a small change in enzyme activity may matter near
            a functional threshold, even if $d$ is small. Nor does $d$ alone tell us
            how precisely the difference is known. The next chapter introduces
            **inferential statistics** (using a sample to learn about a wider
            population) and ways to describe uncertainty in our estimates.
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

                Keeping the observations and selected group fixed, you increase
                the histogram from 10 bins to 25 bins and its shape
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
            {
                "data": "Binning only assigns the existing heights to intervals; it does not change a recorded "
                "height. Keep the dataset and selected group fixed and compare the unchanged summary "
                "table.",
                "population": "The target population is determined by whom the observations represent, not by the "
                "plot settings. Changing bins changes the display of the same sample.",
                "mean": "The arithmetic mean is calculated from the individual heights, not the histogram bars. "
                "With the same observations, their sum and count stay fixed when the bins change.",
            }
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

                One implausibly large height is accidentally added to a dataset.
                Which pair contains two summaries that are sensitive to how far
                this observation lies from the rest?
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
            "**Correct.** The mean uses the size of every value, and the standard "
            "deviation gives especially large weight to distant values by squaring "
            "their deviations. The median and IQR depend mainly on positions in the sorted data and are "
            "therefore more resistant."
        ),
        incorrect_text=(
            {
                "median_iqr": "Both the median and IQR are resistant to one extreme value: they depend on "
                "positions in the ordered data. They may shift when a value is added, but do not "
                "grow without bound as that one value increases.",
                "median_sd": "SD is sensitive to the extreme value, but the median is resistant because it is "
                "determined by the middle positions in the sorted data. The mean is the sensitive measure of center to "
                "pair with SD here.",
                "mean_iqr": "The mean is sensitive to the extreme value, but the IQR describes the middle "
                "half of the ordered observations. SD, which uses every squared deviation, is the "
                "sensitive measure of spread here.",
            }
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
            {
                "both_cm": "SD is in cm, but variance averages squared deviations. Each squared deviation has "
                "units cm², and averaging does not remove those units.",
                "reversed": "These units are reversed: squaring deviations produces variance in cm²; taking its "
                "square root gives SD in cm.",
                "unitless": "Dividing by n or n − 1 removes no measurement units. Variance remains in cm² and SD "
                "in cm. Ratios such as CV and Cohen’s d are unitless because matching units cancel.",
            }
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
            {
                "doubled": "The mean difference doubles, but so does the pooled SD in the denominator. Thus (2 × "
                "difference)/(2 × pooled SD) equals the original d.",
                "halved": "The denominator doubles, but the numerator doubles too. Halving d would require "
                "changing only the denominator, not converting all observations to a new unit.",
                "zero": "Changing units does not remove a difference between the groups. The factor of two "
                "cancels in d; it is zero only if the original mean difference was zero.",
            }
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
    - Use a weighted mean when the experiment justifies unequal contributions.
      Consider a geometric mean when positive values combine by multiplication
      and fold differences answer the scientific question.
    - Mean and standard deviation use the size of every value and are sensitive
      to extremes; median and IQR are more resistant.
    - Histograms, box plots, and violin plots preserve different aspects of a
      distribution. None is a complete description by itself.
    - Cohen's $d$ expresses the difference between means relative to spread.
      Biological importance and uncertainty still need separate consideration.

    **Next:** Chapter 2 treats a dataset as a sample from a wider population and
    asks how precisely its summaries estimate unknown population quantities.
    """)


if __name__ == "__main__":
    app.run()
