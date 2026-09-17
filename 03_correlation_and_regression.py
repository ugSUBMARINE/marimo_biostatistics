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
    app_title="Correlation and regression",
    css_file="site/assets/notebook.css",
)


@app.cell
def chapter_header():
    from companion_style import notebook_header

    notebook_header(
        3,
        "Correlation and regression",
        "Learning from paired measurements of river water quality",
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
    from scipy import optimize, stats

    from companion_data import read_csv
    from companion_style import (
        COLORS,
        FIGURE_SIZE_LINKED,
        FIGURE_SIZE_STANDARD,
        compact_table,
        review_feedback,
        two_column_panel,
    )

    return (
        COLORS,
        FIGURE_SIZE_LINKED,
        FIGURE_SIZE_STANDARD,
        compact_table,
        np,
        optimize,
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
    A single variable has a distribution. When two measurements are recorded on the
    same observation unit, we can also ask whether and how they vary together. This
    chapter develops two related but different tools:

    - **correlation** summarizes the direction and strength of an association;
    - **regression** specifies a directional model for a response conditional on one
      or more predictors.

    By the end of the chapter, you should be able to:

    1. interpret covariance, Pearson correlation, and Spearman rank correlation;
    2. explain why every correlation must be inspected together with a plot;
    3. quantify uncertainty in a correlation with Fisher's transformation or a
       paired bootstrap;
    4. interpret slopes, residuals, $R^2$, confidence bands, and prediction
       intervals;
    5. distinguish statistical adjustment from a causal claim about confounding.

    > **Two-minute preview:** Correlation is symmetric and unitless. Regression is
    > directional and retains the units of the response and predictor. Both can be
    > distorted by unusual observations, restricted ranges, nonlinear structure,
    > and omitted variables, so the scatter plot always comes first.
    """)


@app.cell
def _(mo, np, pd, read_csv, stats):
    water_data_path = mo.notebook_location() / "public" / "Wasserqualitaet.csv"
    raw_water_data = read_csv(water_data_path, encoding="utf-8-sig")

    german_to_english = {
        "Nummer des Flusses": "river_id",
        "Temperatur (°C)": "temperature_c",
        "Fließgeschwindigkeit (m/s)": "flow_speed_m_s",
        "Sauerstoffkonzentration (mg/l)": "oxygen_mg_l",
        "Nitratkonzentration (mg/l)": "nitrate_mg_l",
        "Phosphatkonzentration (mug/l)": "phosphate_ug_l",
        "Wassergüte (1-20)": "water_quality_score",
        "Besiedlungsdichte (1-20)": "population_density_score",
        "Entfernung von der Quelle (km)": "distance_km",
        "Vorkommen der Flussnapfschnecke (0/1)": "river_snail_present",
        "Eutrophierungszustand (0/1)": "eutrophic",
    }
    water_data = raw_water_data.rename(columns=german_to_english).copy()

    variable_labels = {
        "flow_speed_m_s": "Flow speed [m/s]",
        "oxygen_mg_l": "Oxygen concentration [mg/L]",
        "temperature_c": "Water temperature [°C]",
        "nitrate_mg_l": "Nitrate concentration [mg/L]",
        "phosphate_ug_l": "Phosphate concentration [µg/L]",
        "distance_km": "Distance from source [km]",
        "water_quality_score": "Water-quality score [1–20]",
        "population_density_score": "Population-density score [1–20]",
    }

    def as_float_array(values):
        array = np.asarray(values, dtype=float)
        if array.ndim != 1 or array.size < 3:
            raise ValueError("values must be a one-dimensional sample of size >= 3")
        if not np.isfinite(array).all():
            raise ValueError("values must be finite")
        return array

    def pearson_details(x_values, y_values):
        x_array = as_float_array(x_values)
        y_array = as_float_array(y_values)
        if x_array.shape != y_array.shape:
            raise ValueError("paired variables must have the same shape")
        x_centered = x_array - x_array.mean()
        y_centered = y_array - y_array.mean()
        contributions = x_centered * y_centered
        covariance = float(contributions.sum() / (x_array.size - 1))
        correlation = float(covariance / (x_array.std(ddof=1) * y_array.std(ddof=1)))
        return {
            "x": x_array,
            "y": y_array,
            "x_mean": float(x_array.mean()),
            "y_mean": float(y_array.mean()),
            "x_centered": x_centered,
            "y_centered": y_centered,
            "contributions": contributions,
            "covariance": covariance,
            "correlation": correlation,
        }

    def fisher_correlation_interval(correlation, sample_size, confidence=0.95):
        if sample_size <= 3:
            raise ValueError("Fisher interval requires sample_size > 3")
        clipped = float(np.clip(correlation, -0.999999, 0.999999))
        alpha = 1.0 - confidence
        critical = float(stats.norm.ppf(1.0 - alpha / 2.0))
        transformed = float(np.arctanh(clipped))
        margin = critical / np.sqrt(sample_size - 3)
        return tuple(np.tanh([transformed - margin, transformed + margin]))

    def paired_bootstrap_correlation(x_values, y_values, repetitions, rng):
        x_array = as_float_array(x_values)
        y_array = as_float_array(y_values)
        if x_array.shape != y_array.shape:
            raise ValueError("paired variables must have the same shape")
        indices = rng.integers(0, x_array.size, size=(int(repetitions), x_array.size))
        x_resamples = x_array[indices]
        y_resamples = y_array[indices]
        x_centered = x_resamples - x_resamples.mean(axis=1, keepdims=True)
        y_centered = y_resamples - y_resamples.mean(axis=1, keepdims=True)
        numerators = np.sum(x_centered * y_centered, axis=1)
        denominators = np.sqrt(
            np.sum(x_centered**2, axis=1) * np.sum(y_centered**2, axis=1)
        )
        correlations = numerators / denominators
        return correlations[np.isfinite(correlations)]

    def simulate_correlation_sampling(
        population_correlation, sample_size, repetitions, rng
    ):
        first = rng.normal(size=(int(repetitions), int(sample_size)))
        independent = rng.normal(size=(int(repetitions), int(sample_size)))
        second = (
            population_correlation * first
            + np.sqrt(1.0 - population_correlation**2) * independent
        )
        first_centered = first - first.mean(axis=1, keepdims=True)
        second_centered = second - second.mean(axis=1, keepdims=True)
        correlations = np.sum(first_centered * second_centered, axis=1) / np.sqrt(
            np.sum(first_centered**2, axis=1) * np.sum(second_centered**2, axis=1)
        )
        correlations = np.clip(correlations, -0.999999, 0.999999)
        return {
            "correlations": correlations,
            "fisher_z": np.arctanh(correlations),
            "population_correlation": float(population_correlation),
            "sample_size": int(sample_size),
            "repetitions": int(repetitions),
        }

    def generate_point_cloud(slope, noise_sd, sample_size, seed=82431):
        rng = np.random.default_rng(seed)
        x_values = rng.uniform(-2.5, 2.5, int(sample_size))
        y_values = slope * x_values + rng.normal(0.0, noise_sd, int(sample_size))
        return x_values, y_values

    def simple_regression(x_values, y_values):
        x_array = as_float_array(x_values)
        y_array = as_float_array(y_values)
        if x_array.shape != y_array.shape:
            raise ValueError("paired variables must have the same shape")
        x_centered = x_array - x_array.mean()
        y_centered = y_array - y_array.mean()
        slope = float(np.sum(x_centered * y_centered) / np.sum(x_centered**2))
        intercept = float(y_array.mean() - slope * x_array.mean())
        fitted = intercept + slope * x_array
        residuals = y_array - fitted
        total_sum_squares = float(np.sum(y_centered**2))
        regression_sum_squares = float(np.sum((fitted - y_array.mean()) ** 2))
        error_sum_squares = float(np.sum(residuals**2))
        r_squared = float(1.0 - error_sum_squares / total_sum_squares)
        residual_sd = float(np.sqrt(error_sum_squares / (x_array.size - 2)))
        return {
            "x": x_array,
            "y": y_array,
            "slope": slope,
            "intercept": intercept,
            "fitted": fitted,
            "residuals": residuals,
            "sst": total_sum_squares,
            "ssr": regression_sum_squares,
            "sse": error_sum_squares,
            "r_squared": r_squared,
            "residual_sd": residual_sd,
        }

    def regression_intervals(model, x_grid, confidence=0.95):
        x_values = model["x"]
        x_grid_array = np.asarray(x_grid, dtype=float)
        sample_size = x_values.size
        x_mean = x_values.mean()
        x_sum_squares = np.sum((x_values - x_mean) ** 2)
        critical = float(
            stats.t.ppf(1.0 - (1.0 - confidence) / 2.0, df=sample_size - 2)
        )
        fitted_grid = model["intercept"] + model["slope"] * x_grid_array
        mean_se = model["residual_sd"] * np.sqrt(
            1.0 / sample_size + (x_grid_array - x_mean) ** 2 / x_sum_squares
        )
        prediction_se = model["residual_sd"] * np.sqrt(
            1.0 + 1.0 / sample_size + (x_grid_array - x_mean) ** 2 / x_sum_squares
        )
        return {
            "x": x_grid_array,
            "fitted": fitted_grid,
            "mean_lower": fitted_grid - critical * mean_se,
            "mean_upper": fitted_grid + critical * mean_se,
            "prediction_lower": fitted_grid - critical * prediction_se,
            "prediction_upper": fitted_grid + critical * prediction_se,
        }

    def residualize(values, adjustment):
        adjustment_model = simple_regression(adjustment, values)
        return adjustment_model["residuals"]

    def partial_correlation(x_values, y_values, adjustment):
        x_residuals = residualize(x_values, adjustment)
        y_residuals = residualize(y_values, adjustment)
        residual_correlation = pearson_details(x_residuals, y_residuals)["correlation"]
        return residual_correlation, x_residuals, y_residuals

    def make_diagnostic_dataset(pattern):
        rng = np.random.default_rng(82437)
        x_values = np.linspace(0.0, 10.0, 50)
        if pattern == "Linear with constant variance":
            y_values = 2.0 + 0.8 * x_values + rng.normal(0.0, 0.9, x_values.size)
        elif pattern == "Curvature":
            y_values = (
                2.0
                + 0.8 * x_values
                + 0.16 * (x_values - 5.0) ** 2
                + rng.normal(0.0, 0.7, x_values.size)
            )
        elif pattern == "Increasing variance":
            y_values = (
                2.0
                + 0.8 * x_values
                + rng.normal(0.0, 0.2 + 0.22 * x_values, x_values.size)
            )
        elif pattern == "Two clusters":
            group = x_values >= 5.0
            y_values = (
                2.0
                + 0.55 * x_values
                + 2.5 * group
                + rng.normal(0.0, 0.55, x_values.size)
            )
        elif pattern == "Influential outlier":
            y_values = 2.0 + 0.8 * x_values + rng.normal(0.0, 0.7, x_values.size)
            y_values[-1] += 8.0
        else:
            raise ValueError(f"Unknown residual pattern: {pattern}")
        return x_values, y_values

    anscombe_x = np.array([10, 8, 13, 9, 11, 14, 6, 4, 12, 7, 5], dtype=float)
    anscombe_data = {
        "I": (
            anscombe_x,
            np.array(
                [8.04, 6.95, 7.58, 8.81, 8.33, 9.96, 7.24, 4.26, 10.84, 4.82, 5.68]
            ),
        ),
        "II": (
            anscombe_x,
            np.array(
                [9.14, 8.14, 8.74, 8.77, 9.26, 8.10, 6.13, 3.10, 9.13, 7.26, 4.74]
            ),
        ),
        "III": (
            anscombe_x,
            np.array(
                [7.46, 6.77, 12.74, 7.11, 7.81, 8.84, 6.08, 5.39, 8.15, 6.42, 5.73]
            ),
        ),
        "IV": (
            np.array([8, 8, 8, 8, 8, 8, 8, 19, 8, 8, 8], dtype=float),
            np.array(
                [6.58, 5.76, 7.71, 8.84, 8.47, 7.04, 5.25, 12.50, 5.56, 7.91, 6.89]
            ),
        ),
    }

    assert raw_water_data.shape == (24, 11)
    oxygen_flow_r = pearson_details(
        water_data["flow_speed_m_s"], water_data["oxygen_mg_l"]
    )["correlation"]
    oxygen_temperature_r = pearson_details(
        water_data["temperature_c"], water_data["oxygen_mg_l"]
    )["correlation"]
    river_regression = simple_regression(
        water_data["flow_speed_m_s"], water_data["oxygen_mg_l"]
    )
    river_partial_r, nitrate_residuals, phosphate_residuals = partial_correlation(
        water_data["nitrate_mg_l"],
        water_data["phosphate_ug_l"],
        water_data["distance_km"],
    )
    assert np.isclose(oxygen_flow_r, 0.6790226114)
    assert np.isclose(oxygen_temperature_r, -0.2917539945)
    assert np.isclose(river_regression["slope"], 7.3791311878)
    assert np.isclose(
        river_regression["sst"], river_regression["ssr"] + river_regression["sse"]
    )
    assert np.isclose(river_partial_r, 0.3365627665)
    return (
        anscombe_data,
        fisher_correlation_interval,
        generate_point_cloud,
        make_diagnostic_dataset,
        nitrate_residuals,
        paired_bootstrap_correlation,
        pearson_details,
        phosphate_residuals,
        raw_water_data,
        regression_intervals,
        river_partial_r,
        river_regression,
        simple_regression,
        simulate_correlation_sampling,
        variable_labels,
        water_data,
        water_data_path,
    )


@app.cell
def _(mo):
    mo.md(r"""
    ---

    ## The river dataset

    The 24 rows below are paired observations from rivers reported in the course
    material after Rudolf and Kuhlisch. Each row is one river. The original German
    headings are retained in the source file and in this table; shorter English
    aliases are used only inside calculations.

    The dataset mixes metric measurements, ordinal scores, and binary indicators.
    That distinction matters: Pearson correlation is designed for metric variables,
    while ranks provide a more defensible summary for ordinal scores.

    **Explore the table:** use the search box to find a row, the page controls to
    see all 24 rivers, and horizontal scrolling to reach the remaining columns.
    Compare *Fließgeschwindigkeit* (flow speed) with *Sauerstoffkonzentration*
    (oxygen). Sorting or searching this table changes only its display; the
    analyses below continue to use all 24 rivers.
    """)


@app.cell
def _(mo, raw_water_data, water_data_path):
    river_table = mo.ui.table(
        raw_water_data,
        pagination=True,
        page_size=12,
        selection=None,
        show_search=True,
        show_column_summaries=False,
        show_data_types=False,
        show_download=True,
        label="Observed river measurements",
    )
    mo.vstack(
        [
            mo.callout(
                mo.md(
                    f"**Source file:** `{water_data_path.name}` · "
                    f"{len(raw_water_data)} rivers · {raw_water_data.shape[1]} variables"
                ),
                kind="info",
            ),
            river_table,
        ]
    )


@app.cell
def _(mo):
    mo.md(r"""
    ---

    ## 1. Bivariate data, covariance, and Pearson correlation

    A scatter plot preserves the paired structure: every point combines an $x$ and
    a $y$ measurement from the same river. Marginal distributions describe each
    variable separately, but they do not show how the measurements are paired.

    Sample covariance averages products of centered values:

    $$s_{xy}=\frac{1}{n-1}\sum_{i=1}^{n}(x_i-\bar{x})(y_i-\bar{y}).$$

    Points in the upper-right and lower-left quadrants contribute positively;
    points in the other two quadrants contribute negatively. Pearson's correlation
    standardizes covariance by both sample standard deviations:

    $$r=\frac{s_{xy}}{s_xs_y}.$$

    This makes $r$ dimensionless and constrains it to the interval $[-1,1]$.

    **Try this:** start with flow speed on the horizontal axis and oxygen on the
    vertical axis. Predict the sign of $r$, then switch the horizontal variable to
    temperature. Keep that pair selected and multiply $x$ by 100: covariance
    scales by 100, whereas $r$ stays the same. Finally try shifting both variables;
    neither covariance nor $r$ changes.

    The left plot shows the observations and their means (dashed lines). The right
    plot subtracts those means: larger markers indicate larger absolute products
    $(x_i-\bar x)(y_i-\bar y)$. Every selector updates the plots immediately.
    """)


@app.cell
def _(mo):
    correlation_x_variable = mo.ui.dropdown(
        {
            "Flow speed": "flow_speed_m_s",
            "Temperature": "temperature_c",
            "Nitrate": "nitrate_mg_l",
            "Distance from source": "distance_km",
        },
        value="Flow speed",
        label="Horizontal variable",
        full_width=True,
    )
    correlation_y_variable = mo.ui.dropdown(
        {
            "Oxygen": "oxygen_mg_l",
            "Phosphate": "phosphate_ug_l",
            "Nitrate": "nitrate_mg_l",
        },
        value="Oxygen",
        label="Vertical variable",
        full_width=True,
    )
    unit_transformation = mo.ui.dropdown(
        {
            "Original units": (1.0, 1.0, "original"),
            "Multiply x by 100": (100.0, 1.0, "x × 100"),
            "Multiply y by 1,000": (1.0, 1000.0, "y × 1,000"),
            "Shift both variables": (1.0, 1.0, "shifted"),
        },
        value="Original units",
        label="Linear unit transformation",
        full_width=True,
    )
    return correlation_x_variable, correlation_y_variable, unit_transformation


@app.cell
def _(
    COLORS,
    FIGURE_SIZE_LINKED,
    correlation_x_variable,
    correlation_y_variable,
    mo,
    np,
    pearson_details,
    plt,
    two_column_panel,
    unit_transformation,
    variable_labels,
    water_data,
):
    selected_x_name = correlation_x_variable.value
    selected_y_name = correlation_y_variable.value
    x_multiplier, y_multiplier, transformation_name = unit_transformation.value
    selected_x = water_data[selected_x_name].to_numpy(dtype=float) * x_multiplier
    selected_y = water_data[selected_y_name].to_numpy(dtype=float) * y_multiplier
    if transformation_name == "shifted":
        selected_x = selected_x + 25.0
        selected_y = selected_y - 10.0
    selected_details = pearson_details(selected_x, selected_y)

    covariance_figure, covariance_axes = plt.subplots(1, 2, figsize=FIGURE_SIZE_LINKED)
    covariance_axes[0].scatter(
        selected_x,
        selected_y,
        color=COLORS["orange"],
        edgecolor="white",
        linewidth=0.5,
        s=42,
    )
    covariance_axes[0].axvline(
        selected_details["x_mean"], color=COLORS["blue"], linestyle="--"
    )
    covariance_axes[0].axhline(
        selected_details["y_mean"], color=COLORS["blue"], linestyle="--"
    )
    covariance_axes[0].set(
        title="Observed paired measurements",
        xlabel=variable_labels[selected_x_name],
        ylabel=variable_labels[selected_y_name],
    )

    positive_contribution = selected_details["contributions"] >= 0
    contribution_colors = np.where(
        positive_contribution, COLORS["sky"], COLORS["vermillion"]
    )
    contribution_sizes = 22 + 75 * (
        np.abs(selected_details["contributions"])
        / max(np.abs(selected_details["contributions"]).max(), 1e-12)
    )
    covariance_axes[1].scatter(
        selected_details["x_centered"],
        selected_details["y_centered"],
        color=contribution_colors,
        s=contribution_sizes,
        edgecolor="white",
        linewidth=0.5,
    )
    covariance_axes[1].axvline(0, color=COLORS["gray"], linestyle=":")
    covariance_axes[1].axhline(0, color=COLORS["gray"], linestyle=":")
    covariance_axes[1].set(
        title="Centered covariance contributions",
        xlabel=r"$x_i-\bar{x}$",
        ylabel=r"$y_i-\bar{y}$",
    )
    for covariance_axis in covariance_axes:
        covariance_axis.grid(linestyle=":", alpha=0.25)
    covariance_figure.tight_layout()

    covariance_controls = mo.vstack(
        [
            correlation_x_variable,
            correlation_y_variable,
            unit_transformation,
            mo.stat(
                f"{selected_details['covariance']:.3f}",
                label="Sample covariance",
                bordered=True,
            ),
            mo.stat(
                f"{selected_details['correlation']:.3f}",
                label="Pearson correlation r",
                bordered=True,
            ),
        ]
    )
    covariance_note = mo.callout(
        mo.md(
            f"The covariance is **{selected_details['covariance']:.3f}**, measured "
            f"in x-units × y-units, while $r$ is "
            f"**{selected_details['correlation']:.3f}** and has no unit. Positive "
            "linear rescaling and shifting do not change the correlation. Blue "
            "points contribute positively; vermillion points contribute negatively."
        ),
        kind="info",
    )
    two_column_panel(
        covariance_controls,
        mo.vstack([covariance_figure, covariance_note]),
        widths=(1, 3),
    )


@app.cell
def _(mo):
    mo.md(r"""
    ### Interactive laboratory: what controls the observed correlation?

    Before moving the controls, predict what will happen to $r$ when noise increases
    or when the sample size increases. Sample size changes the stability of the
    estimate, but it does not mechanically force an observed correlation upward.

    **Try this:** keep the slope at 1 and increase noise from 0.1 to 4. Then restore
    noise to 1 and move the slope through 0 to −1. Watch the cloud tilt and $r$
    change sign. Finally compare $n=8$ and $n=200$ at the same slope and noise.
    Each change updates immediately; each point is one simulated pair. Sample
    size changes the dataset, so $r$ may fluctuate rather than change monotonically.
    """)


@app.cell
def _(mo):
    cloud_slope = mo.ui.slider(
        -2.0,
        2.0,
        step=0.1,
        value=1.0,
        show_value=True,
        full_width=True,
        label="Underlying slope",
    )
    cloud_noise = mo.ui.slider(
        0.1,
        4.0,
        step=0.1,
        value=1.0,
        show_value=True,
        full_width=True,
        label="Noise SD",
    )
    cloud_sample_size = mo.ui.slider(
        8,
        200,
        step=4,
        value=40,
        show_value=True,
        full_width=True,
        label="Sample size n",
    )
    return cloud_noise, cloud_sample_size, cloud_slope


@app.cell
def _(
    COLORS,
    FIGURE_SIZE_STANDARD,
    cloud_noise,
    cloud_sample_size,
    cloud_slope,
    generate_point_cloud,
    mo,
    pearson_details,
    plt,
    two_column_panel,
):
    cloud_x, cloud_y = generate_point_cloud(
        cloud_slope.value, cloud_noise.value, cloud_sample_size.value
    )
    cloud_details = pearson_details(cloud_x, cloud_y)
    cloud_figure, cloud_axis = plt.subplots(figsize=FIGURE_SIZE_STANDARD)
    cloud_axis.scatter(
        cloud_x,
        cloud_y,
        color=COLORS["orange"],
        edgecolor="white",
        linewidth=0.5,
        alpha=0.85,
    )
    cloud_axis.set(
        xlabel="Predictor x [arbitrary units]",
        ylabel="Response y [arbitrary units]",
        xlim=(-2.8, 2.8),
        ylim=(min(-8, cloud_y.min() - 1), max(8, cloud_y.max() + 1)),
    )
    cloud_axis.grid(linestyle=":", alpha=0.3)
    cloud_figure.tight_layout()
    two_column_panel(
        mo.vstack(
            [
                cloud_slope,
                cloud_noise,
                cloud_sample_size,
                mo.stat(
                    f"{cloud_details['correlation']:.3f}",
                    label="Observed Pearson r",
                    bordered=True,
                ),
            ]
        ),
        mo.vstack(
            [
                cloud_figure,
                mo.callout(
                    mo.md(
                        "The underlying random numbers are held fixed while the "
                        "controls move, so changes reflect the selected model rather "
                        "than unrelated redraws. More noise weakens the linear "
                        "signal; a larger sample makes estimates more stable across "
                        "repeated samples."
                    ),
                    kind="warn",
                ),
            ]
        ),
    )


@app.cell
def _(mo):
    mo.md(r"""
    ---

    ## 2. Plot first: outliers, nonlinear structure, and range restriction

    A correlation coefficient compresses an entire point cloud into one number.
    Anscombe's quartet shows why that compression can be dangerous: four datasets
    have nearly identical means, variances, correlations, and regression lines but
    fundamentally different structures.
    """)


@app.cell
def _(
    COLORS,
    FIGURE_SIZE_LINKED,
    anscombe_data,
    compact_table,
    mo,
    np,
    pd,
    pearson_details,
    plt,
    stats,
):
    anscombe_figure, anscombe_axes = plt.subplots(2, 2, figsize=FIGURE_SIZE_LINKED)
    anscombe_rows = []
    quartet_colors = ["blue", "green", "vermillion", "purple"]
    for anscombe_axis, (quartet_name, quartet_values), color_name in zip(
        anscombe_axes.flat, anscombe_data.items(), quartet_colors
    ):
        quartet_x, quartet_y = quartet_values
        quartet_model = stats.linregress(quartet_x, quartet_y)
        quartet_grid = np.array([3.0, 20.0])
        anscombe_axis.scatter(
            quartet_x,
            quartet_y,
            color=COLORS[color_name],
            edgecolor="white",
            linewidth=0.4,
        )
        anscombe_axis.plot(
            quartet_grid,
            quartet_model.intercept + quartet_model.slope * quartet_grid,
            color=COLORS["gray"],
            linestyle="--",
        )
        anscombe_axis.set(
            title=f"Dataset {quartet_name}",
            xlim=(2, 20),
            ylim=(3, 14),
            xlabel="x",
            ylabel="y",
        )
        anscombe_axis.grid(linestyle=":", alpha=0.25)
        anscombe_rows.append(
            {
                "Dataset": quartet_name,
                "Mean x": quartet_x.mean(),
                "Mean y": quartet_y.mean(),
                "Variance x": quartet_x.var(ddof=1),
                "Variance y": quartet_y.var(ddof=1),
                "Pearson r": pearson_details(quartet_x, quartet_y)["correlation"],
            }
        )
    anscombe_figure.tight_layout()
    anscombe_summary = compact_table(
        pd.DataFrame(anscombe_rows),
        column_widths={
            "Dataset": 70,
            "Mean x": 75,
            "Mean y": 75,
            "Variance x": 85,
            "Variance y": 85,
            "Pearson r": 80,
        },
        format_mapping={
            "Mean x": "{:.2f}",
            "Mean y": "{:.2f}",
            "Variance x": "{:.2f}",
            "Variance y": "{:.2f}",
            "Pearson r": "{:.3f}",
        },
    )
    mo.vstack(
        [
            anscombe_figure,
            anscombe_summary,
            mo.callout(
                mo.md(
                    "**Plot first, summarize second.** Dataset II is curved, III is "
                    "dominated by an unusual response, and IV by a high-leverage "
                    "predictor. None of those facts is visible in $r$ alone."
                ),
                kind="warn",
            ),
        ]
    )


@app.cell
def _(mo):
    mo.md(r"""
    ### Two experiments: one extra point and a narrower range

    **Left plot:** the first two sliders position one added, hypothetical river
    (the diamond). Keep its flow speed at 1.0 m/s and lower its oxygen to 1 mg/L.
    Then move its flow speed towards the center of the cloud. Watch the fitted
    line and the “r after added point” value: a point far from the predictor mean
    can exert more influence on the fit.

    **Right plot:** the third slider retains an increasingly narrow central range
    of the original flow measurements. Lower it from 1.0 to 0.35 and compare the
    retained sample size and correlation. Faded points are excluded from this
    calculation; the shaded area marks the included range. This experiment uses
    only the original rivers, so the added point has no effect on the right plot.
    """)


@app.cell
def _(mo):
    influence_x = mo.ui.slider(
        0.0,
        1.4,
        step=0.02,
        value=1.0,
        show_value=True,
        full_width=True,
        label="Influential point: flow speed [m/s]",
    )
    influence_y = mo.ui.slider(
        0.0,
        18.0,
        step=0.2,
        value=7.5,
        show_value=True,
        full_width=True,
        label="Influential point: oxygen [mg/L]",
    )
    range_fraction = mo.ui.slider(
        0.35,
        1.0,
        step=0.05,
        value=1.0,
        show_value=True,
        full_width=True,
        label="Retained central predictor range",
    )
    return influence_x, influence_y, range_fraction


@app.cell
def _(
    COLORS,
    FIGURE_SIZE_LINKED,
    influence_x,
    influence_y,
    mo,
    np,
    pearson_details,
    plt,
    range_fraction,
    simple_regression,
    two_column_panel,
    water_data,
):
    base_flow = water_data["flow_speed_m_s"].to_numpy(dtype=float)
    base_oxygen = water_data["oxygen_mg_l"].to_numpy(dtype=float)
    augmented_flow = np.append(base_flow, influence_x.value)
    augmented_oxygen = np.append(base_oxygen, influence_y.value)
    augmented_details = pearson_details(augmented_flow, augmented_oxygen)
    augmented_model = simple_regression(augmented_flow, augmented_oxygen)

    range_center = float(np.mean(base_flow))
    range_half_width = 0.5 * (base_flow.max() - base_flow.min()) * range_fraction.value
    range_mask = np.abs(base_flow - range_center) <= range_half_width
    restricted_flow = base_flow[range_mask]
    restricted_oxygen = base_oxygen[range_mask]
    restricted_r = pearson_details(restricted_flow, restricted_oxygen)["correlation"]

    influence_figure, influence_axes = plt.subplots(1, 2, figsize=FIGURE_SIZE_LINKED)
    influence_axes[0].scatter(
        base_flow, base_oxygen, color=COLORS["orange"], label="Original rivers"
    )
    influence_axes[0].scatter(
        [influence_x.value],
        [influence_y.value],
        color=COLORS["vermillion"],
        marker="D",
        s=55,
        label="Added point",
    )
    influence_grid = np.linspace(0.1, 1.4, 100)
    influence_axes[0].plot(
        influence_grid,
        augmented_model["intercept"] + augmented_model["slope"] * influence_grid,
        color=COLORS["blue"],
    )
    influence_axes[0].set(
        title="Influence of one point",
        xlabel="Flow speed [m/s]",
        ylabel="Oxygen concentration [mg/L]",
        xlim=(0.1, 1.45),
        ylim=(0, 18.5),
    )
    influence_axes[0].legend(frameon=False, fontsize=7)

    influence_axes[1].scatter(
        base_flow[~range_mask],
        base_oxygen[~range_mask],
        color=COLORS["gray"],
        alpha=0.25,
        label="Excluded by range",
    )
    influence_axes[1].scatter(
        restricted_flow, restricted_oxygen, color=COLORS["orange"], label="Retained"
    )
    influence_axes[1].axvspan(
        range_center - range_half_width,
        range_center + range_half_width,
        color=COLORS["sky"],
        alpha=0.12,
    )
    influence_axes[1].set(
        title="Range restriction",
        xlabel="Flow speed [m/s]",
        ylabel="Oxygen concentration [mg/L]",
        xlim=(0.1, 1.1),
        ylim=(2, 14),
    )
    influence_axes[1].legend(frameon=False, fontsize=7)
    for influence_axis in influence_axes:
        influence_axis.grid(linestyle=":", alpha=0.25)
    influence_figure.tight_layout()

    two_column_panel(
        mo.vstack(
            [
                influence_x,
                influence_y,
                range_fraction,
                mo.stat(
                    f"{augmented_details['correlation']:.3f}",
                    label="r after added point",
                    bordered=True,
                ),
                mo.stat(
                    f"{restricted_r:.3f}",
                    label=f"r in retained n={range_mask.sum()}",
                    bordered=True,
                ),
            ]
        ),
        mo.vstack(
            [
                influence_figure,
                mo.callout(
                    mo.md(
                        "A point far from the predictor mean has high leverage and "
                        "can rotate the fitted line. Restricting the observed range "
                        "can weaken or otherwise change $r$ even when the underlying "
                        "biological relationship is unchanged."
                    ),
                    kind="warn",
                ),
            ]
        ),
        widths=(1.15, 2.85),
    )


@app.cell
def _(mo):
    mo.md(r"""
    ---

    ## 3. Uncertainty in a correlation coefficient

    An observed $r$ is an estimate. Across repeated samples, its distribution is
    asymmetric when the population correlation is near $-1$ or $1$. Fisher's
    transformation moves the bounded coefficient to an unbounded scale:

    $$z=\operatorname{atanh}(r)=\frac{1}{2}\ln\left(\frac{1+r}{1-r}\right),
    \qquad SE_z\approx\frac{1}{\sqrt{n-3}}.$$

    An approximate interval is constructed on the $z$ scale and transformed back
    with $r=\tanh(z)$. A bootstrap offers a different approximation by resampling
    the observed **rows** with replacement.

    **Try the sampling experiment:** choose $\rho=0.9$ and $n=10$, then click
    **Run a new simulation**. Each histogram summarizes correlations from many
    independent simulated samples of that size. Compare the asymmetry on the raw
    $r$ scale with the Fisher scale. Increase $n$ to 100 and click again: estimates
    should cluster more tightly around the population value.

    The repetition count controls how precisely the simulation describes the
    sampling distribution; it does not add observations to any one sample. Settings
    take effect only when you press the button. Press it again without changing
    settings to see Monte Carlo variation; the caption records the settings used
    for the displayed result.
    """)


@app.cell
def _(mo, np, simulate_correlation_sampling):
    initial_sampling_result = simulate_correlation_sampling(
        0.70, sample_size=20, repetitions=5_000, rng=np.random.default_rng(82433)
    )
    get_sampling_result, set_sampling_result = mo.state(initial_sampling_result)
    return get_sampling_result, set_sampling_result


@app.cell
def _(mo, np, set_sampling_result, simulate_correlation_sampling):
    sampling_population_r = mo.ui.slider(
        -0.95,
        0.95,
        step=0.05,
        value=0.70,
        show_value=True,
        full_width=True,
        label="Population correlation ρ",
    )
    sampling_n = mo.ui.slider(
        5,
        100,
        step=5,
        value=20,
        show_value=True,
        full_width=True,
        label="Sample size n",
    )
    sampling_repetitions = mo.ui.slider(
        500,
        10_000,
        step=500,
        value=5_000,
        show_value=True,
        full_width=True,
        label="Repeated samples",
    )

    def run_sampling_simulation(_value):
        set_sampling_result(
            simulate_correlation_sampling(
                float(sampling_population_r.value),
                int(sampling_n.value),
                int(sampling_repetitions.value),
                np.random.default_rng(),
            )
        )

    run_sampling_button = mo.ui.button(
        label="Run a new simulation",
        kind="success",
        full_width=True,
        on_click=run_sampling_simulation,
    )
    return (
        run_sampling_button,
        sampling_n,
        sampling_population_r,
        sampling_repetitions,
    )


@app.cell
def _(
    COLORS,
    FIGURE_SIZE_LINKED,
    get_sampling_result,
    mo,
    np,
    plt,
    run_sampling_button,
    sampling_n,
    sampling_population_r,
    sampling_repetitions,
    two_column_panel,
):
    sampling_result = get_sampling_result()
    simulated_rs = sampling_result["correlations"]
    simulated_zs = sampling_result["fisher_z"]
    sampling_figure, sampling_axes = plt.subplots(1, 2, figsize=FIGURE_SIZE_LINKED)
    sampling_axes[0].hist(
        simulated_rs,
        bins=50,
        range=[-1.0, 1.0],
        density=True,
        color=COLORS["orange"],
        edgecolor="white",
    )
    sampling_axes[0].axvline(
        sampling_result["population_correlation"],
        color=COLORS["green"],
        linestyle="--",
        label="Population ρ",
    )
    sampling_axes[0].set(
        title="Sampling distribution of r",
        xlabel="Sample correlation r",
        ylabel="Density",
        xlim=(-1, 1),
    )
    sampling_axes[0].legend(frameon=False, fontsize=8)

    expected_z = np.arctanh(sampling_result["population_correlation"])
    sampling_axes[1].hist(
        simulated_zs, bins=42, density=True, color=COLORS["purple"], edgecolor="white"
    )
    sampling_axes[1].axvline(
        expected_z, color=COLORS["green"], linestyle="--", label="atanh(ρ)"
    )
    sampling_axes[1].set(
        title="After Fisher transformation", xlabel="Fisher z", ylabel="Density"
    )
    sampling_axes[1].legend(frameon=False, fontsize=8)
    for sampling_axis in sampling_axes:
        sampling_axis.grid(axis="y", linestyle=":", alpha=0.3)
    sampling_figure.tight_layout()

    simulated_se = float(np.std(simulated_rs, ddof=1))
    sampling_mcse = simulated_se / np.sqrt(2 * (sampling_result["repetitions"] - 1))
    two_column_panel(
        mo.vstack(
            [
                sampling_population_r,
                sampling_n,
                sampling_repetitions,
                run_sampling_button,
                mo.stat(
                    f"{simulated_se:.3f}",
                    label="Simulated SD of r",
                    caption=f"MC uncertainty of SD ≈ {sampling_mcse:.4f}",
                    bordered=True,
                ),
            ]
        ),
        mo.vstack(
            [
                sampling_figure,
                mo.callout(
                    mo.md(
                        f"The displayed result uses ρ = "
                        f"**{sampling_result['population_correlation']:.2f}**, "
                        f"$n={sampling_result['sample_size']}$, and "
                        f"{sampling_result['repetitions']:,} repetitions. The raw "
                        "$r$ distribution is bounded; Fisher's scale is more nearly "
                        "symmetric. Change settings, then press the button to draw "
                        "fresh samples."
                    ),
                    kind="info",
                ),
            ]
        ),
    )


@app.cell
def _(mo):
    mo.md(r"""
    ### Compare two confidence-interval procedures

    **Try this:** select oxygen versus flow speed and click **Run paired
    bootstrap**. Compare the Fisher and bootstrap limits in the table. Next
    select oxygen versus temperature, run again, and check whether the intervals
    include zero. Finally choose the example with one outlier and inspect how
    the bootstrap distribution and interval agreement change.

    Here we resample the observed dataset rather than generate new rivers from a
    known population. Each histogram value comes from resampling whole $(x,y)$
    rows and recalculating $r$. The shaded region spans the middle 95% of these
    estimates. The selector and repetition count are applied only when you click
    the button; its caption identifies the currently displayed dataset.
    Increasing repetitions makes the bootstrap calculation more stable, but does
    not remove outlier influence or increase the original sample size.
    """)


@app.cell
def _(mo, np, paired_bootstrap_correlation, water_data):
    initial_bootstrap_x = water_data["flow_speed_m_s"].to_numpy(dtype=float)
    initial_bootstrap_y = water_data["oxygen_mg_l"].to_numpy(dtype=float)
    initial_bootstrap_distribution = paired_bootstrap_correlation(
        initial_bootstrap_x,
        initial_bootstrap_y,
        repetitions=5_000,
        rng=np.random.default_rng(82434),
    )
    initial_correlation_bootstrap = {
        "name": "Oxygen versus flow speed",
        "x": initial_bootstrap_x,
        "y": initial_bootstrap_y,
        "distribution": initial_bootstrap_distribution,
        "repetitions": 5_000,
    }
    get_correlation_bootstrap, set_correlation_bootstrap = mo.state(
        initial_correlation_bootstrap
    )
    return get_correlation_bootstrap, set_correlation_bootstrap


@app.cell
def _(
    mo,
    np,
    paired_bootstrap_correlation,
    set_correlation_bootstrap,
    water_data,
):
    bootstrap_pair = mo.ui.dropdown(
        [
            "Oxygen versus flow speed",
            "Oxygen versus temperature",
            "Oxygen versus flow with one outlier",
        ],
        value="Oxygen versus flow speed",
        label="Variable pair",
        full_width=True,
    )
    correlation_bootstrap_repetitions = mo.ui.slider(
        500,
        10_000,
        step=500,
        value=5_000,
        show_value=True,
        full_width=True,
        label="Bootstrap repetitions B",
    )

    def run_correlation_bootstrap(_value):
        if bootstrap_pair.value == "Oxygen versus temperature":
            x_values = water_data["temperature_c"].to_numpy(dtype=float)
            y_values = water_data["oxygen_mg_l"].to_numpy(dtype=float)
        else:
            x_values = water_data["flow_speed_m_s"].to_numpy(dtype=float)
            y_values = water_data["oxygen_mg_l"].to_numpy(dtype=float)
            if bootstrap_pair.value == "Oxygen versus flow with one outlier":
                x_values = np.append(x_values, 1.35)
                y_values = np.append(y_values, 1.0)
        repetitions = int(correlation_bootstrap_repetitions.value)
        distribution = paired_bootstrap_correlation(
            x_values, y_values, repetitions, np.random.default_rng()
        )
        set_correlation_bootstrap(
            {
                "name": bootstrap_pair.value,
                "x": x_values,
                "y": y_values,
                "distribution": distribution,
                "repetitions": repetitions,
            }
        )

    run_correlation_bootstrap_button = mo.ui.button(
        label="Run paired bootstrap",
        kind="success",
        full_width=True,
        on_click=run_correlation_bootstrap,
    )
    return (
        bootstrap_pair,
        correlation_bootstrap_repetitions,
        run_correlation_bootstrap_button,
    )


@app.cell
def _(
    COLORS,
    FIGURE_SIZE_STANDARD,
    bootstrap_pair,
    compact_table,
    correlation_bootstrap_repetitions,
    fisher_correlation_interval,
    get_correlation_bootstrap,
    mo,
    np,
    pd,
    pearson_details,
    plt,
    run_correlation_bootstrap_button,
    two_column_panel,
):
    correlation_bootstrap = get_correlation_bootstrap()
    observed_bootstrap_r = pearson_details(
        correlation_bootstrap["x"], correlation_bootstrap["y"]
    )["correlation"]
    bootstrap_lower, bootstrap_upper = np.percentile(
        correlation_bootstrap["distribution"], [2.5, 97.5]
    )
    fisher_lower, fisher_upper = fisher_correlation_interval(
        observed_bootstrap_r, len(correlation_bootstrap["x"])
    )

    bootstrap_figure, bootstrap_axis = plt.subplots(figsize=FIGURE_SIZE_STANDARD)
    bootstrap_axis.hist(
        correlation_bootstrap["distribution"],
        bins=100,
        range=[-1.0, 1.0],
        density=True,
        color=COLORS["purple"],
        edgecolor="white",
        label="Paired bootstrap r",
    )
    bootstrap_axis.axvline(
        observed_bootstrap_r, color=COLORS["green"], linewidth=2, label="Observed r"
    )
    bootstrap_axis.axvspan(
        bootstrap_lower,
        bootstrap_upper,
        color=COLORS["orange"],
        alpha=0.22,
        label="95% bootstrap interval",
    )
    bootstrap_axis.set(xlabel="Bootstrap correlation r", ylabel="Density", xlim=(-1, 1))
    bootstrap_axis.grid(axis="y", linestyle=":", alpha=0.3)
    bootstrap_axis.legend(frameon=False, fontsize=8)
    bootstrap_figure.tight_layout()

    interval_comparison = compact_table(
        pd.DataFrame(
            {
                "Method": ["Fisher z", "Paired bootstrap"],
                "Lower": [fisher_lower, bootstrap_lower],
                "Upper": [fisher_upper, bootstrap_upper],
            }
        ),
        column_widths={"Method": 120, "Lower": 75, "Upper": 75},
        format_mapping={"Lower": "{:.3f}", "Upper": "{:.3f}"},
    )
    bootstrap_controls = mo.vstack(
        [
            bootstrap_pair,
            correlation_bootstrap_repetitions,
            run_correlation_bootstrap_button,
            mo.stat(f"{observed_bootstrap_r:.3f}", label="Observed r", bordered=True),
            interval_comparison,
        ]
    )
    bootstrap_note = mo.callout(
        mo.md(
            f"Displayed: **{correlation_bootstrap['name']}** from "
            f"{correlation_bootstrap['repetitions']:,} paired resamples. Resampling "
            "$x$ and $y$ independently would break which measurements belong to the "
            "same river and would therefore answer the wrong question."
        ),
        kind="warn",
    )
    two_column_panel(
        bootstrap_controls,
        mo.vstack([bootstrap_figure, bootstrap_note]),
        widths=(1.3, 2.7),
    )


@app.cell
def _(mo):
    mo.md(r"""
    ---

    ## 4. Spearman rank correlation

    Spearman's coefficient is Pearson correlation applied to the ranks of both
    variables. Tied observations receive their average rank. Ranking discards
    metric distances but preserves order, so Spearman correlation describes a
    **monotonic** rather than specifically linear association.

    This is useful for ordinal variables such as the water-quality and
    population-density scores, and for metric relationships that are monotonic but
    strongly curved. It is not a general detector of every nonlinear pattern.

    **Try this:** use the synthetic example with curvature 1 (an approximately
    straight relationship), then increase it to 2.5. Compare the two coefficients
    as the relationship bends while retaining its increasing trend. Select
    **Round y to create ties** and look for repeated, averaged ranks in the table
    (the first eight rows are shown). At curvature 0 the systematic response is
    constant and only noise remains, so neither coefficient need be large.

    Switch to **Ordinal river scores** to calculate ranks from the observed
    water-quality and population-density scores. The curvature and rounding
    controls apply only to the synthetic example; river scores are used as
    recorded. Small plotting offsets separate overlapping points without
    changing any values used in the calculation.
    """)


@app.cell
def _(mo):
    rank_curvature = mo.ui.slider(
        0.0,
        2.5,
        step=0.1,
        value=1.0,
        show_value=True,
        full_width=True,
        label="Monotonic curvature",
    )
    rank_ties = mo.ui.checkbox(value=False, label="Round y to create ties")
    rank_example = mo.ui.radio(
        ["Synthetic monotonic data", "Ordinal river scores"],
        value="Synthetic monotonic data",
        label="Example",
    )
    return rank_curvature, rank_example, rank_ties


@app.cell
def _(
    COLORS,
    FIGURE_SIZE_STANDARD,
    compact_table,
    mo,
    np,
    pd,
    pearson_details,
    plt,
    rank_curvature,
    rank_example,
    rank_ties,
    stats,
    two_column_panel,
    water_data,
):
    if rank_example.value == "Ordinal river scores":
        rank_x = water_data["water_quality_score"].to_numpy(dtype=float)
        rank_y = water_data["population_density_score"].to_numpy(dtype=float)
        rank_x_label = "Water-quality score [1–20]"
        rank_y_label = "Population-density score [1–20]"
    else:
        rank_rng = np.random.default_rng(82435)
        rank_x = np.linspace(0.2, 4.0, 30)
        rank_y = rank_x**rank_curvature.value + rank_rng.normal(0, 0.18, rank_x.size)
        if rank_ties.value:
            rank_y = np.round(rank_y, 0)
        rank_x_label = "x [arbitrary units]"
        rank_y_label = "y [arbitrary units]"

    x_ranks = stats.rankdata(rank_x, method="average")
    y_ranks = stats.rankdata(rank_y, method="average")
    rank_pearson = pearson_details(rank_x, rank_y)["correlation"]
    rank_spearman = pearson_details(x_ranks, y_ranks)["correlation"]

    rank_preview_data = pd.DataFrame(
        {"x": rank_x, "Rank x": x_ranks, "y": rank_y, "Rank y": y_ranks}
    ).head(8)
    rank_preview = compact_table(
        rank_preview_data,
        column_widths={"x": 65, "Rank x": 70, "y": 65, "Rank y": 70},
        format_mapping={
            "x": "{:.2f}",
            "Rank x": "{:.1f}",
            "y": "{:.2f}",
            "Rank y": "{:.1f}",
        },
    )

    rank_figure, rank_axis = plt.subplots(figsize=FIGURE_SIZE_STANDARD)
    rank_figure_jitter = np.random.default_rng(82436).uniform(-0.04, 0.04, rank_x.size)
    rank_axis.scatter(
        rank_x,
        rank_y + rank_figure_jitter
        if rank_ties.value or rank_example.value == "Ordinal river scores"
        else rank_y,
        color=COLORS["orange"],
        edgecolor="white",
        linewidth=0.5,
    )
    rank_axis.set(xlabel=rank_x_label, ylabel=rank_y_label)
    rank_axis.grid(linestyle=":", alpha=0.3)
    rank_figure.tight_layout()

    two_column_panel(
        mo.vstack(
            [
                rank_example,
                rank_curvature,
                rank_ties,
                mo.stat(f"{rank_pearson:.3f}", label="Pearson r", bordered=True),
                mo.stat(f"{rank_spearman:.3f}", label="Spearman ρs", bordered=True),
                rank_preview,
            ]
        ),
        mo.vstack(
            [
                rank_figure,
                mo.callout(
                    mo.md(
                        "Spearman correlation remains high when order is preserved "
                        "through a curved monotonic transformation. Creating ties "
                        "shows why average ranks can be fractional. Ranking gains "
                        "robustness to metric spacing but loses information about "
                        "the sizes of differences."
                    ),
                    kind="info",
                ),
            ]
        ),
        widths=(1.25, 2.75),
    )


@app.cell
def _(mo):
    mo.md(r"""
    ---

    ## 5. Regression as a directional model

    Correlation treats $x$ and $y$ symmetrically. Regression instead defines a
    response and models its conditional mean from one or more predictors. For simple
    linear regression,

    $$Y_i=\beta_0+\beta_1x_i+\varepsilon_i,\qquad
    \hat{y}_i=b_0+b_1x_i,\qquad e_i=y_i-\hat{y}_i.$$

    Here oxygen concentration is the response and flow speed is the predictor. The
    label “independent variable” does not establish experimental independence or
    causality; those claims require the study design and biological reasoning.

    **Try this:** move the river-row slider and follow the highlighted diamond
    (observed oxygen) and square (fitted mean at the same flow speed). Their
    vertical separation is the residual shown in the third tile. Find one river
    above the line and one below it, then check the residual signs. Selecting a
    row changes the highlight and readout; the line is always fitted to all rivers.
    """)


@app.cell
def _(mo, water_data):
    regression_point = mo.ui.slider(
        1,
        len(water_data),
        value=1,
        show_value=True,
        full_width=True,
        label="Selected river row",
    )
    return (regression_point,)


@app.cell
def _(
    COLORS,
    FIGURE_SIZE_STANDARD,
    mo,
    np,
    plt,
    regression_point,
    river_regression,
    two_column_panel,
):
    selected_regression_index = int(regression_point.value) - 1
    selected_regression_x = river_regression["x"][selected_regression_index]
    selected_regression_y = river_regression["y"][selected_regression_index]
    selected_regression_fitted = river_regression["fitted"][selected_regression_index]
    selected_regression_residual = river_regression["residuals"][
        selected_regression_index
    ]
    regression_grid = np.linspace(0.1, 1.05, 200)

    anatomy_figure, anatomy_axis = plt.subplots(figsize=FIGURE_SIZE_STANDARD)
    anatomy_axis.scatter(
        river_regression["x"],
        river_regression["y"],
        color=COLORS["orange"],
        edgecolor="white",
        linewidth=0.5,
        label="Observed rivers",
    )
    anatomy_axis.plot(
        regression_grid,
        river_regression["intercept"] + river_regression["slope"] * regression_grid,
        color=COLORS["blue"],
        linewidth=2,
        label="Fitted mean",
    )
    anatomy_axis.scatter(
        [selected_regression_x],
        [selected_regression_y],
        color=COLORS["vermillion"],
        s=62,
        marker="D",
        zorder=4,
        label="Selected observation",
    )
    anatomy_axis.scatter(
        [selected_regression_x],
        [selected_regression_fitted],
        color=COLORS["green"],
        s=50,
        marker="s",
        zorder=4,
        label="Fitted value",
    )
    anatomy_axis.vlines(
        selected_regression_x,
        selected_regression_fitted,
        selected_regression_y,
        color=COLORS["gray"],
        linestyle="--",
        linewidth=2,
        label="Residual",
    )
    anatomy_axis.set(
        xlabel="Flow speed [m/s]",
        ylabel="Oxygen concentration [mg/L]",
        xlim=(0.1, 1.08),
        ylim=(2, 14),
    )
    anatomy_axis.grid(linestyle=":", alpha=0.3)
    anatomy_axis.legend(frameon=False, fontsize=8, ncols=2)
    anatomy_figure.tight_layout()

    slope_sentence = (
        f"The fitted slope is {river_regression['slope']:.2f} "
        "mg/L per m/s: within the observed range, an increase of 0.10 m/s in flow "
        f"speed corresponds to a fitted mean oxygen increase of "
        f"{0.1 * river_regression['slope']:.2f} mg/L."
    )
    two_column_panel(
        mo.vstack(
            [
                regression_point,
                mo.stat(
                    f"{selected_regression_y:.2f} mg/L",
                    label="Observed y",
                    bordered=True,
                ),
                mo.stat(
                    f"{selected_regression_fitted:.2f} mg/L",
                    label="Fitted ŷ",
                    bordered=True,
                ),
                mo.stat(
                    f"{selected_regression_residual:+.2f} mg/L",
                    label="Residual e",
                    bordered=True,
                ),
            ]
        ),
        mo.vstack([anatomy_figure, mo.callout(mo.md(slope_sentence), kind="info")]),
    )


@app.cell
def _(mo):
    mo.md(r"""
    ---

    ## 6. Least squares and regression coefficients

    Least squares chooses the intercept and slope that minimize the sum of squared
    residuals:

    $$SSE(b_0,b_1)=\sum_{i=1}^{n}(y_i-b_0-b_1x_i)^2.$$

    Try to fit the line manually before revealing the optimum. Squaring prevents
    positive and negative residuals from cancelling and penalizes large residuals
    strongly.

    **Try this:** begin with slope 0, a horizontal line, and move the intercept to
    reduce SSE. Then increase the slope and readjust the intercept. The vertical
    segments show the residuals whose squares are being added. In the right plot,
    each contour joins parameter pairs with the same SSE; move the diamond towards
    the green star to approach the minimum. Select **Reveal least-squares solution**
    to overlay the fitted line and read its equation and minimum SSE below the plots.
    The derivation below can be expanded to see how that solution is calculated.
    """)


@app.cell
def _(mo):
    manual_slope = mo.ui.slider(
        -2.0,
        14.0,
        step=0.2,
        value=0.0,
        show_value=True,
        full_width=True,
        label="Candidate slope b₁ [mg/L per m/s]",
    )
    manual_intercept = mo.ui.slider(
        0.0,
        12.0,
        step=0.2,
        value=7.0,
        show_value=True,
        full_width=True,
        label="Candidate intercept b₀ [mg/L]",
    )
    reveal_least_squares = mo.ui.checkbox(
        value=False, label="Reveal least-squares solution"
    )
    return manual_intercept, manual_slope, reveal_least_squares


@app.cell
def _(
    COLORS,
    FIGURE_SIZE_LINKED,
    manual_intercept,
    manual_slope,
    mo,
    np,
    plt,
    reveal_least_squares,
    river_regression,
    two_column_panel,
):
    candidate_fitted = (
        manual_intercept.value + manual_slope.value * river_regression["x"]
    )
    candidate_residuals = river_regression["y"] - candidate_fitted
    candidate_sse = float(np.sum(candidate_residuals**2))
    least_squares_grid = np.linspace(0.1, 1.05, 200)

    fit_figure, fit_axes = plt.subplots(1, 2, figsize=FIGURE_SIZE_LINKED)
    fit_axes[0].scatter(
        river_regression["x"], river_regression["y"], color=COLORS["orange"]
    )
    fit_axes[0].plot(
        least_squares_grid,
        manual_intercept.value + manual_slope.value * least_squares_grid,
        color=COLORS["vermillion"],
        linewidth=2,
        label="Candidate line",
    )
    fit_axes[0].vlines(
        river_regression["x"],
        candidate_fitted,
        river_regression["y"],
        color=COLORS["gray"],
        alpha=0.55,
        linewidth=1,
    )
    if reveal_least_squares.value:
        fit_axes[0].plot(
            least_squares_grid,
            river_regression["intercept"]
            + river_regression["slope"] * least_squares_grid,
            color=COLORS["green"],
            linestyle="--",
            linewidth=2,
            label="Least-squares line",
        )
    fit_axes[0].set(
        title="Candidate model and residuals",
        xlabel="Flow speed [m/s]",
        ylabel="Oxygen concentration [mg/L]",
        xlim=(0.1, 1.08),
        ylim=(0, 18),
    )
    fit_axes[0].legend(frameon=False, fontsize=7)

    slope_grid_values = np.linspace(-2, 14, 100)
    intercept_grid_values = np.linspace(0, 12, 100)
    slope_mesh, intercept_mesh = np.meshgrid(slope_grid_values, intercept_grid_values)
    landscape_predictions = (
        intercept_mesh[:, :, None]
        + slope_mesh[:, :, None] * river_regression["x"][None, None, :]
    )
    sse_landscape = np.sum(
        (river_regression["y"][None, None, :] - landscape_predictions) ** 2,
        axis=2,
    )
    fit_axes[1].contour(
        slope_mesh,
        intercept_mesh,
        sse_landscape,
        levels=14,
        colors=COLORS["gray"],
        linewidths=0.65,
    )
    fit_axes[1].scatter(
        [manual_slope.value],
        [manual_intercept.value],
        color=COLORS["vermillion"],
        marker="D",
        label="Candidate",
    )
    fit_axes[1].scatter(
        [river_regression["slope"]],
        [river_regression["intercept"]],
        color=COLORS["green"],
        marker="*",
        s=90,
        label="Optimum",
    )
    fit_axes[1].set(title="SSE landscape", xlabel="Slope b₁", ylabel="Intercept b₀")
    fit_axes[1].legend(frameon=False, fontsize=7)
    for fit_axis in fit_axes:
        fit_axis.grid(linestyle=":", alpha=0.2)
    fit_figure.tight_layout()

    fit_note_text = (
        f"The least-squares solution is $\\hat y={river_regression['intercept']:.2f}"
        f"+{river_regression['slope']:.2f}x$ with "
        f"$SSE={river_regression['sse']:.2f}$."
        if reveal_least_squares.value
        else "Move both sliders to reduce SSE, then reveal the optimum. The contour "
        "plot shows why slope and intercept must be optimized together."
    )
    two_column_panel(
        mo.vstack(
            [
                manual_slope,
                manual_intercept,
                reveal_least_squares,
                mo.stat(
                    f"{candidate_sse:.2f}",
                    label="Candidate SSE [(mg/L)²]",
                    bordered=True,
                ),
            ]
        ),
        mo.vstack([fit_figure, mo.callout(mo.md(fit_note_text), kind="info")]),
        widths=(1, 3),
    )


@app.cell
def _(mo):
    least_squares_derivation = mo.accordion(
        {
            "Derivation: solving the normal equations": mo.md(r"""
            Differentiating $SSE$ with respect to $b_0$ and $b_1$ and setting both
            derivatives to zero yields

            $$\sum_i(y_i-b_0-b_1x_i)=0,$$
            $$\sum_i x_i(y_i-b_0-b_1x_i)=0.$$

            Solving these equations gives

            $$b_1=\frac{\sum_i(x_i-\bar{x})(y_i-\bar{y})}
            {\sum_i(x_i-\bar{x})^2}=\frac{s_{xy}}{s_x^2},
            \qquad b_0=\bar{y}-b_1\bar{x}.$$

            Therefore the fitted line passes through $(\bar{x},\bar{y})$. The slope
            has units “response units per predictor unit”; the intercept has response
            units but may lack a useful scientific interpretation when $x=0$ lies
            outside the observed range.
            """)
        }
    )
    least_squares_derivation  # noqa: B018


@app.cell
def _(mo):
    mo.md(r"""
    ---

    ## 7. Decomposing variation and interpreting $R^2$

    With an intercept, every total deviation can be split into a fitted and a
    residual deviation:

    $$y_i-\bar y=(\hat y_i-\bar y)+(y_i-\hat y_i).$$

    Least squares makes the cross-product vanish, giving

    $$SST=SSR+SSE,\qquad R^2=\frac{SSR}{SST}=1-\frac{SSE}{SST}.$$

    $R^2$ is the proportion of observed response variation accounted for by this
    fitted model. It does not identify a causal mechanism.

    **Try this:** select different river rows and follow the green segment from
    the response mean to the fitted value, and the vermillion segment from the
    fitted value to the observation. These are signed deviations and can point
    in opposite directions. The table adds squared deviations across **all**
    rivers, so its totals and $R^2$ stay fixed when the highlight changes. Check
    that explained and residual sums add to the total; this identity applies to
    the sums, not to the squared deviations of each individual point.
    """)


@app.cell
def _(mo, water_data):
    decomposition_point = mo.ui.slider(
        1,
        len(water_data),
        value=9,
        show_value=True,
        full_width=True,
        label="Selected river row",
    )
    return (decomposition_point,)


@app.cell
def _(
    COLORS,
    FIGURE_SIZE_STANDARD,
    compact_table,
    decomposition_point,
    mo,
    np,
    pd,
    pearson_details,
    plt,
    river_regression,
    two_column_panel,
):
    decomposition_index = int(decomposition_point.value) - 1
    decomposition_x = river_regression["x"][decomposition_index]
    decomposition_y = river_regression["y"][decomposition_index]
    decomposition_fitted = river_regression["fitted"][decomposition_index]
    decomposition_mean = float(np.mean(river_regression["y"]))

    decomposition_table = compact_table(
        pd.DataFrame(
            {
                "Quantity": ["Total (SST)", "Explained (SSR)", "Residual (SSE)"],
                "Sum of squares": [
                    river_regression["sst"],
                    river_regression["ssr"],
                    river_regression["sse"],
                ],
                "Share of SST": [
                    1.0,
                    river_regression["r_squared"],
                    1.0 - river_regression["r_squared"],
                ],
            }
        ),
        column_widths={"Quantity": 120, "Sum of squares": 105, "Share of SST": 95},
        format_mapping={"Sum of squares": "{:.2f}", "Share of SST": "{:.1%}"},
    )

    decomposition_figure, decomposition_axis = plt.subplots(
        figsize=FIGURE_SIZE_STANDARD
    )
    decomposition_axis.scatter(
        river_regression["x"], river_regression["y"], color=COLORS["orange"]
    )
    decomposition_grid = np.linspace(0.1, 1.05, 200)
    decomposition_axis.plot(
        decomposition_grid,
        river_regression["intercept"] + river_regression["slope"] * decomposition_grid,
        color=COLORS["blue"],
        label="Fitted line",
    )
    decomposition_axis.axhline(
        decomposition_mean, color=COLORS["gray"], linestyle=":", label="Response mean"
    )
    decomposition_axis.vlines(
        decomposition_x,
        decomposition_mean,
        decomposition_fitted,
        color=COLORS["green"],
        linewidth=4,
        label="Explained deviation",
    )
    decomposition_axis.vlines(
        decomposition_x,
        decomposition_fitted,
        decomposition_y,
        color=COLORS["vermillion"],
        linewidth=4,
        label="Residual deviation",
    )
    decomposition_axis.scatter(
        [decomposition_x], [decomposition_y], color=COLORS["vermillion"], s=55, zorder=4
    )
    decomposition_axis.set(
        xlabel="Flow speed [m/s]",
        ylabel="Oxygen concentration [mg/L]",
        xlim=(0.1, 1.08),
        ylim=(2, 14),
    )
    decomposition_axis.grid(linestyle=":", alpha=0.25)
    decomposition_axis.legend(frameon=False, fontsize=8, ncols=2)
    decomposition_figure.tight_layout()

    river_r_squared_from_r = (
        pearson_details(river_regression["x"], river_regression["y"])["correlation"]
        ** 2
    )
    assert np.isclose(river_regression["r_squared"], river_r_squared_from_r)
    two_column_panel(
        mo.vstack(
            [
                decomposition_point,
                mo.stat(
                    f"{river_regression['r_squared']:.3f}",
                    label="Regression R²",
                    bordered=True,
                ),
                decomposition_table,
            ]
        ),
        mo.vstack(
            [
                decomposition_figure,
                mo.callout(
                    mo.md(
                        f"For this simple regression with an intercept, "
                        f"$R^2=r^2={river_regression['r_squared']:.3f}$. It is valid "
                        "to say that the model accounts for about "
                        f"{100 * river_regression['r_squared']:.1f}% of the observed "
                        "oxygen variation. It is not valid to say that flow speed "
                        "causes that percentage."
                    ),
                    kind="warn",
                ),
            ]
        ),
        widths=(1.25, 2.75),
    )


@app.cell
def _(mo):
    mo.md(r"""
    ---

    ## 8. Uncertainty, prediction, and residual checks

    A **confidence interval** at a selected predictor value concerns the unknown mean
    response. A **prediction interval** concerns one future observation and must also
    include observation-to-observation scatter, so it is wider.

    Regression calculations also rely on a model for the residuals. Diagnostic plots
    can reveal curvature, changing variance, clusters, or influential observations.
    They expose tension with the model; they do not mechanically prove assumptions.

    **Try the interval explorer:** move the flow-speed slider from 0.60 m/s towards
    either end of the observed range. The green vertical line marks the location
    read out in the tiles. Compare the narrow band for the mean with the wider band
    for a new observation, and notice how both widen away from the average flow
    speed. Increase the confidence level from 0.80 to 0.99: greater coverage needs
    wider intervals, while the fitted mean stays unchanged. These are pointwise
    intervals at each flow speed, not a simultaneous guarantee for the whole curve.
    """)


@app.cell
def _(mo):
    interval_x = mo.ui.slider(
        0.2,
        1.0,
        step=0.02,
        value=0.60,
        show_value=True,
        full_width=True,
        label="Flow speed for interpretation [m/s]",
    )
    interval_confidence = mo.ui.slider(
        0.80,
        0.99,
        step=0.01,
        value=0.95,
        show_value=True,
        full_width=True,
        label="Confidence level",
    )
    return interval_confidence, interval_x


@app.cell
def _(
    COLORS,
    FIGURE_SIZE_STANDARD,
    interval_confidence,
    interval_x,
    mo,
    np,
    plt,
    regression_intervals,
    river_regression,
    two_column_panel,
):
    interval_grid = np.linspace(0.15, 1.05, 250)
    interval_results = regression_intervals(
        river_regression, interval_grid, interval_confidence.value
    )
    selected_interval_results = regression_intervals(
        river_regression, np.array([interval_x.value]), interval_confidence.value
    )
    selected_mean_fit = selected_interval_results["fitted"][0]
    selected_mean_lower = selected_interval_results["mean_lower"][0]
    selected_mean_upper = selected_interval_results["mean_upper"][0]
    selected_prediction_lower = selected_interval_results["prediction_lower"][0]
    selected_prediction_upper = selected_interval_results["prediction_upper"][0]

    interval_figure, interval_axis = plt.subplots(figsize=FIGURE_SIZE_STANDARD)
    interval_axis.scatter(
        river_regression["x"],
        river_regression["y"],
        color=COLORS["orange"],
        edgecolor="white",
        linewidth=0.5,
        label="Observed rivers",
    )
    interval_axis.fill_between(
        interval_grid,
        interval_results["prediction_lower"],
        interval_results["prediction_upper"],
        color=COLORS["purple"],
        alpha=0.14,
        label="Prediction interval",
    )
    interval_axis.fill_between(
        interval_grid,
        interval_results["mean_lower"],
        interval_results["mean_upper"],
        color=COLORS["sky"],
        alpha=0.35,
        label="Confidence band for mean",
    )
    interval_axis.plot(
        interval_grid,
        interval_results["fitted"],
        color=COLORS["blue"],
        linewidth=2,
        label="Fitted mean",
    )
    interval_axis.axvline(interval_x.value, color=COLORS["green"], linestyle="--")
    interval_axis.set(
        xlabel="Flow speed [m/s]",
        ylabel="Oxygen concentration [mg/L]",
        xlim=(0.1, 1.08),
        ylim=(0, 16),
    )
    interval_axis.grid(linestyle=":", alpha=0.25)
    interval_axis.legend(frameon=False, fontsize=8, ncols=2)
    interval_figure.tight_layout()

    interval_controls = mo.vstack(
        [
            interval_x,
            interval_confidence,
            mo.stat(
                f"{selected_mean_fit:.2f} mg/L",
                label="Estimated mean oxygen",
                bordered=True,
            ),
            mo.stat(
                f"{selected_mean_lower:.2f} to {selected_mean_upper:.2f}",
                label="Mean-response interval [mg/L]",
                bordered=True,
            ),
            mo.stat(
                f"{selected_prediction_lower:.2f} to {selected_prediction_upper:.2f}",
                label="New-observation interval [mg/L]",
                bordered=True,
            ),
        ]
    )
    interval_note = mo.callout(
        mo.md(
            f"At {interval_x.value:.2f} m/s, the model estimates the average oxygen "
            f"concentration as **{selected_mean_fit:.2f} mg/L**. The confidence "
            "interval addresses that average; the wider prediction interval "
            "addresses where one new river measurement might fall under the model."
        ),
        kind="info",
    )
    two_column_panel(interval_controls, mo.vstack([interval_figure, interval_note]))


@app.cell
def _(mo):
    mo.md(r"""
    ### Read the residuals

    **Try this:** start with constant variance, then select **Curvature** and
    **Increasing variance**. For each dataset, first inspect the observations and
    fitted line on the left, then look for a curve or fan in the residuals on the
    right. The zero line represents a perfect prediction at an observation.
    Compare **Two clusters** and **Influential outlier** next: can you identify
    the issue from $R^2$ alone?

    The selector replaces the simulated scenario and refits its line immediately.
    Returning to a scenario restores the same points, so comparisons are repeatable.
    These plots address the shape and spread of residuals; independence must be
    assessed from how observations were collected.
    """)


@app.cell
def _(mo):
    diagnostic_pattern = mo.ui.dropdown(
        [
            "Linear with constant variance",
            "Curvature",
            "Increasing variance",
            "Two clusters",
            "Influential outlier",
        ],
        value="Linear with constant variance",
        label="Residual pattern",
        full_width=True,
    )
    return (diagnostic_pattern,)


@app.cell
def _(
    COLORS,
    FIGURE_SIZE_LINKED,
    diagnostic_pattern,
    make_diagnostic_dataset,
    mo,
    plt,
    simple_regression,
    two_column_panel,
):
    diagnostic_x, diagnostic_y = make_diagnostic_dataset(diagnostic_pattern.value)
    diagnostic_model = simple_regression(diagnostic_x, diagnostic_y)
    diagnostic_figure, diagnostic_axes = plt.subplots(1, 2, figsize=FIGURE_SIZE_LINKED)
    diagnostic_axes[0].scatter(diagnostic_x, diagnostic_y, color=COLORS["orange"])
    diagnostic_axes[0].plot(
        diagnostic_x, diagnostic_model["fitted"], color=COLORS["blue"], linewidth=2
    )
    diagnostic_axes[0].set(
        title="Data and fitted line", xlabel="Predictor x", ylabel="Response y"
    )
    diagnostic_axes[1].scatter(
        diagnostic_model["fitted"], diagnostic_model["residuals"], color=COLORS["gray"]
    )
    diagnostic_axes[1].axhline(0, color=COLORS["blue"], linestyle="--")
    diagnostic_axes[1].set(
        title="Residuals versus fitted values",
        xlabel="Fitted response ŷ",
        ylabel="Residual e",
    )
    for diagnostic_axis in diagnostic_axes:
        diagnostic_axis.grid(linestyle=":", alpha=0.3)
    diagnostic_figure.tight_layout()

    diagnostic_messages = {
        "Linear with constant variance": "A roughly patternless residual cloud is compatible with the simple linear mean and constant-spread model.",
        "Curvature": "The curved residual pattern says that a straight conditional mean misses systematic structure.",
        "Increasing variance": "The fan shape indicates that residual spread increases with the fitted response.",
        "Two clusters": "Separated residual groups suggest an omitted grouping variable or mixture of processes.",
        "Influential outlier": "One unusual response can strongly affect coefficients, uncertainty, and diagnostics; investigate it scientifically rather than deleting it automatically.",
    }
    two_column_panel(
        mo.vstack(
            [
                diagnostic_pattern,
                mo.stat(
                    f"{diagnostic_model['r_squared']:.3f}",
                    label="Fitted R²",
                    bordered=True,
                ),
                mo.stat(
                    f"{diagnostic_model['residual_sd']:.3f}",
                    label="Residual SD",
                    bordered=True,
                ),
            ]
        ),
        mo.vstack(
            [
                diagnostic_figure,
                mo.callout(
                    mo.md(diagnostic_messages[diagnostic_pattern.value]), kind="warn"
                ),
            ]
        ),
    )


@app.cell
def _(mo):
    mo.md(r"""
    ---

    ## 9. Confounding, residualization, and partial correlation

    Nitrate and phosphate concentrations are strongly correlated in the raw river
    data. Both also increase with distance from the river source. To describe their
    remaining **linear** association after adjusting for distance, regress each
    concentration on distance and correlate the two sets of residuals.

    This residual correlation equals the three-correlation formula

    $$r_{xy\cdot z}=\frac{r_{xy}-r_{xz}r_{yz}}
    {\sqrt{(1-r_{xz}^2)(1-r_{yz}^2)}}.$$

    Calculation alone does not make distance a causal confounder. That label also
    requires temporal and biological knowledge.

    **Try this:** first inspect the two concentration-versus-distance plots.
    Switch from **Raw nitrate–phosphate association** to **Adjusted for distance**.
    Only the third plot changes: its coordinates become the residuals from the
    first two regressions, and the correlation tile updates accordingly. A positive
    nitrate residual means more nitrate than predicted at that river's distance;
    a negative one means less. Compare the raw and adjusted coefficients in the
    table, then describe the remaining association without assigning a causal role
    to distance from this calculation alone.
    """)


@app.cell
def _(mo):
    adjustment_view = mo.ui.radio(
        ["Raw nitrate–phosphate association", "Adjusted for distance"],
        value="Raw nitrate–phosphate association",
        label="Association to display",
    )
    return (adjustment_view,)


@app.cell
def _(
    COLORS,
    FIGURE_SIZE_LINKED,
    adjustment_view,
    compact_table,
    mo,
    nitrate_residuals,
    np,
    pd,
    pearson_details,
    phosphate_residuals,
    plt,
    river_partial_r,
    simple_regression,
    two_column_panel,
    water_data,
):
    nitrate_values = water_data["nitrate_mg_l"].to_numpy(dtype=float)
    phosphate_values = water_data["phosphate_ug_l"].to_numpy(dtype=float)
    distance_values = water_data["distance_km"].to_numpy(dtype=float)
    nitrate_phosphate_r = pearson_details(nitrate_values, phosphate_values)[
        "correlation"
    ]
    nitrate_distance_r = pearson_details(nitrate_values, distance_values)["correlation"]
    phosphate_distance_r = pearson_details(phosphate_values, distance_values)[
        "correlation"
    ]
    formula_partial_r = (
        nitrate_phosphate_r - nitrate_distance_r * phosphate_distance_r
    ) / np.sqrt((1.0 - nitrate_distance_r**2) * (1.0 - phosphate_distance_r**2))
    assert np.isclose(formula_partial_r, river_partial_r)

    association_summary = compact_table(
        pd.DataFrame(
            {
                "Association": [
                    "Nitrate–phosphate",
                    "Nitrate–distance",
                    "Phosphate–distance",
                    "Nitrate–phosphate | distance",
                ],
                "Correlation": [
                    nitrate_phosphate_r,
                    nitrate_distance_r,
                    phosphate_distance_r,
                    river_partial_r,
                ],
            }
        ),
        column_widths={"Association": 190, "Correlation": 95},
        wrapped_columns=["Association"],
        format_mapping={"Correlation": "{:.3f}"},
    )

    adjustment_figure, adjustment_axes = plt.subplots(1, 3, figsize=FIGURE_SIZE_LINKED)
    adjustment_axes[0].scatter(distance_values, nitrate_values, color=COLORS["orange"])
    nitrate_distance_model = simple_regression(distance_values, nitrate_values)
    adjustment_axes[0].plot(
        np.sort(distance_values),
        nitrate_distance_model["intercept"]
        + nitrate_distance_model["slope"] * np.sort(distance_values),
        color=COLORS["blue"],
    )
    adjustment_axes[0].set(
        title="Nitrate and distance", xlabel="Distance [km]", ylabel="Nitrate [mg/L]"
    )

    adjustment_axes[1].scatter(
        distance_values, phosphate_values, color=COLORS["purple"]
    )
    phosphate_distance_model = simple_regression(distance_values, phosphate_values)
    adjustment_axes[1].plot(
        np.sort(distance_values),
        phosphate_distance_model["intercept"]
        + phosphate_distance_model["slope"] * np.sort(distance_values),
        color=COLORS["blue"],
    )
    adjustment_axes[1].set(
        title="Phosphate and distance",
        xlabel="Distance [km]",
        ylabel="Phosphate [µg/L]",
    )

    if adjustment_view.value == "Adjusted for distance":
        displayed_nitrate = nitrate_residuals
        displayed_phosphate = phosphate_residuals
        displayed_title = "Residual association"
        displayed_x_label = "Nitrate residual [mg/L]"
        displayed_y_label = "Phosphate residual [µg/L]"
        displayed_r = river_partial_r
    else:
        displayed_nitrate = nitrate_values
        displayed_phosphate = phosphate_values
        displayed_title = "Raw association"
        displayed_x_label = "Nitrate [mg/L]"
        displayed_y_label = "Phosphate [µg/L]"
        displayed_r = nitrate_phosphate_r
    displayed_adjustment_model = simple_regression(
        displayed_nitrate, displayed_phosphate
    )
    displayed_adjustment_grid = np.linspace(
        displayed_nitrate.min(), displayed_nitrate.max(), 100
    )
    adjustment_axes[2].scatter(
        displayed_nitrate, displayed_phosphate, color=COLORS["vermillion"]
    )
    adjustment_axes[2].plot(
        displayed_adjustment_grid,
        displayed_adjustment_model["intercept"]
        + displayed_adjustment_model["slope"] * displayed_adjustment_grid,
        color=COLORS["blue"],
    )
    adjustment_axes[2].set(
        title=displayed_title, xlabel=displayed_x_label, ylabel=displayed_y_label
    )
    for adjustment_axis in adjustment_axes:
        adjustment_axis.grid(linestyle=":", alpha=0.25)
    adjustment_figure.tight_layout()

    two_column_panel(
        mo.vstack(
            [
                adjustment_view,
                mo.stat(
                    f"{displayed_r:.3f}", label="Displayed correlation", bordered=True
                ),
                association_summary,
            ]
        ),
        mo.vstack(
            [
                adjustment_figure,
                mo.callout(
                    mo.md(
                        f"The raw nitrate–phosphate correlation is "
                        f"**{nitrate_phosphate_r:.3f}**; after linear adjustment for "
                        f"distance it is **{river_partial_r:.3f}**. This change is "
                        "consistent with shared linear structure involving distance, "
                        "but it does not by itself establish a causal pathway."
                    ),
                    kind="warn",
                ),
            ]
        ),
        widths=(1.35, 2.65),
    )


@app.cell
def _(mo):
    mo.md(r"""
    ---

    ## 10. Regression extensions

    The same modeling language extends beyond a straight line:

    - **nonlinear regression** uses a nonlinear mean function, such as
      Michaelis–Menten enzyme kinetics;
    - **logistic regression** models the probability of a binary outcome through
      log odds;
    - **multiple regression** conditions a response on several predictors;
    - **errors-in-variables models** are needed when predictor measurement error is
      not negligible for the scientific goal.

    ### Fit an enzyme-kinetics curve

    The Michaelis–Menten model relates substrate concentration $s$ to reaction
    rate $v$:

    $$\hat v(s)=\frac{V_{max}s}{K_m+s},\qquad
    SSE(V_{max},K_m)=\sum_i\left(v_i-\frac{V_{max}s_i}{K_m+s_i}\right)^2.$$

    **Try this:** move $V_{max}$ while holding $K_m$ fixed, then move $K_m$ while
    holding $V_{max}$ fixed. The solid blue curve is your candidate model. Gray
    vertical segments show deviations from the seven observations to that curve;
    their squared lengths sum to the candidate SSE. Adjust both sliders to reduce
    it and compare with the dashed green best-fitting curve and the minimum SSE.
    All controls update immediately; the observed points and optimal fit stay fixed.

    The table reports both parameter pairs and their sums of squared deviations
    in $(\mathrm{µmol/min})^2$. The optimizer searches positive parameter values
    continuously, so the slider steps may prevent an exact match to its solution.
    Equal weighting is used because all simulated rates have the same noise SD.
    """)


@app.cell
def _(np, optimize):
    def michaelis_rate(substrate, vmax, km):
        return vmax * substrate / (km + substrate)

    michaelis_rng = np.random.default_rng(82438)
    michaelis_observed_x = np.array([0.2, 0.5, 1.0, 2.0, 4.0, 7.0, 10.0])
    michaelis_reference_y = michaelis_rate(michaelis_observed_x, 8.0, 1.5)
    michaelis_observed_y = michaelis_reference_y + michaelis_rng.normal(
        0, 0.3, michaelis_observed_x.size
    )
    michaelis_best_parameters, _parameter_covariance = optimize.curve_fit(
        michaelis_rate,
        michaelis_observed_x,
        michaelis_observed_y,
        p0=(8.0, 1.5),
        bounds=([0.0, 1e-8], [np.inf, np.inf]),
    )
    michaelis_best_sse = float(
        np.sum(
            (
                michaelis_observed_y
                - michaelis_rate(michaelis_observed_x, *michaelis_best_parameters)
            )
            ** 2
        )
    )
    return (
        michaelis_best_parameters,
        michaelis_best_sse,
        michaelis_observed_x,
        michaelis_observed_y,
        michaelis_rate,
    )


@app.cell
def _(mo):
    michaelis_vmax = mo.ui.slider(
        2.0,
        12.0,
        step=0.2,
        value=8.0,
        show_value=True,
        full_width=True,
        label="Vmax [µmol/min]",
    )
    michaelis_km = mo.ui.slider(
        0.2,
        5.0,
        step=0.1,
        value=1.5,
        show_value=True,
        full_width=True,
        label="Km [mM]",
    )
    return michaelis_km, michaelis_vmax


@app.cell
def _(
    COLORS,
    FIGURE_SIZE_STANDARD,
    compact_table,
    michaelis_best_parameters,
    michaelis_best_sse,
    michaelis_km,
    michaelis_observed_x,
    michaelis_observed_y,
    michaelis_rate,
    michaelis_vmax,
    mo,
    np,
    pd,
    plt,
    two_column_panel,
):
    substrate_grid = np.linspace(0.0, 10.0, 300)
    michaelis_curve = michaelis_rate(
        substrate_grid, michaelis_vmax.value, michaelis_km.value
    )
    michaelis_candidate_predictions = michaelis_rate(
        michaelis_observed_x, michaelis_vmax.value, michaelis_km.value
    )
    michaelis_candidate_sse = float(
        np.sum((michaelis_observed_y - michaelis_candidate_predictions) ** 2)
    )
    michaelis_comparison = compact_table(
        pd.DataFrame(
            {
                "Model": ["Candidate", "Best fit"],
                "Vmax [µmol/min]": [michaelis_vmax.value, michaelis_best_parameters[0]],
                "Km [mM]": [michaelis_km.value, michaelis_best_parameters[1]],
                "SSE [(µmol/min)²]": [michaelis_candidate_sse, michaelis_best_sse],
            }
        ),
        column_widths={
            "Model": 75,
            "Vmax [µmol/min]": 100,
            "Km [mM]": 75,
            "SSE [(µmol/min)²]": 110,
        },
        format_mapping={
            "Vmax [µmol/min]": "{:.3f}",
            "Km [mM]": "{:.3f}",
            "SSE [(µmol/min)²]": "{:.4f}",
        },
    )

    michaelis_figure, michaelis_axis = plt.subplots(figsize=FIGURE_SIZE_STANDARD)
    michaelis_axis.scatter(
        michaelis_observed_x,
        michaelis_observed_y,
        color=COLORS["orange"],
        edgecolor="white",
        linewidth=0.5,
        label="Simulated observations",
        zorder=4,
    )
    michaelis_axis.vlines(
        michaelis_observed_x,
        michaelis_candidate_predictions,
        michaelis_observed_y,
        color=COLORS["gray"],
        linewidth=1.5,
        label="Candidate residuals",
    )
    michaelis_axis.plot(
        substrate_grid,
        michaelis_curve,
        color=COLORS["blue"],
        linewidth=2,
        label="Candidate Michaelis–Menten curve",
    )
    michaelis_axis.plot(
        substrate_grid,
        michaelis_rate(substrate_grid, *michaelis_best_parameters),
        color=COLORS["green"],
        linestyle="--",
        linewidth=2,
        label="Best least-squares fit",
    )
    michaelis_axis.axhline(
        michaelis_vmax.value, color=COLORS["blue"], linestyle=":", alpha=0.5
    )
    michaelis_axis.axvline(
        michaelis_km.value, color=COLORS["purple"], linestyle=":", alpha=0.5
    )
    michaelis_axis.set(
        xlabel="Substrate concentration [mM]",
        ylabel="Reaction rate [µmol/min]",
        xlim=(0, 10),
        ylim=(0, 13),
    )
    michaelis_axis.grid(linestyle=":", alpha=0.3)
    michaelis_axis.legend(frameon=False, fontsize=8)
    michaelis_figure.tight_layout()

    two_column_panel(
        mo.vstack(
            [
                mo.md("**Adjust the candidate parameters**"),
                michaelis_vmax,
                michaelis_km,
                mo.stat(
                    f"{0.5 * michaelis_vmax.value:.2f} µmol/min",
                    label="Rate at substrate = Km",
                    bordered=True,
                ),
            ]
        ),
        mo.vstack(
            [
                michaelis_figure,
                michaelis_comparison,
                mo.callout(
                    mo.md(
                        "$V_{max}$ is the asymptotic rate, while $K_m$ is the "
                        "substrate concentration at half of $V_{max}$; dotted guides "
                        "mark your candidate values. "
                        f"Your candidate SSE is **{michaelis_candidate_sse:.4f}**, "
                        f"compared with **{michaelis_best_sse:.4f}** for the fit "
                        "to these observations. The data were simulated using "
                        "$V_{max}=8$ µmol/min and $K_m=1.5$ mM; noise means those "
                        "generating values need not minimize SSE for this sample."
                    ),
                    kind="info",
                ),
            ]
        ),
    )
    return (michaelis_candidate_sse,)


@app.cell
def _(mo):
    mo.md(r"""
    ---

    ## 11. Review questions

    Choose one answer for each question. Feedback appears immediately and explains
    the underlying reasoning.
    """)


@app.cell
def _(mo):
    review_question_1 = mo.ui.radio(
        {
            "The correlation coefficient is sufficient": "coefficient_only",
            "The scatter plot reveals structure hidden by the summaries": "plot_first",
            "Only the sample means matter": "means_only",
        },
        value=None,
        label="Choose one answer",
    )
    mo.vstack(
        [
            mo.md(
                "**1. Why must Anscombe's quartet be plotted even though its datasets have nearly identical correlations?**"
            ),
            review_question_1,
        ]
    )
    return (review_question_1,)


@app.cell
def _(review_feedback, review_question_1):
    review_feedback(
        review_question_1.value,
        correct_value="plot_first",
        correct_text="**Correct.** Identical numerical summaries can accompany linear, curved, clustered, or outlier-driven patterns.",
        incorrect_text={
            "coefficient_only": "The quartet has nearly identical r values despite very different point "
            "patterns. A single linear-association coefficient cannot reveal curvature or "
            "show that one point drives the fit.",
            "means_only": "Means locate the centers of the separate variables, but do not show how x and y "
            "are paired. The quartet also has nearly identical means; its scatter plots reveal "
            "the differences.",
        },
    )


@app.cell
def _(mo):
    review_question_2 = mo.ui.radio(
        {
            "Pearson correlation": "pearson",
            "Spearman rank correlation": "spearman",
            "Neither coefficient can summarize monotonic order": "neither",
        },
        value=None,
        label="Choose one answer",
    )
    mo.vstack(
        [
            mo.md(
                "**2. Two ordinal scores tend to increase together, but the distances between score levels are not meaningful. Which coefficient naturally summarizes this monotonic association?**"
            ),
            review_question_2,
        ]
    )
    return (review_question_2,)


@app.cell
def _(review_feedback, review_question_2):
    review_feedback(
        review_question_2.value,
        correct_value="spearman",
        correct_text="**Correct.** Spearman correlation uses ranks and measures monotonic association without requiring metric distances or a straight-line relationship.",
        incorrect_text={
            "pearson": "Pearson uses numerical distances between values and summarizes linear association. "
            "Arbitrary numerical codes for ordinal categories do not supply meaningful distances; "
            "Spearman uses their ordering instead.",
            "neither": "Spearman is designed to summarize monotonic association through ranks, even when "
            "numerical spacing is not meaningful. Tied categories receive average ranks.",
        },
    )


@app.cell
def _(mo):
    review_question_3 = mo.ui.radio(
        {
            "A confidence interval for the conditional mean": "mean_interval",
            "A prediction interval for a new observation": "prediction_interval",
            "The fitted value alone": "point_only",
        },
        value=None,
        label="Choose one answer",
    )
    mo.vstack(
        [
            mo.md(
                "**3. Which interval answers: ‘Where might the oxygen concentration of the next river measured at this flow speed fall?’**"
            ),
            review_question_3,
        ]
    )
    return (review_question_3,)


@app.cell
def _(review_feedback, review_question_3):
    review_feedback(
        review_question_3.value,
        correct_value="prediction_interval",
        correct_text="**Correct.** A prediction interval includes uncertainty in the fitted mean and residual variation among individual observations.",
        incorrect_text={
            "mean_interval": "A confidence interval for the conditional mean addresses average oxygen at that "
            "flow speed. A new river also varies around that mean, so its prediction "
            "interval includes residual scatter.",
            "point_only": "The fitted value is a point prediction and expresses no uncertainty. A prediction "
            "interval includes uncertainty in the fitted mean plus variation of a new "
            "observation around it.",
        },
    )


@app.cell
def _(mo):
    review_question_4 = mo.ui.radio(
        {
            "Flow speed causes 46% of oxygen": "causal",
            "The fitted model accounts for about 46% of observed oxygen variation": "model_statement",
            "46% of rivers have high oxygen": "river_fraction",
        },
        value=None,
        label="Choose one answer",
    )
    mo.vstack(
        [
            mo.md(
                "**4. What is a defensible interpretation of $R^2=0.46$ for oxygen regressed on flow speed?**"
            ),
            review_question_4,
        ]
    )
    return (review_question_4,)


@app.cell
def _(review_feedback, review_question_4):
    review_feedback(
        review_question_4.value,
        correct_value="model_statement",
        correct_text="**Correct.** $R^2$ describes the share of observed response variation accounted for by the fitted model, not a causal percentage or a fraction of observations.",
        incorrect_text={
            "causal": "R² describes the fitted model’s reduction in squared residual variation in this "
            "sample. An observational association does not show that intervening on flow speed "
            "would cause a corresponding oxygen change.",
            "river_fraction": "The 46% refers to a proportion of squared variation in oxygen, not a count of "
            "rivers. No threshold defining “high oxygen” enters the calculation of R².",
        },
    )


@app.cell
def _(mo):
    review_question_5 = mo.ui.radio(
        {
            "Distance is proven to be a causal confounder": "proven",
            "There is less remaining linear association after adjustment, but causal interpretation needs subject knowledge": "conditional",
            "Nitrate and phosphate are now independent": "independent",
        },
        value=None,
        label="Choose one answer",
    )
    mo.vstack(
        [
            mo.md(
                "**5. The nitrate–phosphate correlation falls from 0.862 to 0.337 after linear adjustment for distance. What follows?**"
            ),
            review_question_5,
        ]
    )
    return (review_question_5,)


@app.cell
def _(review_feedback, review_question_5):
    review_feedback(
        review_question_5.value,
        correct_value="conditional",
        correct_text="**Correct.** Partial correlation measures the association between the residuals after linearly regressing each variable on distance. The smaller coefficient describes less remaining linear association; it does not establish a causal explanation.",
        incorrect_text={
            "proven": "The reduction shows that linear adjustment removes part of the shared pattern. It does "
            "not establish distance’s causal role; that requires knowledge of the processes and a "
            "defensible causal model.",
            "independent": "The residual correlation is still 0.337, not zero. Even zero correlation would "
            "generally rule out only linear association, not every form of dependence.",
        },
    )


@app.cell
def _(mo):
    mo.md(r"""
    ---

    ## 12. Summary and bridge

    - Inspect the paired observations before compressing them into a coefficient.
    - Covariance retains product units; Pearson correlation is standardized,
      dimensionless, and specific to linear association.
    - Spearman correlation replaces values with ranks and summarizes monotonic order.
    - An observed correlation has sampling uncertainty. Fisher intervals use an
      approximately normal transformed scale; bootstrap intervals must resample
      complete paired rows.
    - Regression defines a directional response model. Least squares minimizes SSE,
      slopes retain scientific units, and residuals record model discrepancies.
    - $R^2$ describes in-sample variation accounted for by a model, not causation.
    - Confidence bands concern a conditional mean; prediction intervals concern a
      future observation.
    - Partial correlation describes remaining linear association after statistical
      adjustment. Causal language requires study design and biological knowledge.

    **Next:** Chapter 4 introduces null models and p-values. Chapter 5 will use those
    ideas to test a correlation coefficient and to compare regression-based models.
    """)


if __name__ == "__main__":
    app.run()
