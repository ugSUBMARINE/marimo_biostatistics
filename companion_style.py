"""Shared visual components for the biostatistics companion notebooks.

The helpers in this module standardize presentation without hiding statistical
calculations.  Keep dataset-specific values, prose, and plotting logic in the
individual notebooks.
"""

from collections.abc import Callable, Sequence
from typing import Any

import marimo as mo

COLORS = {
    "orange": "#E69F00",
    "sky": "#56B4E9",
    "green": "#009E73",
    "yellow": "#F0E442",
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "purple": "#CC79A7",
    "gray": "#6B7280",
}

FIGURE_SIZE_STANDARD = (6.8, 3.8)
FIGURE_SIZE_SHALLOW = (6.8, 3.4)
FIGURE_SIZE_COMPACT = (5.8, 3.2)
FIGURE_SIZE_LINKED = (7.6, 3.8)


def two_column_panel(
    controls: object,
    results: object,
    *,
    widths: Sequence[float] = (1, 3),
    gap: float = 0.5,
) -> mo.Html:
    """Place controls and results in the standard responsive panel."""

    return mo.hstack(
        [controls, results],
        widths=list(widths),
        gap=gap,
        align="center",
        wrap=True,
    )


def compact_table(
    dataframe: Any,
    *,
    column_widths: dict[str, int] | None = None,
    format_mapping: dict[str, str | Callable[..., Any]] | None = None,
    wrapped_columns: list[str] | None = None,
    label: str = "",
) -> Any:
    """Render a small result table without exploratory-table controls."""

    return mo.ui.table(
        dataframe,
        pagination=False,
        selection=None,
        show_search=False,
        show_column_summaries=False,
        show_data_types=False,
        show_download=False,
        column_widths=column_widths,
        format_mapping=format_mapping,
        wrapped_columns=wrapped_columns,
        label=label,
    )


def dataset_action_buttons(
    *,
    on_generate: Callable[[Any], Any],
    on_reset: Callable[[Any], Any],
    generate_label: str = "Generate a new dataset",
    reset_label: str = "Reset to lecture dataset",
) -> tuple[Any, Any]:
    """Create consistently styled generate and reset buttons."""

    generate_button = mo.ui.button(
        label=generate_label,
        kind="success",
        on_click=on_generate,
        full_width=True,
    )
    reset_button = mo.ui.button(
        label=reset_label,
        on_click=on_reset,
        full_width=True,
    )
    return generate_button, reset_button


def review_feedback(
    selected_value: Any,
    *,
    correct_value: Any,
    correct_text: str,
    incorrect_text: str,
    unanswered_text: str = "Select an answer above.",
) -> mo.Html:
    """Return the standard immediate-feedback callout for a review question."""

    if selected_value is None:
        text, kind = unanswered_text, "neutral"
    elif selected_value == correct_value:
        text, kind = correct_text, "success"
    else:
        text, kind = incorrect_text, "danger"
    return mo.callout(mo.md(text), kind=kind)


__all__ = [
    "COLORS",
    "FIGURE_SIZE_COMPACT",
    "FIGURE_SIZE_LINKED",
    "FIGURE_SIZE_SHALLOW",
    "FIGURE_SIZE_STANDARD",
    "compact_table",
    "dataset_action_buttons",
    "review_feedback",
    "two_column_panel",
]
