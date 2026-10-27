"""
Interactive review of the recovery intervals.

Serves a small Dash app for a result folder produced by the main CLI. Start and end of each
recovery interval can be dragged; the secant slope is recalculated live and can be saved
back to the summary files.

    python -m rocuronium_recovery_curve review <result_folder>
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import polars as pl
from dash import Dash, Input, Output, State, dcc, html, no_update
from plotly import graph_objects as pgo

from rocuronium_recovery_curve.constants import (
    MANUAL_COLS,
    MANUAL_INTERVAL_ID_COL,
    MANUAL_SLOPE_WINDOW_COL,
)
from rocuronium_recovery_curve.plotting import (
    base_figure,
    plot_processed_data,
)
from rocuronium_recovery_curve.processing import (
    ELAPSED_SEC_COL,
    GAUSSIAN_SIGNAL_SMOOTHING_SLOPE_INTERVALS_ID_COL,
    _as_int,
    default_slope_window,
    recovery_intervals,
)

PROCESSED_SUFFIX = ".processed.tofs.csv"
EDITOR_SHAPE_COLOR = "rgb(213, 94, 0)"

logger = logging.getLogger(__name__)


def patient_names(result_folder: Path) -> List[str]:
    return sorted(
        p.name.removesuffix(PROCESSED_SUFFIX)
        for p in result_folder.glob(f"*{PROCESSED_SUFFIX}")
    )


def _path(result_folder: Path, patient: str, suffix: str) -> Path:
    return result_folder / f"{patient}{suffix}"


Window = Tuple[int, int]
Groups = Dict[
    int, Tuple[int, ...]
]  # first interval number -> all merged interval numbers


def load_bounds(result_folder: Path, patient: str) -> Tuple[Dict[int, Window], Groups]:
    """
    Reads the saved user bounds and merged intervals from the manual columns of the processed CSV,
    or empty dicts (= initial windows) if there are none.
    """
    saved = pl.read_csv(_path(result_folder, patient, PROCESSED_SUFFIX))
    bounds: Dict[int, Window] = {}
    groups: Groups = {}
    if not set(MANUAL_COLS) <= set(saved.columns):
        return bounds, groups

    number_of = {
        interval_id: number
        for number, (interval_id, _) in enumerate(recovery_intervals(saved), start=1)
    }
    edited = saved.filter(pl.col(MANUAL_SLOPE_WINDOW_COL))
    for first, rows in edited.group_by(MANUAL_INTERVAL_ID_COL):
        first = int(first[0])
        bounds[first] = (
            _as_int(rows[ELAPSED_SEC_COL].min()),
            _as_int(rows[ELAPSED_SEC_COL].max()),
        )
        members = tuple(
            sorted(
                {
                    number_of[i]
                    for i in saved.filter(pl.col(MANUAL_INTERVAL_ID_COL) == first)[
                        GAUSSIAN_SIGNAL_SMOOTHING_SLOPE_INTERVALS_ID_COL
                    ]
                }
            )
        )
        if len(members) > 1:
            groups[first] = members
    return bounds, groups


def load_state(
    result_folder: Path, patient: str, data: pl.DataFrame
) -> Tuple[Dict[int, Window], Groups]:
    """
    Editor state of a patient: initial windows, replaced by the saved ones, without the intervals
    that were merged into another one.
    """
    saved_bounds, groups = load_bounds(result_folder, patient)
    bounds = default_bounds(data) | saved_bounds
    for members in groups.values():
        for absorbed in members[1:]:
            bounds.pop(absorbed, None)
    return bounds, groups


def load_patient(
    result_folder: Path, patient: str
) -> Tuple[pl.DataFrame, Optional[pl.DataFrame]]:
    data = pl.read_csv(_path(result_folder, patient, PROCESSED_SUFFIX)).drop(
        MANUAL_COLS, strict=False
    )
    med_path = _path(result_folder, patient, ".medication.csv")
    meds = pl.read_csv(med_path) if med_path.is_file() else None
    return data, meds


def default_bounds(data: pl.DataFrame) -> Dict[int, Window]:
    """Starting slope window of each recovery interval, numbered from 1 like the plot."""
    return {
        i + 1: default_slope_window(interval)
        for i, (_, interval) in enumerate(recovery_intervals(data))
    }


def edited_bounds(
    data: pl.DataFrame, bounds: Dict[int, Window], groups: Groups
) -> Dict[int, Window]:
    """Only the intervals the user changed: other window than initially, or merged."""
    defaults = default_bounds(data)
    return {i: b for i, b in bounds.items() if b != defaults.get(i) or i in groups}


def merge_overlapping(
    data: pl.DataFrame, bounds: Dict[int, Window], groups: Groups
) -> Tuple[Dict[int, Window], Groups]:
    """
    Merges changed intervals whose windows overlap (touching counts) into one window spanning both.
    The merged interval is numbered like its first member; the others disappear from the bounds.
    """
    bounds = dict(bounds)
    groups = dict(groups)

    edited = sorted(edited_bounds(data, bounds, groups).items(), key=lambda i: i[1][0])
    chains: List[Tuple[Window, List[int]]] = []
    for interval, (start_s, end_s) in edited:
        if chains and start_s <= chains[-1][0][1]:
            (chain_start, chain_end), chain_members = chains[-1]
            chains[-1] = (
                (chain_start, max(chain_end, end_s)),
                chain_members + [interval],
            )
        else:
            chains.append(((start_s, end_s), [interval]))

    for window, chain_members in chains:
        if len(chain_members) < 2:
            continue
        members = sorted(m for i in chain_members for m in groups.pop(i, (i,)))
        for interval in chain_members:
            del bounds[interval]
        bounds[members[0]] = window
        groups[members[0]] = tuple(members)

    return bounds, groups


def expand_bounds(bounds: Dict[int, Window], groups: Groups) -> Dict[int, Window]:
    """Gives every member of a merged interval the merged window, as the plot expects it."""
    return {
        member: window
        for interval, window in bounds.items()
        for member in groups.get(interval, (interval,))
    }


def with_manual_columns(
    data: pl.DataFrame, bounds: Dict[int, Window], groups: Groups
) -> pl.DataFrame:
    """
    Adds the manual interval number (as in the plot, merged intervals share the first member's
    number) and whether the row lies in the user's slope window. Only for edited intervals.
    """
    expanded = expand_bounds(edited_bounds(data, bounds, groups), groups)
    if not expanded:
        return data

    first_member = {m: g[0] for g in groups.values() for m in g}
    rows = []
    for number, (interval_id, _) in enumerate(recovery_intervals(data), start=1):
        window = expanded.get(number)
        rows.append(
            {
                GAUSSIAN_SIGNAL_SMOOTHING_SLOPE_INTERVALS_ID_COL: interval_id,
                MANUAL_INTERVAL_ID_COL: first_member.get(number, number),
                "_start": window[0] if window else None,
                "_end": window[1] if window else None,
            }
        )
    numbers = pl.DataFrame(
        rows,
        schema={
            GAUSSIAN_SIGNAL_SMOOTHING_SLOPE_INTERVALS_ID_COL: data.schema[
                GAUSSIAN_SIGNAL_SMOOTHING_SLOPE_INTERVALS_ID_COL
            ],
            MANUAL_INTERVAL_ID_COL: pl.Int64,
            "_start": pl.Int64,
            "_end": pl.Int64,
        },
    )
    return (
        data.join(
            numbers,
            on=GAUSSIAN_SIGNAL_SMOOTHING_SLOPE_INTERVALS_ID_COL,
            how="left",
            maintain_order="left",
        )
        .with_columns(
            pl.col(ELAPSED_SEC_COL)
            .is_between(pl.col("_start"), pl.col("_end"))
            .fill_null(False)
            .alias(MANUAL_SLOPE_WINDOW_COL)
        )
        .drop("_start", "_end")
    )


def build_figure(
    patient: str,
    data: pl.DataFrame,
    meds: Optional[pl.DataFrame],
    bounds: Dict[int, Window],
    groups: Groups,
) -> Tuple[pgo.Figure, pl.DataFrame, pl.DataFrame, int]:
    """
    Returns the figure, both summaries and the index of the first editable shape.

    Editable start/end lines are the shapes from that index on, two per interval in order.
    """
    fig = base_figure(patient, data)
    fig, recovery_summary, medication_summary = plot_processed_data(
        data,
        meds,
        fig,
        logger,
        interval_bounds=expand_bounds(edited_bounds(data, bounds, groups), groups),
    )
    first_editable = len(fig.layout.shapes)
    fig.update_layout(uirevision=patient)  # keep zoom while dragging
    for start_s, end_s in (bounds[i] for i in sorted(bounds)):
        for x in (start_s, end_s):
            fig.add_shape(
                type="line",
                xref="x",
                yref="paper",
                x0=x / 60,
                x1=x / 60,
                y0=0.0,
                y1=1.0,
                line=dict(color=EDITOR_SHAPE_COLOR, width=2),
                editable=True,
            )
    return fig, recovery_summary, medication_summary, first_editable


def bounds_from_relayout(
    relayout: Dict[str, float],
    bounds: Dict[int, Window],
    first_editable: int,
) -> Dict[int, Window]:
    """Applies `shapes[n].x0` edits from Plotly's relayoutData onto the interval bounds."""
    updated = dict(bounds)
    interval_numbers = sorted(bounds)
    for key, value in relayout.items():
        if not (key.startswith("shapes[") and key.endswith("].x0")):
            continue
        shape_idx = int(key[len("shapes[") : -len("].x0")]) - first_editable
        if not 0 <= shape_idx < 2 * len(interval_numbers):
            continue
        interval = interval_numbers[shape_idx // 2]
        start_s, end_s = updated[interval]
        seconds = int(round(value * 60))
        updated[interval] = (
            (seconds, end_s) if shape_idx % 2 == 0 else (start_s, seconds)
        )
    return updated


def save_results(
    result_folder: Path,
    patient: str,
    data: pl.DataFrame,
    meds: Optional[pl.DataFrame],
    bounds: Dict[int, Window],
    groups: Groups,
) -> None:
    """Writes the processed data with the manual interval columns, the summaries and the plots."""
    fig, recovery_summary, medication_summary, first_editable = build_figure(
        patient, data, meds, bounds, groups
    )
    # Saved plots should not show the editing handles
    fig.layout.shapes = fig.layout.shapes[:first_editable]

    with_manual_columns(data, bounds, groups).write_csv(
        _path(result_folder, patient, PROCESSED_SUFFIX)
    )
    recovery_summary.write_csv(
        _path(result_folder, patient, ".revocery_summary.tofs.tsv"), separator="\t"
    )
    medication_summary.write_csv(
        _path(result_folder, patient, ".medication_summary.tofs.tsv"), separator="\t"
    )
    fig.write_html(_path(result_folder, patient, ".html"))
    fig.write_image(_path(result_folder, patient, ".png"), width=1920, height=800)


def create_app(result_folder: Path) -> Dash:
    app = Dash(__name__)
    patients = patient_names(result_folder)

    app.layout = html.Div(
        [
            dcc.Dropdown(
                patients,
                patients[0] if patients else None,
                id="patient",
                clearable=False,
            ),
            html.Button("Save", id="save"),
            html.Span(id="status", style={"marginLeft": "1em"}),
            dcc.Graph(
                id="graph",
                config={"edits": {"shapePosition": True}},
                style={"height": "80vh"},
            ),
            # user bounds per interval: {"1": [start_s, end_s], ...}
            dcc.Store(id="bounds"),
            # merged intervals: {"9": [9, 10], ...}
            dcc.Store(id="groups"),
            dcc.Store(id="first-editable"),
        ]
    )

    @app.callback(
        Output("graph", "figure"),
        Output("bounds", "data"),
        Output("groups", "data"),
        Output("first-editable", "data"),
        Output("status", "children"),
        Input("patient", "value"),
        Input("graph", "relayoutData"),
        Input("save", "n_clicks"),
        State("bounds", "data"),
        State("groups", "data"),
        State("first-editable", "data"),
        prevent_initial_call=False,
    )
    def update(
        patient, relayout, _n_clicks, stored_bounds, stored_groups, first_editable
    ):
        unchanged = (no_update,) * 5
        if patient is None:
            return unchanged

        data, meds = load_patient(result_folder, patient)
        triggered = _triggered_id()
        status = no_update

        if triggered not in ("graph", "save") or stored_bounds is None:
            bounds, groups = load_state(result_folder, patient, data)
        else:
            bounds = {int(k): (v[0], v[1]) for k, v in stored_bounds.items()}
            groups = {int(k): tuple(v) for k, v in (stored_groups or {}).items()}
            if triggered == "graph":
                if not relayout or first_editable is None:
                    return unchanged
                new_bounds = bounds_from_relayout(relayout, bounds, first_editable)
                if new_bounds == bounds:
                    return unchanged
                bounds = new_bounds
            else:
                bounds, groups = merge_overlapping(data, bounds, groups)
                save_results(result_folder, patient, data, meds, bounds, groups)
                # Show what was written, as if the patient was opened anew
                bounds, groups = load_state(result_folder, patient, data)
                status = f"Saved and reloaded {patient}"

        fig, _, _, first = build_figure(patient, data, meds, bounds, groups)
        return (
            fig,
            {str(k): list(v) for k, v in bounds.items()},
            {str(k): list(v) for k, v in groups.items()},
            first,
            status,
        )

    return app


def _triggered_id() -> Optional[str]:
    from dash import ctx

    return ctx.triggered_id


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Adds the `review` arguments to the given parser."""

    parser.add_argument(
        "result_folder",
        type=Path,
        help="Path to the result folder of the `calc` command.",
    )
    parser.add_argument("--port", type=int, default=8050, help="Port of the Dash app.")


def run(args: argparse.Namespace) -> None:
    """
    Starts the review app.
    """

    if not args.result_folder.is_dir():
        raise ValueError(f"{args.result_folder} is not a directory.")

    create_app(args.result_folder).run(port=args.port)
