"""
Interactive review of the recovery intervals.

Serves a small Dash app for a result folder produced by the main CLI. Start and end of each
recovery interval can be dragged; the secant slope is recalculated live and can be saved
back to the summary files.

    python -m rocuronium_recovery_curve review <result_folder>
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import polars as pl
from dash import Dash, Input, Output, State, dcc, html, no_update
from plotly import graph_objects as pgo

from rocuronium_recovery_curve.plotting import (
    base_figure,
    plot_processed_data,
)
from rocuronium_recovery_curve.processing import (
    ELAPSED_SEC_COL,
    GAUSSIAN_SIGNAL_SMOOTHING_SLOPE_INTERVALS_ID_COL,
    IS_RECOVERY_COL,
    _as_int,
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


def load_bounds(result_folder: Path, patient: str) -> Dict[int, Tuple[int, int]]:
    """Reads the saved user bounds, or an empty dict (= defaults) if there are none."""
    path = _path(result_folder, patient, ".intervals.json")
    if not path.is_file():
        return {}
    return {
        int(e["interval"]): (int(e["start_s"]), int(e["end_s"]))
        for e in json.loads(path.read_text())
    }


def save_bounds(
    result_folder: Path, patient: str, bounds: Dict[int, Tuple[int, int]]
) -> None:
    entries = [
        {"interval": i, "start_s": s, "end_s": e}
        for i, (s, e) in sorted(bounds.items())
    ]
    _path(result_folder, patient, ".intervals.json").write_text(
        json.dumps(entries, indent=2)
    )


def load_patient(
    result_folder: Path, patient: str
) -> Tuple[pl.DataFrame, Optional[pl.DataFrame]]:
    data = pl.read_csv(_path(result_folder, patient, PROCESSED_SUFFIX))
    med_path = _path(result_folder, patient, ".medication.csv")
    meds = pl.read_csv(med_path) if med_path.is_file() else None
    return data, meds


def default_bounds(data: pl.DataFrame) -> Dict[int, Tuple[int, int]]:
    """First and last elapsed second of each recovery interval, numbered from 1 like the plot."""
    recoveries = data.filter(pl.col(IS_RECOVERY_COL))
    groups = sorted(
        recoveries.group_by(GAUSSIAN_SIGNAL_SMOOTHING_SLOPE_INTERVALS_ID_COL),
        key=lambda g: g[0][0],
    )
    return {
        i + 1: (_as_int(g[ELAPSED_SEC_COL].first()), _as_int(g[ELAPSED_SEC_COL].last()))
        for i, (_, g) in enumerate(groups)
    }


def build_figure(
    patient: str,
    data: pl.DataFrame,
    meds: Optional[pl.DataFrame],
    bounds: Dict[int, Tuple[int, int]],
) -> Tuple[pgo.Figure, pl.DataFrame, pl.DataFrame, int]:
    """
    Returns the figure, both summaries and the index of the first editable shape.

    Editable start/end lines are the shapes from that index on, two per interval in order.
    """
    fig = base_figure(patient, data)
    fig, recovery_summary, medication_summary = plot_processed_data(
        data, meds, fig, logger, interval_bounds=bounds
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
    bounds: Dict[int, Tuple[int, int]],
    first_editable: int,
) -> Dict[int, Tuple[int, int]]:
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
            dcc.Store(id="first-editable"),
        ]
    )

    @app.callback(
        Output("graph", "figure"),
        Output("bounds", "data"),
        Output("first-editable", "data"),
        Input("patient", "value"),
        Input("graph", "relayoutData"),
        State("bounds", "data"),
        State("first-editable", "data"),
        prevent_initial_call=False,
    )
    def update(patient, relayout, stored_bounds, first_editable):
        if patient is None:
            return no_update, no_update, no_update

        data, meds = load_patient(result_folder, patient)
        triggered = _triggered_id()

        if triggered != "graph" or stored_bounds is None:
            bounds = default_bounds(data) | load_bounds(result_folder, patient)
        else:
            bounds = {int(k): (v[0], v[1]) for k, v in stored_bounds.items()}
            if not relayout or first_editable is None:
                return no_update, no_update, no_update
            new_bounds = bounds_from_relayout(relayout, bounds, first_editable)
            if new_bounds == bounds:
                return no_update, no_update, no_update
            bounds = new_bounds

        fig, _, _, first = build_figure(patient, data, meds, bounds)
        return fig, {str(k): list(v) for k, v in bounds.items()}, first

    @app.callback(
        Output("status", "children"),
        Input("save", "n_clicks"),
        State("patient", "value"),
        State("bounds", "data"),
        prevent_initial_call=True,
    )
    def save(_n_clicks, patient, stored_bounds):
        if patient is None or stored_bounds is None:
            return no_update
        bounds = {int(k): (v[0], v[1]) for k, v in stored_bounds.items()}
        data, meds = load_patient(result_folder, patient)
        fig, recovery_summary, medication_summary, _ = build_figure(
            patient, data, meds, bounds
        )
        # Saved plots should not show the editing handles
        fig.layout.shapes = fig.layout.shapes[: -2 * len(bounds)]

        save_bounds(result_folder, patient, bounds)
        recovery_summary.write_csv(
            _path(result_folder, patient, ".revocery_summary.tofs.tsv"), separator="\t"
        )
        medication_summary.write_csv(
            _path(result_folder, patient, ".medication_summary.tofs.tsv"),
            separator="\t",
        )
        fig.write_html(_path(result_folder, patient, ".html"))
        fig.write_image(_path(result_folder, patient, ".png"), width=1920, height=800)
        return f"Saved {patient}"

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
