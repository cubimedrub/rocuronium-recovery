"""Plotly figures for the recovery curves."""

import logging
from typing import Dict, List, Optional, Tuple

import plotly.graph_objects as pgo
import polars as pl

from rocuronium_recovery_curve.constants import (
    BLACK,
    DARK_GREY,
    LIGHT_GREY,
    MA_DOSE,
    MA_DOSE_UNIT,
    MA_MED,
    MA_TIME_SEC,
    RC_ELAPSED_TIME_COL,
    SU_INTERVAL_END_TIME,
    SU_INTERVAL_SLOPE_MAX,
    SU_INTERVAL_SLOPE_MEAN,
    SU_INTERVAL_SLOPE_STD,
    SU_INTERVAL_START_TIME,
    SU_INTERVAL_TIME_TO_NEXT,
    SU_MEDICATION_DOSE,
    SU_MEDICATION_NAME,
    SU_MEDICATION_ORIGINAL_TIME,
    SU_MEDICATION_TIME_DIFF,
    SU_MEDICATION_UNIT,
)
from rocuronium_recovery_curve.processing import (
    ELAPSED_SEC_COL,
    GAUSSIAN_SIGNAL_SMOOTHING_SLOPE_INTERVALS_ID_COL,
    IS_RECOVERY_COL,
    NORMALIZED_AMPLITUDE_COL,
    IntervalStats,
    _as_float,
    _as_int,
    interval_stats,
)


def plot_processed_data(
    data: pl.DataFrame,
    case_med_administration: Optional[pl.DataFrame],
    fig: pgo.Figure,
    logger: logging.Logger,
    interval_bounds: Optional[Dict[int, Tuple[int, int]]] = None,
) -> Tuple[pgo.Figure, pl.DataFrame, pl.DataFrame]:
    """
    Plots the given data with Plotly, highlighting the recovery intervals

    Parameters
    ----------
    interval_bounds: Optional[Dict[int, Tuple[int, int]]]
        User adjusted (start, end) in elapsed seconds per 1-based recovery interval number.
        Intervals without an entry use their first and last data point.
    data: pl.DataFrame
        A DataFrame containing the first TOF events with a normalized amplitude above zero which is smoothed
        and derived to get the slope for each data point and marked intervals.
    case_med_administration: pl.DataFrame
        A DataFrame containing the medication administration for the case, which can be used to add vertical lines to the plot.
    fig: pgo.Figure
        A Plotly figure to which the intervals should be added.
    logger: logging.Logger
        Logger

    Returns
    -------
    (pgo.Figure, pl.DataFrame)
        The given figure with the intervals added and the patient summary with interval information added.
    """

    # Select recoveries
    recoveries = data.filter(pl.col(IS_RECOVERY_COL))

    # Group interval
    recovery_interval: List[Tuple[int, pl.DataFrame]] = [
        (interval_id[0], interval)
        for interval_id, interval in recoveries.group_by(
            GAUSSIAN_SIGNAL_SMOOTHING_SLOPE_INTERVALS_ID_COL
        )
    ]
    recovery_interval.sort(key=lambda g: g[0])

    # fig.add_scatter(
    #     x=data[ELAPSED_SEC_COL].to_list(),
    #     y=data[GAUSSIAN_SIGNAL_SMOOTHING_COL],
    #     text=data["Elapsed time (hh:mm:ss)"].to_list(),
    #     mode="lines",
    #     name="gaussian smoothed norm. TOF ampl. (> 0)",
    #     visible="legendonly",
    # )

    # Summary columns
    medication_original_time: List[float] = []
    medication_time_diff: List[float] = []
    medication_name: List[str] = []
    medication_dose: List[float] = []
    medication_unit: List[str] = []
    interval_start_time: List[int] = []
    interval_end_time: List[int] = []
    interval_slope_mean: List[float] = []
    interval_slope_max: List[float] = []
    interval_slope_std: List[float] = []
    interval_time_to_next: List[float] = []

    stats_per_interval: List[IntervalStats] = []
    for interval_idx, (_, interval) in enumerate(recovery_interval):
        default_start = _as_int(interval[ELAPSED_SEC_COL].first())
        default_end = _as_int(interval[ELAPSED_SEC_COL].last())
        start_s, end_s = (interval_bounds or {}).get(
            interval_idx + 1, (default_start, default_end)
        )
        stats_per_interval.append(interval_stats(data, start_s, end_s))

    for interval_idx in range(len(recovery_interval)):
        # fig.add_scatter(
        #     x=interval[ELAPSED_SEC_COL].to_list(),
        #     y=interval[GAUSSIAN_SIGNAL_SMOOTHING_COL],
        #     text=interval["Elapsed time (hh:mm:ss)"].to_list(),
        #     mode="lines",
        #     line=dict(color=CB_BLUISH_GREEN),
        #     name=f"gaussian smoothed norm. TOF ampl. (> 0) pos slope ({interval_id})",
        # )
        stats = stats_per_interval[interval_idx]
        interval_start_time.append(stats.start_s)
        interval_end_time.append(stats.end_s)
        interval_slope_mean.append(stats.slope_per_min)
        interval_slope_max.append(stats.slope_max)
        interval_slope_std.append(stats.slope_std)

        fig.add_scatter(
            x=[stats.start_s / 60, stats.end_s / 60],
            y=[stats.y_start, stats.y_end],
            mode="lines",
            line_dash="dash",
            line=dict(color=BLACK),
            name=f"interval: {interval_idx + 1}; slope {stats.slope_per_min:.4f} /min",
        )

        if interval_idx < len(recovery_interval) - 1:
            next_start_s = stats_per_interval[interval_idx + 1].start_s
            time_to_next = float(next_start_s - stats.end_s)

            interval_time_to_next.append(time_to_next / 60)

            fig.add_shape(
                type="line",
                xref="x",
                yref="paper",
                x0=stats.end_s / 60,
                x1=next_start_s / 60,
                y0=0.0,
                y1=0.0,
                line=dict(
                    color="MediumPurple",
                    width=4,
                    dash="dot",
                ),
            )

            time_to_next_midpoint = (stats.end_s + time_to_next / 2) / 60
            label_y = 0.05 if interval_idx % 2 == 0 else 0.1
            fig.add_scatter(
                x=[time_to_next_midpoint],
                y=[label_y],
                text=[f"{round(time_to_next / 60, 2)} min"],
                mode="text",
                textfont=dict(color="MediumPurple"),
                showlegend=False,
            )

        else:
            interval_time_to_next.append(0.0)

        if case_med_administration is not None and case_med_administration.height > 0:
            medcation = case_med_administration.filter(
                (pl.col(MA_TIME_SEC) >= stats.start_s)
                & (pl.col(MA_TIME_SEC) <= stats.end_s)
            )

            if medcation.height > 0:
                time_sec_col_idx = case_med_administration.columns.index(MA_TIME_SEC)
                med_col_idx = case_med_administration.columns.index(MA_MED)
                med_dose_col_idx = case_med_administration.columns.index(MA_DOSE)
                med_dose_unit_col_idx = case_med_administration.columns.index(
                    MA_DOSE_UNIT
                )

                for row in medcation.iter_rows():
                    med: str = row[med_col_idx][:2]
                    label: str = (
                        f"{med}: {row[med_dose_col_idx]} {row[med_dose_unit_col_idx]}"
                    )

                    medication_original_time.append(
                        _as_float(row[time_sec_col_idx]) / 60
                    )
                    medication_time_diff.append(
                        (float(stats.end_s) - _as_float(row[time_sec_col_idx])) / 60
                    )
                    medication_name.append(row[med_col_idx])
                    medication_dose.append(row[med_dose_col_idx])
                    medication_unit.append(row[med_dose_unit_col_idx])

                    fig.add_vline(
                        stats.end_s / 60,
                        line_dash="dash",
                        line_color=LIGHT_GREY,
                        annotation_text=label,
                        annotation_position="top right",
                        annotation_textangle=-90,
                    )

    patient_interval_summary = pl.DataFrame(
        {
            SU_INTERVAL_START_TIME: interval_start_time,
            SU_INTERVAL_END_TIME: interval_end_time,
            SU_INTERVAL_SLOPE_MEAN: interval_slope_mean,
            SU_INTERVAL_SLOPE_MAX: interval_slope_max,
            SU_INTERVAL_SLOPE_STD: interval_slope_std,
            SU_INTERVAL_TIME_TO_NEXT: interval_time_to_next,
        }
    )

    patient_medication_summery = pl.DataFrame(
        {
            SU_MEDICATION_ORIGINAL_TIME: medication_original_time,
            SU_MEDICATION_TIME_DIFF: medication_time_diff,
            SU_MEDICATION_NAME: medication_name,
            SU_MEDICATION_DOSE: medication_dose,
            SU_MEDICATION_UNIT: medication_unit,
        }
    )

    return (fig, patient_interval_summary, patient_medication_summery)


def base_figure(title: str, data: pl.DataFrame) -> pgo.Figure:
    """
    Creates the figure with the normalized TOF amplitude and the 25/50/75/100% reference lines.

    Parameters
    ----------
    title: str
        Figure title, usually the patient file name.
    data: pl.DataFrame
        TOF events with a normalized amplitude above zero.

    Returns
    -------
    pgo.Figure
    """
    fig = pgo.Figure()
    fig.update_layout(
        xaxis=dict(title=dict(text="minutes")),
        yaxis=dict(title=dict(text="amplitude")),
        title=dict(text=title),
    )

    fig.add_scatter(
        x=[s / 60 for s in data[ELAPSED_SEC_COL].to_list()],
        y=data[NORMALIZED_AMPLITUDE_COL].to_list(),
        text=data[RC_ELAPSED_TIME_COL].to_list(),
        mode="lines",
        line=dict(color=DARK_GREY),
        name="norm. TOF ampl. filtered",
    )

    for percentage in range(25, 101, 25):
        fig.add_hline(
            percentage / 100,
            line_dash="dash",
            line_color=LIGHT_GREY,
            annotation_text=f"{percentage}%",
            annotation_position="top right",
        )

    return fig
