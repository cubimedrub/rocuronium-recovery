import argparse
import datetime
import logging
import re
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import kaleido
import kaleido.kaleido
import numpy as np
import plotly.graph_objects as pgo
import polars as pl
from scipy.ndimage import gaussian_filter1d

# Recover curve (RC) original columns
RC_ELAPSED_TIME_COL: str = "Elapsed time (hh:mm:ss)"
RC_MODE_COL: str = "Mode"
RC_PEAK_MV_COL: str = "Peak (mV)"
RC_VALLEY_MV_COL: str = "Valley (mV)"
RC_NOISE_PRESENT_COL: str = "Noise Present"

RC_COLS: Tuple[str, ...] = (
    RC_ELAPSED_TIME_COL,
    RC_MODE_COL,
    RC_NOISE_PRESENT_COL,
    RC_PEAK_MV_COL,
    RC_VALLEY_MV_COL,
)

RC_SHEET_IDX: int = 1
RC_HEADER_ROW: int = 1

# Medicine administration (MA) original columns

MA_CASE_ID_COL: str = "caseid"
MA_TIME: str = "chartTime"
MA_MED: str = "medi"
MA_DOSE: str = "dose"
MA_DOSE_UNIT: str = "unit"

MA_COLS: Tuple[str, ...] = (MA_CASE_ID_COL, MA_TIME, MA_MED, MA_DOSE, MA_DOSE_UNIT)

MA_SHEET_NAME: str = "Meds all"
MA_HEADER_ROW: int = 0

# Patient / case ID (PC) original columns

PC_CASE_ID_COL: str = "caseid"
PC_PATIENT_ID: str = "PateintenID"

PC_COLS: Tuple[str, ...] = (
    PC_CASE_ID_COL,
    PC_PATIENT_ID,
)

PC_SHEET_NAME: str = "Klinik"
PC_HEADER_ROW: int = 0

# New columns for RC
ROW_INDEX_COL: str = "row_index"
TOF_INDEX_COL: str = "tof_index"
ELAPSED_SEC_COL: str = "elapsed_sec"
AMPLITUDE_COL: str = "amplitude"
NORMALIZED_AMPLITUDE_COL: str = "normalized_amplitude"
GAUSSIAN_SIGNAL_SMOOTHING_COL: str = "gaussian_signal_smoothing"
GAUSSIAN_SIGNAL_SMOOTHING_SLOPE_COL: str = "gaussian_signal_smoothing_slope"
GAUSSIAN_SIGNAL_SMOOTHING_SLOPE_INTERVALS_ID_COL: str = (
    "gaussian_signal_smoothing_slope_intervals_id"
)
TOF_COUNT_COL: str = "tof_count"
IS_RECOVERY_COL: str = "is_recovery"
TMP_SIGN_COL: str = "tmp_sign"


# New columns for MA
MA_TIME_SEC = "chartTimeSec"
MA_PATIENT_ID_COL = "patient_id"

# Summary columns SU_MEDICATION_ORIGINAL_TIME: str = "medication_original_time"
SU_MEDICATION_ORIGINAL_TIME: str = "medication_original_time"
SU_MEDICATION_NAME: str = "medication_name"
SU_MEDICATION_DOSE: str = "medication_dose"
SU_MEDICATION_UNIT: str = "medication_unit"
SU_MEDICATION_TIME_DIFF = "medication_time_diff"

SU_INTERVAL_START_TIME = "interval_start_time"
SU_INTERVAL_END_TIME = "interval_end_time"
SU_INTERVAL_SLOPE_MEAN = "interval_slope_mean"
SU_INTERVAL_SLOPE_MAX = "interval_slope_max"
SU_INTERVAL_SLOPE_STD = "interval_slope_std"
SU_INTERVAL_TIME_TO_NEXT = "interval_time_to_next"


# Some colorblind-friendly colors according to Wong https://www.nature.com/articles/nmeth.1618
BLACK = "rgb(0, 0, 0)"
DARK_GREY = "rgb(169, 169, 169)"
LIGHT_GREY = "rgb(211, 211, 211)"

# constants
DEFAULT_AMPLITUDE_CHANGE_RATIO_THRESHOLD_FOR_MERGING: float = 0.1


def cli() -> argparse.Namespace:
    """
    Parses command line arguments for the Rocuronium recovery curve analysis.
    """

    parser = argparse.ArgumentParser(
        description="Process patient files for rocuronium recovery curve analysis."
    )
    parser.add_argument(
        "patient_folder",
        type=Path,
        help="Path to the folder containing patient Excel files with Rocuronium recovery curves",
    )

    parser.add_argument(
        "result_folder", type=Path, help="Path to folder where results getting saved."
    )

    parser.add_argument(
        "--patiend-id-regex",
        type=str,
        default="\\d{4}\\S\\d{2}\\S\\d{2}\\S(?P<patient_id>\\d+)\\S\\d+",
        help="Regex to get case ID from patient file. Name patien ID group `patient_id`.",
    )

    parser.add_argument(
        "--med-administration",
        type=Path,
        help="Path to the Excel file containing medication timestamps",
    )

    parser.add_argument(
        "-v",
        "--verbose",
        default=0,
        action="count",
        help="Can be used multiple time, e.g. `-vvvvv` or `--verbose --verbose`, each time the log level is increased 1 => Warning, 2 => Info, 3 => Debug + Tracebacks",
    )

    parser.add_argument(
        "-t",
        "--merge-threshold",
        type=float,
        default=DEFAULT_AMPLITUDE_CHANGE_RATIO_THRESHOLD_FOR_MERGING,
        help=(
            "Amplitude merge threshold for merging subsequent negative slope intervals with recovery. "
            "If the amplitude change of the negative slope interval is smaller than this ratio of the previous positive slope interval, it gets merged into the recovery curve."
        ),
    )

    return parser.parse_args()


def verbosity_to_error_level(verbosity: int) -> int:
    """
    Converts a verbosity level to a logging error level.

    Parameters
    ----------
    verbosity : int
        The verbosity level, where higher values indicate more verbose output.

    Returns
    -------
    int
        The corresponding logging error level.
    """
    if verbosity <= 0:
        return logging.ERROR
    elif verbosity == 1:
        return logging.WARNING
    elif verbosity == 2:
        return logging.INFO
    else:
        return logging.DEBUG


def _as_float(val: object) -> float:
    """
    Small helper to make sure a value returned by Polars is actually a float.
    This is mostly done to make the typechecker happy as the hughe union typehint
    provided by Polars is not usable.
    """
    assert isinstance(val, (int, float))
    return float(val)


def _as_int(val: object) -> int:
    """
    Small helper to make sure a value returned by Polars is actually a int.
    This is mostly done to make the typechecker happy as the hughe union typehint
    provided by Polars is not usable.
    """
    assert isinstance(val, int)
    return int(val)


def hms_to_seconds(hms: str) -> int:
    """
    Converts a hours:minutes:seconds string into seconds.

    Parameters
    ----------
    hms : str
        A string in the format "hh:mm:ss".

    Returns
    -------
    int
        The total number of seconds represented by the input string.
    """
    time = datetime.datetime.strptime(hms, "%H:%M:%S")
    return time.hour * 3600 + time.minute * 60 + time.second


def chart_time_to_seconds(chart_time: str) -> int:
    """
    Converts a chart time from administration of medicince to seconds, ignoring the date.

    Parameters
    ----------
    chart_time : str
        A string in the format "YYYY-MM-dd hh:mm:ss.".

    Returns
    -------
    int
        The total number of seconds represented by the input string.
    """
    hms = chart_time.replace(".", " ").split(" ")[1]
    time = datetime.datetime.strptime(hms, "%H:%M:%S")
    return time.hour * 3600 + time.minute * 60 + time.second


def load_medication_administration(
    medication_administration_file: Path,
) -> pl.DataFrame:
    """
    Loads the medication administration data from the given Excel file, filters for Rocuronium and Sugammadex administrations, maps case IDs to patient IDs and converts chartTime to seconds.


    Parameters
    ----------
    medication_administration : Path
        The path to the Excel file containing medication administration data, which should have sheets for patient-case

    Returns
    -------
    pl.DataFrame

    Raises
    ------
    ValueError
        If the medication administration data cannot be read or if the case ID to patient ID mapping fails.
    """

    if not medication_administration_file.is_file():
        raise ValueError(
            f"Medication administration file {medication_administration_file} is not a file."
        )

    case_id_patient_id = pl.read_excel(
        medication_administration_file,
        sheet_name=PC_SHEET_NAME,
        columns=PC_COLS,
        read_options={"header_row": PC_HEADER_ROW},
    )
    medication_administration = pl.read_excel(
        medication_administration_file,
        sheet_name=MA_SHEET_NAME,
        columns=MA_COLS,
        read_options={"header_row": MA_HEADER_ROW},
    )

    # Some basic type checks
    if not isinstance(case_id_patient_id, pl.DataFrame):
        raise ValueError(
            f"Failed to read case ID to patient map from {medication_administration}'s sheet {PC_SHEET_NAME}"
        )

    if not isinstance(medication_administration, pl.DataFrame):
        raise ValueError(
            f"Failed to read medication administration from {medication_administration}'s sheet {MA_SHEET_NAME}"
        )

    pc_case_id_col_idx = case_id_patient_id.columns.index(PC_CASE_ID_COL)
    pc_patient_id_col_idx = case_id_patient_id.columns.index(PC_PATIENT_ID)

    case_id_to_patient_id_map: Dict[int, int] = {
        row[pc_case_id_col_idx]: row[pc_patient_id_col_idx]
        for row in case_id_patient_id.iter_rows()
        if row[pc_case_id_col_idx] is not None
        and row[pc_patient_id_col_idx] is not None
    }

    medication_administration = medication_administration.filter(
        (pl.col(MA_MED) == "Rocuronium") | (pl.col(MA_MED) == "Sugammadex")
    )
    medication_administration = medication_administration.with_columns(
        pl.Series(
            MA_TIME_SEC,
            [
                chart_time_to_seconds(chart_time)
                for chart_time in medication_administration[MA_TIME]
            ],
            dtype=pl.datatypes.Int64,
        )
    )

    medication_administration = medication_administration.with_columns(
        pl.Series(
            MA_PATIENT_ID_COL,
            [
                case_id_to_patient_id_map.get(int(case_id), None)
                for case_id in medication_administration[MA_CASE_ID_COL]
            ],
        )
    )

    return medication_administration


def siginal_processing(
    data: pl.DataFrame,
) -> pl.DataFrame:
    """
    Applies signal smoothing to the normalized amplitude of the recovery curve
    and calculates the derivate of the smoothed amplitude at any given point.

    Parameters
    ----------
    data : pl:DataFrame
        A DataFrame containing the first TOF events with amplitude above zero

    Returns
    -------
    pl.DataFrame
        The input DataFrame with additional columns for the smoothed signal and its slope.

    """

    if data.height == 0:
        raise ValueError("No TOF[0] with amplitude larger")

    if NORMALIZED_AMPLITUDE_COL not in data.columns:
        raise ValueError(
            f"Column {NORMALIZED_AMPLITUDE_COL} not found in data DataFrame"
        )

    if data.filter(pl.col(NORMALIZED_AMPLITUDE_COL) <= 0).height > 0:
        raise ValueError(
            "Data contains TOF events with non-positive amplitude, but only positive amplitudes are expected for signal smoothing"
        )

    # Use gaussian signal smoothing to filter noise, sigma controls width of gaussian distirbution. higher sigma == more smoothing
    gaussian_signal_smoothing = gaussian_filter1d(
        data[NORMALIZED_AMPLITUDE_COL], sigma=2
    )
    if len(gaussian_signal_smoothing) == 0:
        raise ValueError("Signal smoothing retuned empty array")

    data.insert_column(
        len(data.columns),
        pl.Series(GAUSSIAN_SIGNAL_SMOOTHING_COL, gaussian_signal_smoothing),
    )

    # Calculate derivate of the gaussian smoothing
    gaussian_signal_smoothing_slope = np.gradient(gaussian_signal_smoothing, 1.0)
    data.insert_column(
        len(data.columns),
        pl.Series(GAUSSIAN_SIGNAL_SMOOTHING_SLOPE_COL, gaussian_signal_smoothing_slope),
    )

    return data


def interval_processing(
    data: pl.DataFrame,
    interval_merge_threshold: float,
) -> pl.DataFrame:
    """
    Divides the datapoint into increasing (recovery) and decreasing (applied Rocuronium) intervals. Smaller decreasing intervals getting
    merged with the previous interval if it is a recovery interval and the change between those to lower or equal the given threshold.

    Parameters
    ----------
    data: pl.DataFrame
        A DataFrame containing the first TOF events with a normalized amplitude above zero which is smoothed
        and derived to get the slope for each data point.
    interval_merge_threshold : float
        Amplitude change threshold for merging subsequent negative slope intervals with recovery.
        If the amplitude change of the negative slope interval is smaller than this ratio of the previous positive

    Returns
    -------
    pl.DataFrame
        The input DataFrame with additional columns to mark the intervals
    """

    if data.height == 0:
        raise ValueError("No TOF[0] with amplitude larger")

    if GAUSSIAN_SIGNAL_SMOOTHING_SLOPE_COL not in data.columns:
        raise ValueError(
            f"Column {GAUSSIAN_SIGNAL_SMOOTHING_SLOPE_COL} not found in data DataFrame"
        )

    # Create a running ID for all intervals with consecutive pos or neg values (0 == pos)
    data = (
        data.with_columns(
            pl.when(pl.col(GAUSSIAN_SIGNAL_SMOOTHING_SLOPE_COL) >= 0)
            .then(1)
            .otherwise(-1)
            .alias(TMP_SIGN_COL)
        )
        .with_columns(
            (pl.col(TMP_SIGN_COL) != pl.col(TMP_SIGN_COL).shift(1))
            .fill_null(True)
            .cast(pl.Int64)  # Convert boolean (True/False) to 1/0
            .cum_sum()  # Cumulative sum creates the running ID
            .alias(GAUSSIAN_SIGNAL_SMOOTHING_SLOPE_INTERVALS_ID_COL),
        )
        .drop(TMP_SIGN_COL)
    )

    # Next we merge recovery intervals (positive slope) with subsequent negative intervals if the negative interval changes by MERGE_WITH_RECOVERY_RATIO_THRESHOLD less
    # Create a new boolean column to indicate recovery curves (mostly positive slope with occasional negative slope)
    data = data.with_columns(
        pl.when(pl.col(GAUSSIAN_SIGNAL_SMOOTHING_SLOPE_COL) >= 0)
        .then(True)
        .otherwise(False)
        .alias(IS_RECOVERY_COL)
    )

    # As long as something has merged we start again, set temporarily to true to start the while loop
    has_merged_intervals = True
    while has_merged_intervals:
        # assume nothing will change
        has_merged_intervals = False

        # Collect intervals
        slope_intervals: List[Tuple[int, pl.DataFrame]] = [
            (interval_id[0], interval)
            for interval_id, interval in data.group_by(
                GAUSSIAN_SIGNAL_SMOOTHING_SLOPE_INTERVALS_ID_COL
            )
        ]
        slope_intervals.sort(key=lambda i: i[0])

        for interval_index, (interval_id, interval) in enumerate(slope_intervals[:-1]):
            if not interval[IS_RECOVERY_COL].first():
                # skip intervals which are not part of the recovery
                continue

            # get subsequent interval
            next_interval_id, next_interval = slope_intervals[interval_index + 1]

            # calc change of this intervals
            interval_change = _as_float(interval[AMPLITUDE_COL].max()) - _as_float(
                interval[AMPLITUDE_COL].min()
            )

            if interval_change == 0:
                continue

            can_merge_with_next = True

            # Check if the next interval is not part of the recovery (negative slope)
            if not next_interval[IS_RECOVERY_COL].first():
                # calc change of next interval
                next_interval_change = abs(
                    _as_float(next_interval[AMPLITUDE_COL].max())
                    - _as_float(next_interval[AMPLITUDE_COL].min())
                )

                can_merge_with_next = (
                    next_interval_change / interval_change <= interval_merge_threshold
                )

            if can_merge_with_next:
                # merge intervals by setting recovery curve to true ...
                data = data.with_columns(
                    pl.when(
                        pl.col(GAUSSIAN_SIGNAL_SMOOTHING_SLOPE_INTERVALS_ID_COL)
                        == next_interval_id
                    )
                    .then(True)
                    .otherwise(pl.col(IS_RECOVERY_COL))
                    .alias(IS_RECOVERY_COL)
                )

                # and the interval ID to the same for both intervals
                data = data.with_columns(
                    pl.when(
                        pl.col(GAUSSIAN_SIGNAL_SMOOTHING_SLOPE_INTERVALS_ID_COL)
                        == next_interval_id
                    )
                    .then(interval_id)
                    .otherwise(pl.col(GAUSSIAN_SIGNAL_SMOOTHING_SLOPE_INTERVALS_ID_COL))
                    .alias(GAUSSIAN_SIGNAL_SMOOTHING_SLOPE_INTERVALS_ID_COL)
                )

                has_merged_intervals = True

                # break this run and start from again with the first interval
                break

    return data


def plot_processed_data(
    data: pl.DataFrame,
    case_med_administration: Optional[pl.DataFrame],
    fig: pgo.Figure,
    logger: logging.Logger,
) -> Tuple[pgo.Figure, pl.DataFrame, pl.DataFrame]:
    """
    Plots the given data with Plotly, highlighting the recovery intervals

    Parameters
    ----------
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

    for interval_idx, (interval_id, interval) in enumerate(recovery_interval):
        # fig.add_scatter(
        #     x=interval[ELAPSED_SEC_COL].to_list(),
        #     y=interval[GAUSSIAN_SIGNAL_SMOOTHING_COL],
        #     text=interval["Elapsed time (hh:mm:ss)"].to_list(),
        #     mode="lines",
        #     line=dict(color=CB_BLUISH_GREEN),
        #     name=f"gaussian smoothed norm. TOF ampl. (> 0) pos slope ({interval_id})",
        # )
        interval_start_time.append(_as_int(interval[ELAPSED_SEC_COL].first()))
        interval_end_time.append(_as_int(interval[ELAPSED_SEC_COL].last()))

        slope_std = _as_float(interval[GAUSSIAN_SIGNAL_SMOOTHING_SLOPE_COL].std())
        interval_slope_std.append(slope_std)

        max_slope = _as_float(interval[GAUSSIAN_SIGNAL_SMOOTHING_SLOPE_COL].max())
        interval_slope_max.append(slope_std)

        slope_threshold = max_slope - slope_std  # 3 sigma rule too loose

        interval_in_slope_capped = interval.filter(
            pl.col(GAUSSIAN_SIGNAL_SMOOTHING_SLOPE_COL) >= slope_threshold
        )

        # fig.add_scatter(
        #     x=interval_in_slope_capped[ELAPSED_SEC_COL],
        #     y=interval_in_slope_capped[GAUSSIAN_SIGNAL_SMOOTHING_COL],
        #     mode="lines",
        #     line_dash="dot",
        #     line=dict(color=CB_REDDISH_PURPLE),
        #     name=f"gaussian smoothed norm. TOF ampl. (> 0) pos slope ({interval_id}) >= slope std",
        # )

        interval_in_slope_capped_median = _as_float(
            interval_in_slope_capped[GAUSSIAN_SIGNAL_SMOOTHING_SLOPE_COL].median()
        )
        interval_slope_mean.append(interval_in_slope_capped_median)

        offset = _as_float(
            interval_in_slope_capped[GAUSSIAN_SIGNAL_SMOOTHING_COL].min()
        )

        marker_line_x = (
            np.array(
                [
                    interval_in_slope_capped[ELAPSED_SEC_COL].first(),
                    interval_in_slope_capped[ELAPSED_SEC_COL].last(),
                ],
                dtype=np.int64,
            )
            / 60
        )

        marker_line_y = np.array(
            [
                offset,
                interval_in_slope_capped_median * interval_in_slope_capped.height
                + offset,
            ],
            np.float64,
        )

        fig.add_scatter(
            x=marker_line_x,
            y=marker_line_y,
            mode="lines",
            line_dash="dash",
            line=dict(color=BLACK),
            name=f"interval: {interval_idx + 1}; slope (median) {interval_in_slope_capped_median}",
        )

        if interval_idx < len(recovery_interval) - 1:
            (_, next_interval) = recovery_interval[interval_idx + 1]
            time_to_next = _as_float(
                next_interval[ELAPSED_SEC_COL].first()
            ) - _as_float(interval[ELAPSED_SEC_COL].last())

            interval_time_to_next.append(time_to_next / 60)

            fig.add_shape(
                type="line",
                xref="x",
                yref="paper",
                x0=_as_float(interval[ELAPSED_SEC_COL].last()) / 60,
                x1=_as_float(next_interval[ELAPSED_SEC_COL].first()) / 60,
                y0=0.0,
                y1=0.0,
                line=dict(
                    color="MediumPurple",
                    width=4,
                    dash="dot",
                ),
            )

            time_to_next_midpoint = (
                _as_float(interval[ELAPSED_SEC_COL].last()) + time_to_next / 2
            ) / 60
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
                (pl.col(MA_TIME_SEC) >= interval[ELAPSED_SEC_COL].first())
                & (pl.col(MA_TIME_SEC) <= interval[ELAPSED_SEC_COL].last())
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
                        (
                            _as_float(interval_in_slope_capped[ELAPSED_SEC_COL].last())
                            - _as_float(row[time_sec_col_idx])
                        )
                        / 60
                    )
                    medication_name.append(row[med_col_idx])
                    medication_dose.append(row[med_dose_col_idx])
                    medication_unit.append(row[med_dose_unit_col_idx])

                    fig.add_vline(
                        _as_float(interval[ELAPSED_SEC_COL].last()) / 60,
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


def process_patient_file(
    patient_file: Path,
    med_administration: Optional[pl.DataFrame],
    base_result_path: Path,
    patiend_id_regex: re.Pattern,
    interval_merge_threshold: float,
    logger: logging.Logger,
):
    """
    Processes a single patient Excel file to extract TOF measurements, calculate amplitudes, and generate a recovery curve plot.

    Parameters
    ----------
    patient_file : Path
        The path to the patient's Excel file containing TOF measurements.
    med_administration : Optional[pl.DataFrame]
        Optional DataFrame containing medication administration timestamps, used for reference in the analysis.
    base_result_path : Path
        The base path for each result file derived by processing the given patient file. Just use with_suffix
    patiend_id_regex : Path
        A compiled regular expression used to extract the patient ID from the patient file name.
    interval_merge_threshold : float
        Amplitude change threshold for merging subsequent negative slope intervals with recovery.
        If the amplitude change of the negative slope interval is smaller than this ratio of the previous positive slope interval, it gets merged into the recovery curve.
    logger : loggging.Logger
        Logger
    """

    logger.info(f"Reading {patient_file}...")

    # Read file. Skip first row as it only contains device and vendor name. Only keep important columns
    muscle_activity = pl.read_excel(
        patient_file,
        sheet_id=RC_SHEET_IDX,
        columns=RC_COLS,
        read_options={"header_row": RC_HEADER_ROW},
    )

    # Make sure there was only one worksheet read.
    if not isinstance(muscle_activity, pl.DataFrame):
        raise ValueError(f"Expected a DataFrame, got {type(muscle_activity)}")

    # Get only TOF rows
    tofs = muscle_activity.filter(
        (pl.col(RC_MODE_COL) == "TOF") & (pl.col(RC_NOISE_PRESENT_COL).is_not_null())
    )

    # Add row index for easy reference later
    tofs = tofs.with_row_index(ROW_INDEX_COL)

    # Create a index over all cohesive measurments, indicated by having the same timestamp
    tof_index_col = np.full(tofs.height, -1, dtype=np.dtypes.Int64DType)
    for _group_name, group in tofs.group_by(RC_ELAPSED_TIME_COL, maintain_order=True):
        row_indexed = group[ROW_INDEX_COL].to_numpy()
        tof_index_col[row_indexed] = [i for i in range(len(row_indexed))]

    # Add the new index after the elsased time
    elapsed_time_col_idx = tofs.columns.index(RC_ELAPSED_TIME_COL)
    tofs.insert_column(
        elapsed_time_col_idx + 1, pl.Series(TOF_INDEX_COL, tof_index_col)
    )

    # Convert the elsapsed time into seconds and add it as new column after the H:M:S one.
    tofs.insert_column(
        elapsed_time_col_idx + 1,
        pl.Series(
            ELAPSED_SEC_COL,
            [hms_to_seconds(hms) for hms in tofs[RC_ELAPSED_TIME_COL]],
            dtype=pl.datatypes.Int64,
        ),
    )

    # Calculate amplitude
    tofs = tofs.with_columns(
        (pl.col(RC_PEAK_MV_COL) - pl.col(RC_VALLEY_MV_COL)).alias(AMPLITUDE_COL)
    )

    # get T_c for Normalization
    tc = tofs[AMPLITUDE_COL].first()

    if tc == 0.0:
        raise ValueError("T_c is zero!")

    tofs = tofs.filter(pl.col(AMPLITUDE_COL) <= tc)

    # Agressive filtering of outliers
    tofs = tofs.with_columns(
        (pl.col(AMPLITUDE_COL) / tc).alias(NORMALIZED_AMPLITUDE_COL)
    )

    # Select only first TOF event without noise
    first_tofs = (
        tofs.group_by(ELAPSED_SEC_COL, maintain_order=True)
        .first()
        .filter(pl.col(RC_NOISE_PRESENT_COL) == "No")
    )

    # Try to get the case ID from the file
    case_med_administration: Optional[pl.DataFrame] = None
    patient_id_match: str = patiend_id_regex.match(patient_file.stem)

    if patient_id_match is not None and med_administration is not None:
        patient_id = patient_id_match.group("patient_id")

        # Select medication for this case
        case_med_administration = med_administration.filter(
            pl.col(MA_PATIENT_ID_COL) == int(patient_id)
        )

        if case_med_administration.height > 0:
            # Sync med administration timestamps with Rucoronium timestamps by calculating the difference between first non calibration and first rocuronium administration
            # and remove it from the administration
            tc_start = tofs[ELAPSED_SEC_COL].first()
            med_administration_start = case_med_administration[MA_TIME_SEC].first()

            if not isinstance(tc_start, int):
                raise ValueError(
                    f"First TOF elapsed time is not an integer but {type(tc_start)}"
                )
            if not isinstance(med_administration_start, int):
                raise ValueError(
                    f"First medication administration timestamp is not an integer but {type(med_administration_start)}"
                )
            start_difference = abs(tc_start - med_administration_start)

            case_med_administration = case_med_administration.with_columns(
                pl.col(MA_TIME_SEC) - start_difference
            )

    # Select only positive amplitude
    first_tofs_above_zero = first_tofs.filter(pl.col(AMPLITUDE_COL) > 0)

    fig = pgo.Figure()
    fig.update_layout(
        xaxis=dict(title=dict(text="minutes")),
        yaxis=dict(title=dict(text="amplitude")),
    )

    # fig.add_scatter(
    #     x=first_tofs[ELAPSED_SEC_COL].to_list(),
    #     y=first_tofs[NORMALIZED_AMPLITUDE_COL].to_list(),
    #     text=first_tofs["Elapsed time (hh:mm:ss)"].to_list(),
    #     mode="lines",
    #     line=dict(color=CB_BLACK),
    #     name="norm. TOF ampl.",
    #     visible="legendonly",
    # )

    fig.add_scatter(
        x=[s / 60 for s in first_tofs_above_zero[ELAPSED_SEC_COL].to_list()],
        y=first_tofs_above_zero[NORMALIZED_AMPLITUDE_COL].to_list(),
        text=first_tofs["Elapsed time (hh:mm:ss)"].to_list(),
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

    # if case_med_administration is not None and case_med_administration.height > 0:
    #     time_sec_col_idx = case_med_administration.columns.index(MA_TIME_SEC)
    #     med_col_idx = case_med_administration.columns.index(MA_MED)
    #     med_dose_col_idx = case_med_administration.columns.index(MA_DOSE)
    #     med_dose_unit_col_idx = case_med_administration.columns.index(MA_DOSE_UNIT)
    #     for row in case_med_administration.iter_rows():
    #         med: str = row[med_col_idx][:2]
    #         label: str = f"{med}: {row[med_dose_col_idx]} {row[med_dose_unit_col_idx]}"
    #         fig.add_vline(
    #             row[time_sec_col_idx] / 60,
    #             line_dash="dash",
    #             line_color=LIGHT_GREY,
    #             annotation_text=label,
    #             annotation_position="top right",
    #             annotation_textangle=-90,
    #         )

    try:
        first_tofs_above_zero = siginal_processing(first_tofs_above_zero)
        first_tofs_above_zero = interval_processing(
            first_tofs_above_zero, interval_merge_threshold
        )
        (fig, patient_recovery_summary, patient_medication_summary) = (
            plot_processed_data(
                first_tofs_above_zero, case_med_administration, fig, logger
            )
        )

        fig.write_html(base_result_path.with_suffix(".html"))
        fig.write_image(base_result_path.with_suffix(".png"), width=1920, height=800)
        processed_tofs_path = base_result_path.with_suffix(".processed.tofs.csv")
        first_tofs_above_zero.write_csv(processed_tofs_path)
        recovery_summary_path = base_result_path.with_suffix(
            ".revocery_summary.tofs.tsv"
        )
        patient_recovery_summary.write_csv(recovery_summary_path, separator="\t")
        medication_summary_path = base_result_path.with_suffix(
            ".medication_summary.tofs.tsv"
        )
        patient_medication_summary.write_csv(medication_summary_path, separator="\t")

    except BaseException as err:
        errored_tofs_path = base_result_path.with_suffix(".errored.tofs.csv")
        tofs.write_csv(errored_tofs_path)
        logger.error(
            f"{err} @ {patient_file}, writing filtered TOF to {errored_tofs_path} for closer inspection"
        )
        if logger.level == logging.DEBUG:
            logger.error(traceback.format_exc())


def main():
    """
    Software entrypoint
    """

    cli_args = cli()
    if cli_args.patient_folder is None:
        raise ValueError("Patient folder is not given.")

    if not cli_args.patient_folder.is_dir():
        raise ValueError(
            f"Patient folder {cli_args.patient_folder} is not a directory."
        )

    if cli_args.result_folder is None:
        raise ValueError("Result folder is not given.")

    if not cli_args.result_folder.is_dir():
        cli_args.result_folder.create(parents=True, exist_ok=True)

    try:
        patiend_id_regex: re.Pattern = re.compile(cli_args.patiend_id_regex)
    except BaseException as err:
        raise ValueError(
            f"{err} @ Failed to compile patient ID regex {cli_args.patiend_id_regex}"
        )

    med_administration: Optional[pl.DataFrame] = None
    if cli_args.med_administration is not None:
        try:
            med_administration = load_medication_administration(
                cli_args.med_administration
            )
        except BaseException as err:
            raise ValueError(
                f"{err} @ Failed to load medication administration from {cli_args.med_administration}"
            )

    # Collect all Excel files and skip open files marked with `~`
    patient_files: List[Path] = list(
        filter(
            lambda p: not p.stem.startswith("~"), cli_args.patient_folder.glob("*.xlsx")
        )
    )

    logger = logging.getLogger(__name__)
    logger.setLevel(verbosity_to_error_level(cli_args.verbose))
    logging_terminal_handler = logging.StreamHandler()
    logger.addHandler(logging_terminal_handler)

    # Starts a "browser" in background to speed up image generation with Plotly (pl.Figure().write_image())
    kaleido.start_sync_server(width=1920, height=800)

    for patient_file in patient_files:
        base_result_path = cli_args.result_folder.joinpath(patient_file.stem)

        # setup log file for this patient file
        log_file = base_result_path.with_suffix(".log")
        logger.info("Logging to %s", log_file)
        logging_file_handler = logging.FileHandler(log_file, encoding="utf-8", mode="w")
        logger.addHandler(logging_file_handler)

        try:
            process_patient_file(
                patient_file,
                med_administration,
                base_result_path,
                patiend_id_regex,
                cli_args.merge_threshold,
                logger,
            )
        except BaseException as err:
            logger.error(f"{err} @ {patient_file}")
            if cli_args.verbose:
                logger.error(traceback.format_exc())

        logger.removeHandler(logging_file_handler)
        logging_file_handler.flush()

    kaleido.stop_sync_server()


if __name__ == "__main__":
    main()
