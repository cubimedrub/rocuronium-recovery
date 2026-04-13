import argparse
import datetime
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

MA_COLS: Tuple[str, ...] = (
    MA_CASE_ID_COL,
    MA_TIME,
    MA_MED,
)

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
GAUSSIAN_SIGNAL_SMOOTHING_SLOPE_INTERVALS_COL: str = (
    "gaussian_signal_smoothing_slope_intervals"
)
TOF_COUNT_COL: str = "tof_count"


# New columns for MA
MA_TIME_SEC = "chartTimeSec"
MA_PATIENT_ID_COL = "patient_id"


# Some colorblind-friendly colors according to Wong https://www.nature.com/articles/nmeth.1618
CB_BLACK = "rgb(0,0,0)"
CB_BLUISH_GREEN = "rgb(0,158,115)"
CB_BLUE = "rgb(0,114,178)"
CB_VERMILLION = "rgb(213,94,0)"
CB_REDDISH_PURPLE = "rgb(204, 121, 167)"


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
        "--verbose",
        default=False,
        action="store_true",
        help="Prints traceback on error for easier debugging",
    )

    return parser.parse_args()


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
        (pl.col(MA_MED) == "Rocuronium") | (pl.col(MA_MED) == "Rocuronium")
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


def siginal_smoothing(first_tofs_above_zero: pl.DataFrame, fig: pgo.Figure):
    """
    Applies signal smoothing to the recovery curve and calculates the intervals with increasing slope.

    Parameters
    ----------
    first_tofs_above_zero : pl:DataFrame
        A DataFrame containing the first TOF events with amplitude above zero
    fig : pgo.Figure
        The plotly figure to which the smoothed signal and intervals will be added
    """

    if first_tofs_above_zero.height == 0:
        raise ValueError("No TOF[0] with amplitude larger")

    # Use gaussian signal smoothing to filter noise, sigma controls width of gaussian distirbution. higher sigma == more smoothing
    gaussian_signal_smoothing = gaussian_filter1d(
        first_tofs_above_zero[NORMALIZED_AMPLITUDE_COL], sigma=2
    )
    if len(gaussian_signal_smoothing) == 0:
        raise ValueError("Signal smoothing retuned empty array")

    first_tofs_above_zero.insert_column(
        len(first_tofs_above_zero.columns),
        pl.Series(GAUSSIAN_SIGNAL_SMOOTHING_COL, gaussian_signal_smoothing),
    )

    # Calculate derivate of the gaussian smoothing
    gaussian_signal_smoothing_slope = np.gradient(gaussian_signal_smoothing, 1.0)
    first_tofs_above_zero.insert_column(
        len(first_tofs_above_zero.columns),
        pl.Series(GAUSSIAN_SIGNAL_SMOOTHING_SLOPE_COL, gaussian_signal_smoothing_slope),
    )

    # Positive value filter/expression for gaussian smoothing
    pos_filter = pl.col(GAUSSIAN_SIGNAL_SMOOTHING_SLOPE_COL) >= 0
    # Find positive value with no positive value before, i.e. the start of a positive derivate
    start_pos = pos_filter & (~pos_filter.shift(1).fill_null(False))

    # Create a running ID for all intervals with consecutive pos/neg values
    first_tofs_above_zero = first_tofs_above_zero.with_columns(
        start_pos.cast(pl.Int64)
        .cum_sum()
        .alias(GAUSSIAN_SIGNAL_SMOOTHING_SLOPE_INTERVALS_COL)
    )

    # Select positive gaussian derivate
    first_tofs_above_zero_pos_slope = first_tofs_above_zero.filter(
        pl.col(GAUSSIAN_SIGNAL_SMOOTHING_SLOPE_COL) > 0
    )

    # Group interval
    pos_slope_intervals: List[Tuple[Tuple[int], pl.DataFrame]] = list(
        iter(
            first_tofs_above_zero_pos_slope.group_by(
                GAUSSIAN_SIGNAL_SMOOTHING_SLOPE_INTERVALS_COL
            )
        )
    )

    pos_slope_intervals.sort(key=lambda g: g[0][0])

    fig.add_scatter(
        x=first_tofs_above_zero[ELAPSED_SEC_COL].to_list(),
        y=first_tofs_above_zero[GAUSSIAN_SIGNAL_SMOOTHING_COL],
        text=first_tofs_above_zero["Elapsed time (hh:mm:ss)"].to_list(),
        mode="lines",
        name="gaussian smoothed norm. TOF ampl. (> 0)",
        visible="legendonly",
    )

    for name, interval in pos_slope_intervals:
        median_slope = interval[GAUSSIAN_SIGNAL_SMOOTHING_SLOPE_COL].median()
        offset = interval[GAUSSIAN_SIGNAL_SMOOTHING_COL].min()

        try:
            median_line_x = np.array(
                [interval[ELAPSED_SEC_COL].first(), interval[ELAPSED_SEC_COL].last()],
                dtype=np.int64,
            )
        except Exception as err:
            raise ValueError(f"Failed to convert median line x values to float: {err}")

        if not isinstance(median_slope, float):
            raise ValueError(
                f"Median of slope is not a floating point numbers but {type(median_slope)}"
            )

        if not isinstance(offset, float):
            raise ValueError(
                f"Interval y-offset is not a floating point numbers but {type(offset)}"
            )

        median_line_y = np.array(
            [
                offset,  # median_slope * 0 + offset
                median_slope * interval.height + offset,
            ],
            np.float64,
        )

        fig.add_scatter(
            x=interval[ELAPSED_SEC_COL].to_list(),
            y=interval[GAUSSIAN_SIGNAL_SMOOTHING_COL],
            text=interval["Elapsed time (hh:mm:ss)"].to_list(),
            mode="lines",
            line=dict(color=CB_BLUISH_GREEN),
            name=f"gaussian smoothed norm. TOF ampl. (> 0) pos slope ({name[0]})",
        )

        fig.add_scatter(
            x=median_line_x,
            y=median_line_y,
            text=[
                format(f"{i} (median slope = {median_slope})") for i in median_line_x
            ],
            mode="lines",
            line_dash="dot",
            line=dict(color=CB_VERMILLION),
            name=f"gaussian smoothed norm. TOF ampl. (> 0) pos slope ({name[0]}) median",
        )


def process_patient_file(
    patient_file: Path,
    med_administration: Optional[pl.DataFrame],
    result_folder: Path,
    patiend_id_regex: re.Pattern,
    is_verbose: bool,
):
    """
    Processes a single patient Excel file to extract TOF measurements, calculate amplitudes, and generate a recovery curve plot.

    Parameters
    ----------
    patient_file : Path
        The path to the patient's Excel file containing TOF measurements.
    med_administration : Optional[pl.DataFrame]
        Optional DataFrame containing medication administration timestamps, used for reference in the analysis.
    result_folder : Path
        The path to the folder where the resulting plot and filtered data will be saved.
    patiend_id_regex : Path
        A compiled regular expression used to extract the patient ID from the patient file name.
    is_verbose : bool
        If True, prints detailed error messages and tracebacks for debugging purposes.
    """

    print(f"Reading {patient_file}...")
    base_result_path = result_folder.joinpath(patient_file.stem)

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
        xaxis=dict(title=dict(text="seconds")),
        yaxis=dict(title=dict(text="amplitude")),
    )

    fig.add_scatter(
        x=first_tofs[ELAPSED_SEC_COL].to_list(),
        y=first_tofs[NORMALIZED_AMPLITUDE_COL].to_list(),
        text=first_tofs["Elapsed time (hh:mm:ss)"].to_list(),
        mode="lines",
        line=dict(color=CB_BLACK),
        name="norm. TOF ampl.",
        visible="legendonly",
    )

    fig.add_scatter(
        x=first_tofs_above_zero[ELAPSED_SEC_COL].to_list(),
        y=first_tofs_above_zero[NORMALIZED_AMPLITUDE_COL].to_list(),
        text=first_tofs["Elapsed time (hh:mm:ss)"].to_list(),
        mode="lines",
        line=dict(color=CB_BLUE),
        name="norm. TOF ampl. > 0",
        visible="legendonly",
    )

    for percentage in range(25, 101, 25):
        fig.add_hline(
            percentage / 100,
            line_dash="dash",
            line_color="gray",
            annotation_text=f"{percentage}%",
            annotation_position="top right",
        )

    if case_med_administration is not None and case_med_administration.height > 0:
        time_sec_col_idx = case_med_administration.columns.index(MA_TIME_SEC)
        med_col_idx = case_med_administration.columns.index(MA_MED)
        for row in case_med_administration.iter_rows():
            label: str = row[med_col_idx]
            fig.add_vline(
                row[time_sec_col_idx],
                line_dash="dash",
                line_color="red",
                annotation_text=label[:2],
                annotation_position="top right",
            )

    try:
        siginal_smoothing(first_tofs_above_zero, fig)
    except BaseException as err:
        filtered_tofs_path = base_result_path.with_suffix(".tofs.csv")
        print(
            f"{err} @ {patient_file}, writing filtered TOF to {filtered_tofs_path} for closer inspection"
        )
        if is_verbose:
            print(traceback.format_exc())
        tofs.write_csv(filtered_tofs_path)

    fig.write_html(base_result_path.with_suffix(".html"))
    fig.write_image(base_result_path.with_suffix(".png"), width=1920, height=800)


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

    # Starts a "browser" in background to speed up image generation with Plotly (pl.Figure().write_image())
    kaleido.start_sync_server(width=1920, height=800)

    for patient_file in patient_files:
        try:
            process_patient_file(
                patient_file,
                med_administration,
                cli_args.result_folder,
                patiend_id_regex,
                cli_args.verbose,
            )
        except BaseException as err:
            print(f"{err} @ {patient_file}")
            if cli_args.verbose:
                print(traceback.format_exc())

    kaleido.stop_sync_server()


if __name__ == "__main__":
    main()
