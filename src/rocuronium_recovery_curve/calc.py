"""Calculation of recovery curves from patient TOF Excel files."""

import argparse
import datetime
import logging
import re
import traceback
from pathlib import Path
from typing import Dict, List, Optional

import kaleido
import kaleido.kaleido
import numpy as np
import polars as pl

from rocuronium_recovery_curve.constants import (
    MA_CASE_ID_COL,
    MA_COLS,
    MA_HEADER_ROW,
    MA_MED,
    MA_PATIENT_ID_COL,
    MA_SHEET_NAME,
    MA_TIME,
    MA_TIME_SEC,
    PC_CASE_ID_COL,
    PC_COLS,
    PC_HEADER_ROW,
    PC_PATIENT_ID,
    PC_SHEET_NAME,
    RC_COLS,
    RC_ELAPSED_TIME_COL,
    RC_HEADER_ROW,
    RC_MODE_COL,
    RC_NOISE_PRESENT_COL,
    RC_PEAK_MV_COL,
    RC_SHEET_IDX,
    RC_VALLEY_MV_COL,
)
from rocuronium_recovery_curve.plotting import (
    base_figure,
    plot_processed_data,
)
from rocuronium_recovery_curve.processing import (
    AMPLITUDE_COL,
    ELAPSED_SEC_COL,
    NORMALIZED_AMPLITUDE_COL,
    ROW_INDEX_COL,
    TOF_INDEX_COL,
    interval_processing,
    siginal_processing,
)

# constants
DEFAULT_AMPLITUDE_CHANGE_RATIO_THRESHOLD_FOR_MERGING: float = 0.1


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Adds the `calc` arguments to the given parser."""

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

    fig = base_figure(base_result_path.name, first_tofs_above_zero)

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
        if case_med_administration is not None and case_med_administration.height > 0:
            # Needed by the interval review app to recalculate the medication summary
            case_med_administration.write_csv(
                base_result_path.with_suffix(".medication.csv")
            )
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


def run(cli_args: argparse.Namespace):
    """
    Runs the calculation for all patient files in the given folder.
    """

    if cli_args.patient_folder is None:
        raise ValueError("Patient folder is not given.")

    if not cli_args.patient_folder.is_dir():
        raise ValueError(
            f"Patient folder {cli_args.patient_folder} is not a directory."
        )

    if cli_args.result_folder is None:
        raise ValueError("Result folder is not given.")

    if not cli_args.result_folder.is_dir():
        cli_args.result_folder.mkdir(parents=True, exist_ok=True)

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
