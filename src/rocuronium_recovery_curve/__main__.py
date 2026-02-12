import argparse
import datetime
from pathlib import Path
from typing import List, Tuple

import numpy as np
import plotly.graph_objects as pgo
import polars as pl
from scipy.ndimage import gaussian_filter1d

# Original columns
ELAPSED_TIME_COL: str = "Elapsed time (hh:mm:ss)"
MODE_COL: str = "Mode"
PEAK_MV_COL: str = "Peak (mV)"
VALLEY_MV_COL: str = "Valley (mV)"
NOISE_PRESENT_COL: str = "Noise Present"

COLS: Tuple[str, ...] = (
    ELAPSED_TIME_COL,
    MODE_COL,
    NOISE_PRESENT_COL,
    PEAK_MV_COL,
    VALLEY_MV_COL,
)

SHEET_IDX: int = 1
HEADER_ROW: int = 1

# New columns
ROW_INDEX_COL: str = "row_index"
TOF_INDEX_COL: str = "tof_index"
ELAPSED_SEC_COL: str = "elapsed_sec"
AMPLITUDE_COL: str = "amplitude"
NORMALIZED_AMPLITUDE_COL: str = "normalized_amplitude"
GAUSSIAN_SIGNAL_SMOOTHING_COL: str = "gaussian_signal_smoothing"
GAUSSIAN_SIGNAL_SMOOTHING_SLOPE_COL: str = "gaussian_signal_smoothing_slope"
GAUSSIAN_SIGNAL_SMOOTHING_SLOPE_INTERVALS_COL: str = "gaussian_signal_smoothing_slope_intervals"
TOF_COUNT_COL: str = "tof_count"


# Some colorblind-friendly colors according to Wong https://www.nature.com/articles/nmeth.1618
CB_BLACK = "rgb(0,0,0)"
CB_BLUISH_GREEN = "rgb(0,158,115)"
CB_BLUE = "rgb(0,114,178)"
CB_VERMILLION = "rgb(213,94,0)"

def cli() -> argparse.Namespace:
    """
    Parses command line arguments for the Rocuronium recovery curve analysis.
    """

    parser = argparse.ArgumentParser(description="Process patient files for rocuronium recovery curve analysis.")
    parser.add_argument(
        "patient_folder",
        type=Path,
        help="Path to the folder containing patient Excel files with Rocuronium recovery curves"
    )

    parser.add_argument(
        "result_folder",
        type=Path,
        help="Path to folder where results getting saved."
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
    return time.hour*3600 + time.minute*60 + time.second

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
    gaussian_signal_smoothing = gaussian_filter1d(first_tofs_above_zero[NORMALIZED_AMPLITUDE_COL], sigma=2)
    if len(gaussian_signal_smoothing) == 0:
        raise ValueError("Signal smoothing retuned empty array")

    first_tofs_above_zero.insert_column(
        len(first_tofs_above_zero.columns),
        pl.Series(GAUSSIAN_SIGNAL_SMOOTHING_COL, gaussian_signal_smoothing)
    )

    # Calculate derivate of the gaussian smoothing
    gaussian_signal_smoothing_slope = np.gradient(gaussian_signal_smoothing, 1.0)
    first_tofs_above_zero.insert_column(
        len(first_tofs_above_zero.columns),
        pl.Series(GAUSSIAN_SIGNAL_SMOOTHING_SLOPE_COL, gaussian_signal_smoothing_slope)
    )

    # Positive value filter/expression for gaussian smoothing
    pos_filter = pl.col(GAUSSIAN_SIGNAL_SMOOTHING_SLOPE_COL) >= 0
    # Find positive value with no positive value before, i.e. the start of a positive derivate
    start_pos = pos_filter & (~pos_filter.shift(1).fill_null(False))

    # Create a running ID for all intervals with consecutive pos/neg values
    first_tofs_above_zero = first_tofs_above_zero.with_columns(
        start_pos.cast(pl.Int64).cum_sum().alias(GAUSSIAN_SIGNAL_SMOOTHING_SLOPE_INTERVALS_COL)
    )

    # Select positive gaussian derivate
    first_tofs_above_zero_pos_slope = first_tofs_above_zero.filter(pl.col(GAUSSIAN_SIGNAL_SMOOTHING_SLOPE_COL) > 0)

    # Group interval
    pos_slope_intervals: List[Tuple[Tuple[int], pl.DataFrame]] = list(iter(first_tofs_above_zero_pos_slope.group_by(GAUSSIAN_SIGNAL_SMOOTHING_SLOPE_INTERVALS_COL)))

    pos_slope_intervals.sort(key=lambda g: g[0][0])


    fig.add_scatter(
        x=first_tofs_above_zero[ELAPSED_SEC_COL].to_list(),
        y=first_tofs_above_zero[GAUSSIAN_SIGNAL_SMOOTHING_COL],
        text=first_tofs_above_zero["Elapsed time (hh:mm:ss)"].to_list(),
        mode="lines",
        name="gaussian smoothed norm. TOF ampl. (> 0)",
        visible="legendonly"
    )

    for (name, interval) in pos_slope_intervals:
        median = interval[GAUSSIAN_SIGNAL_SMOOTHING_SLOPE_COL].median()
        offset = interval[GAUSSIAN_SIGNAL_SMOOTHING_COL].min()

        if not isinstance(median, float):
            raise ValueError(f"Median of slope is not a floating point numbers but {type(median)}")

        if not isinstance(offset, float):
            raise ValueError(f"Interval y-offset is not a floating point numbers but {type(offset)}")

        median_line = np.array([median * x + offset for x in range(interval.height)])

        fig.add_scatter(
            x=interval[ELAPSED_SEC_COL].to_list(),
            y=interval[GAUSSIAN_SIGNAL_SMOOTHING_COL],
            text=interval["Elapsed time (hh:mm:ss)"].to_list(),
            mode="lines",
            line=dict(color=CB_BLUISH_GREEN),
            name=f"gaussian smoothed norm. TOF ampl. (> 0) pos slope ({name[0]})"
        )

        fig.add_scatter(
            x=interval[ELAPSED_SEC_COL].to_list(),
            y=median_line,
            text=interval["Elapsed time (hh:mm:ss)"],
            mode="lines",
            line_dash="dot",
            line=dict(color=CB_VERMILLION),
            name=f"gaussian smoothed norm. TOF ampl. (> 0) pos slope ({name[0]}) median",
        )

def process_patient_file(patient_file: Path, result_folder: Path):
    """
    Processes a single patient Excel file to extract TOF measurements, calculate amplitudes, and generate a recovery curve plot.

    Parameters
    ----------
    patient_file : Path
        The path to the patient's Excel file containing TOF measurements.
    result_folder : Path
        The path to the folder where the resulting plot and filtered data will be saved.
    """

    print(f"Reading {patient_file}...")
    base_result_path = result_folder.joinpath(patient_file.stem)

    # Read file. Skip first row as it only contains device and vendor name. Only keep important columns
    muscle_activity = pl.read_excel(patient_file, sheet_id=SHEET_IDX, columns=COLS, read_options={"header_row": HEADER_ROW})

    # Make sure there was only one worksheet read.
    if not isinstance(muscle_activity, pl.DataFrame):
        raise ValueError(f"Expected a DataFrame, got {type(muscle_activity)}")

    # Get only TOF rows
    tofs = muscle_activity.filter((pl.col(MODE_COL) == "TOF") & (pl.col(NOISE_PRESENT_COL).is_not_null()) )

    # Add row index for easy reference later
    tofs = tofs.with_row_index(ROW_INDEX_COL)

    # Create a index over all cohesive measurments, indicated by having the same timestamp
    tof_index_col = np.full(tofs.height, -1, dtype=np.dtypes.Int64DType)
    for (_group_name, group) in tofs.group_by(ELAPSED_TIME_COL, maintain_order=True):
        row_indexed = group[ROW_INDEX_COL].to_numpy()
        tof_index_col[row_indexed] = [i for i in range(len(row_indexed))]

    # Add the new index after the elsased time
    elapsed_time_col_idx = tofs.columns.index(ELAPSED_TIME_COL)
    tofs.insert_column(
        elapsed_time_col_idx + 1,
        pl.Series(TOF_INDEX_COL, tof_index_col)
    )

    # Convert the elsapsed time into seconds and add it as new column after the H:M:S one.
    tofs.insert_column(
         elapsed_time_col_idx + 1,
         pl.Series(
             ELAPSED_SEC_COL,
             [hms_to_seconds(hms) for hms in tofs[ELAPSED_TIME_COL]],
             dtype=pl.datatypes.Int64
         )
     )

    # Calculate amplitude
    tofs = tofs.with_columns((pl.col(PEAK_MV_COL) - pl.col(VALLEY_MV_COL)).alias(AMPLITUDE_COL))

    # get T_c for Normalization
    tc = tofs[AMPLITUDE_COL].first()

    if tc == 0.0:
        raise ValueError("T_c is zero!")

    tofs = tofs.with_columns((pl.col(AMPLITUDE_COL) / tc).alias(NORMALIZED_AMPLITUDE_COL))

    # Select only first TOF event without noise
    first_tofs = tofs.group_by(ELAPSED_SEC_COL, maintain_order=True).first().filter(pl.col(NOISE_PRESENT_COL) == "No")

    # Select only positive amplitude
    first_tofs_above_zero = first_tofs.filter(pl.col(AMPLITUDE_COL) > 0)

    fig = pgo.Figure()
    fig.update_layout(
         xaxis=dict(
             title=dict(
                 text="seconds"
             )
         ),
         yaxis=dict(
             title=dict(
                 text="amplitude"
             )
         ),
     )

    fig.add_scatter(
         x=first_tofs[ELAPSED_SEC_COL].to_list(),
         y=first_tofs[NORMALIZED_AMPLITUDE_COL].to_list(),
         text=first_tofs["Elapsed time (hh:mm:ss)"].to_list(),
         mode="lines",
         line=dict(color=CB_BLACK),
         name="norm. TOF ampl.",
         visible="legendonly"
     )

    fig.add_scatter(
         x=first_tofs_above_zero[ELAPSED_SEC_COL].to_list(),
         y=first_tofs_above_zero[NORMALIZED_AMPLITUDE_COL].to_list(),
         text=first_tofs["Elapsed time (hh:mm:ss)"].to_list(),
         mode="lines",
         line=dict(color=CB_BLUE),
         name="norm. TOF ampl. > 0",
         visible="legendonly"
     )

    for percentage in range(25, 101, 25):
        fig.add_hline(percentage / 100, line_dash="dash", line_color="gray", annotation_text=f"{percentage}%", annotation_position="top right")

    try:
        siginal_smoothing(first_tofs_above_zero, fig)
    except BaseException as err:
        filtered_tofs_path = base_result_path.with_suffix(".tofs.csv")
        print(f"{err} @ {patient_file}, writing filtered TOF to {filtered_tofs_path} for closer inspection")
        tofs.write_csv(filtered_tofs_path)


    fig.write_html(base_result_path.with_suffix(".html"))

def main():
    """
    Software entrypoint
    """

    cli_args = cli()
    if cli_args.patient_folder is None:
        raise ValueError("Patient folder is not given.")

    if not cli_args.patient_folder.is_dir():
        raise ValueError(f"Patient folder {cli_args.patient_folder} is not a directory.")

    if cli_args.result_folder is None:
        raise ValueError("Result folder is not given.")

    if not cli_args.result_folder.is_dir():
        cli_args.result_folder.create(parents=True, exist_ok=True)

    # Collect all Excel files and skip open files marked with `~`
    patient_files: List[Path] = list(
        filter(
            lambda p: not p.stem.startswith("~"),
            cli_args.patient_folder.glob("*.xlsx")
        )
    )

    for patient_file in patient_files:
        try:
            process_patient_file(patient_file, cli_args.result_folder)
        except BaseException as err:
            print(f"{err} @ {patient_file}")

if __name__ == "__main__":
    main()
