"""
Signal and interval processing: turns normalized TOF amplitudes into smoothed signals with
recovery intervals. No file or CLI handling in here.
"""

from typing import List, Tuple

import numpy as np
import polars as pl
from scipy.ndimage import gaussian_filter1d

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
