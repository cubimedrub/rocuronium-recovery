"""
Signal and interval processing: turns normalized TOF amplitudes into smoothed signals with
recovery intervals. No file or CLI handling in here.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

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


INITIAL_SLOPE_METHOD = "initial"
MANUAL_SLOPE_METHOD = "manual"


@dataclass(frozen=True)
class IntervalStats:
    """Slope of a recovery interval together with the line drawn for it."""

    start_s: int
    end_s: int
    y_start: float
    y_end: float
    slope: float  # initial: per sample; manual: per second (straight line between the two points)
    slope_max: float  # per sample slope of the smoothed signal
    slope_std: float
    method: str  # INITIAL_SLOPE_METHOD or MANUAL_SLOPE_METHOD


def _steepest_part(interval: pl.DataFrame) -> pl.DataFrame:
    """
    Rows of the interval whose slope is at least the maximum slope of the interval minus one
    standard deviation of the slope.
    """
    slope = pl.col(GAUSSIAN_SIGNAL_SMOOTHING_SLOPE_COL)
    max_slope = _as_float(interval[GAUSSIAN_SIGNAL_SMOOTHING_SLOPE_COL].max())
    slope_std = _as_float(interval[GAUSSIAN_SIGNAL_SMOOTHING_SLOPE_COL].std() or 0.0)

    steepest = interval.filter(slope >= max_slope - slope_std)  # 3 sigma rule too loose
    return interval if steepest.height == 0 else steepest


def initial_interval_stats(interval: pl.DataFrame) -> IntervalStats:
    """
    Initial slope of a recovery interval: median slope of its steepest part. The drawn line starts at
    the lowest smoothed value of the steepest part and rises by median slope times number of points.

    Parameters
    ----------
    interval: pl.DataFrame
        Rows of one recovery interval.

    Returns
    -------
    IntervalStats
    """
    steepest = _steepest_part(interval)
    slopes = steepest[GAUSSIAN_SIGNAL_SMOOTHING_SLOPE_COL]
    median_slope = _as_float(slopes.median())
    offset = _as_float(steepest[GAUSSIAN_SIGNAL_SMOOTHING_COL].min())

    return IntervalStats(
        start_s=_as_int(steepest[ELAPSED_SEC_COL].first()),
        end_s=_as_int(steepest[ELAPSED_SEC_COL].last()),
        y_start=offset,
        y_end=median_slope * steepest.height + offset,
        slope=median_slope,
        slope_max=_as_float(slopes.max()),
        slope_std=_as_float(slopes.std() or 0.0),
        method=INITIAL_SLOPE_METHOD,
    )


def interval_stats(data: pl.DataFrame, start_s: float, end_s: float) -> IntervalStats:
    """
    Straight line between the two data points closest to start_s and end_s, used when a user changed the
    interval. The slope is the amplitude change per second between the two points.

    Parameters
    ----------
    data: pl.DataFrame
        Processed TOF events (needs the elapsed seconds, smoothed signal and its slope).
    start_s, end_s: float
        Requested window in elapsed seconds. Snapped to the nearest data points.

    Returns
    -------
    IntervalStats
    """
    times = data[ELAPSED_SEC_COL].to_numpy()
    smoothed = data[GAUSSIAN_SIGNAL_SMOOTHING_COL].to_numpy()
    slopes = data[GAUSSIAN_SIGNAL_SMOOTHING_SLOPE_COL].to_numpy()

    start_idx = int(np.argmin(np.abs(times - start_s)))
    end_idx = int(np.argmin(np.abs(times - end_s)))
    if start_idx > end_idx:
        start_idx, end_idx = end_idx, start_idx

    delta_s = float(times[end_idx] - times[start_idx])
    delta_y = float(smoothed[end_idx] - smoothed[start_idx])
    window_slopes = slopes[start_idx : end_idx + 1]

    return IntervalStats(
        start_s=int(times[start_idx]),
        end_s=int(times[end_idx]),
        y_start=float(smoothed[start_idx]),
        y_end=float(smoothed[end_idx]),
        slope=delta_y / delta_s if delta_s > 0 else 0.0,
        slope_max=float(window_slopes.max()),
        slope_std=float(window_slopes.std(ddof=1)) if window_slopes.size > 1 else 0.0,
        method=MANUAL_SLOPE_METHOD,
    )


def recovery_intervals(data: pl.DataFrame) -> List[Tuple[int, pl.DataFrame]]:
    """
    Splits the processed data into its recovery intervals, ordered by time.

    Returns
    -------
    List[Tuple[int, pl.DataFrame]]
        (interval ID, rows of the interval) per recovery interval.
    """
    recoveries = data.filter(pl.col(IS_RECOVERY_COL))
    intervals: List[Tuple[int, pl.DataFrame]] = [
        (interval_id[0], interval)
        for interval_id, interval in recoveries.group_by(
            GAUSSIAN_SIGNAL_SMOOTHING_SLOPE_INTERVALS_ID_COL
        )
    ]
    intervals.sort(key=lambda g: g[0])
    return intervals


def default_slope_window(interval: pl.DataFrame) -> Tuple[int, int]:
    """
    Starting window for the slope of a recovery interval: first and last data point whose slope
    is at least the maximum slope of the interval minus one standard deviation, i.e. the steepest part.

    Parameters
    ----------
    interval: pl.DataFrame
        Rows of one recovery interval.

    Returns
    -------
    Tuple[int, int]
        Start and end in elapsed seconds.
    """
    steepest = _steepest_part(interval)

    return (
        _as_int(steepest[ELAPSED_SEC_COL].first()),
        _as_int(steepest[ELAPSED_SEC_COL].last()),
    )


@dataclass(frozen=True)
class IntervalEntry:
    """One row of the recovery summary: one or, after merging, several recovery intervals."""

    members: Tuple[int, ...]  # 1-based recovery interval numbers
    extent_start_s: int  # whole recovery interval(s)
    extent_end_s: int
    stats: IntervalStats


def interval_entries(
    data: pl.DataFrame,
    interval_bounds: Optional[Dict[int, Tuple[int, int]]] = None,
) -> List[IntervalEntry]:
    """
    Slope per recovery interval. Intervals changed by the user (present in interval_bounds) get the
    slope of the straight line between their two points; if two such windows overlap, they are merged
    into one entry spanning both windows. All other intervals keep their initial slope.

    Parameters
    ----------
    data: pl.DataFrame
        Processed TOF events with marked recovery intervals.
    interval_bounds: Optional[Dict[int, Tuple[int, int]]]
        User adjusted window (start, end) in elapsed seconds per 1-based recovery interval number.

    Returns
    -------
    List[IntervalEntry]
        Ordered by start of the recovery interval(s).
    """
    interval_bounds = interval_bounds or {}

    initial: List[IntervalEntry] = []
    manual: List[IntervalEntry] = []
    for number, (_, interval) in enumerate(recovery_intervals(data), start=1):
        extent_start_s = _as_int(interval[ELAPSED_SEC_COL].first())
        extent_end_s = _as_int(interval[ELAPSED_SEC_COL].last())
        if number in interval_bounds:
            start_s, end_s = interval_bounds[number]
            manual.append(
                IntervalEntry(
                    (number,),
                    extent_start_s,
                    extent_end_s,
                    interval_stats(data, start_s, end_s),
                )
            )
        else:
            initial.append(
                IntervalEntry(
                    (number,),
                    extent_start_s,
                    extent_end_s,
                    initial_interval_stats(interval),
                )
            )

    # Chain manual windows that overlap (touching counts) into one
    merged: List[IntervalEntry] = []
    for entry in sorted(manual, key=lambda e: e.stats.start_s):
        if merged and entry.stats.start_s <= merged[-1].stats.end_s:
            previous = merged.pop()
            merged.append(
                IntervalEntry(
                    previous.members + entry.members,
                    min(previous.extent_start_s, entry.extent_start_s),
                    max(previous.extent_end_s, entry.extent_end_s),
                    interval_stats(
                        data,
                        previous.stats.start_s,
                        max(previous.stats.end_s, entry.stats.end_s),
                    ),
                )
            )
        else:
            merged.append(entry)

    return sorted(initial + merged, key=lambda e: (e.extent_start_s, e.stats.start_s))
