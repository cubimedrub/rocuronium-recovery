"""Column names and colors shared between calculation, plotting and review."""

from typing import Tuple

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
SU_SLOPE_WINDOW_START_TIME = "slope_window_start_time"
SU_SLOPE_WINDOW_END_TIME = "slope_window_end_time"
SU_INTERVAL_SLOPE_MEAN = "interval_slope_mean"
SU_INTERVAL_SLOPE_MAX = "interval_slope_max"
SU_INTERVAL_SLOPE_STD = "interval_slope_std"
SU_INTERVAL_SLOPE_METHOD = "interval_slope_method"
SU_INTERVAL_TIME_TO_NEXT = "interval_time_to_next"

# Optional columns added to the processed TOF data by the review app
MANUAL_INTERVAL_ID_COL = "manual_interval_id"
MANUAL_SLOPE_WINDOW_COL = "manual_slope_window"

MANUAL_COLS: Tuple[str, ...] = (MANUAL_INTERVAL_ID_COL, MANUAL_SLOPE_WINDOW_COL)


# Some colorblind-friendly colors according to Wong https://www.nature.com/articles/nmeth.1618
BLACK = "rgb(0, 0, 0)"
DARK_GREY = "rgb(169, 169, 169)"
LIGHT_GREY = "rgb(211, 211, 211)"
