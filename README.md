# Rocuronium Recovery Curve

Processes patient TOF (train-of-four) measurement Excel files to derive
Rocuronium muscle-activity recovery curves. For each patient file the tool
normalizes and smooths the TOF amplitude signal, segments it into recovery
and re-administration intervals, optionally cross-references medication
administration timestamps, and produces a plot plus summary tables.

## Installation

Requires Python >= 3.12.

Using conda:

```sh
conda env create -f environment.yml
conda activate rocuronium_recovery_curve
```

Or with pip:

```sh
pip install -e .[dev]
```

## Input files

### Patient file (`patient_folder/*.xlsx`)

Second worksheet (sheet index 1), header on row 2 (row index 1), must contain columns:

- `Elapsed time (hh:mm:ss)`
- `Mode`
- `Noise Present`
- `Peak (mV)`
- `Valley (mV)`

### Medication administration file (`--med-administration`)

Excel file with two worksheets, header on row 1 (row index 0):

- Sheet `Klinik`:
  - `caseid`
  - `PateintenID`
- Sheet `Meds all`:
  - `caseid`
  - `chartTime`
  - `medi`
  - `dose`
  - `unit`

## Usage

Two subcommands: `calc` calculates the recovery curves, `review` adjusts the intervals afterwards.

### `calc`

```sh
python -m rocuronium_recovery_curve calc <patient_folder> <result_folder> [options]
```

Positional arguments:

- `patient_folder` — folder containing patient Excel files with Rocuronium recovery curves
- `result_folder` — folder where results are saved

Options:

- `--patiend-id-regex REGEX` — regex to extract the patient ID from a patient file name; must define a named group `patient_id` (default: ``\d{4}\S\d{2}\S\d{2}\S(?P<patient_id>\d+)\S\d+``)
- `--med-administration PATH` — path to an Excel file with medication administration timestamps (sheets `Klinik` and `Meds all`)
- `-t, --merge-threshold FLOAT` — amplitude merge threshold for merging subsequent negative-slope intervals into the recovery curve (default: `0.1`)
- `-v, --verbose` — repeatable verbosity flag (`-v` warning, `-vv` info, `-vvv` debug + tracebacks)

Example:

```sh
python -m rocuronium_recovery_curve calc ./patients ./results --med-administration ./meds.xlsx -vv
```

## Output

For each patient Excel file in `patient_folder`, the tool writes to `result_folder`:

- `<patient>.html` / `<patient>.png` — recovery curve plot
- `<patient>.processed.tofs.csv` — processed TOF data; after saving in the review app with changed intervals it also has the columns `manual_interval_id` (interval number as in the plot, merged intervals share the first number, empty outside recovery intervals) and `manual_slope_window` (row lies in the user's slope window)
- `<patient>.revocery_summary.tofs.tsv` — per-interval recovery summary
- `<patient>.medication_summary.tofs.tsv` — medication administrations matched to intervals
- `<patient>.medication.csv` — synced medication administrations (only with `--med-administration`), used by the review app
- `<patient>.log` — per-patient processing log

If processing a patient file fails, the filtered TOF data is written to
`<patient>.errored.tofs.csv` for inspection and the error is logged.

### Recovery summary columns

- `interval_start_time` / `interval_end_time` — whole recovery interval (elapsed seconds)
- `slope_window_start_time` / `slope_window_end_time` — start and end of the slope window: first and last point of the steepest part of the interval (slope >= max slope minus one standard deviation), or the two points chosen in the review app
- `interval_slope_mean` — slope of the smoothed signal, see `interval_slope_method`
- `interval_slope_method`
  - `initial`: median slope of the steepest part, amplitude per sample
  - `manual`: the interval was changed in the review app; slope of the straight line between the two chosen points on the curve, amplitude per second
- `interval_slope_max` / `interval_slope_std` — per-sample slope statistics within the slope window
- `interval_time_to_next` — minutes to the start of the next recovery interval

`initial` and `manual` slopes are in different units (they differ by the sample spacing) and are not directly comparable.

## Adjusting intervals interactively

```sh
python -m rocuronium_recovery_curve review <result_folder> [--port 8050]
```

Opens a local Dash app for the results in `result_folder`. Pick a patient and drag the orange start/end lines of an
interval. The dashed marker line and its slope update live; start/end snap to the nearest measurement and the
interval switches from `initial` to `manual`. The preview already merges changed intervals whose windows overlap.

**Save** adds the `manual_*` columns to `<patient>.processed.tofs.csv` (only if intervals changed) and regenerates the summaries, HTML and PNG.
When saving, changed intervals whose windows overlap (or touch) are merged for good: they become one interval spanning
both windows, listed as one row in the summary and drawn with one pair of lines in the editor. Merged intervals cannot be split again in
the app; re-run `calc` to start over. Saved bounds are read from the `manual_*` columns the next time the patient is opened.
Re-running `calc` overwrites the summaries and the processed CSV, dropping the `manual_*` columns and with them all manual changes.
