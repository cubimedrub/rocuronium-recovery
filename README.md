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
- `<patient>.processed.tofs.csv` — processed TOF data
- `<patient>.revocery_summary.tofs.tsv` — per-interval recovery summary
- `<patient>.medication_summary.tofs.tsv` — medication administrations matched to intervals
- `<patient>.medication.csv` — synced medication administrations (only with `--med-administration`), used by the review app
- `<patient>.intervals.json` — user adjusted interval start/end (only after saving in the review app)
- `<patient>.log` — per-patient processing log

If processing a patient file fails, the filtered TOF data is written to
`<patient>.errored.tofs.csv` for inspection and the error is logged.

In the recovery summary, `interval_slope_mean` is the secant slope of the smoothed signal between
interval start and end in amplitude per minute (earlier versions: median per-sample slope).
`interval_slope_max` and `interval_slope_std` are per-sample slope statistics within the interval.

## Adjusting intervals interactively

```sh
python -m rocuronium_recovery_curve review <result_folder> [--port 8050]
```

Opens a local Dash app for the results in `result_folder`. Pick a patient, drag the orange start/end
lines of an interval; the dashed marker line and its slope update live (start/end snap to the nearest
measurement). **Save** writes `<patient>.intervals.json` and regenerates the summaries, HTML and PNG.
Saved bounds are loaded the next time the patient is opened. Re-running the main CLI overwrites the
summaries with the automatic intervals but keeps `.intervals.json`.
