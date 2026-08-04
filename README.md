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

```sh
python -m rocuronium_recovery_curve <patient_folder> <result_folder> [options]
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
python -m rocuronium_recovery_curve ./patients ./results --med-administration ./meds.xlsx -vv
```

## Output

For each patient Excel file in `patient_folder`, the tool writes to `result_folder`:

- `<patient>.html` / `<patient>.png` — recovery curve plot
- `<patient>.processed.tofs.csv` — processed TOF data
- `<patient>.revocery_summary.tofs.tsv` — per-interval recovery summary
- `<patient>.medication_summary.tofs.tsv` — medication administrations matched to intervals
- `<patient>.log` — per-patient processing log

If processing a patient file fails, the filtered TOF data is written to
`<patient>.errored.tofs.csv` for inspection and the error is logged.
