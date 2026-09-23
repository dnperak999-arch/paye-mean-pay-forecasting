# Data

This project uses UK PAYE Real Time Information (RTI) statistics broken down by industry, supplied as a single Excel workbook. The source workbook is intentionally not included in this repository, so it must be supplied locally before running the pipeline.

## Expected file

Place the Excel workbook in this directory:

```text
data/
└─ dataset.xlsx
```

`dataset.xlsx` is the example filename used in the main README, but it is not mandatory. `main.py` looks for any `.xlsx` file in `data/` and uses the first one it finds, so the workbook can keep its original name.

## Required workbook structure

The workbook must contain two sheets, read by name:

| Sheet | Contents |
|---|---|
| `Employees (Industry)` | Monthly payrolled employee counts by industry |
| `Mean Pay (industry)` | Monthly mean pay by industry |

Both sheets follow the same layout:

- several metadata rows at the top, with the real table header on **Excel row 6**;
- a `Date` column holding monthly observations;
- one column per industry, plus a UK total column.

Observed coverage in the source workbook:

- **Employees:** July 2014 to July 2025
- **Mean Pay:** July 2014 to June 2025

June 2025 is therefore the latest month present in both sheets, and it is the snapshot used by the cross-industry clustering analysis.

## Missing values

Industry columns are coerced to numeric during preprocessing. In the prediction pipeline, missing monthly mean-pay values are filled by linear interpolation along the time axis, with forward and backward filling applied to any remaining gaps at the start or end of a series.

The June 2025 clustering snapshot contains complete values for all included industries, so no imputation is applied there. Missing or non-positive values in that snapshot raise an explicit error rather than being filled.

## Data availability

The dataset is not committed to Git. Excel files under `data/` are excluded by the repository's `.gitignore`, so no workbook is distributed with the code.

Users must obtain an appropriate PAYE RTI workbook separately and place it in this directory. Any workbook matching the sheet names and layout described above will work.
