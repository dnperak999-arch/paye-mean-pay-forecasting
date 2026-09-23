from __future__ import annotations

from pathlib import Path
import pandas as pd


def project_root() -> Path:
    # src/utils.py -> src -> project root
    return Path(__file__).resolve().parents[1]


def outputs_path(filename: str) -> Path:
    out_dir = project_root() / "figures"
    out_dir.mkdir(exist_ok=True)
    return out_dir / filename


def load_clean_sheet(file_path: Path, sheet_name: str, header_row_idx: int) -> pd.DataFrame:
    """
    Load an Excel sheet where the real header is inside the sheet (not row 0),
    then clean Date column and keep monthly timestamps.
    """
    raw = pd.read_excel(file_path, sheet_name=sheet_name, header=None)

    df = raw.copy()
    df.columns = df.iloc[header_row_idx]

    # Drop metadata rows (up to and including the header row)
    df = df.drop(index=list(range(header_row_idx + 1))).reset_index(drop=True)

    # Drop fully empty columns
    df = df.dropna(axis=1, how="all")

    # Clean column names
    df.columns = [str(c).strip() for c in df.columns]

    # Clean Date strings then parse
    df["Date"] = (
        df["Date"]
        .astype(str)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
        .replace({"nan": pd.NA, "NaT": pd.NA})
    )

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce", format="mixed")

    # Keep only valid dates and sort
    df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)

    # Align dates to month start (monthly series)
    df["Date"] = df["Date"].dt.to_period("M").dt.to_timestamp()

    return df