from __future__ import annotations

from pathlib import Path
import pandas as pd

from src.utils import load_clean_sheet


def load_employees_and_pay(file_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    employees = load_clean_sheet(file_path, "Employees (Industry)", header_row_idx=5)
    pay = load_clean_sheet(file_path, "Mean Pay (industry)", header_row_idx=5)
    return employees, pay