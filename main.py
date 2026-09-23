from __future__ import annotations

from pathlib import Path

from src.preprocessing import load_employees_and_pay
from src.models.clustering import run_clustering
from src.models.forecasting import run_forecasting
from src.utils import project_root

def dataset_path() -> Path:
    """Locate the Excel workbook to analyse inside the local `data/` folder."""
    data_dir = project_root() / "data"

    candidates = sorted(
        p for p in data_dir.glob("*.xlsx") if not p.name.startswith("~$")
    )

    if not candidates:
        raise FileNotFoundError(
            f"No Excel workbook found in {data_dir}. Place the required "
            "workbook there before running main.py (see data/README.md)."
        )

    if len(candidates) > 1:
        names = ", ".join(p.name for p in candidates)
        raise ValueError(
            f"Found {len(candidates)} Excel workbooks in {data_dir}: {names}. "
            "Keep only the workbook you want to analyse in that folder."
        )

    return candidates[0]


def main() -> None:
    file_path = dataset_path()
    print(f"Loading dataset: {file_path.name}")

    employees, pay = load_employees_and_pay(file_path)

    results = run_forecasting(pay)
    print(results.to_string(index=False))

    clusters = run_clustering(employees, pay)
    print(f"Selected k (best silhouette): {clusters['Cluster'].nunique()}")
    print(clusters.head().to_string(index=False))


if __name__ == "__main__":
    main()