from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from src.utils import outputs_path

# Survey answer columns in the Mean Pay sheet; they are not industries.
NON_INDUSTRY_COLUMNS = [
    "Was there any payroll error or Data Issue this month",
    "Majority of the payroll were contributed by people from region North East, North West or East Midlands",
]

# The UK column is the exact sum of the individual industry employee counts,
# so it is a deterministic aggregate rather than a comparable observation.
UK_AGGREGATE_COLUMN = "UK"

K_VALUES = (2, 3, 4, 5)


def _numeric_by_date(df: pd.DataFrame, industries: list[str]) -> pd.DataFrame:
    """Index a sheet by Date and coerce the industry columns to numeric."""
    return df.set_index("Date")[industries].apply(pd.to_numeric, errors="coerce")


def build_industry_snapshot(
    employees: pd.DataFrame,
    pay: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Timestamp]:
    """Build a one-row-per-industry snapshot for the latest shared month.

    Employees and pay are joined on Date (not on row position, since the two
    sheets do not cover the same number of months). The UK aggregate is dropped.

    Returns
    -------
    (snapshot, month)
        snapshot is indexed by industry with columns
        ['Employees', 'MeanPay', 'LogEmployees'].
    """
    industries = [
        c
        for c in pay.columns
        if c not in NON_INDUSTRY_COLUMNS
        and c not in ("Date", UK_AGGREGATE_COLUMN)
    ]

    missing = [c for c in industries if c not in employees.columns]
    if missing:
        raise KeyError(f"Industries missing from the Employees sheet: {missing}")

    emp_by_date = _numeric_by_date(employees, industries)
    pay_by_date = _numeric_by_date(pay, industries)

    shared_months = emp_by_date.index.intersection(pay_by_date.index)
    if shared_months.empty:
        raise ValueError("Employees and Mean Pay sheets share no common month.")

    month = shared_months.max()
    snapshot = pd.DataFrame(
        {
            "Employees": emp_by_date.loc[month],
            "MeanPay": pay_by_date.loc[month],
        }
    )

    unusable = snapshot[snapshot.isna().any(axis=1) | (snapshot["Employees"] <= 0)]
    if not unusable.empty:
        raise ValueError(
            f"Missing or non-positive values for {month:%Y-%m}: {list(unusable.index)}"
        )

    snapshot["LogEmployees"] = np.log10(snapshot["Employees"])
    return snapshot, month


def _group_label(cluster: int) -> str:
    """Letter labels, so group identifiers are not read as an ordering."""
    return chr(ord("A") + int(cluster))


def _short(name: str, width: int = 26) -> str:
    return name if len(name) <= width else name[: width - 1] + "…"


def run_clustering(employees: pd.DataFrame, pay: pd.DataFrame) -> pd.DataFrame:
    """Cluster UK industries by employment size and mean pay.

    One observation is one industry, described by log10(Employees) and mean pay
    in the latest month common to both sheets. k is chosen by silhouette score.
    """
    snapshot, month = build_industry_snapshot(employees, pay)

    if len(snapshot) <= max(K_VALUES):
        raise ValueError(
            f"Need more than {max(K_VALUES)} industries to evaluate "
            f"k={list(K_VALUES)}, got {len(snapshot)}."
        )

    x_scaled = StandardScaler().fit_transform(
        snapshot[["LogEmployees", "MeanPay"]].to_numpy(dtype=np.float64)
    )

    inertia: dict[int, float] = {}
    sil_scores: dict[int, float] = {}
    for k in K_VALUES:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(x_scaled)
        inertia[k] = float(km.inertia_)
        sil_scores[k] = float(silhouette_score(x_scaled, labels))

    k_opt = max(sil_scores, key=sil_scores.get)

    km_final = KMeans(n_clusters=k_opt, random_state=42, n_init=10)
    snapshot["Cluster"] = km_final.fit_predict(x_scaled)

    # Elbow plot
    plt.figure()
    plt.plot(list(K_VALUES), [inertia[k] for k in K_VALUES], marker="o")
    plt.xticks(list(K_VALUES))
    plt.xlabel("Number of clusters (k)")
    plt.ylabel("Inertia (within-cluster sum of squares)")
    plt.title(f"Elbow method: UK industries, {month:%B %Y}")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(outputs_path("kmeans_elbow.png"), dpi=300)
    plt.close()

    # Cluster scatter: one point per industry, labelled
    fig, ax = plt.subplots(figsize=(12, 7.5))
    for c in sorted(snapshot["Cluster"].unique()):
        subset = snapshot[snapshot["Cluster"] == c]
        ax.scatter(
            subset["Employees"],
            subset["MeanPay"],
            s=70,
            edgecolors="black",
            linewidth=0.5,
            label=f"Group {_group_label(c)} (n={len(subset)})",
        )

    # Label in employment order and alternate above/below, so industries that sit
    # close together on the chart do not overprint each other.
    ordered = snapshot.sort_values("Employees")
    for position, (name, row) in enumerate(ordered.iterrows()):
        to_the_right = position % 2 == 0
        ax.annotate(
            _short(str(name)),
            (row["Employees"], row["MeanPay"]),
            textcoords="offset points",
            xytext=(7, 4) if to_the_right else (-7, -11),
            ha="left" if to_the_right else "right",
            fontsize=7,
        )

    ax.set_xscale("log")
    ax.set_xlim(snapshot["Employees"].min() / 1.8, snapshot["Employees"].max() * 3.0)
    ax.set_xlabel("Employees (log scale)")
    ax.set_ylabel("Mean Pay (£)")
    ax.set_title(
        f"UK industries grouped by employment size and mean pay ({month:%B %Y})\n"
        f"K-Means, k={k_opt} selected by silhouette ({sil_scores[k_opt]:.3f})"
    )
    ax.legend(title="K-Means group")
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(outputs_path("industry_clusters.png"), dpi=300)
    plt.close(fig)

    print(f"Cross-industry clustering ({len(snapshot)} industries, "
          f"UK aggregate excluded) - snapshot {month:%Y-%m}")
    print("Silhouette by k: "
          + " | ".join(f"k={k}: {sil_scores[k]:.3f}" for k in K_VALUES))
    print(f"Selected k={k_opt} (silhouette {sil_scores[k_opt]:.3f})")
    for c in sorted(snapshot["Cluster"].unique()):
        members = snapshot.index[snapshot["Cluster"] == c].tolist()
        print(f"  Group {_group_label(c)} (n={len(members)}): {', '.join(members)}")

    return snapshot.reset_index(names="Industry")
