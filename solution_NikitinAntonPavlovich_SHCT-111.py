from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq


MODIFIED_Z_LIMIT = 3.5
MIN_BRAND_OBS = 20
MAD_TO_STD = 1.4826
EPS = 1e-12
OTS_WEIGHT_COLUMN = "_ots_weight"
PLACEHOLDER_INPUT_PATHS = {"path/to/data_train", "/path/to/data_train"}

CATEGORY_ALIASES = ("CategoryDelivery", "CategoryNameDelivery")

CORE_COLUMNS = [
    "SubjectID",
    "QueryText",
    "BrandID",
    "Category1ID",
    "Category2ID",
    "Category3ID",
    "CategoryDelivery",
    "CategoryNameDelivery",
    "Brand",
    "Category1",
    "Category2",
    "Category3",
    "ResourceName",
    "ResourceType",
    "UseType",
    "Platform",
    "Пол",
    "Возраст",
    "Регион",
    "Федеральный_округ",
    "Количество_детей",
    "Занятость",
    "Доход",
    "Weight",
    "Start",
    "researchdate",
    "week",
    "Month",
    "BrandinDelivery",
]

DEMOGRAPHIC_COLUMNS = [
    "Пол",
    "Возраст",
    "Регион",
    "Федеральный_округ",
    "Количество_детей",
    "Занятость",
    "Доход",
]

RESOURCE_COLUMNS = ["ResourceName", "ResourceType", "Platform", "UseType"]
CATEGORY_COLUMNS = ["CategoryDelivery", "Category1", "Category2", "Category3"]
ANOMALIES_COLUMNS = ["SubjectID", "researchdate"]
REASON_COLUMNS = [
    "SubjectID",
    "researchdate",
    "BrandID",
    "Brand",
    "CategoryDelivery",
    "daily_ots",
    "score",
    "threshold",
    "reason",
]


def find_default_input_dir() -> Path:
    candidates = [
        Path.cwd() / "data_train",
        Path.cwd().parent / "data_train",
        Path.home() / "Downloads" / "data_train",
    ]
    for path in candidates:
        if path.exists():
            return path
    return Path.cwd() / "data_train"


def is_placeholder_input_path(input_path: Path) -> bool:
    normalized = str(input_path).replace("\\", "/").strip().rstrip("/").lower()
    return normalized in PLACEHOLDER_INPUT_PATHS


def resolve_input_path(input_path: Path) -> Path:
    if not is_placeholder_input_path(input_path):
        return input_path

    default_path = find_default_input_dir()
    if default_path.exists():
        print(f"--input {input_path} is a placeholder; using {default_path}")
        return default_path

    raise FileNotFoundError(
        "--input /path/to/data_train is a placeholder. Put data_train next to the script "
        "or pass a real path to the folder with parquet files."
    )


def parquet_files(input_path: Path) -> list[Path]:
    if input_path.is_file() and input_path.suffix.lower() == ".parquet":
        return [input_path]
    files = [
        path
        for path in input_path.rglob("*.parquet")
        if "examples" not in {part.lower() for part in path.parts}
    ]
    return sorted(files)


def read_source_data(input_path: Path) -> pd.DataFrame:
    input_path = resolve_input_path(input_path)
    files = parquet_files(input_path)
    if not files:
        raise FileNotFoundError(f"No parquet files found in {input_path}")

    frames: list[pd.DataFrame] = []
    for file_path in files:
        available = set(pq.read_schema(file_path).names)
        columns = [column for column in CORE_COLUMNS if column in available]
        frames.append(pd.read_parquet(file_path, columns=columns))

    data = pd.concat(frames, ignore_index=True)
    data = normalize_columns(data)
    required = {"SubjectID", "BrandID", "Brand", "CategoryDelivery", "Weight", "researchdate", "BrandinDelivery"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    return data


def normalize_columns(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    if "CategoryDelivery" not in data.columns and "CategoryNameDelivery" in data.columns:
        data = data.rename(columns={"CategoryNameDelivery": "CategoryDelivery"})
    elif "CategoryDelivery" in data.columns and "CategoryNameDelivery" in data.columns:
        data["CategoryDelivery"] = data["CategoryDelivery"].fillna(data["CategoryNameDelivery"])

    data["researchdate"] = pd.to_datetime(data["researchdate"], errors="coerce").dt.normalize()
    if "Start" in data.columns:
        data["Start"] = pd.to_datetime(data["Start"], errors="coerce")
    data["SubjectID"] = pd.to_numeric(data["SubjectID"], errors="coerce").astype("Int64")
    data["Weight"] = pd.to_numeric(data["Weight"], errors="coerce")
    data["BrandinDelivery"] = pd.to_numeric(data["BrandinDelivery"], errors="coerce").fillna(0).astype(int)
    data["BrandID"] = data["BrandID"].astype("string")
    data["CategoryDelivery"] = data["CategoryDelivery"].astype("string")
    return data


def analysis_rows(data: pd.DataFrame) -> pd.DataFrame:
    category = data["CategoryDelivery"].astype("string").str.strip()
    mask = (
        data["BrandinDelivery"].eq(1)
        & data["researchdate"].notna()
        & data["SubjectID"].notna()
        & data["BrandID"].notna()
        & category.notna()
        & category.ne("")
        & data["Weight"].notna()
        & data["Weight"].gt(0)
    )
    filtered = data.loc[mask].copy()
    filtered["CategoryDelivery"] = category.loc[mask]
    filtered[OTS_WEIGHT_COLUMN] = filtered.groupby(["SubjectID", "researchdate"], observed=True)["Weight"].transform(
        "median"
    )
    return filtered


def robust_value_stats(values: Iterable[float]) -> dict[str, float]:
    values = np.asarray(list(values), dtype=float)
    if values.size == 0:
        return {"n": 0, "median": 0.0, "mad": 0.0}
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    return {"n": int(values.size), "median": median, "mad": mad}


def robust_group_stats(data: pd.DataFrame, group_columns: str | list[str]) -> pd.DataFrame:
    columns = [group_columns] if isinstance(group_columns, str) else group_columns
    stats = (
        data.groupby(columns, observed=True, sort=False)["log_ots"]
        .agg(n="size", median="median")
        .reset_index()
    )
    deviations = data[columns + ["log_ots"]].merge(stats[columns + ["median"]], on=columns, how="left")
    deviations["_abs_deviation"] = (deviations["log_ots"] - deviations["median"]).abs()
    mad = deviations.groupby(columns, observed=True, sort=False)["_abs_deviation"].median().rename("mad").reset_index()
    return stats.merge(mad, on=columns, how="left")


def build_trigger_table(data: pd.DataFrame) -> pd.DataFrame:
    filtered = analysis_rows(data)

    grouped = (
        filtered.groupby(["SubjectID", "researchdate", "BrandID", "CategoryDelivery"], observed=True, sort=False)
        .agg(
            Brand=("Brand", "first"),
            row_count=("BrandID", "size"),
            Weight=(OTS_WEIGHT_COLUMN, "first"),
        )
        .reset_index()
    )
    grouped["daily_ots"] = grouped["Weight"] * grouped["row_count"]
    grouped["log_ots"] = np.log1p(grouped["daily_ots"])
    return grouped


def add_thresholds(trigger_table: pd.DataFrame) -> pd.DataFrame:
    if trigger_table.empty:
        return trigger_table.assign(
            threshold_log=pd.Series(dtype=float),
            threshold=pd.Series(dtype=float),
            score=pd.Series(dtype=float),
            threshold_source=pd.Series(dtype="string"),
        )

    brand_stats = robust_group_stats(trigger_table, ["CategoryDelivery", "BrandID"]).rename(
        columns={"n": "brand_n", "median": "brand_median", "mad": "brand_mad"}
    )
    category_stats = robust_group_stats(trigger_table, "CategoryDelivery").rename(
        columns={"n": "category_n", "median": "category_median", "mad": "category_mad"}
    )
    global_stats = robust_value_stats(trigger_table["log_ots"])

    scored = trigger_table.merge(brand_stats, on=["CategoryDelivery", "BrandID"], how="left")
    scored = scored.merge(category_stats, on="CategoryDelivery", how="left")

    brand_scale = MAD_TO_STD * scored["brand_mad"].where(scored["brand_mad"].gt(EPS))
    category_scale = MAD_TO_STD * scored["category_mad"].where(scored["category_mad"].gt(EPS))
    global_scale = float(MAD_TO_STD * global_stats["mad"])
    if not np.isfinite(global_scale) or global_scale <= EPS:
        global_scale = float(scored["log_ots"].std(ddof=0))
    if not np.isfinite(global_scale) or global_scale <= EPS:
        global_scale = 1.0

    brand_limit = scored["brand_median"] + MODIFIED_Z_LIMIT * brand_scale
    brand_limit = brand_limit.where(scored["brand_n"].ge(MIN_BRAND_OBS), np.nan)
    category_limit = scored["category_median"] + MODIFIED_Z_LIMIT * category_scale
    global_limit = float(global_stats["median"] + MODIFIED_Z_LIMIT * global_scale)

    limits = pd.concat(
        [
            brand_limit.rename("brand"),
            category_limit.rename("category"),
            pd.Series(global_limit, index=scored.index, name="global"),
        ],
        axis=1,
    )
    scored["threshold_log"] = limits.max(axis=1, skipna=True)
    scored["threshold"] = np.expm1(scored["threshold_log"])
    scored["score"] = scored["daily_ots"] / scored["threshold"].replace(0, np.nan)
    scored["threshold_source"] = limits.idxmax(axis=1)
    return scored


def detect_anomalies(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    trigger_table = build_trigger_table(data)
    scored = add_thresholds(trigger_table)
    reasons = scored.loc[scored["log_ots"].gt(scored["threshold_log"])].copy()
    reasons = reasons.sort_values(["researchdate", "SubjectID", "score"], ascending=[True, True, False])

    reasons["reason"] = (
        f"daily_ots выше robust-z порога {MODIFIED_Z_LIMIT} по log1p(daily_ots); threshold_source="
        + reasons["threshold_source"].astype(str)
        + "; row_count="
        + reasons["row_count"].astype(str)
    )

    reasons = reasons[REASON_COLUMNS]
    anomalies = reasons[ANOMALIES_COLUMNS].drop_duplicates().sort_values(["researchdate", "SubjectID"])
    return anomalies, reasons, scored


def remove_anomaly_days(data: pd.DataFrame, anomalies: pd.DataFrame) -> pd.DataFrame:
    if anomalies.empty:
        return data.copy()
    marker = anomalies[ANOMALIES_COLUMNS].drop_duplicates().assign(_drop_anomaly_day=1)
    merged = data.merge(marker, on=["SubjectID", "researchdate"], how="left")
    return merged.loc[merged["_drop_anomaly_day"].isna()].drop(columns=["_drop_anomaly_day"])


def save_csv_outputs(anomalies: pd.DataFrame, reasons: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    anomalies_out = anomalies.reindex(columns=ANOMALIES_COLUMNS).copy()
    reasons_out = reasons.reindex(columns=REASON_COLUMNS).copy()
    anomalies_out["researchdate"] = pd.to_datetime(anomalies_out["researchdate"]).dt.strftime("%Y-%m-%d")
    reasons_out["researchdate"] = pd.to_datetime(reasons_out["researchdate"]).dt.strftime("%Y-%m-%d")
    reasons_out["daily_ots"] = reasons_out["daily_ots"].round(6)
    reasons_out["score"] = reasons_out["score"].round(6)
    reasons_out["threshold"] = reasons_out["threshold"].round(6)
    anomalies_out.to_csv(output_dir / "anomalies.csv", index=False)
    reasons_out.to_csv(output_dir / "anomaly_reasons.csv", index=False)


def validate_required_outputs(output_dir: Path) -> None:
    anomalies_path = output_dir / "anomalies.csv"
    reasons_path = output_dir / "anomaly_reasons.csv"
    required_plots = [
        output_dir / "plots" / "total_ots_before_after.png",
        output_dir / "plots" / "category_ots_change.png",
        output_dir / "plots" / "daily_anomaly_count.png",
    ]
    missing = [path for path in [anomalies_path, reasons_path, *required_plots] if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required output files: {[str(path) for path in missing]}")

    anomalies = pd.read_csv(anomalies_path)
    reasons = pd.read_csv(reasons_path)
    if list(anomalies.columns) != ANOMALIES_COLUMNS:
        raise ValueError(f"Invalid anomalies.csv columns: {list(anomalies.columns)}")
    if list(reasons.columns) != REASON_COLUMNS:
        raise ValueError(f"Invalid anomaly_reasons.csv columns: {list(reasons.columns)}")
    if anomalies.duplicated(ANOMALIES_COLUMNS).any():
        raise ValueError("anomalies.csv contains duplicate SubjectID/researchdate pairs")


def plot_total_ots_before_after(data: pd.DataFrame, cleaned: pd.DataFrame, plots_dir: Path) -> None:
    before = analysis_rows(data).groupby("researchdate", observed=True)[OTS_WEIGHT_COLUMN].sum()
    after = analysis_rows(cleaned).groupby("researchdate", observed=True)[OTS_WEIGHT_COLUMN].sum()
    index = before.index.union(after.index).sort_values()

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(index, before.reindex(index, fill_value=0), label="before", linewidth=2)
    ax.plot(index, after.reindex(index, fill_value=0), label="after", linewidth=2)
    ax.set_title("Total OTS before and after anomaly removal")
    ax.set_xlabel("researchdate")
    ax.set_ylabel("OTS")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(plots_dir / "total_ots_before_after.png", dpi=160)
    plt.close(fig)


def plot_category_ots_change(data: pd.DataFrame, cleaned: pd.DataFrame, plots_dir: Path) -> None:
    before_rows = analysis_rows(data)
    after_rows = analysis_rows(cleaned)
    before = before_rows.groupby("CategoryDelivery", observed=True)[OTS_WEIGHT_COLUMN].sum()
    after = after_rows.groupby("CategoryDelivery", observed=True)[OTS_WEIGHT_COLUMN].sum().reindex(before.index, fill_value=0)
    change_pct = ((after / before) - 1.0).fillna(0.0) * 100.0
    change_pct = change_pct.sort_values()

    height = max(6, min(14, 0.35 * len(change_pct)))
    fig, ax = plt.subplots(figsize=(11, height))
    ax.barh(change_pct.index.astype(str), change_pct.values, color="#4f7cac")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title("OTS change by CategoryDelivery, %")
    ax.set_xlabel("after / before - 1, %")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(plots_dir / "category_ots_change.png", dpi=160)
    plt.close(fig)


def plot_daily_anomaly_count(anomalies: pd.DataFrame, plots_dir: Path) -> None:
    counts = anomalies.groupby("researchdate", observed=True)["SubjectID"].nunique().sort_index()
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.bar(counts.index, counts.values, width=0.85, color="#c85a54")
    ax.set_title("Anomalous respondents by day")
    ax.set_xlabel("researchdate")
    ax.set_ylabel("respondents")
    ax.grid(axis="y", alpha=0.25)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(plots_dir / "daily_anomaly_count.png", dpi=160)
    plt.close(fig)


def save_required_plots(data: pd.DataFrame, anomalies: pd.DataFrame, output_dir: Path) -> None:
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    cleaned = remove_anomaly_days(data, anomalies)
    plot_total_ots_before_after(data, cleaned, plots_dir)
    plot_category_ots_change(data, cleaned, plots_dir)
    plot_daily_anomaly_count(anomalies, plots_dir)


def safe_filename(value: str) -> str:
    return re.sub(r"[^0-9A-Za-zА-Яа-яЁё_.-]+", "_", value).strip("_") or "value"


def plot_before_after_by_dimension(
    data: pd.DataFrame,
    anomalies: pd.DataFrame,
    column: str,
    output_path: Path,
    top_n: int = 25,
) -> pd.DataFrame:
    if column not in data.columns:
        raise ValueError(f"Column {column!r} is not present in the data")

    before_rows = analysis_rows(data)
    cleaned = remove_anomaly_days(data, anomalies)
    after_rows = analysis_rows(cleaned)
    before = before_rows.groupby(column, observed=True)[OTS_WEIGHT_COLUMN].sum().rename("before_ots")
    after = after_rows.groupby(column, observed=True)[OTS_WEIGHT_COLUMN].sum().rename("after_ots")
    table = pd.concat([before, after], axis=1).fillna(0.0)
    table["change_pct"] = np.where(table["before_ots"].gt(0), (table["after_ots"] / table["before_ots"] - 1) * 100, 0)
    table = table.sort_values("before_ots", ascending=False).head(top_n)

    fig, ax = plt.subplots(figsize=(12, max(5, 0.35 * len(table))))
    y = np.arange(len(table))
    ax.barh(y - 0.18, table["before_ots"], height=0.35, label="before", color="#4f7cac")
    ax.barh(y + 0.18, table["after_ots"], height=0.35, label="after", color="#c85a54")
    ax.set_yticks(y)
    ax.set_yticklabels(table.index.astype(str))
    ax.invert_yaxis()
    ax.set_title(f"OTS before/after by {column}")
    ax.set_xlabel("OTS")
    ax.legend()
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return table.reset_index()


def export_queries_for_anomaly(
    data: pd.DataFrame,
    subject_id: int,
    researchdate: str,
    output_csv: Path | None = None,
) -> pd.DataFrame:
    normalized = normalize_columns(data)
    date_value = pd.to_datetime(researchdate).normalize()
    rows = normalized.loc[(normalized["SubjectID"].eq(subject_id)) & (normalized["researchdate"].eq(date_value))].copy()
    sort_columns = [column for column in ["Start", "Brand", "CategoryDelivery"] if column in rows.columns]
    if sort_columns:
        rows = rows.sort_values(sort_columns)
    columns = [
        column
        for column in [
            "SubjectID",
            "researchdate",
            "Start",
            "QueryText",
            "BrandID",
            "Brand",
            "CategoryDelivery",
            "ResourceName",
            "ResourceType",
            "UseType",
            "Platform",
            "Weight",
            "BrandinDelivery",
        ]
        if column in rows.columns
    ]
    result = rows[columns]
    if output_csv is not None:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        out = result.copy()
        out["researchdate"] = pd.to_datetime(out["researchdate"]).dt.strftime("%Y-%m-%d")
        out.to_csv(output_csv, index=False)
    return result


def plot_brand_daily_ots(
    data: pd.DataFrame,
    anomalies: pd.DataFrame,
    brand_id: str,
    output_path: Path,
    category_delivery: str | None = None,
) -> pd.DataFrame:
    rows = analysis_rows(data)
    rows = rows.loc[rows["BrandID"].astype(str).eq(str(brand_id))].copy()
    if category_delivery is not None:
        rows = rows.loc[rows["CategoryDelivery"].astype(str).eq(category_delivery)]

    cleaned = remove_anomaly_days(rows, anomalies)
    before = rows.groupby("researchdate", observed=True)[OTS_WEIGHT_COLUMN].sum().rename("before_ots")
    after = cleaned.groupby("researchdate", observed=True)[OTS_WEIGHT_COLUMN].sum().rename("after_ots")
    table = pd.concat([before, after], axis=1).fillna(0.0).sort_index()

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(table.index, table["before_ots"], label="before", linewidth=2)
    ax.plot(table.index, table["after_ots"], label="after", linewidth=2)
    title = f"Daily OTS for BrandID={brand_id}"
    if category_delivery:
        title += f", CategoryDelivery={category_delivery}"
    ax.set_title(title)
    ax.set_xlabel("researchdate")
    ax.set_ylabel("OTS")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.autofmt_xdate()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return table.reset_index()


def make_demo_analytics(data: pd.DataFrame, anomalies: pd.DataFrame, output_dir: Path) -> None:
    analytics_dir = output_dir / "plots" / "analytics"
    for column in DEMOGRAPHIC_COLUMNS + RESOURCE_COLUMNS + CATEGORY_COLUMNS:
        if column in data.columns:
            plot_before_after_by_dimension(
                data,
                anomalies,
                column,
                analytics_dir / f"before_after_{safe_filename(column)}.png",
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detect anomalous respondent-day activities in SoS search data.")
    parser.add_argument("--input", type=Path, default=find_default_input_dir(), help="Path to data_train directory or parquet file.")
    parser.add_argument("--output", type=Path, default=Path("output"), help="Directory for output files.")
    parser.add_argument("--make-analytics", action="store_true", help="Build additional before/after plots by dimensions.")
    parser.add_argument("--query-subject", type=int, help="SubjectID for exporting QueryText table.")
    parser.add_argument("--query-date", type=str, help="researchdate YYYY-MM-DD for exporting QueryText table.")
    parser.add_argument("--brand-id", type=str, help="BrandID for optional before/after daily OTS plot.")
    parser.add_argument("--brand-category", type=str, default=None, help="Optional CategoryDelivery filter for --brand-id.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = read_source_data(args.input)
    anomalies, reasons, _scored = detect_anomalies(data)

    save_csv_outputs(anomalies, reasons, args.output)
    save_required_plots(data, anomalies, args.output)
    validate_required_outputs(args.output)

    if args.make_analytics:
        make_demo_analytics(data, anomalies, args.output)

    if args.query_subject is not None and args.query_date is not None:
        export_queries_for_anomaly(
            data,
            args.query_subject,
            args.query_date,
            args.output / f"query_text_{args.query_subject}_{args.query_date}.csv",
        )

    if args.brand_id is not None:
        plot_brand_daily_ots(
            data,
            anomalies,
            args.brand_id,
            args.output / "plots" / f"brand_daily_ots_{safe_filename(args.brand_id)}.png",
            args.brand_category,
        )

    print(f"Saved {len(anomalies)} anomaly respondent-days to {args.output / 'anomalies.csv'}")
    print(f"Saved {len(reasons)} trigger reasons to {args.output / 'anomaly_reasons.csv'}")


if __name__ == "__main__":
    main()
