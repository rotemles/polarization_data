"""Utility script for exploring an SPSS .sav dataset.

This script uses pyreadstat to load the dataset and pandas to explore
its structure. Adjust ``DATASET_PATH`` to point to the desired file.
"""
from __future__ import annotations

import argparse
import json
import sys
import textwrap
from pathlib import Path
from typing import Any

import pandas as pd
import pyreadstat


# Update this path to point at the location of the SPSS file on the local machine.
DATASET_PATH = Path(r"G:\\My Drive\\osfstorage-archive\\waves 1-7 Feb 2025.sav")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Explore an SPSS .sav dataset by printing metadata, value labels, "
            "and basic descriptive statistics."
        )
    )
    parser.add_argument(
        "dataset",
        nargs="?",
        default=DATASET_PATH,
        type=Path,
        help=(
            "Path to the SPSS .sav file. Defaults to the path specified in "
            "the script."
        ),
    )
    parser.add_argument(
        "--preview-rows",
        type=int,
        default=5,
        help="Number of rows to display in dataset preview (default: 5)",
    )
    parser.add_argument(
        "--max-categories",
        type=int,
        default=10,
        help="Maximum number of categories to show in frequency tables (default: 10)",
    )
    parser.add_argument(
        "--correlations",
        action="store_true",
        help="Show correlation matrix for numeric variables",
    )
    parser.add_argument(
        "--export-json",
        type=Path,
        metavar="FILE",
        help="Export analysis results to JSON file",
    )
    parser.add_argument(
        "--export-summary",
        type=Path,
        metavar="FILE",
        help="Export summary statistics to CSV file",
    )
    parser.add_argument(
        "--columns",
        nargs="+",
        help="Analyze only specific columns (space-separated list)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress detailed output, only show summary",
    )
    return parser.parse_args()


def print_section(title: str, char: str = "=") -> None:
    """Print a formatted section header."""
    print(f"\n{char * 60}")
    print(f"{title:^60}")
    print(f"{char * 60}\n")


def print_subsection(title: str) -> None:
    """Print a formatted subsection header."""
    print(f"\n--- {title} ---")


def calculate_data_quality(df: pd.DataFrame) -> dict[str, Any]:
    """Calculate comprehensive data quality metrics."""
    total_cells = df.shape[0] * df.shape[1]
    missing_cells = df.isna().sum().sum()

    quality_metrics = {
        "total_rows": df.shape[0],
        "total_columns": df.shape[1],
        "total_cells": total_cells,
        "missing_cells": int(missing_cells),
        "completeness_percentage": round((1 - missing_cells / total_cells) * 100, 2),
        "memory_usage_mb": round(df.memory_usage(deep=True).sum() / 1024**2, 2),
        "duplicate_rows": int(df.duplicated().sum()),
    }

    # Per-column quality
    column_quality = []
    for col in df.columns:
        col_info = {
            "name": col,
            "dtype": str(df[col].dtype),
            "missing_count": int(df[col].isna().sum()),
            "missing_percentage": round((df[col].isna().sum() / len(df)) * 100, 2),
            "unique_values": int(df[col].nunique()),
            "uniqueness_percentage": round((df[col].nunique() / len(df)) * 100, 2),
        }
        column_quality.append(col_info)

    quality_metrics["column_quality"] = column_quality
    return quality_metrics


def get_variable_types(df: pd.DataFrame) -> dict[str, list[str]]:
    """Categorize variables by type."""
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    datetime_cols = df.select_dtypes(include=["datetime"]).columns.tolist()

    return {
        "numeric": numeric_cols,
        "categorical": categorical_cols,
        "datetime": datetime_cols,
    }


def export_to_json(data: dict[str, Any], filepath: Path) -> None:
    """Export analysis results to JSON file."""
    # Convert non-serializable types
    def convert_types(obj):
        if isinstance(obj, (pd.Series, pd.DataFrame)):
            return obj.to_dict()
        elif isinstance(obj, (pd.Timestamp, pd.Timedelta)):
            return str(obj)
        elif pd.isna(obj):
            return None
        return obj

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=convert_types)
    print(f"\n✓ Analysis exported to JSON: {filepath}")


def main() -> None:
    args = parse_args()
    dataset_path: Path = args.dataset

    if not args.quiet:
        print(f"Loading dataset from: {dataset_path}")

    if not dataset_path.exists():
        print(
            "Error: The specified dataset file was not found. "
            "Please ensure the path is correct and accessible."
        )
        sys.exit(1)

    if not dataset_path.is_file():
        print(
            f"Error: The path '{dataset_path}' is not a file. "
            "Please provide a path to an SPSS .sav file, not a directory."
        )
        sys.exit(1)

    try:
        df, meta = pyreadstat.read_sav(dataset_path)
    except (pyreadstat.ReadstatError, OSError) as exc:
        print("Failed to load dataset:", exc)
        sys.exit(1)

    # Create column labels dictionary (column_labels is a list, not a dict)
    column_labels_dict = {}
    if hasattr(meta, 'column_names') and hasattr(meta, 'column_labels'):
        column_labels_dict = dict(zip(meta.column_names, meta.column_labels))

    # Filter columns if specified
    if args.columns:
        available_cols = df.columns.tolist()
        invalid_cols = [c for c in args.columns if c not in available_cols]
        if invalid_cols:
            print(f"Warning: Invalid columns will be ignored: {invalid_cols}")
        valid_cols = [c for c in args.columns if c in available_cols]
        if not valid_cols:
            print("Error: No valid columns specified.")
            sys.exit(1)
        df = df[valid_cols]
        if not args.quiet:
            print(f"Analyzing {len(valid_cols)} selected columns")

    # Calculate data quality metrics
    quality_metrics = calculate_data_quality(df)
    var_types = get_variable_types(df)

    # Prepare export data
    export_data = {
        "file_metadata": {
            "file_label": meta.file_label,
            "number_columns": meta.number_columns,
            "number_rows": meta.number_rows,
            "creation_time": str(meta.creation_time) if getattr(meta, 'creation_time', None) else None,
            "modified_time": str(getattr(meta, 'modified_time', None)) if getattr(meta, 'modified_time', None) else None,
            "author": getattr(meta, 'author', None),
        },
        "data_quality": quality_metrics,
        "variable_types": var_types,
    }

    # === File Metadata ===
    if not args.quiet:
        print_section("FILE METADATA")
        print(f"File label:         {meta.file_label or '<none>'}")
        print(f"Number of variables: {meta.number_columns}")
        print(f"Number of cases:     {meta.number_rows}")
        creation_time = getattr(meta, 'creation_time', None)
        if creation_time:
            print(f"Created:             {creation_time}")
        modified_time = getattr(meta, 'modified_time', None)
        if modified_time:
            print(f"Modified:            {modified_time}")
        author = getattr(meta, 'author', None)
        if author:
            print(f"Author:              {author}")

    # === Data Quality Overview ===
    print_section("DATA QUALITY OVERVIEW")
    print(f"Total rows:              {quality_metrics['total_rows']:,}")
    print(f"Total columns:           {quality_metrics['total_columns']:,}")
    print(f"Total cells:             {quality_metrics['total_cells']:,}")
    print(f"Missing cells:           {quality_metrics['missing_cells']:,}")
    print(f"Data completeness:       {quality_metrics['completeness_percentage']}%")
    print(f"Duplicate rows:          {quality_metrics['duplicate_rows']:,}")
    print(f"Memory usage:            {quality_metrics['memory_usage_mb']} MB")

    print(f"\nVariable type breakdown:")
    print(f"  Numeric:      {len(var_types['numeric'])}")
    print(f"  Categorical:  {len(var_types['categorical'])}")
    print(f"  DateTime:     {len(var_types['datetime'])}")

    # === Variable Information ===
    if not args.quiet:
        print_section("VARIABLE INFORMATION")
        print(f"{'Variable':<30} {'Label':<40} {'Type':<10} {'Missing %':<10} {'Unique':<10}")
        print("-" * 100)
        for col in df.columns:
            label = column_labels_dict.get(col, '<no label>')
            col_quality = next((c for c in quality_metrics['column_quality'] if c['name'] == col), {})
            print(
                f"{col:<30} {label[:40]:<40} {str(df[col].dtype):<10} "
                f"{col_quality.get('missing_percentage', 0):<10.1f} "
                f"{col_quality.get('unique_values', 0):<10}"
            )

    # === Value Labels ===
    if not args.quiet and meta.variable_value_labels:
        print_section("VALUE LABELS")
        for variable, labels in meta.variable_value_labels.items():
            if args.columns and variable not in args.columns:
                continue
            print_subsection(variable)
            for value, label in sorted(labels.items()):
                print(f"  {value:>5}: {label}")
        export_data["value_labels"] = meta.variable_value_labels

    # === Data Quality Issues ===
    issues = []
    for col_quality in quality_metrics['column_quality']:
        if col_quality['missing_percentage'] > 50:
            issues.append(f"  ⚠ {col_quality['name']}: {col_quality['missing_percentage']}% missing data")
        if col_quality['unique_values'] == 1:
            issues.append(f"  ⚠ {col_quality['name']}: constant column (only 1 unique value)")
        if col_quality['uniqueness_percentage'] == 100 and col_quality['name'] not in ['id', 'ID']:
            issues.append(f"  ℹ {col_quality['name']}: all values are unique (possible identifier)")

    if issues:
        print_section("DATA QUALITY ALERTS")
        for issue in issues[:20]:  # Limit to top 20 issues
            print(issue)
        if len(issues) > 20:
            print(f"\n  ... and {len(issues) - 20} more issues")

    # === Dataset Preview ===
    if not args.quiet:
        print_section("DATASET PREVIEW")
        print(f"First {args.preview_rows} rows:\n")
        print(df.head(args.preview_rows).to_string(index=True))

    # === Descriptive Statistics ===
    print_section("DESCRIPTIVE STATISTICS")

    if var_types['numeric']:
        print_subsection("Numeric Variables")
        numeric_stats = df[var_types['numeric']].describe()
        print(numeric_stats.to_string())
        export_data["numeric_statistics"] = numeric_stats.to_dict()

        # Additional statistics
        if not args.quiet:
            print("\nAdditional Statistics:")
            print(f"{'Variable':<30} {'Median':<12} {'Std Dev':<12} {'Skewness':<12}")
            print("-" * 66)
            for col in var_types['numeric']:
                median = df[col].median()
                std = df[col].std()
                skew = df[col].skew()
                print(f"{col:<30} {median:<12.2f} {std:<12.2f} {skew:<12.2f}")

    if var_types['categorical']:
        print_subsection("Categorical Variables")
        cat_summary = []
        for col in var_types['categorical'][:20]:  # Limit to 20 categorical vars
            n_unique = df[col].nunique()
            most_common = df[col].mode()[0] if len(df[col].mode()) > 0 else "N/A"
            cat_summary.append({
                "variable": col,
                "unique_values": n_unique,
                "most_common": str(most_common)
            })
            print(f"{col:<30} {n_unique:>8} unique values, mode: {most_common}")
        export_data["categorical_summary"] = cat_summary

    # === Frequency Tables ===
    if not args.quiet and var_types['categorical']:
        print_section("FREQUENCY TABLES")
        for column in var_types['categorical'][:10]:  # Limit to 10 tables
            print_subsection(f"{column}")
            counts = df[column].value_counts(dropna=False).head(args.max_categories)
            total = len(df)
            print(f"{'Value':<30} {'Count':<10} {'Percentage':<10}")
            print("-" * 50)
            for value, count in counts.items():
                pct = (count / total) * 100
                print(f"{str(value)[:30]:<30} {count:<10} {pct:>6.2f}%")

    # === Correlations ===
    if args.correlations and len(var_types['numeric']) > 1:
        print_section("CORRELATION ANALYSIS")
        corr_matrix = df[var_types['numeric']].corr()
        print("\nCorrelation Matrix:")
        print(corr_matrix.to_string())

        # Find strong correlations
        print("\nStrong Correlations (|r| > 0.5):")
        strong_corr = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) > 0.5:
                    var1 = corr_matrix.columns[i]
                    var2 = corr_matrix.columns[j]
                    strong_corr.append((var1, var2, corr_val))
                    print(f"  {var1} <-> {var2}: {corr_val:.3f}")

        if not strong_corr:
            print("  No strong correlations found.")

        export_data["correlations"] = corr_matrix.to_dict()

    # === Export Results ===
    if args.export_json:
        export_to_json(export_data, args.export_json)

    if args.export_summary:
        summary_df = pd.DataFrame(quality_metrics['column_quality'])
        summary_df.to_csv(args.export_summary, index=False)
        print(f"\n✓ Summary statistics exported to CSV: {args.export_summary}")

    print_section("ANALYSIS COMPLETE")


if __name__ == "__main__":
    main()
