"""Utility script for exploring an SPSS .sav dataset.

This script uses pyreadstat to load the dataset and pandas to explore
its structure. Adjust ``DATASET_PATH`` to point to the desired file.
"""
from __future__ import annotations

import argparse
import sys
import textwrap
from pathlib import Path

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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_path: Path = args.dataset

    print("Loading dataset from:", dataset_path)
    if not dataset_path.exists():
        print(
            "Error: The specified dataset file was not found. "
            "Please ensure the path is correct and accessible."
        )
        sys.exit(1)

    try:
        df, meta = pyreadstat.read_sav(dataset_path)
    except (pyreadstat.PyreadstatError, OSError) as exc:
        print("Failed to load dataset:", exc)
        sys.exit(1)

    print("\n=== File Metadata ===")
    print("File label:", meta.file_label or "<none>")
    print("Number of variables:", meta.number_columns)
    print("Number of cases:", meta.number_rows)
    if meta.creation_time:
        print("Created:", meta.creation_time)
    if meta.modified_time:
        print("Modified:", meta.modified_time)
    if meta.author:
        print("Author:", meta.author)

    print("\n=== Variable Information ===")
    print("Total columns:", df.shape[1])
    print("Column names:")
    for name in df.columns:
        label = meta.column_labels.get(name)
        print(f"  - {name}: {label or '<no label>'}")

    print("\nData types (pandas dtypes):")
    print(df.dtypes.to_string())

    print("\nMissing values per column:")
    missing_counts = df.isna().sum()
    print(missing_counts.to_string())

    print("\n=== Value Labels ===")
    if meta.variable_value_labels:
        for variable, labels in meta.variable_value_labels.items():
            print(f"\n{variable}:")
            for value, label in labels.items():
                print(f"  {value}: {label}")
    else:
        print("No value labels present.")

    print("\n=== Dataset Preview ===")
    print("First 5 rows:")
    print(df.head().to_string(index=False))

    print("\nDescriptive statistics for numeric columns:")
    numeric_description = df.describe(include=["number"])
    print(numeric_description.to_string())

    print("\nFrequency tables for categorical columns (up to top 10 values):")
    categorical_cols = df.select_dtypes(include=["object", "category"])
    if not categorical_cols.empty:
        for column in categorical_cols.columns:
            print(f"\n{column}:")
            counts = df[column].value_counts(dropna=False).head(10)
            formatted = textwrap.indent(counts.to_string(), prefix="  ")
            print(formatted)
    else:
        print("No categorical columns detected.")


if __name__ == "__main__":
    main()
