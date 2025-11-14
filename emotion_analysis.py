"""Analyze negative emotions toward different groups across time waves.

This script performs longitudinal analysis of emotion data from SPSS files,
including descriptive statistics, visualization, and statistical testing.
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyreadstat
import seaborn as sns
from scipy import stats

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=FutureWarning)

# Configuration
WAVES = ['T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'T7']
GROUPS = {
    'arab': 'Arabs (ערבים)',
    'jew': 'Jews (יהודים)',
    'ashk': 'Ashkenazi (אשכנזים)',
    'miz': 'Mizrahi (מזרחיים)',
    'lef': 'Left (שמאל)',
    'rig': 'Right (ימין)',
    'reli': 'Religious (דתיים)',
    'un': 'Secular (חילוניים)',
}
EMOTIONS = {
    '1': 'Contempt (בוז)',
    '2': 'Disgust (גועל)',
    '3': 'Fear (פחד)',
    '4': 'Anger (כעס)',
    '5': 'Hatred (שנאה)',
}

# Default SPSS file path
DATASET_PATH = Path(r"G:\My Drive\osfstorage-archive\waves 1-7 Feb 2025.sav")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze negative emotions toward different groups across time waves."
    )
    parser.add_argument(
        "dataset",
        nargs="?",
        default=DATASET_PATH,
        type=Path,
        help="Path to the SPSS .sav file",
    )
    parser.add_argument(
        "--groups",
        nargs="+",
        choices=list(GROUPS.keys()),
        help="Analyze only specific groups (default: all groups)",
    )
    parser.add_argument(
        "--political-only",
        action="store_true",
        help="Analyze only political groups (left, right)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("emotion_analysis_results"),
        help="Directory for output files (default: emotion_analysis_results)",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip generating plots",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Significance level for statistical tests (default: 0.05)",
    )
    return parser.parse_args()


def load_emotion_data(dataset_path: Path) -> tuple[pd.DataFrame, dict[str, str]]:
    """Load SPSS file and extract only emotion-related variables."""
    print(f"Loading dataset from: {dataset_path}")

    if not dataset_path.exists():
        print("Error: Dataset file not found.")
        sys.exit(1)

    if not dataset_path.is_file():
        print("Error: Path is not a file.")
        sys.exit(1)

    try:
        df, meta = pyreadstat.read_sav(dataset_path)
    except (pyreadstat.ReadstatError, OSError) as exc:
        print(f"Failed to load dataset: {exc}")
        sys.exit(1)

    # Filter to emotion columns only
    emotion_cols = [col for col in df.columns if col.startswith('negfeel.')]

    if not emotion_cols:
        print("Error: No emotion variables found (expected columns starting with 'negfeel.')")
        sys.exit(1)

    df_emotions = df[emotion_cols].copy()

    # Create column labels dictionary
    column_labels_dict = {}
    if hasattr(meta, 'column_names') and hasattr(meta, 'column_labels'):
        column_labels_dict = dict(zip(meta.column_names, meta.column_labels))

    print(f"Loaded {len(df_emotions)} respondents with {len(emotion_cols)} emotion variables")

    return df_emotions, column_labels_dict


def reshape_to_long(df: pd.DataFrame, groups_filter: list[str] | None = None) -> pd.DataFrame:
    """Reshape wide format to long format for analysis.

    Converts from: negfeel.{group}_{emotion}_T{wave}
    To long format with columns: respondent_id, group, emotion, wave, value
    """
    print("\nReshaping data to long format...")

    records = []

    for col in df.columns:
        # Parse column name: negfeel.{group}_{emotion}_T{wave}
        parts = col.split('.')
        if len(parts) != 2:
            continue

        name_parts = parts[1].split('_')
        if len(name_parts) < 3:
            continue

        group = name_parts[0]
        emotion = name_parts[1]
        wave = name_parts[2]

        # Apply group filter if specified
        if groups_filter and group not in groups_filter:
            continue

        # Extract non-null values
        values = df[col].dropna()

        for idx, value in values.items():
            records.append({
                'respondent_id': idx,
                'group': group,
                'emotion': emotion,
                'wave': wave,
                'value': value,
            })

    df_long = pd.DataFrame(records)

    print(f"Reshaped to {len(df_long)} observations")
    print(f"Groups: {sorted(df_long['group'].unique())}")
    print(f"Emotions: {sorted(df_long['emotion'].unique())}")
    print(f"Waves: {sorted(df_long['wave'].unique())}")

    return df_long


def calculate_descriptives(df_long: pd.DataFrame) -> pd.DataFrame:
    """Calculate descriptive statistics for each emotion×group×wave combination."""
    print("\n" + "="*60)
    print("STEP 1: DESCRIPTIVE STATISTICS")
    print("="*60)

    stats_list = []

    for group in sorted(df_long['group'].unique()):
        for emotion in sorted(df_long['emotion'].unique()):
            for wave in WAVES:
                subset = df_long[
                    (df_long['group'] == group) &
                    (df_long['emotion'] == emotion) &
                    (df_long['wave'] == wave)
                ]['value']

                if len(subset) > 0:
                    stats_list.append({
                        'group': group,
                        'emotion': emotion,
                        'wave': wave,
                        'mean': subset.mean(),
                        'std': subset.std(),
                        'n': len(subset),
                        'min': subset.min(),
                        'max': subset.max(),
                    })

    df_stats = pd.DataFrame(stats_list)

    # Create summary view
    print("\nSample descriptive statistics (first 20 rows):")
    print(df_stats.head(20).to_string(index=False))

    return df_stats


def create_line_plots(df_stats: pd.DataFrame, output_dir: Path) -> None:
    """Create line plots showing emotion trends over time."""
    print("\n" + "="*60)
    print("STEP 2: VISUALIZATION - LINE PLOTS")
    print("="*60)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Plot 1: Separate plot for each group
    for group_code, group_name in GROUPS.items():
        df_group = df_stats[df_stats['group'] == group_code]

        if df_group.empty:
            continue

        fig, ax = plt.subplots(figsize=(12, 6))

        for emotion_code, emotion_name in EMOTIONS.items():
            df_emotion = df_group[df_group['emotion'] == emotion_code]

            if df_emotion.empty:
                continue

            # Sort by wave
            df_emotion = df_emotion.sort_values('wave')

            # Plot mean with error bars (standard error)
            x = range(len(df_emotion))
            y = df_emotion['mean']
            se = df_emotion['std'] / np.sqrt(df_emotion['n'])

            ax.plot(x, y, marker='o', label=emotion_name, linewidth=2)
            ax.fill_between(x, y - se, y + se, alpha=0.2)

        ax.set_xlabel('Wave', fontsize=12)
        ax.set_ylabel('Mean Emotion Intensity', fontsize=12)
        ax.set_title(f'Negative Emotions Toward {group_name} Over Time', fontsize=14, fontweight='bold')
        ax.set_xticks(range(len(WAVES)))
        ax.set_xticklabels(WAVES)
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)

        filename = output_dir / f"lineplot_{group_code}.png"
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"✓ Saved: {filename}")

    # Plot 2: Faceted plot - all emotions, all groups
    groups_present = sorted(df_stats['group'].unique())
    emotions_present = sorted(df_stats['emotion'].unique())

    n_emotions = len(emotions_present)
    fig, axes = plt.subplots(n_emotions, 1, figsize=(14, 3*n_emotions), sharex=True)

    if n_emotions == 1:
        axes = [axes]

    for idx, emotion_code in enumerate(emotions_present):
        ax = axes[idx]

        for group_code in groups_present:
            df_subset = df_stats[
                (df_stats['group'] == group_code) &
                (df_stats['emotion'] == emotion_code)
            ].sort_values('wave')

            if df_subset.empty:
                continue

            x = range(len(df_subset))
            y = df_subset['mean']

            group_name = GROUPS.get(group_code, group_code)
            ax.plot(x, y, marker='o', label=group_name, linewidth=2)

        emotion_name = EMOTIONS.get(emotion_code, emotion_code)
        ax.set_ylabel('Mean Intensity', fontsize=11)
        ax.set_title(f'{emotion_name}', fontsize=12, fontweight='bold')
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(range(len(WAVES)))
        ax.set_xticklabels(WAVES)

    axes[-1].set_xlabel('Wave', fontsize=12)
    fig.suptitle('Emotion Trends Across All Groups', fontsize=16, fontweight='bold', y=1.001)

    filename = output_dir / "lineplot_all_emotions_faceted.png"
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ Saved: {filename}")


def create_heatmap(df_stats: pd.DataFrame, output_dir: Path) -> None:
    """Create heatmap showing emotion intensity across groups and waves."""
    print("\n" + "="*60)
    print("STEP 2: VISUALIZATION - HEATMAP")
    print("="*60)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Create pivot table: rows = group_emotion, columns = wave
    df_stats['group_emotion'] = df_stats.apply(
        lambda row: f"{GROUPS.get(row['group'], row['group'])} - {EMOTIONS.get(row['emotion'], row['emotion'])}",
        axis=1
    )

    pivot = df_stats.pivot_table(
        index='group_emotion',
        columns='wave',
        values='mean',
        aggfunc='first'
    )

    # Reorder columns to match wave order
    wave_order = [w for w in WAVES if w in pivot.columns]
    pivot = pivot[wave_order]

    # Create heatmap
    fig, ax = plt.subplots(figsize=(10, 16))

    sns.heatmap(
        pivot,
        cmap='YlOrRd',
        annot=True,
        fmt='.2f',
        linewidths=0.5,
        cbar_kws={'label': 'Mean Emotion Intensity'},
        ax=ax
    )

    ax.set_title('Emotion Intensity Heatmap: All Groups × All Waves', fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Wave', fontsize=12)
    ax.set_ylabel('Group - Emotion', fontsize=12)

    filename = output_dir / "heatmap_emotions.png"
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ Saved: {filename}")


def run_anova_tests(df_long: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
    """Run one-way ANOVA to test differences across waves."""
    print("\n" + "="*60)
    print("STEP 3: STATISTICAL TESTS - ANOVA")
    print("="*60)

    results = []

    for group in sorted(df_long['group'].unique()):
        for emotion in sorted(df_long['emotion'].unique()):
            # Get data for all waves
            wave_groups = []
            for wave in WAVES:
                subset = df_long[
                    (df_long['group'] == group) &
                    (df_long['emotion'] == emotion) &
                    (df_long['wave'] == wave)
                ]['value'].dropna()

                if len(subset) > 0:
                    wave_groups.append(subset.values)

            if len(wave_groups) < 2:
                continue

            # Run one-way ANOVA
            f_stat, p_value = stats.f_oneway(*wave_groups)

            # Also test T1 vs T7 specifically
            t1_data = df_long[
                (df_long['group'] == group) &
                (df_long['emotion'] == emotion) &
                (df_long['wave'] == 'T1')
            ]['value'].dropna()

            t7_data = df_long[
                (df_long['group'] == group) &
                (df_long['emotion'] == emotion) &
                (df_long['wave'] == 'T7')
            ]['value'].dropna()

            if len(t1_data) > 0 and len(t7_data) > 0:
                t_stat, t1_t7_p = stats.ttest_ind(t1_data, t7_data)
                t1_mean = t1_data.mean()
                t7_mean = t7_data.mean()
                delta = t7_mean - t1_mean
            else:
                t1_t7_p = np.nan
                t1_mean = np.nan
                t7_mean = np.nan
                delta = np.nan

            results.append({
                'group': GROUPS.get(group, group),
                'emotion': EMOTIONS.get(emotion, emotion),
                'F_statistic': f_stat,
                'ANOVA_p_value': p_value,
                'significant': 'Yes' if p_value < alpha else 'No',
                'T1_mean': t1_mean,
                'T7_mean': t7_mean,
                'change_T1_to_T7': delta,
                'T1_vs_T7_p_value': t1_t7_p,
                'T1_vs_T7_significant': 'Yes' if t1_t7_p < alpha else 'No',
            })

    df_anova = pd.DataFrame(results)

    print(f"\nCompleted ANOVA tests for {len(df_anova)} group×emotion combinations")
    print(f"Significant differences across waves (p < {alpha}): {(df_anova['significant'] == 'Yes').sum()}")
    print(f"Significant T1 vs T7 differences (p < {alpha}): {(df_anova['T1_vs_T7_significant'] == 'Yes').sum()}")

    return df_anova


def run_trend_analysis(df_long: pd.DataFrame) -> pd.DataFrame:
    """Run linear regression to detect trends over time."""
    print("\n" + "="*60)
    print("STEP 4: TREND ANALYSIS - LINEAR REGRESSION")
    print("="*60)

    results = []

    # Convert wave to numeric (T1=1, T2=2, etc.)
    wave_to_num = {f'T{i}': i for i in range(1, 8)}
    df_long['wave_num'] = df_long['wave'].map(wave_to_num)

    for group in sorted(df_long['group'].unique()):
        for emotion in sorted(df_long['emotion'].unique()):
            subset = df_long[
                (df_long['group'] == group) &
                (df_long['emotion'] == emotion)
            ][['wave_num', 'value']].dropna()

            if len(subset) < 10:  # Need reasonable sample size
                continue

            # Linear regression: value ~ wave_num
            slope, intercept, r_value, p_value, std_err = stats.linregress(
                subset['wave_num'],
                subset['value']
            )

            # Determine trend direction
            if p_value < 0.05:
                if slope > 0:
                    trend = 'Increasing ↑'
                else:
                    trend = 'Decreasing ↓'
            else:
                trend = 'No trend'

            results.append({
                'group': GROUPS.get(group, group),
                'emotion': EMOTIONS.get(emotion, emotion),
                'slope': slope,
                'intercept': intercept,
                'r_squared': r_value**2,
                'p_value': p_value,
                'std_error': std_err,
                'trend': trend,
            })

    df_trends = pd.DataFrame(results)

    print(f"\nCompleted trend analysis for {len(df_trends)} group×emotion combinations")
    print(f"Increasing trends: {(df_trends['trend'] == 'Increasing ↑').sum()}")
    print(f"Decreasing trends: {(df_trends['trend'] == 'Decreasing ↓').sum()}")
    print(f"No significant trend: {(df_trends['trend'] == 'No trend').sum()}")

    return df_trends


def export_results(
    df_stats: pd.DataFrame,
    df_anova: pd.DataFrame,
    df_trends: pd.DataFrame,
    output_dir: Path
) -> None:
    """Export all results to CSV files."""
    print("\n" + "="*60)
    print("STEP 5: EXPORTING RESULTS")
    print("="*60)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Export descriptive statistics
    filename = output_dir / "descriptive_statistics.csv"
    df_stats.to_csv(filename, index=False)
    print(f"✓ Saved: {filename}")

    # Export ANOVA results
    filename = output_dir / "anova_results.csv"
    df_anova.to_csv(filename, index=False)
    print(f"✓ Saved: {filename}")

    # Export trend analysis
    filename = output_dir / "trend_analysis.csv"
    df_trends.to_csv(filename, index=False)
    print(f"✓ Saved: {filename}")

    # Create summary table combining key results
    summary = df_anova.merge(
        df_trends[['group', 'emotion', 'slope', 'trend']],
        on=['group', 'emotion'],
        how='left'
    )

    filename = output_dir / "summary_all_results.csv"
    summary.to_csv(filename, index=False)
    print(f"✓ Saved: {filename}")


def print_key_findings(df_anova: pd.DataFrame, df_trends: pd.DataFrame) -> None:
    """Print summary of key findings."""
    print("\n" + "="*60)
    print("KEY FINDINGS SUMMARY")
    print("="*60)

    # Most significant changes T1 to T7
    print("\n--- Largest Changes from T1 to T7 ---")
    top_changes = df_anova.nlargest(5, 'change_T1_to_T7')[
        ['group', 'emotion', 'T1_mean', 'T7_mean', 'change_T1_to_T7', 'T1_vs_T7_p_value']
    ]
    print(top_changes.to_string(index=False))

    # Steepest trends
    print("\n--- Steepest Increasing Trends ---")
    increasing = df_trends[df_trends['trend'] == 'Increasing ↑'].nlargest(5, 'slope')[
        ['group', 'emotion', 'slope', 'p_value']
    ]
    print(increasing.to_string(index=False))

    print("\n--- Steepest Decreasing Trends ---")
    decreasing = df_trends[df_trends['trend'] == 'Decreasing ↓'].nsmallest(5, 'slope')[
        ['group', 'emotion', 'slope', 'p_value']
    ]
    print(decreasing.to_string(index=False))


def main() -> None:
    args = parse_args()

    # Determine which groups to analyze
    if args.political_only:
        groups_filter = ['lef', 'rig']
        print("Analyzing political groups only: Left, Right")
    elif args.groups:
        groups_filter = args.groups
        print(f"Analyzing selected groups: {', '.join(groups_filter)}")
    else:
        groups_filter = None
        print("Analyzing all groups")

    # Load data
    df_emotions, column_labels = load_emotion_data(args.dataset)

    # Reshape to long format
    df_long = reshape_to_long(df_emotions, groups_filter)

    # Step 1: Descriptive statistics
    df_stats = calculate_descriptives(df_long)

    # Step 2: Visualization
    if not args.no_plots:
        create_line_plots(df_stats, args.output_dir)
        create_heatmap(df_stats, args.output_dir)

    # Step 3: Statistical tests
    df_anova = run_anova_tests(df_long, alpha=args.alpha)

    # Step 4: Trend analysis
    df_trends = run_trend_analysis(df_long)

    # Step 5: Export results
    export_results(df_stats, df_anova, df_trends, args.output_dir)

    # Print key findings
    print_key_findings(df_anova, df_trends)

    print("\n" + "="*60)
    print("ANALYSIS COMPLETE!")
    print("="*60)
    print(f"\nAll results saved to: {args.output_dir.absolute()}")


if __name__ == "__main__":
    main()
