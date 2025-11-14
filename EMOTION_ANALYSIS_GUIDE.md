# Emotion Analysis Guide

## Quick Start

### Basic Usage
```bash
python emotion_analysis.py "G:/My Drive/osfstorage-archive/waves 1-7 Feb 2025.sav"
```

### Analyze Only Political Groups (Left vs Right)
```bash
python emotion_analysis.py "G:/My Drive/osfstorage-archive/waves 1-7 Feb 2025.sav" --political-only
```

### Analyze Specific Groups
```bash
python emotion_analysis.py "path/to/file.sav" --groups lef rig arab jew
```

### Custom Output Directory
```bash
python emotion_analysis.py "path/to/file.sav" --output-dir my_results
```

## What the Script Does

### שלב 1: Descriptive Statistics (ניתוח תיאורי)
- Calculates **mean**, **standard deviation**, and **N** for each emotion×group×wave
- Handles missing data automatically
- Output: `descriptive_statistics.csv`

### שלב 2: Visualization (ויזואליזציה)
Creates multiple plots:

1. **Line plots per group** (`lineplot_{group}.png`)
   - 5 emotions over 7 waves
   - Error bars showing standard error
   - Separate file for each group

2. **Faceted line plot** (`lineplot_all_emotions_faceted.png`)
   - All groups compared for each emotion
   - Easy to see which groups show similar trends

3. **Heatmap** (`heatmap_emotions.png`)
   - All emotions × all waves in one view
   - Color intensity shows emotion strength

### שלב 3: Statistical Tests (מבחנים סטטיסטיים)
- **One-way ANOVA**: Tests if emotion differs across T1-T7
- **T1 vs T7 t-test**: Specific comparison of first and last wave
- **Independent samples** approach (correct for different people per wave)
- Output: `anova_results.csv`

### שלב 4: Trend Analysis (ניתוח מגמות)
- **Linear regression**: emotion ~ wave number
- Detects:
  - ↑ Increasing trends (slope > 0, p < 0.05)
  - ↓ Decreasing trends (slope < 0, p < 0.05)
  - No significant trend
- Output: `trend_analysis.csv`

### שלב 5: Summary Export (סיכום)
- `summary_all_results.csv`: Combined ANOVA + trend results
- Shows T1 mean, T7 mean, change, p-values, and trend direction

## Output Files

All files saved to `emotion_analysis_results/` (or custom directory):

```
emotion_analysis_results/
├── descriptive_statistics.csv      # Mean, SD, N for all combinations
├── anova_results.csv                # Statistical test results
├── trend_analysis.csv               # Linear regression results
├── summary_all_results.csv          # Combined key findings
├── lineplot_lef.png                 # Left group emotions over time
├── lineplot_rig.png                 # Right group emotions over time
├── lineplot_arab.png                # Arab group emotions over time
├── ... (one per group)
├── lineplot_all_emotions_faceted.png # All groups compared
└── heatmap_emotions.png             # Overview heatmap
```

## Understanding the Results

### Key Columns in Output Files

**descriptive_statistics.csv:**
- `group`: Target group (e.g., "Left", "Right")
- `emotion`: Specific emotion (1-5)
- `wave`: T1-T7
- `mean`: Average intensity (probably 1-7 scale)
- `std`: Standard deviation
- `n`: Number of respondents

**anova_results.csv:**
- `ANOVA_p_value`: Does emotion differ across T1-T7?
- `T1_vs_T7_p_value`: Specific comparison first vs last
- `change_T1_to_T7`: Positive = increase, Negative = decrease

**trend_analysis.csv:**
- `slope`: Change per wave (positive = increasing)
- `p_value`: Is trend significant?
- `trend`: "Increasing ↑", "Decreasing ↓", or "No trend"

## Groups Available

```
arab  = Arabs (ערבים)
jew   = Jews (יהודים)
ashk  = Ashkenazi (אשכנזים)
miz   = Mizrahi (מזרחיים)
lef   = Left (שמאל)          ← Political
rig   = Right (ימין)         ← Political
reli  = Religious (דתיים)
un    = Secular (חילוניים)
```

## Emotions Measured

```
1 = Contempt (בוז)
2 = Disgust (גועל)
3 = Fear (פחד)
4 = Anger (כעס)
5 = Hatred (שנאה)
```

## Advanced Options

```bash
# Skip plot generation (faster)
python emotion_analysis.py file.sav --no-plots

# Change significance level
python emotion_analysis.py file.sav --alpha 0.01

# Get help
python emotion_analysis.py --help
```

## Interpreting Political Polarization

For research on political polarization, focus on:

1. **Emotions toward Left** (`negfeel.lef_*`)
2. **Emotions toward Right** (`negfeel.rig_*`)

Look for:
- ✓ Increasing trends → Growing polarization
- ✓ Divergence between groups → Asymmetric polarization
- ✓ Spikes in specific waves → Event-driven changes

## Example Questions This Answers

1. **Did anger toward the political outgroup increase over time?**
   → Check `trend_analysis.csv` for "Right (ימין) - Anger" slope

2. **Which emotion changed most from T1 to T7?**
   → Check `anova_results.csv`, sort by `change_T1_to_T7`

3. **Are there differences in polarization by emotion type?**
   → Compare slopes across 5 emotions in political groups

4. **When did the biggest changes occur?**
   → Look at line plots to visually identify inflection points
