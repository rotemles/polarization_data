# Polarization Data Analysis

A comprehensive tool for exploring and analyzing SPSS (.sav) datasets. This repository contains utilities for inspecting polarization poll data with advanced data quality checks, statistical analysis, and export capabilities.

## Overview

The `analyze_spss.py` script provides an in-depth exploration of SPSS datasets, offering insights into data quality, variable distributions, correlations, and more. It's designed to help researchers quickly understand their data structure and identify potential issues before conducting detailed analysis.

## Features

- **Comprehensive Data Quality Metrics**
  - Data completeness percentage
  - Missing value analysis per column
  - Duplicate row detection
  - Memory usage tracking
  - Automatic quality alerts for problematic columns

- **Statistical Analysis**
  - Descriptive statistics for numeric variables (mean, median, std, skewness)
  - Frequency tables with percentages for categorical variables
  - Correlation matrix with automatic detection of strong correlations
  - Variable type classification (numeric, categorical, datetime)

- **Flexible Output Control**
  - Customizable preview row counts
  - Quiet mode for summary-only output
  - Column-specific analysis
  - Structured, formatted output with clear sections

- **Export Capabilities**
  - JSON export of full analysis results
  - CSV export of column-level summary statistics
  - Easy integration with downstream analysis pipelines

## Installation

### Create Conda Environment

1. Create a new conda environment with the required packages:

```bash
conda create -n spss-analysis python=3.10 pandas pyreadstat -c conda-forge
```

2. Activate the environment:

```bash
conda activate spss-analysis
```

### Alternative: Using pip

If you prefer pip, create a virtual environment and install the required packages:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install pandas pyreadstat
```

### Required Packages

- **pandas**: Data manipulation and analysis library
- **pyreadstat**: Library for reading SPSS files (.sav format)
- **Python**: 3.8 or higher

## Usage

### Basic Usage

```bash
python analyze_spss.py path/to/your/dataset.sav
```

### Command-Line Arguments

```
positional arguments:
  dataset               Path to the SPSS .sav file. Defaults to the path
                        specified in the script.

optional arguments:
  -h, --help            Show this help message and exit

  --preview-rows N      Number of rows to display in dataset preview
                        (default: 5)

  --max-categories N    Maximum number of categories to show in frequency
                        tables (default: 10)

  --correlations        Show correlation matrix for numeric variables and
                        identify strong correlations (|r| > 0.5)

  --export-json FILE    Export complete analysis results to JSON file

  --export-summary FILE Export column-level summary statistics to CSV file

  --columns COL1 COL2   Analyze only specific columns (space-separated list)

  --quiet               Suppress detailed output, only show summary metrics
```

### Examples

#### 1. Basic analysis with default settings
```bash
python analyze_spss.py survey_data.sav
```

#### 2. Quick summary without detailed output
```bash
python analyze_spss.py survey_data.sav --quiet
```

#### 3. Analyze specific columns with correlation analysis
```bash
python analyze_spss.py survey_data.sav --columns age income education --correlations
```

#### 4. Export results for further processing
```bash
python analyze_spss.py survey_data.sav --export-json analysis.json --export-summary summary.csv
```

#### 5. Preview more rows and show more categories
```bash
python analyze_spss.py survey_data.sav --preview-rows 10 --max-categories 20
```

#### 6. Comprehensive analysis with all features
```bash
python analyze_spss.py survey_data.sav --correlations --export-json full_analysis.json --preview-rows 15
```

## What the Script Does

### 1. Data Loading & Validation
- Loads SPSS .sav files using pyreadstat
- Validates file existence and accessibility
- Extracts both data and metadata (variable labels, value labels, etc.)

### 2. File Metadata Display
- Shows file label, creation/modification dates, and author
- Reports number of variables and cases in the dataset
- Displays only when not in quiet mode

### 3. Data Quality Overview
The script automatically calculates and displays:
- **Total rows, columns, and cells**: Overall dataset dimensions
- **Missing cells count**: Number and percentage of missing values
- **Data completeness**: Percentage of non-missing data
- **Duplicate rows**: Number of exact duplicate records
- **Memory usage**: Dataset size in memory
- **Variable type breakdown**: Count of numeric, categorical, and datetime variables

### 4. Variable Information Table
For each variable, displays:
- Variable name
- Variable label (from SPSS metadata)
- Data type
- Missing data percentage
- Number of unique values

### 5. Value Labels
Displays SPSS value labels (coded responses) for categorical variables, showing the mapping between numeric codes and their meanings (e.g., 1: "Strongly Disagree", 2: "Disagree", etc.)

### 6. Data Quality Alerts
Automatically identifies and warns about:
- **High missing data**: Columns with >50% missing values
- **Constant columns**: Variables with only one unique value
- **Potential identifiers**: Columns where all values are unique

### 7. Dataset Preview
Shows the first N rows (default: 5) of the dataset with all columns, helping you understand the data structure visually.

### 8. Descriptive Statistics

#### For Numeric Variables:
- Standard statistics: count, mean, std, min, 25%, 50%, 75%, max
- Additional metrics: median, standard deviation, skewness
- Skewness helps identify asymmetric distributions

#### For Categorical Variables:
- Number of unique values per variable
- Most common value (mode)
- Limited to first 20 categorical variables to avoid overwhelming output

### 9. Frequency Tables
- Shows value counts and percentages for categorical variables
- Displays top N categories (default: 10, customizable)
- Includes missing values (NaN) in the counts
- Limited to first 10 categorical variables

### 10. Correlation Analysis (Optional)
When `--correlations` flag is used:
- Computes full correlation matrix for all numeric variables
- Identifies and highlights strong correlations (|r| > 0.5)
- Helps identify multicollinearity and relationships between variables

### 11. Export Functionality

#### JSON Export (`--export-json`)
Creates a JSON file containing:
- File metadata
- Complete data quality metrics
- Variable type classifications
- Numeric statistics
- Categorical summaries
- Value labels
- Correlation matrix (if computed)

#### CSV Export (`--export-summary`)
Creates a CSV file with one row per column, including:
- Column name and data type
- Missing count and percentage
- Unique value count and percentage
- Ideal for Excel analysis or further processing

## Output Structure

The script organizes output into clear sections:

1. **FILE METADATA** - Basic file information
2. **DATA QUALITY OVERVIEW** - High-level quality metrics
3. **VARIABLE INFORMATION** - Detailed variable table
4. **VALUE LABELS** - SPSS value label mappings
5. **DATA QUALITY ALERTS** - Automatic problem detection
6. **DATASET PREVIEW** - Sample rows from the dataset
7. **DESCRIPTIVE STATISTICS** - Statistical summaries
8. **FREQUENCY TABLES** - Category distributions
9. **CORRELATION ANALYSIS** - Variable relationships (if requested)
10. **ANALYSIS COMPLETE** - Confirmation message

## Configuration

To set a default dataset path, edit the `DATASET_PATH` variable in `analyze_spss.py`:

```python
DATASET_PATH = Path(r"path/to/your/default/dataset.sav")
```

This allows you to run the script without specifying the file path each time.

## Troubleshooting

### "ModuleNotFoundError: No module named 'pandas'"
Make sure you've activated your conda environment:
```bash
conda activate spss-analysis
```

### "File not found" error
- Verify the file path is correct
- Use absolute paths or ensure you're in the correct directory
- For Windows paths with spaces, use quotes: `python analyze_spss.py "C:\My Files\data.sav"`

### Memory issues with large datasets
- Use `--quiet` mode to reduce output
- Use `--columns` to analyze only specific columns
- Consider analyzing subsets of your data

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests for improvements.

## License

This project is open source and available for research and educational purposes.
