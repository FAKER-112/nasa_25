# Copilot Instructions for AI Coding Agents

## Project Overview
This workspace contains CSV datasets related to NASA's Kepler mission and a Jupyter notebook (`notebook1.ipynb`) for data analysis. The main workflow is exploratory data analysis (EDA) using pandas in Python.

## Key Files
- `cumulative_2025.09.30_03.59.13.csv`: Main dataset for analysis
- `nasa_Kepler Objects of Interest (KOI).csv`: Output or secondary dataset
- `notebook1.ipynb`: Central location for all code and analysis

## Data Loading & Manipulation
- Use pandas (`import pandas as pd`) for all data operations.
- CSVs are loaded with `pd.read_csv`, often with custom parameters:
  - `sep=","`, `quotechar='"'`, `on_bad_lines="skip"`, `low_memory=False`
  - Example: `df = pd.read_csv(file, sep=",", quotechar='"', on_bad_lines="skip", low_memory=False)`
- For skipping header or metadata rows, use `skiprows` (e.g., `skiprows=144`).
- Data is sometimes exported with `df.to_csv()`.

## Analysis Patterns
- Use `df.describe()` for summary statistics.
- Use `df.describe(include=[object])` for object-type columns.
- Use `df.info()`, `df.shape`, and `df.head()` for quick inspection.
- Null values are checked with `df.isnull().sum()`.

## Notebook Conventions
- All code is written in Python within Jupyter notebook cells.
- Dataframes are named `df` by convention.
- Output files are written to the workspace root.

## Developer Workflow
- No build or test scripts; all work is interactive in the notebook.
- No external dependencies beyond pandas (install with `pip install pandas` if needed).
- No custom modules or package structure; all logic is in the notebook.

## Integration Points
- No API calls, web services, or cross-component communication.
- All analysis is local and file-based.

## Example Workflow
1. Load CSV with pandas
2. Inspect data (`df.head()`, `df.info()`)
3. Analyze statistics (`df.describe()`)
4. Export results if needed (`df.to_csv()`)

## Special Notes
- If you encounter errors loading CSVs, check for header rows or malformed lines and use `skiprows` or `on_bad_lines`.
- All file paths are absolute and Windows-style (e.g., `C:\Users\User\Downloads\nasa25\...`).

---

If any section is unclear or missing, please provide feedback to improve these instructions.