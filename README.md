# ISYE 7406 Final Project — Production Scale, Box Office Revenue and Oscar Recognition

This repository holds the code and outputs for the ISYE 7406 course project (Team 51, Jacob Schroeder). The midterm notebook builds the analysis dataset and produces every figure and table in the midterm report:
- **Data preparation**: joins the TMDB 5000 movies and credits files, engineers cast, crew and genre features, cleans the data, adjusts dollars to 2017 values and merges Oscar nominations and wins
- **Exploratory analysis**: time trends (RQ1), production scale vs. budget (RQ2), non-linear patterns in return on budget (RQ3) and the nomination and win stages of the awards question (RQ4)
- **Report tables**: exports every table as a LaTeX file for the report

---

## Prerequisites

The notebook was run on Python 3.14. Install the required packages:

```bash
pip install -r requirements.txt
```

The notebook imports from a local `models/` folder included in this repository. Keep it in the same directory as the notebook; it needs no other setup.

---

## Data

Place the following files in a `Data/` directory at the project root:

| File | Source | Used for |
|---|---|---|
| `tmdb_5000_movies.csv` | [Kaggle: TMDB 5000 Movie Dataset](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata) | Budget, revenue, genres, release date, ratings |
| `tmdb_5000_credits.csv` | [Kaggle: TMDB 5000 Movie Dataset](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata) | Cast and crew lists |
| `the_oscar_award.csv` | [Kaggle: The Oscar Award, 1927–2020](https://www.kaggle.com/datasets/unanimad/the-oscar-award) | Oscar nominations and wins (optional; the awards analysis is skipped without it) |

Dollar amounts are converted to 2017 dollars with the U.S. Bureau of Labor Statistics CPI-U annual averages, which are stored in `models/data_loader.py`.

---

## Running the Notebook

1. Launch Jupyter:
   ```bash
   jupyter notebook midterm_eda.ipynb
   ```

2. Run all cells top to bottom (**Kernel → Restart & Run All**). The `outputs/` directories are created automatically.

3. Optionally, run the smoke tests:
   ```bash
   python tests.py
   ```

---

## Code Structure

| File | Contents |
|---|---|
| `midterm_eda.ipynb` | Runs the full pipeline and shows every figure and table |
| `models/output_writer.py` | `OutputWriter`: logs messages and saves figures, tables and the console log |
| `models/data_loader.py` | `TMDBDataLoader`: loading, feature engineering, cleaning, inflation adjustment and the Oscar merge |
| `models/exploratory_analysis.py` | `ExploratoryAnalysis`: one method per part of the exploratory analysis |
| `models/latex_export.py` | `LatexTableExporter`: converts the CSV tables to LaTeX for the report |
| `tests.py` | Smoke tests that check the cleaning and Oscar-merge counts |

---

## Outputs

All results are saved to the `outputs/` directory:

| Path | Description |
|---|---|
| `figures/*.png` | Every figure in the report and appendix |
| `tables/*.csv` | Every table in the report and appendix, plus `oscar_unmatched_check.csv` for reviewing the Oscar match |
| `tables_tex/*.tex` | The same tables as LaTeX `tabular` code, read by the report with `\input{}` |
| `tmdb_clean.csv` | The analysis-ready dataset (3,139 films) |
| `eda_summary.txt` | The full console log of the run |
