# Bayesian Inference & Decision Theory — Group Project

Semester group project applying Bayesian inference to a real-world dataset.

## Project Setup

- **Topic:** Hierarchical Bayesian Models
- **Dataset:** Diabetes 130-US Hospitals (1999–2008) — readmission data
- **Grouping variable:** `medical_specialty` (73 categories; hospital ID is not available in this dataset)
- **Target:** `readmitted_30d` (binary — readmitted within 30 days vs. not)

## Structure

- `data/` — raw and cleaned datasets (`diabetic_data_cleaned.csv` is the preprocessed version used downstream)
- `notebooks/` — exploratory analysis, model fitting, diagnostics
- `slides/` — final presentation deck
- `docs/` — brief, topic decision notes, preprocessing & EDA report

## Status

- [x] Topic chosen
- [x] Dataset identified
- [x] Data cleaned and preprocessed (96,436 rows × 48 columns, zero missing values)
- [x] EDA complete (specialty-level readmission patterns, candidate predictors, correlation check)
- [ ] Priors specified
- [ ] Model fitted (complete pooling / no pooling / partial pooling comparison)
- [ ] Diagnostics run
- [ ] Results interpreted
- [ ] Slides prepared

See `docs/Preprocessing_EDA_Report.docx` for the full writeup of preprocessing decisions and EDA findings.