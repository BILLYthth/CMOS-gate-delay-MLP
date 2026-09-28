# CMOS Gate Delay Prediction using Machine Learning

Predicting propagation delay of CMOS logic gates (NAND, NOR, EXOR) with a Multi-Layer Perceptron (MLP), as a fast alternative to repeated SPICE simulation.

**Senior Project 1 — HCMUTE, Faculty of International Education**
Advisor: PhD. Pham Van Khoa · Co-author: Nguyen Trong Duc

---

## Overview

Circuit designers typically rely on repeated SPICE/Cadence simulations to characterize how gate delay changes with process, voltage, and load conditions — an accurate but slow process, especially early in the design cycle. This project explores replacing repeated simulation with a trained ML model: an MLP that predicts propagation delay (and equivalent resistance) directly from circuit parameters, trained on data generated in **Cadence Virtuoso (90nm GPDK)**.

## Pipeline

1. **Data generation** — Parametric sweeps (VDD, load capacitance, transistor width, k-ratio, temperature) run in Cadence ADE XL across NAND, NOR, and EXOR gates, exported as CSV.
2. **Parsing & feature engineering** — Custom regex-based parser to handle Cadence's comma-embedded headers; derived features include `Wp = k × Wn`, `Tp = (TpHL + TpLH) / 2`, and an Ohm's-law-derived equivalent resistance (`Req = 0.5 × VDD × (mult/I_D)`), computed independently of the delay target to avoid data leakage.
3. **Model training** — `scikit-learn` `MLPRegressor` with `GridSearchCV` (5-fold CV) to tune architecture/activation/regularization; log-space target transform.
4. **Evaluation** — MAPE, R², and Q² (cross-validated, pooled out-of-fold predictions) to check for overfitting.

<img width="597" height="847" alt="pipeline_flowchart" src="https://github.com/user-attachments/assets/bb823e9e-c48b-4b2e-bf8b-b60732b9a333" />


## Results

| Gate | Test MAPE | R² | Q² |
|------|-----------|-----|-----|
| NAND | 3.73% | 0.997 | 0.992 |
| NOR  | 4.51% | 0.996 | 0.991 |
| EXOR | 3.66% | 0.996 | 0.989 |

The small gap between R² (in-sample) and Q² (cross-validated) across all three gates indicates limited overfitting.

## Repository Structure

```
├── src/                # Parsing, feature engineering, training pipeline
├── notebooks/           # Exploratory analysis and result visualization
├── models/              # Trained model artifacts (.pkl)
├── data/                # Small sample of the dataset (full data omitted — generated via licensed Cadence PDK)
├── results/              # Metric tables and plots
├── requirements.txt
└── README.md
```

## Tech Stack

- **Circuit simulation:** Cadence Virtuoso, ADE XL (90nm GPDK)
- **ML:** Python, scikit-learn (MLPRegressor, GridSearchCV), pandas, numpy
- **Visualization:** matplotlib

## Running the Project

```bash
pip install -r requirements.txt
python src/mlp_pipeline.py
```

## Notes

- Raw Cadence simulation data is not included in full, as it depends on a licensed foundry PDK. A small representative sample is provided in `data/` to illustrate the format.
- This project was the top-performing Senior Project 1 submission in its cohort at HCMUTE.

## License

MIT
