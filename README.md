# Census Income Classification Project

## Overview

This project develops a machine learning pipeline for predicting high-income individuals (income ≥ $50,000) using the 1994-1995 U.S. Census Income dataset. The analysis includes comprehensive exploratory data analysis (EDA), model development using CatBoost gradient boosting, and business segmentation for actionable targeting strategies.

**Key Results:**
- CatBoost model achieving 0.697 weighted PR-AUC (16% improvement over logistic regression baseline)
- Identified high-value segment representing 7.1% of population with 57.4% high-income rate
- Robust cross-validation with low variance (σ = 0.0032) confirming generalizability

> **Complete Project Download:** For full reproducibility including the raw dataset files (census-bureau.data and census-bureau.columns), download the `Census data.zip` file from this repository.

---

## Requirements

### Software Dependencies

The project requires Python 3.10+ with the following packages:

```
pandas
numpy
matplotlib
seaborn
scikit-learn
catboost
jupyter
```

### Installation

Install all dependencies using pip:

```bash
pip install -r requirements.txt
```

---

## Project Structure

```
.
├── Census data.zip             # Complete project archive (includes all files below + data files)
├── census-bureau.data      # Raw dataset (199,523 observations, 43 features) [IN ZIP ONLY]
└── census-bureau.columns   # Column metadata and descriptions [IN ZIP ONLY]
│
├── Census_data.ipynb           # Main analysis notebook (ALL code and results)
├── report.pdf                 # Project report
├── ML-TakehomeProject.pdf      # Project assignment description
├── requirements.txt            # Python package dependencies
│
├── figures/                    # Generated visualizations
│   ├── Imbalance.png           # Class distribution
│   ├── pos_rate_*.png          # Weighted univariate analysis plots
│   ├── CatBoostFI.png          # Feature importance
│   └── pr_curve.png            # Precision-recall curve
│
├── tables/                     # Generated summary statistics
│   └── pos_rate_*.csv          # Univariate analysis tables
│
├── models/                     # Trained model artifacts
│   ├── catboost_*.cbm          # Saved CatBoost models
│   ├── feature_cols.json       # Feature metadata
│   └── label_map.json          # Label encoding
│
└── README.md                   # This file
```

---

## Execution Instructions

### Step 1: Data Preparation

**Note:** Due to GitHub file size limitations, the dataset files are not directly available in the repository but are included in the `Census data.zip` file.

To run the analysis:

1. **Download and extract** the `Census data.zip` file from the repository
2. Ensure the following dataset files are in the project root directory:
   - `census-bureau.data` (raw data file - 25MB)
   - `census-bureau.columns` (feature metadata)

Alternatively, if reviewing the complete project, simply extract all contents from the zip file.

### Step 2: Run the Analysis Notebook

The entire analysis pipeline is contained in a single Jupyter notebook: `Census_data.ipynb`

**Option A: Using Jupyter Notebook**

```bash
jupyter notebook Census_data.ipynb
```

Then run all cells sequentially using: `Cell > Run All`

**Option B: Using JupyterLab**

```bash
jupyter lab Census_data.ipynb
```

**Option C: Using VS Code**

Open `Census_data.ipynb` in VS Code with the Jupyter extension and click "Run All".

**Expected Runtime:** 40-60 minutes (depending on hardware; includes cross-validation and hyperparameter tuning)

**Note:** The notebook includes optional hyperparameter tuning sections ("Additional trials for potential improvement" and "Tuning") that can be skipped for faster execution (~20-30 minutes). These sections validate that the original parameters were optimal but are not required to reproduce the core results and report figures.

### Step 3: Review Generated Outputs

After successful execution, the notebook generates:

**Figures** (in `figures/`):
- Class distribution plot
- Weighted positive rate analyses by feature
- Feature importance visualizations

**Tables** (in `tables/`):
- Segment profiles and summary statistics
- Validation predictions

**Models** (in `models/`):
- Trained CatBoost models (`.cbm` files)
- Feature metadata and label encoding

---

## Notebook Contents

The `Census_data.ipynb` notebook contains the following sections:

1. **Data Loading and Inspection**
   - Initial data shape, types, and quality assessment
   - 199,523 rows × 43 columns
   
2. **Data Preprocessing**
   - Handling missing values and "Not in universe" categories
   - Conflict resolution (379 rows removed) and duplicate aggregation (53,499 duplicates)
   - Final cleaned dataset: 152,718 unique observations

3. **Exploratory Data Analysis**
   - Class imbalance visualization (93.8% vs. 6.2%)
   - Weighted univariate analysis by feature (8 categorical features analyzed)
   - Multivariate correlation analysis (Spearman and weighted Pearson)

4. **Baseline Modeling: Logistic Regression**
   - One-hot encoding and standardization
   - Weighted PR-AUC: 0.601

5. **Primary Model: CatBoost**
   - Single holdout validation: 0.689 weighted PR-AUC
   - 5-fold cross-validation: 0.697 ± 0.003 weighted PR-AUC
   - Out-of-fold predictions: 0.697 weighted PR-AUC
   - Feature importance analysis
   - Threshold optimization: 0.3513 (F1 = 0.621)
   - Hyperparameter tuning validation

6. **Business Segmentation**
   - Model-aligned K-means clustering (K=5)
   - Weighted segment profiling
   - Targeting recommendations

All results, figures, and tables referenced in the report are generated directly from this notebook.

---

## Key Findings

### Model Performance
- **CatBoost weighted PR-AUC:** 0.697 (mean across 5 folds)
- **16% improvement** over logistic regression baseline (0.601)
- **Low variance:** σ = 0.0032 across folds, indicating robust generalization

### Feature Importance
Top predictive features (by CatBoost gain):
1. Family members under 18
2. Age
3. Num persons worked for employer
4. Dividends from stocks
5. Education

### Business Segmentation
Five distinct population segments identified:
- **Segment 3 (High-Value):** 7.1% population, 57.4% income rate, 81.5% high education
- **Segment 0 (Public Sector):** 6.9% population, 8.5% income rate, 75.8% full-year work
- **Segment 1 (Working Class):** 30.9% population, 4.7% income rate
- **Segments 2 & 4 (Retired/Dependents):** 55% population, <1.2% income rate

**Recommendation:** Prioritize Segment 3 for high-value targeting (9× higher conversion than population baseline).

---

## Deployment Considerations

1. **Threshold Calibration:** The optimal F1-maximizing threshold is 0.3513. Adjust based on business-specific precision/recall tradeoffs.

2. **Fairness Auditing:** The model encodes historical disparities from 1994-1995 data (e.g., male income rate 10.3% vs. female 2.7%). Deployment requires fairness monitoring and potential debiasing.

3. **Temporal Drift:** The 30-year gap between training data (1994-1995) and present requires periodic retraining to account for labor market evolution.

4. **Survey Weights:** All production predictions should be aggregated using population weights for representative inference.

---

## Troubleshooting

**Issue:** Notebook fails to run due to missing packages
- **Solution:** Ensure all dependencies are installed: `pip install -r requirements.txt`

**Issue:** Data file not found
- **Solution:** Verify `census-bureau.data` is in the project root directory

**Issue:** Out of memory during model training
- **Solution:** Reduce CatBoost `iterations` parameter or use a machine with more RAM (8GB+ recommended)

---

## Questions

For questions or clarifications, please refer to the project report (`report.pdf`) and the detailed analysis in `Census_data.ipynb`.
