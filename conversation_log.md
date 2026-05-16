# Conversation Log — AIoT HW3 Project

**Workspace:** `c:\作業\智慧物聯網應用與實作\HW3`  
**Course:** 智慧物聯網應用與實作 (AIoT Applications and Practice)

---

## Session 1 — 2026-05-14

**Conversation ID:** dfc9554d-983a-4891-bbe0-079e6ce95a27  
**Time:** 2026-05-14T01:07 (UTC+8)

### Objective

Build a complete Linear Regression report for HW3, covering the following requirements:

- **Topic:** House Price Prediction using Multiple Linear Regression with Feature Selection
- **Models:** Simple Linear Regression (SLR) and Multiple Linear Regression (MLR)
- **Feature Selection:** Pearson Correlation + SelectKBest (F-regression, k=8)
- **Evaluation Metrics:** R² Score, RMSE, MAE
- **Methodology:** CRISP-DM framework (6 phases)
- **Language:** Python with scikit-learn, pandas, numpy, matplotlib, seaborn

### Actions Taken

1. **Inspected dataset** (`Housing.csv`):
   - 545 samples, 13 columns (1 target: `price`, 12 features)
   - No missing values
   - Mix of continuous numeric and binary yes/no categorical features
   - `furnishingstatus` is a 3-class ordinal variable

2. **Wrote `housing_regression.py`** — a full CRISP-DM pipeline script:
   - Phase 1: Business Understanding (problem statement, objectives, KPIs)
   - Phase 2: Data Understanding (EDA, price distribution, categorical counts)
   - Phase 3: Data Preparation (binary encoding, ordinal encoding, correlation analysis, SelectKBest)
   - Phase 4: Modeling (SLR on `area`, MLR on top-8 features)
   - Phase 5: Evaluation (R², RMSE, MAE, actual vs predicted, residual plots)
   - Phase 6: Deployment (model summary, prediction function)
   - Generated 10 publication-quality dark-themed figures in `report_figures/`

3. **Executed `housing_regression.py`** successfully (resolved Windows CP950 encoding issue using `python -X utf8` flag and `io.TextIOWrapper`).

4. **Key Results:**

   | Metric | SLR (area only) | MLR (8 features) |
   |--------|-----------------|------------------|
   | R² Score | 0.2729 | **0.6194** |
   | RMSE | ₹1,917,104 | **₹1,387,029** |
   | MAE | ₹1,474,748 | **₹1,027,656** |

   - Best single predictor: `area` (Pearson r = 0.536)
   - Top-8 selected features (by both Correlation and SelectKBest): `area`, `bathrooms`, `airconditioning`, `stories`, `parking`, `bedrooms`, `prefarea`, `furnishingstatus`
   - Highest coefficient in MLR: `bathrooms` (~₹1.15M per bathroom)

5. **Wrote `report.md`** — full structured Markdown report containing:
   - All 6 CRISP-DM phases with explanations
   - Feature description table
   - Descriptive statistics
   - Correlation table and SelectKBest F-score table
   - Embedded figure references (10 figures)
   - MLR coefficient interpretation table
   - Metrics comparison table
   - Deployment strategy (REST API + web form)
   - Limitations and future work
   - Final conclusion

### Files Created

| File | Purpose |
|------|---------|
| `housing_regression.py` | Python analysis script (CRISP-DM pipeline) |
| `report.md` | Full Markdown report |
| `conversation_log.md` | This log file |
| `report_figures/fig1_price_distribution.png` | Price histogram (raw + log-transformed) |
| `report_figures/fig2_categorical_counts.png` | Categorical feature bar charts |
| `report_figures/fig3_correlation_heatmap.png` | Pearson correlation heatmap |
| `report_figures/fig4_corr_bar.png` | Feature-price correlation bar chart |
| `report_figures/fig5_selectkbest.png` | SelectKBest F-scores (k=8 highlighted) |
| `report_figures/fig6_mlr_coefficients.png` | MLR feature coefficients |
| `report_figures/fig7_actual_vs_predicted.png` | Actual vs Predicted scatter (SLR + MLR) |
| `report_figures/fig8_residuals.png` | Residual analysis plots |
| `report_figures/fig9_metrics_comparison.png` | R²/RMSE/MAE comparison bar charts |
| `report_figures/fig10_cv_scores.png` | 5-fold cross-validation R² scores |

### Technical Notes

- Windows console encoding is CP950; resolved by running `python -X utf8` and wrapping stdout with `io.TextIOWrapper`
- matplotlib backend set to `"Agg"` (non-interactive) to avoid display issues in headless execution
- All figures use a dark-theme color palette for visual clarity
- Train/test split: 80/20 with `random_state=42`

---

## Session 2 — 2026-05-16

**Conversation ID:** dfc9554d-983a-4891-bbe0-079e6ce95a27 (continued)  
**Time:** 2026-05-16T11:32 – 11:44 (UTC+8)

### Actions Taken

1. **Added Research Context to `report.md`** (lines 8–14):
   - Inserted a bilingual `### 研究脈絡 Research Context` section between the dataset metadata and the first `---` separator.
   - Chinese paragraph: explains dataset source (Kaggle, 545 records, 13 features), methods (Pearson + SelectKBest, SLR vs MLR), and CRISP-DM framework motivation.
   - User subsequently refined the Chinese text (removed "印度" modifier, removed the English blockquote version).

2. **Created `README.md`**:
   - Project overview and bilingual header
   - Full directory/file structure tree
   - CRISP-DM phase summary table
   - Feature selection results table (Pearson r + F-score for all 12 features)
   - Model results comparison table (SLR vs MLR: R², RMSE, MAE)
   - Embedded sample figure previews (2×2 grid)
   - Setup & usage instructions (pip install + `python -X utf8` run command)
   - Key findings and business insights
   - References section (dataset, CRISP-DM, scikit-learn docs)

3. **Pushed project to GitHub**:
   - Repository: https://github.com/hsu24/AIoT_HW3.git
   - Initialized local git repository (`git init`)
   - Staged all 16 files (`git add .`)
   - Created initial commit: `"Initial commit: CRISP-DM house price prediction (SLR + MLR)"`
   - Set remote origin, renamed branch to `main`, pushed successfully (`git push -u origin main`)

### Files Pushed to GitHub

| File | Notes |
|------|-------|
| `Housing.csv` | Raw dataset |
| `housing_regression.py` | Full CRISP-DM analysis pipeline |
| `report.md` | Structured report with research context |
| `README.md` | Project overview for GitHub |
| `conversation_log.md` | This log |
| `report_figures/*.png` | 10 visualization figures |

---

*Log maintained in English per project requirements.*
