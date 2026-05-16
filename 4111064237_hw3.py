# -*- coding: utf-8 -*-
"""
House Price Prediction using Linear Regression
============================================================
Topic    : Multi-Variable Linear Regression with Feature Selection
Models   : Simple Linear Regression, Multiple Linear Regression
Feature  : Correlation + SelectKBest
Metrics  : R2, RMSE, MAE
Process  : CRISP-DM
"""
import io, sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

warnings.filterwarnings("ignore")

# ── Colour palette ──────────────────────────────────────────────────────────
PALETTE = {
    "bg":      "#0F1117",
    "card":    "#1A1D2E",
    "accent1": "#7C5CBF",
    "accent2": "#4ECDC4",
    "accent3": "#FF6B6B",
    "accent4": "#FFE66D",
    "text":    "#E0E0E0",
    "muted":   "#8B8FA8",
}
plt.rcParams.update({
    "figure.facecolor":  PALETTE["bg"],
    "axes.facecolor":    PALETTE["card"],
    "axes.edgecolor":    PALETTE["muted"],
    "axes.labelcolor":   PALETTE["text"],
    "xtick.color":       PALETTE["text"],
    "ytick.color":       PALETTE["text"],
    "text.color":        PALETTE["text"],
    "grid.color":        "#2A2D3E",
    "grid.linestyle":    "--",
    "grid.alpha":        0.5,
    "font.family":       "DejaVu Sans",
    "axes.titlesize":    13,
    "axes.labelsize":    11,
})

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "report_figures")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def savefig(name):
    path = os.path.join(OUTPUT_DIR, name)
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=PALETTE["bg"])
    plt.close()
    print(f"  [saved] {name}")
    return path

# ==============================================================
# PHASE 1 - Business Understanding
# ==============================================================
print("\n" + "="*60)
print("PHASE 1 | Business Understanding")
print("="*60)
print("""
Objective  : Predict house prices using property attributes.
Stakeholder: Real-estate agencies, home buyers, policy analysts.
Success KPI: R2 >= 0.70 on unseen test set.
Approach   : Build Simple Linear Regression (baseline) and
             Multiple Linear Regression (main model) with
             feature selection via Correlation + SelectKBest.
""")

# ==============================================================
# PHASE 2 - Data Understanding
# ==============================================================
print("="*60)
print("PHASE 2 | Data Understanding")
print("="*60)

CSV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Housing.csv")
df_raw = pd.read_csv(CSV_PATH)
print(f"\nShape      : {df_raw.shape}")
print(f"Columns    : {list(df_raw.columns)}")
print(f"\nMissing values:\n{df_raw.isnull().sum()}")
print(f"\nDescriptive statistics (price):")
print(df_raw["price"].describe())

# -- Figure 1: Price distribution --------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.patch.set_facecolor(PALETTE["bg"])
fig.suptitle("Figure 1 - House Price Distribution", fontsize=15,
             color=PALETTE["text"], y=1.02)

axes[0].hist(df_raw["price"], bins=40,
             color=PALETTE["accent1"], edgecolor=PALETTE["bg"], alpha=0.85)
axes[0].set_title("Raw Price")
axes[0].set_xlabel("Price (INR)")
axes[0].set_ylabel("Count")
axes[0].yaxis.grid(True)

axes[1].hist(np.log1p(df_raw["price"]), bins=40,
             color=PALETTE["accent2"], edgecolor=PALETTE["bg"], alpha=0.85)
axes[1].set_title("Log-Transformed Price")
axes[1].set_xlabel("log(1 + Price)")
axes[1].set_ylabel("Count")
axes[1].yaxis.grid(True)

plt.tight_layout()
fig1_path = savefig("fig1_price_distribution.png")

# -- Figure 2: Categorical feature counts ------------------------------------
cat_cols = ["mainroad","guestroom","basement","hotwaterheating",
            "airconditioning","prefarea","furnishingstatus"]
fig, axes = plt.subplots(2, 4, figsize=(18, 7))
fig.patch.set_facecolor(PALETTE["bg"])
fig.suptitle("Figure 2 - Categorical Feature Counts", fontsize=15,
             color=PALETTE["text"])
axes = axes.flatten()
colors = [PALETTE["accent1"], PALETTE["accent2"], PALETTE["accent3"],
          PALETTE["accent4"]]
for i, col in enumerate(cat_cols):
    vc = df_raw[col].value_counts()
    bars = axes[i].bar(vc.index, vc.values,
                       color=colors[i % len(colors)], alpha=0.85,
                       edgecolor=PALETTE["bg"])
    axes[i].set_title(col)
    axes[i].set_ylabel("Count")
    axes[i].yaxis.grid(True)
    for bar in bars:
        h = bar.get_height()
        axes[i].text(bar.get_x() + bar.get_width()/2., h + 2,
                     f'{int(h)}', ha='center', va='bottom',
                     color=PALETTE["text"], fontsize=9)
axes[-1].set_visible(False)
plt.tight_layout()
fig2_path = savefig("fig2_categorical_counts.png")

# ==============================================================
# PHASE 3 - Data Preparation
# ==============================================================
print("\n" + "="*60)
print("PHASE 3 | Data Preparation")
print("="*60)

df = df_raw.copy()

# Encode binary yes/no columns
binary_cols = ["mainroad","guestroom","basement","hotwaterheating",
               "airconditioning","prefarea"]
for col in binary_cols:
    df[col] = (df[col] == "yes").astype(int)

# Ordinal encode furnishingstatus
furnish_map = {"unfurnished": 0, "semi-furnished": 1, "furnished": 2}
df["furnishingstatus"] = df["furnishingstatus"].map(furnish_map)

print("\nEncoded DataFrame (first 3 rows):")
print(df.head(3).to_string())

# -- Correlation Analysis ----------------------------------------------------
corr = df.corr()
price_corr = corr["price"].drop("price").sort_values(ascending=False)
print(f"\nCorrelation with price:\n{price_corr}")

# -- Figure 3: Correlation heatmap -------------------------------------------
fig, ax = plt.subplots(figsize=(13, 10))
fig.patch.set_facecolor(PALETTE["bg"])
mask = np.triu(np.ones_like(corr, dtype=bool))
cmap = sns.diverging_palette(250, 15, s=90, l=40, n=12, center="light",
                              as_cmap=True)
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=0.8, vmin=-0.4,
            center=0, square=True, linewidths=0.5,
            linecolor=PALETTE["bg"], annot=True, fmt=".2f",
            annot_kws={"size": 8}, ax=ax,
            cbar_kws={"shrink": 0.8})
ax.set_title("Figure 3 - Pearson Correlation Heatmap", fontsize=15,
             color=PALETTE["text"], pad=12)
ax.tick_params(labelsize=9)
plt.tight_layout()
fig3_path = savefig("fig3_correlation_heatmap.png")

# -- Figure 4: Top correlations bar chart ------------------------------------
fig, ax = plt.subplots(figsize=(10, 5))
fig.patch.set_facecolor(PALETTE["bg"])
bar_colors = [PALETTE["accent1"] if v >= 0 else PALETTE["accent3"]
              for v in price_corr.values]
ax.barh(price_corr.index, price_corr.values,
        color=bar_colors, alpha=0.85, edgecolor=PALETTE["bg"])
ax.axvline(0, color=PALETTE["muted"], linewidth=1)
ax.set_title("Figure 4 - Feature Correlation with Price", fontsize=15,
             color=PALETTE["text"])
ax.set_xlabel("Pearson r")
ax.xaxis.grid(True)
plt.tight_layout()
fig4_path = savefig("fig4_corr_bar.png")

# -- Feature / Target split --------------------------------------------------
X = df.drop("price", axis=1)
y = df["price"]

# -- SelectKBest (F-regression, k=8) -----------------------------------------
selector = SelectKBest(f_regression, k=8)
selector.fit(X, y)
f_scores = pd.Series(selector.scores_, index=X.columns).sort_values(ascending=False)
selected_features = list(f_scores.nlargest(8).index)
print(f"\nSelectKBest F-scores:\n{f_scores}")
print(f"\nSelected top-8 features: {selected_features}")

# -- Figure 5: SelectKBest F-scores ------------------------------------------
fig, ax = plt.subplots(figsize=(10, 5))
fig.patch.set_facecolor(PALETTE["bg"])
palette_list = [PALETTE["accent1"] if feat in selected_features
                else PALETTE["muted"] for feat in f_scores.index]
ax.barh(f_scores.index, f_scores.values,
        color=palette_list, alpha=0.85, edgecolor=PALETTE["bg"])
ax.axvline(f_scores.iloc[7], color=PALETTE["accent4"],
           linestyle="--", linewidth=1.5, label="k=8 cut-off")
ax.legend(facecolor=PALETTE["card"], edgecolor=PALETTE["muted"])
ax.set_title("Figure 5 - SelectKBest F-Scores (k=8 highlighted)", fontsize=15,
             color=PALETTE["text"])
ax.set_xlabel("F-score")
ax.xaxis.grid(True)
plt.tight_layout()
fig5_path = savefig("fig5_selectkbest.png")

# -- Train / Test split ------------------------------------------------------
X_sel = X[selected_features]
X_train, X_test, y_train, y_test = train_test_split(
    X_sel, y, test_size=0.2, random_state=42)
print(f"\nTrain size: {len(X_train)}, Test size: {len(X_test)}")

# ==============================================================
# PHASE 4 - Modeling
# ==============================================================
print("\n" + "="*60)
print("PHASE 4 | Modeling")
print("="*60)

# -- Model A: Simple Linear Regression (best single feature) -----------------
best_single = price_corr.abs().idxmax()
print(f"\n[SLR] Best single feature: '{best_single}'  "
      f"(r = {price_corr[best_single]:.3f})")

slr = LinearRegression()
slr.fit(X_train[[best_single]], y_train)
y_pred_slr = slr.predict(X_test[[best_single]])

slr_r2   = r2_score(y_test, y_pred_slr)
slr_rmse = np.sqrt(mean_squared_error(y_test, y_pred_slr))
slr_mae  = mean_absolute_error(y_test, y_pred_slr)
print(f"[SLR] R2={slr_r2:.4f}  RMSE={slr_rmse:,.0f}  MAE={slr_mae:,.0f}")

# -- Model B: Multiple Linear Regression (all selected features) -------------
mlr = LinearRegression()
mlr.fit(X_train, y_train)
y_pred_mlr = mlr.predict(X_test)

mlr_r2   = r2_score(y_test, y_pred_mlr)
mlr_rmse = np.sqrt(mean_squared_error(y_test, y_pred_mlr))
mlr_mae  = mean_absolute_error(y_test, y_pred_mlr)
print(f"\n[MLR] R2={mlr_r2:.4f}  RMSE={mlr_rmse:,.0f}  MAE={mlr_mae:,.0f}")

# -- MLR coefficients --------------------------------------------------------
coef_df = pd.DataFrame({
    "Feature": selected_features,
    "Coefficient": mlr.coef_
}).sort_values("Coefficient", key=abs, ascending=False)
print(f"\n[MLR] Intercept: {mlr.intercept_:,.0f}")
print(coef_df.to_string(index=False))

# -- Figure 6: MLR Coefficients ----------------------------------------------
fig, ax = plt.subplots(figsize=(10, 5))
fig.patch.set_facecolor(PALETTE["bg"])
bar_cols = [PALETTE["accent1"] if v >= 0 else PALETTE["accent3"]
            for v in coef_df["Coefficient"]]
ax.barh(coef_df["Feature"], coef_df["Coefficient"],
        color=bar_cols, alpha=0.85, edgecolor=PALETTE["bg"])
ax.axvline(0, color=PALETTE["muted"], linewidth=1)
ax.set_title("Figure 6 - MLR Feature Coefficients", fontsize=15,
             color=PALETTE["text"])
ax.set_xlabel("Coefficient value")
ax.xaxis.grid(True)
plt.tight_layout()
fig6_path = savefig("fig6_mlr_coefficients.png")

# ==============================================================
# PHASE 5 - Evaluation
# ==============================================================
print("\n" + "="*60)
print("PHASE 5 | Evaluation")
print("="*60)

# -- Cross-validation (5-fold) -----------------------------------------------
cv_scores = cross_val_score(mlr, X_sel, y, cv=5, scoring="r2")
print(f"\n[MLR] 5-Fold CV R2: {cv_scores}")
print(f"[MLR] Mean CV R2  : {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")

# -- Figure 7: Actual vs Predicted -------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.patch.set_facecolor(PALETTE["bg"])
fig.suptitle("Figure 7 - Actual vs Predicted Price", fontsize=15,
             color=PALETTE["text"])

for ax, y_pred, title, color in zip(
    axes,
    [y_pred_slr, y_pred_mlr],
    ["Simple Linear Regression", "Multiple Linear Regression"],
    [PALETTE["accent2"], PALETTE["accent1"]]
):
    ax.scatter(y_test, y_pred, alpha=0.6, color=color, edgecolors="none", s=40)
    mn = min(y_test.min(), float(y_pred.min()))
    mx = max(y_test.max(), float(y_pred.max()))
    ax.plot([mn, mx], [mn, mx], color=PALETTE["accent3"],
            linewidth=2, linestyle="--", label="Perfect fit")
    r2 = r2_score(y_test, y_pred)
    ax.set_title(f"{title}\nR2 = {r2:.4f}")
    ax.set_xlabel("Actual Price")
    ax.set_ylabel("Predicted Price")
    ax.legend(facecolor=PALETTE["card"], edgecolor=PALETTE["muted"])
    ax.xaxis.grid(True); ax.yaxis.grid(True)

plt.tight_layout()
fig7_path = savefig("fig7_actual_vs_predicted.png")

# -- Figure 8: Residual plots ------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(16, 5))
fig.patch.set_facecolor(PALETTE["bg"])
fig.suptitle("Figure 8 - Residual Analysis", fontsize=15,
             color=PALETTE["text"])

for ax, y_pred, title, color in zip(
    axes,
    [y_pred_slr, y_pred_mlr],
    ["SLR Residuals", "MLR Residuals"],
    [PALETTE["accent2"], PALETTE["accent1"]]
):
    residuals = y_test - y_pred
    ax.scatter(y_pred, residuals, alpha=0.5, color=color,
               edgecolors="none", s=35)
    ax.axhline(0, color=PALETTE["accent3"], linewidth=1.5, linestyle="--")
    ax.set_title(title)
    ax.set_xlabel("Fitted values")
    ax.set_ylabel("Residuals")
    ax.xaxis.grid(True); ax.yaxis.grid(True)

plt.tight_layout()
fig8_path = savefig("fig8_residuals.png")

# -- Figure 9: Metrics comparison bar chart ----------------------------------
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.patch.set_facecolor(PALETTE["bg"])
fig.suptitle("Figure 9 - Model Metrics Comparison", fontsize=15,
             color=PALETTE["text"])

metrics_data = [
    ("R2 Score",  slr_r2,         mlr_r2),
    ("RMSE (M)",  slr_rmse/1e6,   mlr_rmse/1e6),
    ("MAE (M)",   slr_mae/1e6,    mlr_mae/1e6),
]
for ax, (metric, v_slr, v_mlr) in zip(axes, metrics_data):
    bars = ax.bar(["SLR", "MLR"], [v_slr, v_mlr],
                  color=[PALETTE["accent2"], PALETTE["accent1"]],
                  edgecolor=PALETTE["bg"], alpha=0.85, width=0.5)
    for bar in bars:
        h = bar.get_height()
        label = f'{h:.4f}' if "R2" in metric else f'{h:.3f}'
        ax.text(bar.get_x() + bar.get_width()/2., h * 1.01,
                label, ha='center', va='bottom',
                color=PALETTE["text"], fontsize=10)
    ax.set_title(metric)
    ax.set_ylabel("Value")
    ax.yaxis.grid(True)

plt.tight_layout()
fig9_path = savefig("fig9_metrics_comparison.png")

# -- Figure 10: 5-Fold CV Scores ---------------------------------------------
fig, ax = plt.subplots(figsize=(9, 5))
fig.patch.set_facecolor(PALETTE["bg"])
folds = [f"Fold {i+1}" for i in range(5)]
bars = ax.bar(folds, cv_scores, color=PALETTE["accent1"],
              edgecolor=PALETTE["bg"], alpha=0.85)
ax.axhline(cv_scores.mean(), color=PALETTE["accent4"],
           linewidth=2, linestyle="--", label=f"Mean = {cv_scores.mean():.4f}")
for bar, val in zip(bars, cv_scores):
    ax.text(bar.get_x() + bar.get_width()/2., val + 0.002,
            f'{val:.4f}', ha='center', va='bottom',
            color=PALETTE["text"], fontsize=10)
ax.set_title("Figure 10 - MLR 5-Fold Cross-Validation R2 Scores", fontsize=13,
             color=PALETTE["text"])
ax.set_ylabel("R2 Score")
ax.legend(facecolor=PALETTE["card"], edgecolor=PALETTE["muted"])
ax.yaxis.grid(True)
plt.tight_layout()
fig10_path = savefig("fig10_cv_scores.png")

# ==============================================================
# PHASE 6 - Deployment
# ==============================================================
print("\n" + "="*60)
print("PHASE 6 | Deployment")
print("="*60)

print(f"""
+--------------------------------------------------+
|          MODEL EVALUATION SUMMARY               |
+-------------------+--------------+---------------+
| Metric            |     SLR      |     MLR       |
+-------------------+--------------+---------------+
| R2 Score          |  {slr_r2:>8.4f}  |  {mlr_r2:>9.4f}  |
| RMSE              | {slr_rmse:>10,.0f}  | {mlr_rmse:>11,.0f}  |
| MAE               | {slr_mae:>10,.0f}  | {mlr_mae:>11,.0f}  |
| CV Mean R2        |     -        |  {cv_scores.mean():>9.4f}  |
+-------------------+--------------+---------------+
| Best single feat  : {best_single}
| Selected features : {len(selected_features)} (SelectKBest k=8)
+--------------------------------------------------+
""")

# Expose results for report generation
RESULTS = {
    "slr_r2": slr_r2, "slr_rmse": slr_rmse, "slr_mae": slr_mae,
    "mlr_r2": mlr_r2, "mlr_rmse": mlr_rmse, "mlr_mae": mlr_mae,
    "cv_mean": cv_scores.mean(), "cv_std": cv_scores.std(),
    "best_single": best_single, "selected_features": selected_features,
    "price_corr": price_corr, "f_scores": f_scores,
    "coef_df": coef_df, "intercept": mlr.intercept_,
}

print("\n[OK] All figures saved to:", OUTPUT_DIR)
print("[OK] Analysis complete.\n")
