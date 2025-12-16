# ===========================================================
# HOUSE PRICE PREDICTION
# ===========================================================


import os
import time
import warnings
import itertools
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import skew
import scipy.stats as stats

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso

from openpyxl import load_workbook
from openpyxl.utils.dataframe import dataframe_to_rows

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ===========================================================
# USER PATHS
# ===========================================================
data_path = r"C:\Users\Admin\OneDrive\Desktop\NCSU_Semester 1\Machine Learning in Finance\Projects\Project 1\IA_House_Price_Original_Data.xlsx"
sheet_name = "Sheet1"

results_folder = r"C:\Users\Admin\OneDrive\Desktop\NCSU_Semester 1\Machine Learning in Finance\Projects\Project 1\Project Analysis"
os.makedirs(results_folder, exist_ok=True)

# subfolders for plots
plots_root = os.path.join(results_folder, "Plots")
scatter_folder = os.path.join(plots_root, "Scatter_Plots")
linear_plots = os.path.join(plots_root, "Linear")
ridge_plots = os.path.join(plots_root, "Ridge")
lasso_plots = os.path.join(plots_root, "Lasso")
compare_plots = os.path.join(plots_root, "Comparisons")

for p in [scatter_folder, linear_plots, ridge_plots, lasso_plots, compare_plots]:
    os.makedirs(p, exist_ok=True)

gd_folder = os.path.join(results_folder, "GD_Tuning")
os.makedirs(gd_folder, exist_ok=True)

# ===========================================================
# STEP 1: LOAD DATA
# ===========================================================
df = pd.read_excel(data_path, sheet_name=sheet_name)
print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")

# ===========================================================
# STEP 2: BASIC CLEANING & VARIABLE SELECTION
# ===========================================================
df["FinishedBsmtSF"] = df["BsmtFinSF1"].fillna(0) + df["BsmtFinSF2"].fillna(0)

column_mapping = {
    "LotArea": "Lot area (sqft)",
    "OverallQual": "Overall quality",
    "OverallCond": "Overall condition",
    "YearBuilt": "Year built",
    "YearRemodAdd": "Year remodeled",
    "FinishedBsmtSF": "Finished basement (sqft)",
    "BsmtUnfSF": "Unfinished basement (sqft)",
    "TotalBsmtSF": "Total basement (sqft)",
    "1stFlrSF": "First floor (sqft)",
    "2ndFlrSF": "Second floor (sqft)",
    "GrLivArea": "Living area (sqft)",
    "FullBath": "Full bathrooms",
    "HalfBath": "Half bathrooms",
    "BedroomAbvGr": "Bedrooms",
    "TotRmsAbvGrd": "Total rooms",
    "Fireplaces": "Fireplaces",
    "GarageCars": "Garage parking spaces",
    "GarageArea": "Garage area (sqft)",
    "WoodDeckSF": "Wood deck (sqft)",
    "OpenPorchSF": "Open porch (sqft)",
    "EnclosedPorch": "Enclosed porch (sqft)",
    "Neighborhood": "Neighborhood",
    "BsmtQual": "Basement quality"
}

cols_to_keep = [col for col in column_mapping.keys() if col in df.columns]
df_selected = df[cols_to_keep + ["SalePrice", "Id"]].copy()
df_selected.rename(columns=column_mapping, inplace=True)

# drop rows with problematic ids as you did earlier
df_selected = df_selected[~df_selected["Id"].isin([2218, 2219])]
print("Cleaned dataset shape:", df_selected.shape)

# ===========================================================
# STEP 3: HANDLE MISSING VALUES
# ===========================================================
df_selected["Basement quality"] = df_selected["Basement quality"].fillna("NoBsmt")

basement_quality_map = {"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1, "NoBsmt": 0}
df_selected["Basement_Quality_Encoded"] = df_selected["Basement quality"].map(basement_quality_map)

print("\nBasement quality counts:\n", df_selected["Basement quality"].value_counts())

# ===========================================================
# STEP 4: LOG TRANSFORM CHECK & APPLY
# ===========================================================
def log_transform_comparison(df, target="SalePrice", skew_threshold=1.0, corr_improvement=0.02):
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    results = []
    for col in numeric_cols:
        if col == target:
            continue
        vals = df[col].dropna()
        if (vals <= 0).any():
            continue
        sk = skew(vals)
        corr_orig = df[col].corr(df[target])
        corr_log = np.log1p(vals).corr(df[target])
        corr_diff = corr_log - corr_orig
        results.append({
            "Feature": col,
            "Skewness": sk,
            "Corr_Original": corr_orig,
            "Corr_Log": corr_log,
            "Corr_Improvement": corr_diff,
            "Recommend_Both": (sk > skew_threshold) and (abs(corr_diff) > corr_improvement)
        })
    res_df = pd.DataFrame(results)
    to_log = res_df[res_df["Recommend_Both"]]["Feature"].tolist()
    df_trans = df.copy()
    for v in to_log:
        df_trans[v] = np.log1p(df_trans[v])
        print(f"Applied log transform to: {v}")
    return df_trans, res_df

df_transformed, log_results = log_transform_comparison(df_selected, target="SalePrice")

# ===========================================================
# STEP 5: ONE-HOT ENCODE NEIGHBORHOOD (drop OldTown baseline)
# ===========================================================
neigh_dummies = pd.get_dummies(df_transformed["Neighborhood"], prefix="Neighborhood", drop_first=False)
if "Neighborhood_OldTown" in neigh_dummies.columns:
    neigh_dummies.drop("Neighborhood_OldTown", axis=1, inplace=True)
df_transformed = pd.concat([df_transformed.drop(columns=["Neighborhood"]), neigh_dummies], axis=1)
print("Neighborhood dummies added:", neigh_dummies.shape[1])

# ===========================================================
# STEP 6: STANDARDIZE & PREPARE X,y
# ===========================================================
X = df_transformed.drop(columns=["SalePrice", "Id", "Basement quality"])
y = df_transformed["SalePrice"].values.reshape(-1, 1)

scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y).ravel()

print("X standardized:", X_scaled.shape, "| y standardized:", y_scaled.shape)

# ===========================================================
# STEP 7: SPLIT DATA: Train 1800, Val 600, Test rest
# ===========================================================
X_train, X_temp, y_train, y_temp = train_test_split(X_scaled, y_scaled, train_size=1800, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, train_size=600, random_state=42)
print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

# ===========================================================
# Save scatter plots & correlation heatmap (PNGs)
# ===========================================================
sns.set(style="whitegrid")
for col in X.columns:
    plt.figure(figsize=(5,4))
    plt.scatter(df_transformed[col], df_transformed["SalePrice"], alpha=0.6)
    plt.xlabel(col)
    plt.ylabel("SalePrice")
    plt.title(f"Scatter: {col} vs SalePrice")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(scatter_folder, f"Scatter_{col}.png"))
    plt.close()

# correlation heatmap (numeric only)
numeric_df = df_transformed.select_dtypes(include=[np.number]).copy()
corr_matrix = numeric_df.corr()
plt.figure(figsize=(14,10))
sns.heatmap(corr_matrix, annot=False, cmap="coolwarm")
plt.title("Correlation Heatmap (All Features)")
plt.tight_layout()
plt.savefig(os.path.join(plots_root, "Correlation_Heatmap.png"))
plt.close()
corr_matrix.to_excel(os.path.join(results_folder, "Correlation_Matrix.xlsx"))

print("Saved scatter plots and correlation heatmap.")

# ===========================================================
# PARAMETERS: Lambda
# ===========================================================

ridge_lambdas = [0.10, 0.30, 0.60]
lasso_lambdas = [0.02, 0.06, 0.10]

# ===========================================================
# UTILS: bias, predict, metrics
# ===========================================================
def add_bias(X):
    return np.c_[np.ones((X.shape[0], 1)), X]

def predict_theta(theta, X):
    X_b = add_bias(X)
    return X_b.dot(theta).ravel()

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

# ===========================================================
# GRADIENT DESCENT: Linear & Ridge implementations
# ===========================================================
def gd_linear(X, y, lr, n_iter, theta_init=None):
    m, n = X.shape
    X_b = add_bias(X)
    theta = np.zeros((n+1, 1)) if theta_init is None else theta_init.copy()
    cost_hist = []
    for _ in range(n_iter):
        y_pred = X_b.dot(theta).ravel()
        grad = (1 / m) * X_b.T.dot((y_pred - y).reshape(-1,1))
        theta -= lr * grad
        cost = (1 / (2 * m)) * np.sum((y_pred - y) ** 2)
        cost_hist.append(cost)
    return theta, cost_hist

def gd_ridge(X, y, lr, n_iter, alpha=0.1, theta_init=None):
    m, n = X.shape
    X_b = add_bias(X)
    theta = np.zeros((n+1, 1)) if theta_init is None else theta_init.copy()
    cost_hist = []
    for _ in range(n_iter):
        y_pred = X_b.dot(theta).ravel()
        grad = (1 / m) * X_b.T.dot((y_pred - y).reshape(-1,1))
        # regularization: do not penalize intercept
        reg = np.vstack(([[0.0]], (alpha / m) * theta[1:]))
        grad += reg
        theta -= lr * grad
        cost = (1 / (2 * m)) * np.sum((y_pred - y) ** 2) + (alpha / (2 * m)) * np.sum(theta[1:].ravel()**2)
        cost_hist.append(cost)
    return theta, cost_hist

# ===========================================================
# ADAPTIVE LEARNING RATE AND CONVERGENCE FINDER
# ===========================================================
def adaptive_gd_search(
        X, y,
        model_type="linear",
        alpha=0.0,
        lr_start=0.001,
        lr_max=0.1,
        tol=1e-6,
        max_iter=2000,
        verbose=True
):
    def compute_cost(theta, Xb, y, alpha):
        y_pred = Xb.dot(theta).ravel()
        m = len(y)
        mse = (1/(2*m)) * np.sum((y_pred - y)**2)
        if model_type == "ridge":
            mse += (alpha / (2*m)) * np.sum(theta[1:]**2)
        return mse

    Xb = np.c_[np.ones((X.shape[0], 1)), X]
    m, n = X.shape
    best_lr = None
    best_iter = None
    best_theta = None
    best_curve = None
    best_cost = np.inf

    # Generate candidate learning rates
    candidate_lrs = np.logspace(np.log10(lr_start), np.log10(lr_max), 8)

    # Track already-tested learning rates to avoid duplicates
    tested_lrs = set()

    for lr in candidate_lrs:
        rounded_lr = round(lr, 6)
        if rounded_lr in tested_lrs:
            continue
        tested_lrs.add(rounded_lr)

        theta = np.zeros((n+1, 1))
        cost_history = []

        for i in range(max_iter):
            y_pred = Xb.dot(theta).ravel()
            grad = (1/m) * Xb.T.dot((y_pred - y).reshape(-1,1))

            if model_type == "ridge":
                reg = np.vstack(([[0.0]], (alpha/m) * theta[1:]))
                grad += reg

            theta -= lr * grad
            cost = compute_cost(theta, Xb, y, alpha)
            cost_history.append(cost)

            # Check convergence
            if i > 0 and abs(cost_history[-2] - cost_history[-1]) < tol:
                if verbose:
                    print(f"✅ Converged | {model_type} | lr={lr:.6f} | iter={i}")
                # Update only if cost improves meaningfully
                if cost < best_cost - 1e-8:
                    best_lr, best_iter, best_theta, best_cost, best_curve = lr, i, theta.copy(), cost, cost_history
                break

            # Divergence (early stop if cost explodes)
            if i > 2 and cost_history[-1] > 2 * cost_history[0]:
                if verbose:
                    print(f"⚠️ Diverging | {model_type} | lr={lr:.6f}")
                break


    if best_curve is not None:
        plt.figure(figsize=(6,4))
    plt.plot(best_curve, label=f"lr={best_lr:.5f}")
    plt.xlabel("Iteration")
    plt.ylabel("Cost (J)")
    plt.title(f"{model_type.capitalize()} GD Convergence")
    plt.legend()
    plt.tight_layout()
    # save instead of show
    out_path = os.path.join(gd_folder, f"{model_type}_GD_Convergence_alpha_{alpha}.png")
    plt.savefig(out_path)
    plt.close()


    return best_lr, best_iter, best_theta, best_cost


# ===========================================================
# NEW sweep_gd FUNCTION USING ADAPTIVE LR & CONVERGENCE
# ===========================================================
def sweep_gd(model_name, X_train, y_train, X_val, y_val, alpha_list=[0.0]):
    results = []
    for alpha in alpha_list:
        model_type = "ridge" if alpha > 0 else "linear"
        print(f"\n🔹 Running adaptive GD for {model_name} | alpha={alpha}")

        best_lr, best_iter, best_theta, best_cost = adaptive_gd_search(
            X_train, y_train,
            model_type=model_type,
            alpha=alpha,
            lr_start=0.001,
            lr_max=0.1,
            tol=1e-6,
            max_iter=2000
        )

        # Validation predictions
        Xb_val = np.c_[np.ones((X_val.shape[0], 1)), X_val]
        y_val_pred = Xb_val.dot(best_theta).ravel()
        y_train_pred = np.c_[np.ones((X_train.shape[0], 1)), X_train].dot(best_theta).ravel()

        # Record performance
        train_r2 = r2_score(y_train, y_train_pred)
        val_r2 = r2_score(y_val, y_val_pred)
        train_mse = mean_squared_error(y_train, y_train_pred)
        val_mse = mean_squared_error(y_val, y_val_pred)

        results.append({
            "Model": model_name,
            "Alpha": alpha,
            "LearningRate": best_lr,
            "Iterations": best_iter,
            "Train_R2": train_r2,
            "Val_R2": val_r2,
            "Train_MSE": train_mse,
            "Val_MSE": val_mse
        })
    return pd.DataFrame(results)

# plotting helpers (save to subfolders)
def plot_r2_iter(df, model_label, alpha, out_folder):
    plt.figure(figsize=(9,6))
    for lr in sorted(df["LearningRate"].unique()):
        sub = df[df["LearningRate"]==lr].groupby("Iterations")["Val_R2"].mean()
        if sub.empty:
            continue
        plt.plot(sub.index, sub.values, marker='o', label=f"lr={lr}")
    plt.title(f"{model_label} (alpha={alpha}) — Validation R² vs Iterations")
    plt.xlabel("Iterations"); plt.ylabel("Validation R²"); plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_folder, f"{model_label}_R2_vs_Iter_alpha_{alpha}.png"))
    plt.close()

def plot_r2_lr(df, model_label, alpha, out_folder):
    plt.figure(figsize=(9,6))
    for it in sorted(df["Iterations"].unique()):
        sub = df[df["Iterations"]==it].sort_values("LearningRate")
        if sub.empty:
            continue
        plt.plot(sub["LearningRate"], sub["Val_R2"], marker='o', label=f"iter={it}")
    plt.title(f"{model_label} (alpha={alpha}) — Validation R² vs Learning Rate")
    plt.xlabel("Learning Rate"); plt.ylabel("Validation R²"); plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_folder, f"{model_label}_R2_vs_LR_alpha_{alpha}.png"))
    plt.close()

def plot_heatmap(df, value_col, model_label, alpha, out_folder):
    pivot = df.pivot_table(index="Iterations", columns="LearningRate", values=value_col, aggfunc="mean")
    plt.figure(figsize=(10,5))
    sns.heatmap(pivot, annot=True, fmt=".4f", cmap="viridis")
    plt.title(f"{model_label} (alpha={alpha}) — {value_col} heatmap")
    plt.tight_layout()
    plt.savefig(os.path.join(out_folder, f"{model_label}_{value_col}_heatmap_alpha_{alpha}.png"))
    plt.close()

# ===========================================================
# RUN: Linear GD sweep (on transformed dataset)
# ===========================================================
print("Running Linear GD sweep (transformed base - with log where applied)...")
linear_results = sweep_gd("Linear", X_train, y_train, X_val, y_val, alpha_list=[0.0])

# plots & best pick
best_linear_row = None
alpha = 0.0
df_res = linear_results


plot_r2_iter(df_res, "Linear", alpha, linear_plots)
plot_r2_lr(df_res, "Linear", alpha, linear_plots)


r = df_res.loc[df_res["Val_R2"].idxmax()]
best_linear_row = r

print("Best Linear GD config:", best_linear_row.to_dict())

# ===========================================================
# CONVERGENCE CHECK FOR BEST LINEAR CONFIG
# ===========================================================
print("\nChecking convergence behavior for best Linear GD configuration...")

best_lr = float(best_linear_row["LearningRate"])
best_iter = int(best_linear_row["Iterations"])

# Run with many more iterations to observe cost decay
theta_long, cost_history_long = gd_linear(X_train, y_train, lr=best_lr, n_iter=1000)

# Compute relative cost improvement
improvements = np.abs(np.diff(cost_history_long))
convergence_threshold = 1e-6  # you can tweak (1e-5, 1e-7)
converged_iter = None

for i, imp in enumerate(improvements):
    if imp < convergence_threshold:
        converged_iter = i
        break

# Plot cost curve
plt.figure(figsize=(8,5))
plt.plot(range(len(cost_history_long)), cost_history_long, label="Cost")
plt.xlabel("Iteration")
plt.ylabel("Cost (J)")
plt.title(f"Gradient Descent Convergence — lr={best_lr}")
if converged_iter is not None:
    plt.axvline(converged_iter, color='r', linestyle='--', label=f'Converged at {converged_iter}')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(linear_plots, "Linear_Convergence_Check.png"))
plt.close()

if converged_iter is not None:
    print(f"✅ Converged after {converged_iter} iterations (threshold {convergence_threshold}).")
else:
    print(f"⚠️ Did not fully converge within 1000 iterations at lr={best_lr}.")


# ===========================================================
# FEATURE ENGINEERING: squared + interactions (2 methods)
# ===========================================================
print("Creating squared and interaction features (two methods)...")
numeric_df = df_transformed.select_dtypes(include=[np.number]).copy()
corr_with_target = numeric_df.corr()["SalePrice"].drop("SalePrice").abs().sort_values(ascending=False)

# Method 1: correlation-based interactions
high_corr_feats = corr_with_target[corr_with_target > 0.6].index.tolist()
interaction_pairs_method1 = []
for f in high_corr_feats:
    for g in numeric_df.columns:
        if g == "SalePrice" or g == f:
            continue
        corr_fg = numeric_df[[f,g]].corr().iloc[0,1]
        if abs(corr_fg) < 0.3:
            interaction_pairs_method1.append((f,g))
interaction_pairs_method1 = interaction_pairs_method1[:10]

# Method 2: domain-logic pairs
logic_pairs = [
    ("Overall quality", "Living area (sqft)"),
    ("Overall quality", "First floor (sqft)"),
    ("Garage parking spaces", "Garage area (sqft)"),
    ("Finished basement (sqft)", "Total basement (sqft)"),
    ("Fireplaces", "Total rooms")
]

squared_features = ["Overall quality", "Living area (sqft)", "First floor (sqft)", "Lot area (sqft)"]

def build_transformed_df(base_df, interactions, squared_feats):
    df_new = base_df.copy()
    for f in squared_feats:
        if f in df_new.columns:
            df_new[f"{f}_sq"] = df_new[f].values ** 2
    for (a,b) in interactions:
        if (a in df_new.columns) and (b in df_new.columns):
            df_new[f"{a}_x_{b}"] = df_new[a].values * df_new[b].values
    return df_new

df_m1 = build_transformed_df(df_transformed, interaction_pairs_method1, squared_features)
df_m2 = build_transformed_df(df_transformed, logic_pairs, squared_features)

# === Print summary of feature engineering ===
print("\n=== FEATURE ENGINEERING SUMMARY ===")
print(f"Squared features considered: {squared_features}")

print("\nMethod 1 (Correlation-based):")
print(f"Number of interaction terms: {len(interaction_pairs_method1)}")
for a, b in interaction_pairs_method1:
    print(f" - {a} × {b}")

print("\nMethod 2 (Logic-based):")
print(f"Number of interaction terms: {len(logic_pairs)}")
for a, b in logic_pairs:
    print(f" - {a} × {b}")


def prepare_xy_splits(df_trans):
    X_full = df_trans.drop(columns=["SalePrice", "Id", "Basement quality"])
    y_full = df_trans["SalePrice"].values.reshape(-1,1)
    scalerX = StandardScaler(); scalery = StandardScaler()
    X_full_scaled = scalerX.fit_transform(X_full)
    y_full_scaled = scalery.fit_transform(y_full).ravel()
    X_tr, X_temp, y_tr, y_temp = train_test_split(X_full_scaled, y_full_scaled, train_size=1800, random_state=42)
    X_v, X_te, y_v, y_te = train_test_split(X_temp, y_temp, train_size=600, random_state=42)
    return (X_tr, X_v, X_te, y_tr, y_v, y_te, scalerX, scalery, X_full.columns.tolist())

X_m1_tr, X_m1_v, X_m1_te, y_m1_tr, y_m1_v, y_m1_te, scaler_m1_X, scaler_m1_y, cols_m1 = prepare_xy_splits(df_m1)
X_m2_tr, X_m2_v, X_m2_te, y_m2_tr, y_m2_v, y_m2_te, scaler_m2_X, scaler_m2_y, cols_m2 = prepare_xy_splits(df_m2)

# Sweep linear GD on transformed datasets
print("Sweeping GD on transformed method1 (corr-based) ...")
m1_results = [ (0.0, sweep_gd("Linear_Method1", X_m1_tr, y_m1_tr, X_m1_v, y_m1_v, alpha_list=[0.0])) ]
print("Sweeping GD on transformed method2 (logic-based) ...")
m2_results = [ (0.0, sweep_gd("Linear_Method2", X_m2_tr, y_m2_tr, X_m2_v, y_m2_v, alpha_list=[0.0])) ]

best_m1 = None
for alpha, dfm in m1_results:
    plot_r2_iter(dfm, "Linear_Method1", alpha, linear_plots)
    plot_r2_lr(dfm, "Linear_Method1", alpha, linear_plots)

    r = dfm.loc[dfm["Val_R2"].idxmax()]
    if (best_m1 is None) or (r["Val_R2"] > best_m1["Val_R2"]):
        best_m1 = r

best_m2 = None
for alpha, dfm in m2_results:
    plot_r2_iter(dfm, "Linear_Method2", alpha, linear_plots)
    plot_r2_lr(dfm, "Linear_Method2", alpha, linear_plots)

    r = dfm.loc[dfm["Val_R2"].idxmax()]
    if (best_m2 is None) or (r["Val_R2"] > best_m2["Val_R2"]):
        best_m2 = r

print("Best transformed method1:", best_m1.to_dict())
print("Best transformed method2:", best_m2.to_dict())

# ===========================================================
# COMBINED PLOT: Linear Base, Method1, Method2 (R² & MSE)
# ===========================================================
print("Creating combined performance plot for Linear variants...")

# Collect summary metrics
linear_variants_summary = pd.DataFrame([
    {"Model": "Linear", "Val_R2": best_linear_row["Val_R2"], "Val_MSE": best_linear_row["Val_MSE"]},
    {"Model": "Linear_Method1", "Val_R2": best_m1["Val_R2"], "Val_MSE": best_m1["Val_MSE"]},
    {"Model": "Linear_Method2", "Val_R2": best_m2["Val_R2"], "Val_MSE": best_m2["Val_MSE"]}
])

# Create combined plot — R² & MSE on twin y-axis
fig, ax1 = plt.subplots(figsize=(8,6))

color_r2 = "tab:blue"
color_mse = "tab:orange"

ax1.set_xlabel("Model Variant")
ax1.set_ylabel("Validation R²", color=color_r2)
ax1.plot(linear_variants_summary["Model"], linear_variants_summary["Val_R2"],
         marker='o', color=color_r2, label="Validation R²")
ax1.tick_params(axis='y', labelcolor=color_r2)

ax2 = ax1.twinx()  # second y-axis for MSE
ax2.set_ylabel("Validation MSE", color=color_mse)
ax2.plot(linear_variants_summary["Model"], linear_variants_summary["Val_MSE"],
         marker='s', color=color_mse, label="Validation MSE")
ax2.tick_params(axis='y', labelcolor=color_mse)

fig.suptitle("Linear Variants — Validation R² and MSE")
fig.tight_layout()
plt.savefig(os.path.join(linear_plots, "Linear_Variants_R2_MSE.png"))
plt.close()

print("✅ Combined Linear variants plot saved.")


# choose base: original transformed dataset vs method1 vs method2
best_base_choice = "Linear"
best_base_record = best_linear_row
if best_m1["Val_R2"] > best_base_record["Val_R2"]:
    best_base_choice = "Method1"; best_base_record = best_m1
if best_m2["Val_R2"] > best_base_record["Val_R2"]:
    best_base_choice = "Method2"; best_base_record = best_m2

print("Selected base for Ridge/Lasso:", best_base_choice, best_base_record.to_dict())

if best_base_choice == "Method1":
    X_base_train, X_base_val, X_base_test = X_m1_tr, X_m1_v, X_m1_te
    y_base_train, y_base_val, y_base_test = y_m1_tr, y_m1_v, y_m1_te
    scaler_X_base, scaler_y_base = scaler_m1_X, scaler_m1_y
    base_features = cols_m1
elif best_base_choice == "Method2":
    X_base_train, X_base_val, X_base_test = X_m2_tr, X_m2_v, X_m2_te
    y_base_train, y_base_val, y_base_test = y_m2_tr, y_m2_v, y_m2_te
    scaler_X_base, scaler_y_base = scaler_m2_X, scaler_m2_y
    base_features = cols_m2
else:
    X_base_train, X_base_val, X_base_test = X_train, X_val, X_test
    y_base_train, y_base_val, y_base_test = y_train, y_val, y_test
    scaler_X_base, scaler_y_base = scaler_X, scaler_y
    base_features = list(X.columns)

# ===========================================================
# RIDGE: GD sweep for each lambda on chosen base
# ===========================================================
print("Running Ridge GD sweeps on chosen base ...")

ridge_results_all = [
    (alpha, sweep_gd("Ridge", X_base_train, y_base_train, X_base_val, y_base_val, alpha_list=[alpha]))
    for alpha in ridge_lambdas
]

# Collect best R² and MSE for each alpha
ridge_summary_records = []
for alpha, df_r in ridge_results_all:
    row = df_r.loc[df_r["Val_R2"].idxmax()]
    ridge_summary_records.append({
        "Alpha": alpha,
        "Val_R2": row["Val_R2"],
        "Val_MSE": row["Val_MSE"]
    })

ridge_summary_df = pd.DataFrame(ridge_summary_records)

# ✅ Combined chart: R² + MSE vs Alpha
plt.figure(figsize=(8,6))
plt.plot(ridge_summary_df["Alpha"], ridge_summary_df["Val_R2"], marker='o', color='blue', label="Validation R²")
plt.plot(ridge_summary_df["Alpha"], ridge_summary_df["Val_MSE"], marker='s', color='orange', label="Validation MSE")
plt.xlabel("Alpha (λ)")
plt.ylabel("Metric Value")
plt.title("Ridge Regression — Validation R² & MSE vs Alpha")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(ridge_plots, "Ridge_R2_MSE_vs_Alpha.png"))
plt.close()

# Find best ridge config (by Val_R2)
best_ridge_cfg = None
for alpha, df_r in ridge_results_all:
    row = df_r.loc[df_r["Val_R2"].idxmax()]
    row_dict = row.to_dict()
    row_dict["Alpha"] = alpha
    if (best_ridge_cfg is None) or (row["Val_R2"] > best_ridge_cfg["Val_R2"]):
        best_ridge_cfg = row_dict

print("Best Ridge GD config:", best_ridge_cfg)



# ===========================================================
# CONVERGENCE CHECK FOR BEST RIDGE GD
# ===========================================================
print("\nChecking convergence behavior for best Ridge GD configuration...")

ridge_lr = float(best_ridge_cfg["LearningRate"])
ridge_alpha_best = float(best_ridge_cfg["Alpha"])
ridge_iter = int(best_ridge_cfg["Iterations"])

theta_ridge_long, cost_ridge_long = gd_ridge(
    X_base_train, y_base_train,
    lr=ridge_lr,
    n_iter=1000,
    alpha=ridge_alpha_best
)

improvements_ridge = np.abs(np.diff(cost_ridge_long))
convergence_threshold = 1e-6
ridge_converged_iter = None
for i, imp in enumerate(improvements_ridge):
    if imp < convergence_threshold:
        ridge_converged_iter = i
        break

# Plot cost vs iteration
plt.figure(figsize=(8,5))
plt.plot(range(len(cost_ridge_long)), cost_ridge_long, label="Cost (J)")
if ridge_converged_iter is not None:
    plt.axvline(ridge_converged_iter, color='r', linestyle='--', label=f"Converged @ {ridge_converged_iter}")
plt.xlabel("Iteration")
plt.ylabel("Cost")
plt.title(f"Ridge GD Convergence — lr={ridge_lr}, alpha={ridge_alpha_best}")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(ridge_plots, "Ridge_Convergence_Check.png"))
plt.close()

if ridge_converged_iter:
    print(f"✅ Ridge GD converged after {ridge_converged_iter} iterations (ΔJ < {convergence_threshold})")
else:
    print(f"⚠️ Ridge GD did not converge within 1000 iterations.")

# ===========================================================
# LASSO: sklearn coordinate descent (tune alphas) on chosen base
# ===========================================================
print("Running Lasso (sklearn coordinate descent) tuning on chosen base ...")
lasso_tuning_records = []

for alpha in lasso_lambdas:
    model = Lasso(alpha=alpha, max_iter=20000)
    model.fit(X_base_train, y_base_train)

    # Predict on train and val
    yt = model.predict(X_base_train)
    yv = model.predict(X_base_val)

    # Capture how many iterations (coordinate descent steps)
    iters_used = getattr(model, "n_iter_", None)

    # Record key performance metrics
    lasso_tuning_records.append({
        "Alpha": alpha,
        "Iterations": iters_used if iters_used is not None else "NA",
        "Train_R2": r2_score(y_base_train, yt),
        "Val_R2": r2_score(y_base_val, yv),
        "Train_MSE": mean_squared_error(y_base_train, yt),
        "Val_MSE": mean_squared_error(y_base_val, yv),
        "NonZeroCount": int(np.sum(np.abs(model.coef_) > 1e-8))
    })

# Convert to DataFrame and save
df_lasso_tune = pd.DataFrame(lasso_tuning_records)
df_lasso_tune.to_csv(os.path.join(gd_folder, "Lasso_Sklearn_Tuning.csv"), index=False)

# Select best row based on validation R²
best_lasso_row = df_lasso_tune.loc[df_lasso_tune["Val_R2"].idxmax()]
print("Best Lasso alpha selected:", best_lasso_row.to_dict())


# ===========================================================
# CONVERGENCE CHECK FOR LASSO (sklearn)
# ===========================================================
print("\nChecking convergence behavior for best Lasso alpha...")

lasso_alpha_best = float(best_lasso_row["Alpha"])
lasso_model_check = Lasso(alpha=lasso_alpha_best, max_iter=10000, warm_start=True)
lasso_model_check.fit(X_base_train, y_base_train)

print(f"✅ Lasso converged in {lasso_model_check.n_iter_} iterations (coordinate descent steps).")
if lasso_model_check.n_iter_ >= 10000:
    print("⚠️ Warning: Lasso may not have fully converged — consider increasing max_iter.")


# Combined plot for Lasso — R² & MSE vs Alpha
plt.figure(figsize=(8,6))
plt.plot(df_lasso_tune["Alpha"], df_lasso_tune["Val_R2"], marker='o', label="Validation R²")
plt.plot(df_lasso_tune["Alpha"], df_lasso_tune["Val_MSE"], marker='s', color='orange', label="Validation MSE")
plt.xlabel("Alpha (λ)")
plt.ylabel("Metric Value")
plt.title("Lasso — Validation R² & MSE vs Alpha")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(lasso_plots, "Lasso_R2_MSE_vs_Alpha.png"))
plt.close()


# ===========================================================
# CONTROL: Linear (no log transform) - reproduce without the log transform for Lot Area
# ===========================================================
print("Running control: Linear (no log transform) ...")
df_control = df_selected.copy()  # original cleaned without log
neigh_dummies_ctrl = pd.get_dummies(df_control["Neighborhood"], prefix="Neighborhood", drop_first=False)
if "Neighborhood_OldTown" in neigh_dummies_ctrl.columns:
    neigh_dummies_ctrl.drop("Neighborhood_OldTown", axis=1, inplace=True)
df_control = pd.concat([df_control.drop(columns=["Neighborhood"]), neigh_dummies_ctrl], axis=1)
df_control["Basement quality"] = df_control["Basement quality"].fillna("NoBsmt")
df_control["Basement_Quality_Encoded"] = df_control["Basement quality"].map(basement_quality_map)

X_ctrl = df_control.drop(columns=["SalePrice", "Id", "Basement quality"])
y_ctrl = df_control["SalePrice"].values.reshape(-1,1)
scX_ctrl = StandardScaler(); scy_ctrl = StandardScaler()
X_ctrl_s = scX_ctrl.fit_transform(X_ctrl)
y_ctrl_s = scy_ctrl.fit_transform(y_ctrl).ravel()
Xc_tr, Xc_temp, yc_tr, yc_temp = train_test_split(X_ctrl_s, y_ctrl_s, train_size=1800, random_state=42)
Xc_v, Xc_te, yc_v, yc_te = train_test_split(Xc_temp, yc_temp, train_size=600, random_state=42)

ctrl_results = [(0.0, sweep_gd("Linear_NoLog", Xc_tr, yc_tr, Xc_v, yc_v, alpha_list=[0.0]))]
for alpha, dfc in ctrl_results:
    plot_r2_iter(dfc, "Linear_NoLog", alpha, linear_plots)
    plot_r2_lr(dfc, "Linear_NoLog", alpha, linear_plots)
    plot_heatmap(dfc, "Val_R2", "Linear_NoLog", alpha, linear_plots)

best_ctrl = None
for alpha, dfc in ctrl_results:
    r = dfc.loc[dfc["Val_R2"].idxmax()]
    if (best_ctrl is None) or (r["Val_R2"] > best_ctrl["Val_R2"]):
        best_ctrl = r
print("Best control linear (no log):", best_ctrl.to_dict())

# optional: ridge and lasso on control base for comparison (already implemented earlier if desired)
ctrl_ridge_results = [
    (alpha, sweep_gd("Ridge_NoLog", Xc_tr, yc_tr, Xc_v, yc_v, alpha_list=[alpha]))
    for alpha in ridge_lambdas
]

for alpha, dfr in ctrl_ridge_results:
    plot_r2_iter(dfr, "Ridge_NoLog", alpha, ridge_plots)
    plot_r2_lr(dfr, "Ridge_NoLog", alpha, ridge_plots)
    plot_heatmap(dfr, "Val_R2", "Ridge_NoLog", alpha, ridge_plots)

# lasso on control base
ctrl_lasso_records = []
for alpha in lasso_lambdas:
    mdl = Lasso(alpha=alpha, max_iter=20000)
    mdl.fit(Xc_tr, yc_tr)
    ctrl_lasso_records.append({
        "Alpha": alpha,
        "Train_R2": r2_score(yc_tr, mdl.predict(Xc_tr)),
        "Val_R2": r2_score(yc_v, mdl.predict(Xc_v)),
        "Train_MSE": mean_squared_error(yc_tr, mdl.predict(Xc_tr)),
        "Val_MSE": mean_squared_error(yc_v, mdl.predict(Xc_v)),
        "NonZeroCount": int(np.sum(np.abs(mdl.coef_) > 1e-8))
    })
df_ctrl_lasso = pd.DataFrame(ctrl_lasso_records)
df_ctrl_lasso.to_csv(os.path.join(gd_folder, "Lasso_Control_Tuning.csv"), index=False)

# ===========================================================
# FINAL REFIT: Refit best models with sklearn on chosen final base
# ===========================================================
print("Refitting final models on chosen base for evaluation...")

# define final train/val/test and scalers, feature names
if best_base_choice == "Method1":
    X_final_train, X_final_val, X_final_test = X_m1_tr, X_m1_v, X_m1_te
    y_final_train, y_final_val, y_final_test = y_m1_tr, y_m1_v, y_m1_te
    scaler_X_final, scaler_y_final = scaler_m1_X, scaler_m1_y
    final_features = cols_m1
elif best_base_choice == "Method2":
    X_final_train, X_final_val, X_final_test = X_m2_tr, X_m2_v, X_m2_te
    y_final_train, y_final_val, y_final_test = y_m2_tr, y_m2_v, y_m2_te
    scaler_X_final, scaler_y_final = scaler_m2_X, scaler_m2_y
    final_features = cols_m2
else:
    X_final_train, X_final_val, X_final_test = X_train, X_val, X_test
    y_final_train, y_final_val, y_final_test = y_train, y_val, y_test
    scaler_X_final, scaler_y_final = scaler_X, scaler_y
    final_features = list(X.columns)

# Linear (OLS) refit
sk_lin = LinearRegression()
sk_lin.fit(X_final_train, y_final_train)
lin_val_pred = sk_lin.predict(X_final_val); lin_test_pred = sk_lin.predict(X_final_test)
lin_val_r2 = r2_score(y_final_val, lin_val_pred); lin_test_r2 = r2_score(y_final_test, lin_test_pred)
lin_val_mse = mean_squared_error(y_final_val, lin_val_pred); lin_test_mse = mean_squared_error(y_final_test, lin_test_pred)
print(f"Linear refit => Val R2: {lin_val_r2:.4f}, Test R2: {lin_test_r2:.4f}")

# Ridge refit (use best_ridge_cfg['Alpha'])
ridge_alpha = float(best_ridge_cfg["Alpha"]) if best_ridge_cfg and ("Alpha" in best_ridge_cfg) else ridge_lambdas[0]
sk_ridge = Ridge(alpha=ridge_alpha)
sk_ridge.fit(X_final_train, y_final_train)
ridge_val_pred = sk_ridge.predict(X_final_val); ridge_test_pred = sk_ridge.predict(X_final_test)
ridge_val_r2 = r2_score(y_final_val, ridge_val_pred); ridge_test_r2 = r2_score(y_final_test, ridge_test_pred)
ridge_val_mse = mean_squared_error(y_final_val, ridge_val_pred); ridge_test_mse = mean_squared_error(y_final_test, ridge_test_pred)
print(f"Ridge refit (alpha={ridge_alpha}) => Val R2: {ridge_val_r2:.4f}, Test R2: {ridge_test_r2:.4f}")

# Lasso refit (use best_lasso_row alpha)
lasso_alpha_best = float(best_lasso_row["Alpha"])
sk_lasso = Lasso(alpha=lasso_alpha_best, max_iter=20000)
sk_lasso.fit(X_final_train, y_final_train)
lasso_val_pred = sk_lasso.predict(X_final_val); lasso_test_pred = sk_lasso.predict(X_final_test)
lasso_val_r2 = r2_score(y_final_val, lasso_val_pred); lasso_test_r2 = r2_score(y_final_test, lasso_test_pred)
lasso_val_mse = mean_squared_error(y_final_val, lasso_val_pred); lasso_test_mse = mean_squared_error(y_final_test, lasso_test_pred)
print(f"Lasso refit (alpha={lasso_alpha_best}) => Val R2: {lasso_val_r2:.4f}, Test R2: {lasso_test_r2:.4f}")

# ===========================================================
# CONVERT RMSE back to original dollar units (y was z-scaled)
# ===========================================================
def rmse_std_to_dollars(rmse_std, scaler_y_obj):
    # scaler_y_obj.scale_ is array-like length 1
    return float(rmse_std * scaler_y_obj.scale_[0])

lin_test_rmse_dollars = rmse_std_to_dollars(np.sqrt(lin_test_mse), scaler_y_final)
ridge_test_rmse_dollars = rmse_std_to_dollars(np.sqrt(ridge_test_mse), scaler_y_final)
lasso_test_rmse_dollars = rmse_std_to_dollars(np.sqrt(lasso_test_mse), scaler_y_final)

# ===========================================================
# REVERT COEFFICIENTS to original units and produce equations
# ===========================================================
def revert_coefficients(scaler_X_local, scaler_y_local, model, feature_names, log_features=[]):
    coefs = model.coef_.ravel()
    intercept = model.intercept_
    intercept = float(intercept) if isinstance(intercept, (np.ndarray, list)) else intercept
    coefs = np.array([float(c) for c in coefs])
    coef_original = coefs * (scaler_y_local.scale_ / scaler_X_local.scale_)
    intercept_original = float(intercept * scaler_y_local.scale_ + scaler_y_local.mean_ - np.sum(coef_original * scaler_X_local.mean_))
    final_coefs = {'Intercept': intercept_original}
    for fname, c in zip(feature_names, coef_original):
        final_coefs[fname] = float(c)
    # build equation string
    equation = f"SalePrice = {final_coefs['Intercept']:.2f}"
    for fname in feature_names:
        coef_val = final_coefs[fname]
        sign = '+' if coef_val >= 0 else '-'
        if fname in log_transformed_features:
            equation += f" {sign} {abs(coef_val):.2f}*log({fname}+1)"
        else:
            equation += f" {sign} {abs(coef_val):.2f}*{fname}"
    return final_coefs, equation

# keep log transformed features list from earlier
log_transformed_features = []
# find which features were log transformed from log_results recommendation
if "Feature" in log_results.columns:
    log_transformed_features = log_results[log_results["Recommend_Both"] == True]["Feature"].tolist()

lin_coefs_orig, lin_eq = revert_coefficients(scaler_X_final, scaler_y_final, sk_lin, final_features, log_transformed_features)
ridge_coefs_orig, ridge_eq = revert_coefficients(scaler_X_final, scaler_y_final, sk_ridge, final_features, log_transformed_features)
lasso_coefs_orig, lasso_eq = revert_coefficients(scaler_X_final, scaler_y_final, sk_lasso, final_features, log_transformed_features)

# save coefficient sheets
pd.DataFrame({
    "Feature": ["Intercept"] + final_features,
    "Linear_coef_original": [lin_coefs_orig['Intercept']] + [lin_coefs_orig[f] for f in final_features]
}).to_excel(os.path.join(results_folder, "Linear_Coefs_OriginalUnits.xlsx"), index=False)

pd.DataFrame({
    "Feature": ["Intercept"] + final_features,
    "Ridge_coef_original": [ridge_coefs_orig['Intercept']] + [ridge_coefs_orig[f] for f in final_features]
}).to_excel(os.path.join(results_folder, "Ridge_Coefs_OriginalUnits.xlsx"), index=False)

pd.DataFrame({
    "Feature": ["Intercept"] + final_features,
    "Lasso_coef_original": [lasso_coefs_orig['Intercept']] + [lasso_coefs_orig[f] for f in final_features]
}).to_excel(os.path.join(results_folder, "Lasso_Coefs_OriginalUnits.xlsx"), index=False)

with open(os.path.join(results_folder, "Linear_Equation.txt"), 'w') as f:
    f.write(lin_eq)
with open(os.path.join(results_folder, "Ridge_Equation.txt"), 'w') as f:
    f.write(ridge_eq)
with open(os.path.join(results_folder, "Lasso_Equation.txt"), 'w') as f:
    f.write(lasso_eq)

# ===========================================================
# OLS STATISTICS: t-stats & p-values for unregularized Linear
# ===========================================================
Xb_final = add_bias(X_final_train)
theta_ols = np.linalg.pinv(Xb_final.T @ Xb_final) @ (Xb_final.T @ y_final_train.reshape(-1,1))
y_pred_train_lin = sk_lin.predict(X_final_train)
resid = y_final_train - y_pred_train_lin
n_train, p_train = X_final_train.shape
sigma2 = np.sum((resid)**2) / (n_train - p_train - 1)
cov_theta = sigma2 * np.linalg.pinv(Xb_final.T @ Xb_final)
se_theta = np.sqrt(np.diag(cov_theta))
t_stats = theta_ols.ravel() / se_theta
p_values = [2 * (1 - stats.t.cdf(abs(t), df=n_train-p_train-1)) for t in t_stats]

coef_stats_df = pd.DataFrame({
    "Feature": ["Intercept"] + final_features,
    "Theta": theta_ols.ravel(),
    "Std_Error": se_theta,
    "T_stat": t_stats,
    "P_value": p_values
})

# ===========================================================
# SAVE INDIVIDUAL EXCEL WORKBOOKS
# ===========================================================

# --- Linear Workbook ---
linear_summary_file = os.path.join(results_folder, "Linear_Model_Results.xlsx")
with pd.ExcelWriter(linear_summary_file, engine='openpyxl') as writer:
    linear_results.to_excel(writer, sheet_name="Linear_GD", index=False)
    for alpha, dfm in m1_results:
        dfm.to_excel(writer, sheet_name=f"Linear_Method1", index=False)
    for alpha, dfm in m2_results:
        dfm.to_excel(writer, sheet_name=f"Linear_Method2", index=False)
    coef_stats_df.to_excel(writer, sheet_name="OLS_Stats", index=False)
    pd.DataFrame({
        "Feature": ["Intercept"] + final_features,
        "Linear": [lin_coefs_orig['Intercept']] + [lin_coefs_orig[f] for f in final_features]
    }).to_excel(writer, sheet_name="Coefficients", index=False)
print("✅ Linear workbook saved.")

# --- Ridge Workbook ---
ridge_summary_file = os.path.join(results_folder, "Ridge_Model_Results.xlsx")
with pd.ExcelWriter(ridge_summary_file, engine='openpyxl') as writer:
    for alpha, dfr in ridge_results_all:
        dfr.to_excel(writer, sheet_name=f"Ridge_a_{alpha}", index=False)
    pd.DataFrame({
        "Feature": ["Intercept"] + final_features,
        "Ridge": [ridge_coefs_orig['Intercept']] + [ridge_coefs_orig[f] for f in final_features]
    }).to_excel(writer, sheet_name="Coefficients", index=False)
print("✅ Ridge workbook saved.")

# --- Lasso Workbook ---
lasso_summary_file = os.path.join(results_folder, "Lasso_Model_Results.xlsx")
with pd.ExcelWriter(lasso_summary_file, engine='openpyxl') as writer:
    df_lasso_tune.to_excel(writer, sheet_name="Lasso_Tuning", index=False)
    pd.DataFrame({
        "Feature": ["Intercept"] + final_features,
        "Lasso": [lasso_coefs_orig['Intercept']] + [lasso_coefs_orig[f] for f in final_features]
    }).to_excel(writer, sheet_name="Coefficients", index=False)
print("✅ Lasso workbook saved.")




# ===========================================================
# SAVE PLOTS COMPARISON (R² & RMSE)
# ===========================================================
# Comparison bar charts (Val R2 and Test RMSE in dollars)
plt.figure(figsize=(8,5))
models_names = ["Linear", "Ridge", "Lasso"]
val_r2s = [lin_val_r2, ridge_val_r2, lasso_val_r2]
plt.bar(models_names, val_r2s)
plt.title("Validation R² Comparison (Final Refit Models)")
plt.ylabel("Validation R²")
plt.tight_layout()
plt.savefig(os.path.join(compare_plots, "Validation_R2_Comparison.png"))
plt.close()

plt.figure(figsize=(8,5))
test_rmse_dollars = [lin_test_rmse_dollars, ridge_test_rmse_dollars, lasso_test_rmse_dollars]
plt.bar(models_names, test_rmse_dollars)
plt.title("Test RMSE (dollars) Comparison")
plt.ylabel("Test RMSE ($)")
plt.tight_layout()
plt.savefig(os.path.join(compare_plots, "Test_RMSE_Comparison.png"))
plt.close()

print("\n=== CONVERGENCE SUMMARY ===")
print(f"Linear GD converged at iteration: {converged_iter or 'Not within 1000'}")
print(f"Ridge GD converged at iteration: {ridge_converged_iter or 'Not within 1000'}")
print(f"Lasso (coord descent) converged after: {lasso_model_check.n_iter_} iterations")


# ===========================================================
# FINAL TABLE: only best Linear, Ridge, Lasso (to Excel and print)
# ===========================================================
final_table = pd.DataFrame([
    {"Model": "Linear", "Lambda": 0.0,
     "LearningRate": float(best_linear_row["LearningRate"]),
     "Iterations": int(best_linear_row["Iterations"]),
     "Val_R2": float(best_linear_row["Val_R2"]),
     "Test_R2": float(lin_test_r2),
     "Test_MSE": float(lin_test_mse)},

    {"Model": "Ridge", "Lambda": ridge_alpha,
     "LearningRate": float(best_ridge_cfg["LearningRate"]),
     "Iterations": int(best_ridge_cfg["Iterations"]),
     "Val_R2": float(best_ridge_cfg["Val_R2"]),
     "Test_R2": float(ridge_test_r2),
     "Test_MSE": float(ridge_test_mse)},

    {"Model": "Lasso", "Lambda": lasso_alpha_best,
     "LearningRate": None,  # still None, since coordinate descent has no lr
     "Iterations": float(best_lasso_row["Iterations"]),
     "Val_R2": float(best_lasso_row["Val_R2"]),
     "Test_R2": float(lasso_test_r2),
     "Test_MSE": float(lasso_test_mse)}

])


final_table.to_excel(os.path.join(results_folder, "Final_Best_Models_Table.xlsx"), index=False)
print("\n=== FINAL BEST MODELS (selected via Validation set) ===")
print(final_table.round(4))





