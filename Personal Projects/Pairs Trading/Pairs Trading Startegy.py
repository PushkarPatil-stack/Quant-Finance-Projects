import yfinance as yf
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from numpy.ma.extras import average
from statsmodels.tsa.stattools import coint
import time
import openpyxl
from scipy.spatial.distance import pdist, squareform


# Pull the ticker list from excel file. the excel file contains all the companies that are part of the SPX index
def get_sp500_tickers_from_excel(file_path= r"C:\Users\Admin\Downloads\Bloomberg Excel_2025_Sept_04.xlsx"):
    sheet_name = 'SPX'
    df = pd.read_excel( file_path,sheet_name=sheet_name)
    tickers = df['Ticker'].dropna().astype(str).str.replace('/', '-',regex=False).str.strip().tolist()
    return tickers

tickers = get_sp500_tickers_from_excel()
print(f" Loaded {len(tickers)} tickers from Excel.")


data = yf.download(tickers, start='2023-10-01', end='2025-09-01', interval='1d')['Close']
valid_data = data.dropna(how='all') #drops missing data
print(f"Valid rows: {len(valid_data)}")

# Identify failed tickers (those not in the returned columns)
fetched_tickers = set(data.columns.levels[1]) if isinstance(data.columns, pd.MultiIndex) else set(data.columns)
missing_tickers = [t for t in tickers if t not in fetched_tickers]

print(f"Data downloaded for {len(fetched_tickers)} tickers.")
print(f"Missing tickers ({len(missing_tickers)}): {missing_tickers}")

if data.empty:
    print(" No data was downloaded. Check the tickers and date range.")
else:
    print("Data downloaded successfully.")
    print(f" Columns: {data.columns}")
    print(data.head())

# If MultiIndex (like 'Close', 'Open', etc.), focus on 'Close'
if isinstance(data.columns, pd.MultiIndex):
    entry_counts = data['Close'].count()
else:
    entry_counts = data.count()

print("Entries per ticker:")
print(entry_counts.sort_values())

# Drop tickers with fewer than X data points
min_required = 400
valid_tickers = entry_counts[entry_counts >= min_required].index.tolist()
data = data[valid_tickers]
print(f"The number of valid tickers is: {len(data.columns)}")

# Check for tickers that have any missing data points
tickers_with_missing_data = data.columns[data.isna().any()].tolist()

if tickers_with_missing_data:
    missing_data_subset = data[tickers_with_missing_data]
    missing_rows = missing_data_subset[missing_data_subset.isna().any(axis=1)]

    print(f"Rows with missing data for {len(tickers_with_missing_data)} tickers:")
    print(missing_rows)  # Prints all such rows

    # To just see which tickers are missing data at each timestamp
    missing_map = missing_rows.isna()
    print("\nMissing data map (True indicates missing):")
    print(missing_map)
else:
    print("No missing data found for any ticker.")

#Starting with the distance approach

# Step 1: Compute cumulative returns (normalize relative to starting value)
cumulative_returns = data / data.iloc[0]

# Step 2: Min-max normalize each ticker's time series
min_vals = cumulative_returns.min()
max_vals = cumulative_returns.max()
range_vals = max_vals - min_vals

## Step 3. Normalize the returns
normalize_returns = (cumulative_returns - min_vals) / (max_vals - min_vals)

# Step 3a: Drop tickers (columns) where all values are NaN or Inf after normalization
clean_returns = normalize_returns.copy()

# Drop tickers with all NaNs or all Infs
clean_returns = clean_returns.loc[:, clean_returns.notna().any()]
clean_returns = clean_returns.loc[:, np.isfinite(clean_returns).any()]

# Drop tickers (columns) where all values are zero
clean_returns = clean_returns.loc[:, ~(clean_returns == 0).all()]

# Drop timestamps (rows) where any of the remaining tickers has missing data
clean_returns = clean_returns.dropna(axis=0)
clean_returns = clean_returns[np.isfinite(clean_returns).all(axis=1)]

# Final shape
num_tickers = clean_returns.shape[1]
num_rows = clean_returns.shape[0]
num_pairs = num_tickers * (num_tickers - 1) // 2

print(f"Shape after cleaning: {clean_returns.shape} (rows x tickers)")
print(f"Total possible pairs: {num_pairs}")


# ================== SAFETY & DIAGNOSTICS ==================
print("\n--- Distance computation diagnostics ---")
print("clean_returns shape:", clean_returns.shape)
print("Number of tickers:", clean_returns.shape[1])
print("Tickers:", list(clean_returns.columns))
# =========================================================

# Guard: need at least 2 tickers to compute pairwise distances
if clean_returns.shape[1] < 2:
    raise ValueError(
        f"Distance computation aborted: "
        f"need at least 2 tickers, found {clean_returns.shape[1]}"
    )


# Step 4: Compute Euclidean distances between normalized time series
distances = pdist(clean_returns.T, metric='sqeuclidean')  # transpose to get time series as rows

# Step 5: Convert to square matrix
distance_matrix = pd.DataFrame(squareform(distances),
                               index=clean_returns.columns,
                               columns=clean_returns.columns)
median = np.median(distances)
print(f"The median is: {np.median(distances):.4f}")
min_SSD = distances.min()
max_SSD = distances.max()
print(f"Minimum value of SSD is: {min_SSD}")
print(f"Maximum value of SSD is: {max_SSD}")

plt.hist(distances, bins=1000)
plt.title("Histogram of SSD Distances")
plt.xlabel("SSD Distance")
plt.ylabel("Frequency")
plt.show()

# Step 5: Extract all pairs with SSD distance < 1.0
threshold = 35
close_pairs = []

tickers = distance_matrix.columns

for i in range(len(tickers)):
    for j in range(i + 1, len(tickers)):
        dist = distance_matrix.iloc[i, j]
        if dist < threshold:
            close_pairs.append((tickers[i], tickers[j], dist))

# Sort the pairs by distance (ascending)
close_pairs = sorted(close_pairs, key=lambda x: x[2])

# Report only the count (no printing all pairs)
print(f"\nNumber of pairs with SSD distance < {threshold}: {len(close_pairs)}")

# preview
#print("\nclosest pairs (SSD < threshold):")
#for t1, t2, d in close_pairs:
 #   print(f"{t1} - {t2}: Distance = {d:.4f}")


# Starting with Cointegration Approach

cointegrated_pairs = []

for t1, t2, dist in close_pairs:
    series1 = clean_returns[t1]
    series2 = clean_returns[t2]

    score, p_value, _ = coint(series1, series2)

    if p_value < 0.05:
        cointegrated_pairs.append((t1, t2, dist, p_value))

print(f"\nNumber of cointegrated pairs is: {len(cointegrated_pairs)}")

# Now regressing the cointegrated pairs
from statsmodels.tsa.stattools import adfuller

def run_regression_and_adf(y, x):
    x = sm.add_constant(x)
    model = sm.OLS(y, x).fit()
    residuals = model.resid
    r_squared = model.rsquared
    adf_pvalue = adfuller(residuals)[1]
    return model, residuals, r_squared, adf_pvalue

regression_results = []

for t1, t2, dist, pval in cointegrated_pairs:
    s1 = clean_returns[t1]
    s2 = clean_returns[t2]

    # Regress s1 on s2
    model_1, residuals_1, r2_1, adf_1 = run_regression_and_adf(s1, s2)

    # Regress s2 on s1
    model_2, residuals_2, r2_2, adf_2 = run_regression_and_adf(s2, s1)

    # Decide based on R^2 and ADF p-value
    if (r2_1 > r2_2 and adf_1 < adf_2) or (adf_1 < 0.05 and adf_2 >= 0.05):
        dependent, independent = t1, t2
        model = model_1
        residuals = residuals_1
        direction = f"{t1} ~ {t2}"
        r2 = r2_1
        adf_p = adf_1
    else:
        dependent, independent = t2, t1
        model = model_2
        residuals = residuals_2
        direction = f"{t2} ~ {t1}"
        r2 = r2_2
        adf_p = adf_2

    regression_results.append({
        "pair": (t1, t2),
        "direction": direction,
        "beta": model.params.iloc[1],
        "intercept": model.params.iloc[0],
        "R_squared": r2,
        "ADF_pvalue": adf_p,
        "spread": residuals,
        "distance": dist,
        "cointegration_p": pval
    })

print(f"\nCompleted regression and ADF tests for {len(regression_results)} pairs.")

# Identify the zero crossings


# Identify the zero crossings
def count_zero_crossings(series):
    spread_sign = np.sign(series.values)  # convert to NumPy array
    zero_crossings = np.sum(spread_sign[:-1] != spread_sign[1:])
    return zero_crossings

# Lists to hold zero crossing counts and means
zero_crossing_counts = []
spread_means = []

for i, result in enumerate(regression_results):
    spread = result["spread"].dropna()
    mean_spread = float(spread.mean())
    zero_crossings = count_zero_crossings(spread)

    regression_results[i]["mean_spread"] = mean_spread
    regression_results[i]["zero_crossings"] = zero_crossings

    spread_means.append(mean_spread)
    zero_crossing_counts.append(zero_crossings)

# Plot histogram of zero crossings
plt.figure(figsize=(10,6))
plt.hist(zero_crossing_counts, bins=30, color='skyblue', edgecolor='black')
plt.title('Histogram of Zero Crossings in Spread for Cointegrated Pairs')
plt.xlabel('Number of Zero Crossings')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

threshold_zero_crossings = 20
updated_pairs = []

for result in regression_results:
    if result['zero_crossings'] > threshold_zero_crossings:
        updated_pairs.append(result['pair'])

print(f"\nNumber of updated pairs based on zero crossings threshold: {len(updated_pairs)}")

spread_stats = []
for result in regression_results:
    s = result['spread'].dropna()
    m = float(np.mean(s))
    v = float(np.var(s, ddof=0))
    spread_stats.append({
        "pair": result["pair"],
        "direction": result["direction"],
        "mean_spread": m,
        "variance_spread": v,
        "spread": s,
        "adf_pvalue": result["ADF_pvalue"]
    })

# Step 1: Select low-drift pairs
spread_stats_sorted_by_drift = sorted(spread_stats, key=lambda x: abs(x['mean_spread']))
top_drift_pairs = spread_stats_sorted_by_drift[:max(1, int(0.3 * len(spread_stats_sorted_by_drift)))]

# Step 2: From those, pick high variance
top_pairs_for_ou = sorted(top_drift_pairs, key=lambda x: x['variance_spread'], reverse=True)[:50]

print(f"\nSelected {len(top_pairs_for_ou)} pairs with low drift and high variance for OU modeling.")

# === OU fit on levels ===
def fit_ou_on_levels(spread, dt=1.0):
    s = spread.dropna()
    if len(s) < 50:  # too short to fit
        return {"valid": False, "reason": "too few points"}

    y = s.iloc[1:].values
    X = sm.add_constant(s.iloc[:-1].values)  # [const, S_{t-1}]
    ar1 = sm.OLS(y, X).fit()

    alpha = float(ar1.params[0])
    phi   = float(ar1.params[1])

    if not (0 < phi < 1):
        return {"valid": False, "phi": phi, "alpha": alpha, "r_squared": ar1.rsquared}

    k = -np.log(phi) / dt                 # mean-reversion speed
    mu = alpha / (1.0 - phi)              # long-run mean
    half_life = np.log(2.0) / k

    # approximate OU diffusion from AR(1) residuals
    eps_var = float(ar1.mse_resid)
    sigma = np.sqrt(max(1e-12, eps_var) * 2.0 * k / (1.0 - phi**2))

    return {
        "valid": True,
        "alpha": alpha,
        "phi": phi,
        "k": k,
        "mu": mu,
        "sigma": sigma,
        "half_life": half_life,
        "r_squared": ar1.rsquared,
        "p_phi": float(ar1.pvalues[1]),
        "n": len(s)
    }

# === Run OU fit on selected pairs ===
ou_results = []
for item in top_pairs_for_ou:
    res = fit_ou_on_levels(item['spread'], dt=1.0)  # dt=1 = hourly bars
    res.update({
        "pair": item["pair"],
        "direction": item["direction"],
        "adf_pvalue": item["adf_pvalue"]
    })
    ou_results.append(res)

# === Funnel analysis of OU filtering ===
total = len(ou_results)
print(f"\nTotal OU candidate pairs: {total}")

# Count how many pass each stage
stage1 = [r for r in ou_results if r.get("valid", False)]
print(f"After validity check: {len(stage1)} / {total}")

stage2 = [r for r in stage1 if r["adf_pvalue"] < 0.05]
print(f"After ADF p-value < 0.05: {len(stage2)} / {len(stage1)}")

stage3 = [r for r in stage2 if 0.5 < r["phi"] < 0.995]
print(f"After 0.5 < phi < 0.995: {len(stage3)} / {len(stage2)}")

half_lives = [r["half_life"] for r in stage3]
plt.hist(half_lives, bins=30, edgecolor="black")
plt.title("Distribution of Half-lives (hours)")
plt.xlabel("Half-life (hours)")
plt.ylabel("Count")
plt.show()

print("Half-life stats:")
print(f"Min: {np.min(half_lives):.2f}, Max: {np.max(half_lives):.2f}, Median: {np.median(half_lives):.2f}")

# Define buckets for half-life stratification
buckets = [
    (10, 20),
    (20, 30),
    (30, 40)
]

# Apply stratification
bucketed_pairs = {}
for low, high in buckets:
    bucketed_pairs[(low, high)] = [r for r in stage3 if low <= r["half_life"] <= high]

# Print results for each bucket
print("\nHalf-life stratification results:")
for (low, high), pairs in bucketed_pairs.items():
    print(f"  {low} <= half-life <= {high}: {len(pairs)} / {len(stage3)}")



# Apply R^2 > 0.70 filter inside each half-life bucket
stage5_bucketed = {}
for (low, high), pairs in bucketed_pairs.items():
    stage5_bucketed[(low, high)] = [r for r in pairs if r["r_squared"] > 0.70]

print("\nAfter R^2 > 0.70 filtering (per bucket):")
for (low, high), pairs in stage5_bucketed.items():
    print(f"  {low} <= half-life <= {high}: {len(pairs)} / {len(bucketed_pairs[(low, high)])}")

filtered_pairs = [r for pairs in stage5_bucketed.values() for r in pairs]

print(f"\nSelected {len(filtered_pairs)} OU pairs after filtering (merged across all buckets).")

# Print OU pairs per bucket
for (low, high), pairs in stage5_bucketed.items():
    print(f"\nOU pairs in bucket {low} <= half-life <= {high}:")
    for r in pairs:
        print(f"  {r['pair']} | half-life={r['half_life']:.2f}, R²={r['r_squared']:.3f}")

# Trading Strategy using the filtered pairs

# === Trading Simulation on OU Filtered Pairs ===

def backtest_pair(data, pair, beta=None, plot=False):
    t1, t2 = pair
    df = pd.DataFrame(index=data.index)
    df[t1] = data[t1]
    df[t2] = data[t2]

    # Drop missing
    df = df.dropna()

    # Spread (if beta is known from regression use it, else assume 1)
    if beta is None:
        beta = 1.0
    df["Spread"] = df[t1] - beta * df[t2]

    # Z-score
    df["Z-Score"] = (df["Spread"] - df["Spread"].mean()) / df["Spread"].std()

    # Trading signals
    upper_threshold = 2
    lower_threshold = -2
    df["Position"] = 0
    df["Position"] = np.where(df["Z-Score"] > upper_threshold, -1, df["Position"])  # short spread
    df["Position"] = np.where(df["Z-Score"] < lower_threshold, 1, df["Position"])   # long spread
    df["Position"] = np.where((df["Z-Score"].between(-1, 1)), 0, df["Position"])    # exit

    # Returns
    df[f"{t1}_Return"] = df[t1].pct_change()
    df[f"{t2}_Return"] = df[t2].pct_change()
    df["Strategy_Return"] = df["Position"].shift(1) * (df[f"{t1}_Return"] - beta * df[f"{t2}_Return"])
    df["Cumulative_Return"] = (1 + df["Strategy_Return"].fillna(0)).cumprod()

    # Metrics
    sharpe_ratio = df["Strategy_Return"].mean() / df["Strategy_Return"].std() * np.sqrt(252) if df["Strategy_Return"].std() != 0 else 0
    cumulative_max = df["Cumulative_Return"].cummax()
    drawdown = (cumulative_max - df["Cumulative_Return"]) / cumulative_max
    max_drawdown = drawdown.max()

    if plot:
        plt.figure(figsize=(12, 6))
        plt.plot(df.index, df["Cumulative_Return"], label="Cumulative Return")
        plt.title(f"Strategy Cumulative Returns: {t1} & {t2}")
        plt.legend()
        plt.show()

    return {
        "pair": pair,
        "sharpe_ratio": sharpe_ratio,
        "max_drawdown": max_drawdown,
        "final_return": df["Cumulative_Return"].iloc[-1],
        "n_trades": df["Position"].diff().abs().sum() // 2,  # counts entry/exits
    }


# Run backtest for all filtered pairs
results = []
for r in filtered_pairs:
    t1, t2 = r["pair"]
    beta = r.get("beta", 1.0)  # use regression beta if available
    res = backtest_pair(data, (t1, t2), beta=beta, plot=False)
    results.append(res)

# Put results into a DataFrame
results_df = pd.DataFrame(results)
results_df = results_df.sort_values("sharpe_ratio", ascending=False)

print("\n=== Backtest Results for OU Filtered Pairs ===")
print(results_df)

import tabulate
print(tabulate.tabulate(results_df, headers="keys", tablefmt="pretty"))

# Apply performance thresholds to filter pairs
filtered_backtest_results = results_df[
    (results_df["sharpe_ratio"] > 0.5) &
    (results_df["max_drawdown"] < 0.1) &
    (results_df["final_return"] > 1.10)
    ]

print(f"\nFiltered pairs after performance thresholds: {len(filtered_backtest_results)}")
print(filtered_backtest_results)


print(tabulate.tabulate(filtered_backtest_results, headers="keys", tablefmt="pretty"))

# ============================================================================
# SIMPLIFIED PORTFOLIO ANALYSIS
# ============================================================================

import seaborn as sns
from scipy.optimize import minimize

print("\n" + "="*80)
print("PORTFOLIO ANALYSIS")
print("="*80)

# ============================================================================
# 1. Calculate Pair Returns and Build Returns Matrix
# ============================================================================

pair_returns_dict = {}
pair_info = []

for idx, row in filtered_backtest_results.iterrows():
    t1, t2 = row['pair']

    # Find beta
    beta = 1.0
    for reg_result in regression_results:
        if reg_result["pair"] == (t1, t2):
            beta = reg_result["beta"]
            break

    # Reconstruct trading strategy
    df = pd.DataFrame({'t1': data[t1], 't2': data[t2]}).dropna()
    df["Spread"] = df['t1'] - beta * df['t2']
    df["Z"] = (df["Spread"] - df["Spread"].mean()) / df["Spread"].std()

    # Signals: Long at z<-2, Short at z>2, Exit at |z|<1
    df["Pos"] = 0
    df["Pos"] = np.where(df["Z"] > 2, -1, df["Pos"])
    df["Pos"] = np.where(df["Z"] < -2, 1, df["Pos"])
    df["Pos"] = np.where(df["Z"].between(-1, 1), 0, df["Pos"])

    # Returns
    df["Ret"] = df["Pos"].shift(1) * (df['t1'].pct_change() - beta * df['t2'].pct_change())

    pair_name = f"{t1}-{t2}"
    pair_returns_dict[pair_name] = df["Ret"].fillna(0)

    # Stats
    daily_ret = df["Ret"].mean()
    daily_std = df["Ret"].std()
    pair_info.append({
        'pair': pair_name,
        'beta': beta,
        'daily_ret': daily_ret,
        'daily_std': daily_std,
        'ann_ret': daily_ret * 252,
        'ann_std': daily_std * np.sqrt(252),
        'sharpe': (daily_ret / daily_std * np.sqrt(252)) if daily_std > 0 else 0
    })

returns_df = pd.DataFrame(pair_returns_dict)
pair_info_df = pd.DataFrame(pair_info)

print(f"✓ Analyzed {len(pair_info)} pairs | Returns shape: {returns_df.shape}")

# ============================================================================
# 2. Correlation Matrix
# ============================================================================

corr_matrix = returns_df.corr()
avg_corr = corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].mean()

print(f"✓ Avg correlation: {avg_corr:.3f} | Max: {corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].max():.3f}")

plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='RdYlGn', center=0,
            square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
plt.title('Pair Returns Correlation Matrix', fontsize=14, pad=15)
plt.tight_layout()
plt.show()

# ============================================================================
# 3. Calculate Weights (4 Methods)
# ============================================================================

n = len(pair_info)

# Equal Weight
w_equal = np.ones(n) / n

# Sharpe-Weighted
sharpe_vals = pair_info_df['sharpe'].values
w_sharpe = sharpe_vals / sharpe_vals.sum()

# Inverse Volatility
vol_vals = pair_info_df['daily_std'].values
w_invvol = (1/vol_vals) / (1/vol_vals).sum()

# Risk Parity
cov = returns_df.cov() * 252
def risk_parity_obj(w):
    port_vol = np.sqrt(w @ cov @ w)
    risk_contrib = w * (cov @ w) / port_vol
    target = port_vol / len(w)
    return np.sum((risk_contrib - target) ** 2)

result = minimize(risk_parity_obj, np.ones(n)/n, method='SLSQP',
                  bounds=tuple((0,1) for _ in range(n)),
                  constraints={'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
w_riskpar = result.x if result.success else np.ones(n)/n

print(f"Calculated 4 weighting methods")

# ============================================================================
# 4. Portfolio Metrics with Treynor Ratio
# ============================================================================

def calc_metrics(weights, returns, name):
    """Calculate portfolio metrics including Treynor ratio"""
    port_ret = returns @ weights

    # Basic stats
    daily_ret = port_ret.mean()
    daily_std = port_ret.std()
    ann_ret = daily_ret * 252
    ann_std = daily_std * np.sqrt(252)
    sharpe = ann_ret / ann_std if ann_std > 0 else 0

    # Cumulative & drawdown
    cum_ret = (1 + port_ret).cumprod()
    dd = (cum_ret.cummax() - cum_ret) / cum_ret.cummax()
    max_dd = dd.max()

    # Treynor ratio (using SPY as market proxy)
    # For pairs trading (market-neutral), beta should be ~0, making Treynor less meaningful
    # We'll calculate it anyway using the average beta of the pairs
    avg_beta = np.mean([info['beta'] for info in pair_info])
    treynor = ann_ret / avg_beta if avg_beta != 0 else 0

    # Sortino
    downside_ret = port_ret[port_ret < 0]
    downside_std = downside_ret.std() * np.sqrt(252)
    sortino = ann_ret / downside_std if downside_std > 0 else 0

    # Calmar
    calmar = ann_ret / max_dd if max_dd > 0 else 0

    return {
        'Method': name,
        'Ann_Ret_%': ann_ret * 100,
        'Ann_Vol_%': ann_std * 100,
        'Sharpe': sharpe,
        'Sortino': sortino,
        'Treynor': treynor,
        'Max_DD_%': max_dd * 100,
        'Calmar': calmar,
        'cum_ret': cum_ret
    }

metrics = [
    calc_metrics(w_equal, returns_df, 'Equal Weight'),
    calc_metrics(w_sharpe, returns_df, 'Sharpe-Weighted'),
    calc_metrics(w_invvol, returns_df, 'Inv Volatility'),
    calc_metrics(w_riskpar, returns_df, 'Risk Parity')
]

# Display results
summary = pd.DataFrame([{
    'Method': m['Method'],
    'Ann Return': f"{m['Ann_Ret_%']:.2f}%",
    'Ann Vol': f"{m['Ann_Vol_%']:.2f}%",
    'Sharpe': f"{m['Sharpe']:.3f}",
    'Sortino': f"{m['Sortino']:.3f}",
    'Treynor': f"{m['Treynor']:.3f}",
    'Max DD': f"{m['Max_DD_%']:.2f}%",
    'Calmar': f"{m['Calmar']:.3f}"
} for m in metrics])

print("\n" + "="*80)
print("PORTFOLIO PERFORMANCE")
print("="*80)
print(summary.to_string(index=False))

# ============================================================================
# 5. Position Sizing ($100 per pair)
# ============================================================================

positions = []
for info in pair_info:
    t1, t2 = info['pair'].split('-')
    beta = info['beta']
    price_t1 = data[t1].iloc[-1]
    price_t2 = data[t2].iloc[-1]

    notional_t1 = 100
    notional_t2 = beta * 100

    positions.append({
        'Pair': info['pair'],
        'Beta': beta,
        'Price_T1': price_t1,
        'Price_T2': price_t2,
        'Shares_T1': notional_t1 / price_t1,
        'Shares_T2': notional_t2 / price_t2,
        'Capital': notional_t1 + notional_t2
    })

pos_df = pd.DataFrame(positions)
total_capital = pos_df['Capital'].sum()

print("\n" + "="*80)
print("POSITION SIZING (Delta-Hedged, $100 per pair)")
print("="*80)
print(pos_df[['Pair', 'Beta', 'Shares_T1', 'Shares_T2', 'Capital']].to_string(index=False))
print(f"\nTotal Capital Required: ${total_capital:,.2f}")

# ============================================================================
# 6. Pair-Specific Stats
# ============================================================================

pair_stats = pd.DataFrame({
    'Pair': pair_info_df['pair'],
    'Beta': pair_info_df['beta'].round(3),
    'Ann_Ret_%': (pair_info_df['ann_ret'] * 100).round(2),
    'Ann_Vol_%': (pair_info_df['ann_std'] * 100).round(2),
    'Sharpe': pair_info_df['sharpe'].round(3)
})

print("\n" + "="*80)
print("PAIR-SPECIFIC PERFORMANCE")
print("="*80)
print(pair_stats.to_string(index=False))

# ============================================================================
# 7. Visualizations
# ============================================================================

# Cumulative returns
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

for m in metrics:
    ax1.plot(m['cum_ret'].index, m['cum_ret'].values,
             label=f"{m['Method']} (Sharpe: {m['Sharpe']:.2f})", linewidth=2)

ax1.set_title('Portfolio Cumulative Returns by Weighting Method', fontsize=14)
ax1.set_ylabel('Cumulative Return', fontsize=11)
ax1.legend(loc='best')
ax1.grid(True, alpha=0.3)

# Drawdowns
for m in metrics:
    dd = (m['cum_ret'].cummax() - m['cum_ret']) / m['cum_ret'].cummax()
    ax2.plot(dd.index, -dd.values * 100, label=m['Method'], linewidth=1.5)

ax2.set_title('Portfolio Drawdowns', fontsize=14)
ax2.set_xlabel('Date', fontsize=11)
ax2.set_ylabel('Drawdown (%)', fontsize=11)
ax2.legend(loc='best')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Weight comparison
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
weights_data = [
    ('Equal Weight', w_equal),
    ('Sharpe-Weighted', w_sharpe),
    ('Inv Volatility', w_invvol),
    ('Risk Parity', w_riskpar)
]

for idx, (name, weights) in enumerate(weights_data):
    ax = axes[idx//2, idx%2]
    sorted_idx = np.argsort(weights)[::-1]
    ax.barh(range(len(weights)), weights[sorted_idx] * 100, color='steelblue')
    ax.set_yticks(range(len(weights)))
    ax.set_yticklabels([pair_info_df['pair'].iloc[i] for i in sorted_idx], fontsize=8)
    ax.set_xlabel('Weight (%)', fontsize=10)
    ax.set_title(name, fontsize=12)
    ax.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.show()

# ============================================================================
# 8. Summary
# ============================================================================

best = max(metrics, key=lambda x: x['Sharpe'])

print("\n" + "="*80)
print("KEY TAKEAWAYS")
print("="*80)
print(f"\n Best Method: {best['Method']}")
print(f"   • Sharpe: {best['Sharpe']:.3f}")
print(f"   • Ann Return: {best['Ann_Ret_%']:.2f}%")
print(f"   • Max Drawdown: {best['Max_DD_%']:.2f}%")
print(f"\ Portfolio Stats:")
print(f"   • Number of Pairs: {len(pair_info)}")
print(f"   • Avg Correlation: {avg_corr:.3f}")
print(f"   • Total Capital: ${total_capital:,.2f}")

print("="*80 + "\n")