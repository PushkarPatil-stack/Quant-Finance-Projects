"""
Fixed Income Portfolio Risk Modeling and Stress Testing (Python)
Complete script with:
- PCA factors on yield changes
- Credit spread factor
- Nelson-Siegel monthly fit + NS-driven simulations
- Monte Carlo VaR/CVaR (NS-driven) and parametric VaR
- Component / marginal VaR (parametric + MC tail-based)
"""

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm
from scipy.linalg import sqrtm
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import warnings
import statsmodels.api as sm
from datetime import datetime

warnings.filterwarnings("ignore")
np.random.seed(123)

# Optional: try to use tqdm for progress; fallback to no-progress
try:
    from tqdm import tqdm
    use_tqdm = True
except Exception:
    use_tqdm = False
    def tqdm(iterable, total=None):
        return iterable

# ---------------------------
# User settings
# ---------------------------
TICKERS = ["BIL", "SHY", "IEF", "TLT", "LQD", "HYG"]
YIELD_TICKERS = {"3M": "^IRX", "5Y": "^FVX", "10Y": "^TNX", "30Y": "^TYX"}
START = "2005-01-01"
END = datetime.today().strftime("%Y-%m-%d")
WEIGHTS = np.array([0.05, 0.20, 0.25, 0.15, 0.20, 0.15])  # must sum to 1
N_SIM = 20000
ALPHA = 0.01  # 99% VaR
STRESS_PERIODS = {
    "2008 Crisis": ("2008-09-01", "2009-03-01"),
    "COVID-19 Crash": ("2020-02-20", "2020-04-30"),
}

# ---------------------------
# Utilities
# ---------------------------
def download_price_data(tickers, start, end):
    df = yf.download(tickers, start=start, end=end, progress=False, auto_adjust=False)
    # Handle common yf outputs
    if isinstance(df.columns, pd.MultiIndex):
        if "Adj Close" in df.columns:
            df = df["Adj Close"]
        else:
            df = df["Close"]
    elif "Adj Close" in df.columns:
        df = df["Adj Close"]
    elif "Close" in df.columns:
        df = df[["Close"]]
    else:
        raise ValueError("Expected price columns not found from yfinance.")
    if isinstance(df, pd.Series):
        df = df.to_frame()
    return df.dropna(how='all')

def compute_log_returns(price_df):
    return np.log(price_df).diff().dropna()

# Nelson–Siegel param. function
def nelson_siegel(maturity, beta0, beta1, beta2, lambd):
    # maturity (scalar or array), lambd > 0
    x = maturity / lambd
    # guard for zero division
    with np.errstate(divide='ignore', invalid='ignore'):
        term = (1 - np.exp(-x)) / x
        term = np.where(np.isnan(term), 1.0, term)  # when maturity==0 treat term -> 1
        return beta0 + beta1 * term + beta2 * (term - np.exp(-x))

# Helper: fit one NS curve
def fit_ns_curve(yields_row, tenors, p0=None, bounds=None, maxfev=2000):
    try:
        if p0 is None:
            p0 = [yields_row.mean(), -0.02, 0.02, 1.0]
        if bounds is None:
            bounds = ([-1.0, -2.0, -2.0, 0.01], [5.0, 2.0, 2.0, 50.0])
        params, _ = curve_fit(nelson_siegel, tenors, yields_row, p0=p0, bounds=bounds, maxfev=maxfev)
        return params  # beta0, beta1, beta2, lambda
    except Exception:
        return np.array([np.nan, np.nan, np.nan, np.nan])

def ns_yields_from_params(params, tenors):
    return nelson_siegel(tenors, *params)

# ---------------------------
# 1. Data download
# ---------------------------
print("Downloading ETF price data...")
prices = download_price_data(TICKERS, START, END)
print("Downloaded price series shape:", prices.shape)

print("Downloading yield proxies...")
yields_raw = download_price_data(list(YIELD_TICKERS.values()), START, END)
# rename columns to maturities for clarity (order corresponds to YIELD_TICKERS.values())
yields_raw.columns = list(YIELD_TICKERS.keys())[:len(yields_raw.columns)]
print("Yield proxies shape:", yields_raw.shape)

# Standardize yield scale (yfinance often gives percent numbers)
yields = yields_raw.copy()
# use DataFrame.applymap if pandas version supports, otherwise use apply+lambda
try:
    yields = yields.applymap(lambda x: x/100.0 if np.abs(x) > 0.02 else x)
except Exception:
    yields = yields.apply(lambda col: col.map(lambda x: x/100.0 if np.abs(x) > 0.02 else x))

yields = yields.dropna()

# ---------------------------
# 2. Returns & spread
# ---------------------------
etf_rets = compute_log_returns(prices).dropna()
# credit spread proxy (return difference)
if ("LQD" in etf_rets.columns) and ("IEF" in etf_rets.columns):
    spread = (etf_rets["LQD"] - etf_rets["IEF"]).rename("Spread")
else:
    # fallback: zeros
    spread = pd.Series(0.0, index=etf_rets.index, name="Spread")

yield_changes = yields.diff().dropna()

# Align datasets
data = etf_rets.join(yield_changes, how="inner").join(spread, how="inner").dropna()
etf_rets = data[TICKERS]
yield_changes = data[list(yields.columns)]
spread = data["Spread"]

print("Aligned shapes — etf_rets:", etf_rets.shape, "yield_changes:", yield_changes.shape, "spread:", spread.shape)

# ---------------------------
# 3. PCA on yield changes
# ---------------------------
from numpy.linalg import eigh

def compute_pca_factors(X: pd.DataFrame, n_factors=3):
    Xc = X - X.mean(axis=0)
    cov = np.cov(Xc.T)
    vals, vecs = eigh(cov)
    idx = np.argsort(vals)[::-1]
    vals = vals[idx]
    vecs = vecs[:, idx]
    factors = Xc.values.dot(vecs[:, :n_factors])
    factors_df = pd.DataFrame(factors, index=X.index, columns=[f"F{i+1}" for i in range(n_factors)])
    loadings_df = pd.DataFrame(vecs[:, :n_factors], index=X.columns, columns=[f"F{i+1}" for i in range(n_factors)])
    return factors_df, loadings_df, vals

factors_ts, factor_loadings, eigvals = compute_pca_factors(yield_changes, n_factors=3)
print("PCA explained variance (desc):", (eigvals / eigvals.sum())[:3])

# ---------------------------
# 4. Regression of ETF returns on PCA factors + Spread
# ---------------------------
print("Regressing ETF returns on PCA factors + Spread...")

# Build X matrix (factors + spread)
X = pd.concat([factors_ts, spread], axis=1)
# Name columns properly (ensure F1..Fn then Spread)
n_f = factors_ts.shape[1]
X.columns = [f"F{i+1}" for i in range(n_f)] + ["Spread"]
# Add constant
X = sm.add_constant(X)

betas = []
tickers = etf_rets.columns
residuals = pd.DataFrame(index=X.index, columns=tickers)

for ticker in tickers:
    y = etf_rets[ticker].dropna()
    y, X_aligned = y.align(X, join='inner', axis=0)
    model = sm.OLS(y, X_aligned).fit()
    betas.append(model.params.values)
    residuals.loc[y.index, ticker] = model.resid

betas = np.array(betas)  # shape (n_etf, n_regressors)
betas_df = pd.DataFrame(betas, index=tickers, columns=X.columns)

print("\nEstimated Factor Loadings (betas):")
print(betas_df.round(5))

# Residual covariance and factor covariance
resid_cov = residuals.cov()
factor_cov = X.drop(columns="const").cov()


ax = betas_df.drop(columns="const").plot(kind="bar", figsize=(10,6))
ax.set_title("ETF Factor Loadings (Yield PCA + Credit Spread)")
ax.set_ylabel("Beta")
plt.tight_layout()
plt.show()


# ---------------------------
# 5. Nelson–Siegel monthly fits (fast)
# ---------------------------
print("Fitting Nelson–Siegel parameters (monthly sampling, warm-start)...")

# Build tenors array in years, matching yields.columns order
# map "3M" -> 0.25, "5Y" ->5, "10Y"->10, "30Y"->30
tenor_map = {"3M": 0.25, "6M":0.5, "1Y":1.0, "2Y":2.0, "5Y":5.0, "10Y":10.0, "30Y":30.0}
# --- Robust tenor parser ---
def parse_tenor(col):
    """
    Convert column labels like '3M', '6M', '1Y', '5Y', etc. to numeric years.
    """
    col = col.upper().strip()
    if col.endswith('M'):
        return float(col.replace('M', '')) / 12.0
    elif col.endswith('Y'):
        return float(col.replace('Y', ''))
    else:
        # fallback if numeric string
        return float(col)

tenors = np.array([parse_tenor(c) for c in yields.columns])


# monthly resample (average) for fast fitting
yields_m = yields.resample("M").mean().dropna()

params = []
dates = []
p0 = [yields_m.iloc[0].mean(), -0.02, 0.02, 1.0]  # initial guess
iterable = yields_m.iterrows()
if use_tqdm:
    iterable = tqdm(yields_m.iterrows(), total=len(yields_m), desc="NS fit (monthly)")

for date, row in iterable:
    yvals = row.values
    popt = fit_ns_curve(yvals, tenors, p0=p0, maxfev=3000)
    params.append(popt)
    dates.append(date)
    # warm start if successful
    if not np.isnan(popt).any():
        p0 = popt

ns_params = pd.DataFrame(params, index=dates, columns=["beta0","beta1","beta2","lambd"])
print("Fitted NS params (monthly) shape:", ns_params.shape)
# forward fill if any NANs and drop remaining
ns_params = ns_params.fillna(method="ffill").dropna()

# derive NS daily changes series by reindexing to X.index and forward-filling (so we can compute empirical daily deltas)
ns_daily = ns_params.reindex(pd.DatetimeIndex(X.index)).fillna(method="ffill").dropna()
ns_changes = ns_daily.diff().dropna()
ns_mean = ns_changes.mean().values
ns_cov = ns_changes.cov().values

# ---------------------------
# 6. Simulate NS-driven factor shocks
# ---------------------------
def simulate_ns_based_factor_shocks(n_sim, ns_mean, ns_cov, tenors, today_ns_params, pca_loadings_matrix):
    Lns = np.real(sqrtm(ns_cov))  # take only the real part
    z = np.random.normal(size=(n_sim, ns_mean.shape[0]))
    ns_deltas = z.dot(Lns.T) + ns_mean  # n_sim x n_params
    today_yields = ns_yields_from_params(today_ns_params, tenors)
    # pca_loadings_matrix: n_yield_vars x n_pca_factors
    factor_shocks = np.zeros((n_sim, pca_loadings_matrix.shape[1]))
    for i in range(n_sim):
        sim_params = today_ns_params + ns_deltas[i]
        sim_yields = ns_yields_from_params(sim_params, tenors)
        yield_change = (sim_yields - today_yields)  # length n_yield_vars
        # Project yield changes to PCA factor space: yield_change (1 x n_vars) dot loadings (n_vars x n_factors)
        factor_shocks[i, :] = yield_change.dot(pca_loadings_matrix)
    return factor_shocks

# Prepare today's NS (last available)
today_ns = ns_daily.iloc[-1].values
pca_loadings_matrix = factor_loadings.values  # n_yield_vars x n_pca_factors

# Simulate PCA factor shocks from NS dynamics
print("Simulating NS-driven PCA factor shocks...")
pca_factor_shocks_from_ns = simulate_ns_based_factor_shocks(N_SIM, ns_mean, ns_cov, tenors, today_ns, pca_loadings_matrix)

# Simulate spread shocks (empirical mean/std of spread changes)
spread_changes = spread.diff().dropna()
spread_mean = spread_changes.mean()
spread_std = spread_changes.std()
sim_spread_shocks = np.random.normal(loc=spread_mean, scale=spread_std, size=(N_SIM,))

# Compose full factor matrix in regression order [F1,F2,F3,Spread]
sim_full_factors = np.column_stack([
    pca_factor_shocks_from_ns[:, 0],
    pca_factor_shocks_from_ns[:, 1],
    pca_factor_shocks_from_ns[:, 2],
    sim_spread_shocks
])

# ---------------------------
# 7. Map simulated factors -> ETF returns and compute VaR
# ---------------------------
# Extract beta matrix for factors order
regressor_names = [f"F{i+1}" for i in range(factors_ts.shape[1])] + ["Spread"]
beta_matrix = betas_df.loc[:, regressor_names].values  # shape n_etf x n_factors
consts = betas_df["const"].values  # n_etf

# Simulate idiosyncratic noise from residual covariance
resid_L = np.real(sqrtm(resid_cov.fillna(0).values))
sim_idio = np.random.normal(size=(N_SIM, len(TICKERS))).dot(resid_L.T)

# Build simulated returns: N_SIM x n_etf
r_sim_ns = sim_full_factors.dot(beta_matrix.T) + consts + sim_idio

# Portfolio returns and NS-driven MC VaR/CVaR
port_returns_ns = r_sim_ns.dot(WEIGHTS)
var_mc_ns = -np.percentile(port_returns_ns, ALPHA * 100)
cvar_mc_ns = -port_returns_ns[port_returns_ns <= np.percentile(port_returns_ns, ALPHA * 100)].mean()

print(f"NS-driven Monte Carlo VaR (99%): {var_mc_ns:.6f}, CVaR (99%): {cvar_mc_ns:.6f}")

# ---------------------------
# 8. Parametric VaR (delta-normal) from simulated returns covariance
# ---------------------------
cov_r_mat = np.cov(r_sim_ns.T)  # n_etf x n_etf
port_var = WEIGHTS.dot(cov_r_mat).dot(WEIGHTS)
port_sigma = np.sqrt(port_var)
z = norm.ppf(ALPHA)
var_param = -(0 + z * port_sigma)  # mean assumed ~0 for short horizon
print(f"Parametric (delta-normal) VaR (approx): {var_param:.6f}")

# ---------------------------
# 9. Component / marginal VaR
# ---------------------------
# Parametric marginal contributions (approx)
cov_with_port = cov_r_mat.dot(WEIGHTS)  # cov(r_i, r_port)
# mVaR_i approx = -z * (w_i * cov_i,port) / port_sigma
mvar_param = -(z * (WEIGHTS * cov_with_port) / port_sigma)

# MC tail-based contributions
threshold = np.percentile(port_returns_ns, ALPHA*100)
tail_idx = port_returns_ns <= threshold
if isinstance(tail_idx, np.ndarray):
    tail_mean_asset_returns_in_tail = r_sim_ns[tail_idx].mean(axis=0)
else:
    tail_mean_asset_returns_in_tail = r_sim_ns[tail_idx.values].mean(axis=0)
comp_var_mc = -WEIGHTS * tail_mean_asset_returns_in_tail

# normalize to match var_mc_ns magnitude
sum_param = mvar_param.sum()
comp_param_norm = (mvar_param * (var_mc_ns / sum_param)) if (sum_param != 0) else mvar_param
sum_mc = comp_var_mc.sum()
comp_mc_norm = (comp_var_mc * (var_mc_ns / sum_mc)) if (sum_mc != 0) else comp_var_mc

risk_contrib_df = pd.DataFrame({
    "Weight": WEIGHTS,
    "Param_ComponentVaR": comp_param_norm,
    "MC_Tail_ComponentVaR": comp_mc_norm
}, index=TICKERS)

print("\nRisk contributions (parametric normalized & MC tail normalized):")
print(risk_contrib_df.round(6))

# ---------------------------
# 10. Plots for contributions
# ---------------------------
plt.figure(figsize=(8,4))
risk_contrib_df["Param_ComponentVaR"].plot(kind="bar")
plt.title("Parametric Component VaR (normalized)")
plt.ylabel("Contribution to VaR (notional=1)")
plt.tight_layout()
plt.show()

plt.figure(figsize=(8,4))
risk_contrib_df["MC_Tail_ComponentVaR"].plot(kind="bar")
plt.title("MC Tail-based Component VaR (normalized)")
plt.ylabel("Contribution to VaR (notional=1)")
plt.tight_layout()
plt.show()

# ---------------------------
# 11. Historical & Standard MC VaR (original pipeline)
# ---------------------------
# Historical portfolio returns (from data)
hist_port_returns = etf_rets.dot(WEIGHTS)
hist_var = -np.percentile(hist_port_returns, ALPHA * 100)
hist_cvar = -hist_port_returns[hist_port_returns <= np.percentile(hist_port_returns, ALPHA * 100)].mean()

# Standard Monte Carlo using factor historic cov (original PCA method)
# We'll simulate factors from their empirical mean/cov and map through betas (as baseline)
factor_mean = factors_ts.mean().values
factor_cov_emp = factors_ts.cov().values
L = sqrtm(factor_cov_emp)
sim_factors = np.random.normal(size=(N_SIM, factors_ts.shape[1])).dot(L.T) + factor_mean
beta_matrix_pca = betas_df.loc[:, [f"F{i+1}" for i in range(factors_ts.shape[1])]].values
consts_pca = betas_df["const"].values
r_sim_baseline = (sim_factors.dot(beta_matrix_pca.T)) + consts_pca + sim_idio  # reuse sim_idio noise
pnl_sim_baseline = r_sim_baseline.dot(WEIGHTS)
var_mc_baseline = -np.percentile(pnl_sim_baseline, ALPHA*100)
cvar_mc_baseline = -pnl_sim_baseline[pnl_sim_baseline <= np.percentile(pnl_sim_baseline, ALPHA*100)].mean()

print("\nSummary of VaR results:")
print(f"Historical 99% VaR: {hist_var:.6f}, Historical 99% CVaR: {hist_cvar:.6f}")
print(f"Baseline PCA MC VaR (99%): {var_mc_baseline:.6f}, CVaR: {cvar_mc_baseline:.6f}")
print(f"NS-driven MC VaR (99%): {var_mc_ns:.6f}, CVaR: {cvar_mc_ns:.6f}")

# ---------------------------
# 12. Stress testing (historical windows + simple hypothetical shocks)
# ---------------------------
stress_results = {}
for name, (sd, ed) in STRESS_PERIODS.items():
    sub = etf_rets.loc[sd:ed]
    if sub.empty:
        stress_results[name] = {"cumulative_return": np.nan, "worst_day_loss": np.nan}
        continue
    cum_ret = (np.exp(sub).prod() - 1).dot(WEIGHTS)
    worst_day_loss = -sub.dot(WEIGHTS).min()
    stress_results[name] = {"cumulative_return": cum_ret, "worst_day_loss": worst_day_loss}

print("\nHistorical stress results:")
for k,v in stress_results.items():
    print(k, v)

# Hypothetical: parallel +/-100bps and slope shocks (apply via NS -> PCA mapping)
# simple deterministic shocks in factor space (example)
hyp_shocks = {
    "Parallel +100bps": np.array([0.01, 0.0, 0.0, 0.0]),  # NS param shock placeholder (we'll map to PCA)
    "Parallel -100bps": np.array([-0.01, 0.0, 0.0, 0.0])
}
# Map these hypothetical NS param shocks to PCA factor shocks by computing yield changes from (today_ns + delta) and projecting
def ns_param_to_pca_shock(delta_ns):
    sim_params = today_ns + delta_ns
    sim_yields = ns_yields_from_params(sim_params, tenors)
    today_y = ns_yields_from_params(today_ns, tenors)
    yield_change = sim_yields - today_y
    return yield_change.dot(pca_loadings_matrix)  # returns array of PCA factor shocks

for name, delta_ns in hyp_shocks.items():
    # delta_ns is shorter (we provided only delta on beta0) — pad if needed to length 4
    if len(delta_ns) < today_ns.shape[0]:
        delta_ns_full = np.zeros_like(today_ns)
        delta_ns_full[:len(delta_ns)] = delta_ns
    else:
        delta_ns_full = delta_ns
    pca_sh = ns_param_to_pca_shock(delta_ns_full)
    # append spread shock = 0 for simplicity
    combined = np.concatenate([pca_sh, [0.0]])
    delta_r = combined.dot(beta_matrix.T)  # 1 x n_etf
    pnl = delta_r.dot(WEIGHTS)
    print(f"Hypothetical {name}: estimated pnl (notional=1) = {pnl:.6f}")

# ---------------------------
# 13. Save summary
# ---------------------------
summary = pd.DataFrame({
    "Metric": ["Historical VaR (99%)", "Baseline PCA MC VaR (99%)", "NS-driven MC VaR (99%)", "NS-driven MC CVaR (99%)"],
    "Value": [hist_var, var_mc_baseline, var_mc_ns, cvar_mc_ns]
})
summary.to_csv("fixed_income_risk_summary_with_ns.csv", index=False)
print("\nSaved summary to fixed_income_risk_summary_with_ns.csv")
