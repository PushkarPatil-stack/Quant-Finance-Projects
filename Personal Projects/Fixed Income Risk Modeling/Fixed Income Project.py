"""
Fixed Income Portfolio Risk Modeling and Stress Testing (Python)

This script implements a multi-factor risk framework for a fixed income ETF portfolio.
The pipeline consists of the following stages:
    1. Download ETF price data and yield curve proxies from Yahoo Finance.
    2. Compute log returns and a credit spread factor (LQD minus IEF).
    3. Apply Principal Component Analysis (PCA) to daily yield changes to extract
       level, slope, and curvature factors.
    4. Regress each ETF's returns on the PCA factors and the credit spread using OLS,
       producing per-ETF factor loadings (betas) and residual series.
    5. Fit Nelson-Siegel (NS) yield curve parameters on a monthly basis, then derive
       empirical daily deltas of those parameters.
    6. Simulate NS parameter shocks from a multivariate normal distribution, convert
       simulated yield curves back to PCA factor space, and append simulated spread shocks.
    7. Map the simulated factor shocks through the beta matrix (plus idiosyncratic noise)
       to obtain 20,000 simulated portfolio return paths.
    8. Compute NS-driven Monte Carlo VaR and CVaR at the 99% confidence level.
    9. Compute parametric (delta-normal) VaR from the simulated return covariance matrix.
   10. Decompose VaR into per-asset component contributions using both the parametric
       marginal approach and an MC tail-conditional approach.
   11. Compute historical VaR/CVaR and a baseline PCA Monte Carlo VaR for comparison.
   12. Run historical stress tests over the 2008 Crisis and COVID-19 Crash windows, and
       apply hypothetical parallel yield curve shocks (+/-100 bps) through the NS mapping.
   13. Save a summary CSV of all VaR metrics.
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

# Suppress non-critical runtime warnings (e.g., convergence notices during NS fitting)
warnings.filterwarnings("ignore")

# Fix the random seed so that all Monte Carlo results are reproducible across runs
np.random.seed(123)

# Optional: use tqdm for a progress bar during the NS fitting loop.
# If the library is not installed, fall back to a no-op wrapper that returns the iterable unchanged.
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
# Portfolio ETF tickers: short-term T-bills (BIL), short-term Treasuries (SHY),
# intermediate Treasuries (IEF), long-term Treasuries (TLT),
# investment-grade corporates (LQD), and high-yield corporates (HYG).
TICKERS = ["BIL", "SHY", "IEF", "TLT", "LQD", "HYG"]

# Yahoo Finance ticker symbols for U.S. Treasury yield proxies at four maturities.
# These are index tickers that return annualised yields in percent.
YIELD_TICKERS = {"3M": "^IRX", "5Y": "^FVX", "10Y": "^TNX", "30Y": "^TYX"}

# Historical data range: from 2005 to today
START = "2005-01-01"
END = datetime.today().strftime("%Y-%m-%d")

# Portfolio weights assigned to each ETF (must sum to 1.0).
# Order matches TICKERS: BIL, SHY, IEF, TLT, LQD, HYG
WEIGHTS = np.array([0.05, 0.20, 0.25, 0.15, 0.20, 0.15])

# Number of Monte Carlo simulation paths used for VaR estimation
N_SIM = 20000

# Significance level for VaR/CVaR: 0.01 corresponds to 99% confidence
ALPHA = 0.01

# Historical stress test windows: each entry maps a label to a (start_date, end_date) tuple
STRESS_PERIODS = {
    "2008 Crisis": ("2008-09-01", "2009-03-01"),
    "COVID-19 Crash": ("2020-02-20", "2020-04-30"),
}

# ---------------------------
# Utilities
# ---------------------------

def download_price_data(tickers, start, end):
    """
    Download daily price data for a list of tickers using yfinance.

    Adjusted Close prices are preferred over raw Close prices to account for
    corporate actions (dividends, splits). The function handles both single-ticker
    (Series) and multi-ticker (MultiIndex DataFrame) outputs from yfinance.

    Parameters
    ----------
    tickers : list of str
        Yahoo Finance ticker symbols to download.
    start : str
        Start date in 'YYYY-MM-DD' format.
    end : str
        End date in 'YYYY-MM-DD' format.

    Returns
    -------
    pd.DataFrame
        DataFrame with dates as the index and tickers as columns,
        with rows containing any all-NaN entries removed.
    """
    df = yf.download(tickers, start=start, end=end, progress=False, auto_adjust=False)

    # yfinance returns a MultiIndex DataFrame when multiple tickers are requested.
    # Extract the "Adj Close" level if available; otherwise fall back to "Close".
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

    # Ensure the result is always a DataFrame, even for a single ticker
    if isinstance(df, pd.Series):
        df = df.to_frame()

    # Drop rows where every column is NaN (e.g., non-trading days at boundaries)
    return df.dropna(how='all')


def compute_log_returns(price_df):
    """
    Compute daily log returns from a price DataFrame.

    Log returns are defined as ln(P_t / P_{t-1}) = ln(P_t) - ln(P_{t-1}).
    The first row (which is NaN after differencing) is dropped.

    Parameters
    ----------
    price_df : pd.DataFrame
        DataFrame of price levels.

    Returns
    -------
    pd.DataFrame
        DataFrame of daily log returns, aligned to the same columns.
    """
    return np.log(price_df).diff().dropna()


def nelson_siegel(maturity, beta0, beta1, beta2, lambd):
    """
    Compute Nelson-Siegel model yields for given maturities and parameters.

    The Nelson-Siegel parametric yield curve model is:

        y(tau) = beta0
               + beta1 * [(1 - exp(-tau/lambda)) / (tau/lambda)]
               + beta2 * [(1 - exp(-tau/lambda)) / (tau/lambda) - exp(-tau/lambda)]

    where:
        beta0  : long-run (level) factor
        beta1  : short-run (slope) factor
        beta2  : medium-run (curvature) factor
        lambda : decay parameter controlling the location of the hump

    The limiting case as tau -> 0 is handled by setting the loading term to 1.0.

    Parameters
    ----------
    maturity : float or np.ndarray
        Time to maturity in years.
    beta0, beta1, beta2, lambd : float
        Nelson-Siegel parameters.

    Returns
    -------
    float or np.ndarray
        Model-implied yield(s) at the given maturity/maturities.
    """
    x = maturity / lambd  # dimensionless maturity scaled by the decay parameter

    with np.errstate(divide='ignore', invalid='ignore'):
        # Standard NS loading for beta1 and beta2
        term = (1 - np.exp(-x)) / x
        # Replace NaN values (arising from 0/0 when maturity = 0) with the limiting value of 1
        term = np.where(np.isnan(term), 1.0, term)
        return beta0 + beta1 * term + beta2 * (term - np.exp(-x))


def fit_ns_curve(yields_row, tenors, p0=None, bounds=None, maxfev=2000):
    """
    Fit Nelson-Siegel parameters to a single observed yield curve using nonlinear least squares.

    Uses scipy.optimize.curve_fit with optional warm-starting (p0) and
    bounded parameter constraints. Returns NaN parameters if the optimiser fails to converge.

    Parameters
    ----------
    yields_row : array-like
        Observed yields at the tenors given in `tenors`.
    tenors : np.ndarray
        Maturities in years corresponding to each observed yield.
    p0 : list, optional
        Initial parameter guess [beta0, beta1, beta2, lambda].
        Defaults to [mean(yields), -0.02, 0.02, 1.0].
    bounds : tuple, optional
        Lower and upper bounds for each parameter.
        Defaults to reasonable economic constraints.
    maxfev : int
        Maximum number of function evaluations allowed by the optimiser.

    Returns
    -------
    np.ndarray
        Array of fitted [beta0, beta1, beta2, lambda], or NaN array on failure.
    """
    try:
        if p0 is None:
            p0 = [yields_row.mean(), -0.02, 0.02, 1.0]
        if bounds is None:
            # Economic bounds: beta0 in [-1, 5], betas 1&2 in [-2, 2], lambda in [0.01, 50]
            bounds = ([-1.0, -2.0, -2.0, 0.01], [5.0, 2.0, 2.0, 50.0])
        params, _ = curve_fit(nelson_siegel, tenors, yields_row, p0=p0, bounds=bounds, maxfev=maxfev)
        return params  # [beta0, beta1, beta2, lambda]
    except Exception:
        # Return NaNs so that the calling loop can handle failure gracefully
        return np.array([np.nan, np.nan, np.nan, np.nan])


def ns_yields_from_params(params, tenors):
    """
    Reconstruct yield curve values from Nelson-Siegel parameters at given tenors.

    Parameters
    ----------
    params : array-like
        NS parameters [beta0, beta1, beta2, lambda].
    tenors : np.ndarray
        Maturities in years at which to evaluate the yield curve.

    Returns
    -------
    np.ndarray
        Model-implied yields at each tenor.
    """
    return nelson_siegel(tenors, *params)


# ---------------------------
# 1. Data download
# ---------------------------
print("Downloading ETF price data...")
prices = download_price_data(TICKERS, START, END)
print("Downloaded price series shape:", prices.shape)

print("Downloading yield proxies...")
yields_raw = download_price_data(list(YIELD_TICKERS.values()), START, END)

# Rename columns from Yahoo Finance ticker symbols to human-readable maturity labels
# (e.g., "^IRX" -> "3M", "^TNX" -> "10Y"), preserving the order of YIELD_TICKERS.
yields_raw.columns = list(YIELD_TICKERS.keys())[:len(yields_raw.columns)]
print("Yield proxies shape:", yields_raw.shape)

# Yahoo Finance index tickers (^IRX, ^TNX, etc.) return yields in percent (e.g., 5.25 for 5.25%).
# Convert any value greater than 0.02 in absolute terms to decimal form by dividing by 100.
# Values already in decimal form (e.g., 0.05) are left unchanged.
yields = yields_raw.copy()
try:
    # pandas >= 2.1: applymap is the standard element-wise method
    yields = yields.applymap(lambda x: x/100.0 if np.abs(x) > 0.02 else x)
except Exception:
    # Fallback for older pandas versions that do not support applymap
    yields = yields.apply(lambda col: col.map(lambda x: x/100.0 if np.abs(x) > 0.02 else x))

yields = yields.dropna()

# ---------------------------
# 2. Returns & spread
# ---------------------------
# Compute daily log returns for all ETFs and drop the initial NaN row
etf_rets = compute_log_returns(prices).dropna()

# Construct a daily credit spread factor as the return differential between
# investment-grade corporate bonds (LQD) and intermediate Treasuries (IEF).
# This proxy captures changes in credit risk compensation.
if ("LQD" in etf_rets.columns) and ("IEF" in etf_rets.columns):
    spread = (etf_rets["LQD"] - etf_rets["IEF"]).rename("Spread")
else:
    # If either ticker is unavailable, use zeros as a neutral fallback
    spread = pd.Series(0.0, index=etf_rets.index, name="Spread")

# First-difference the yield levels to obtain daily changes in basis points / percentage points
yield_changes = yields.diff().dropna()

# Align all three datasets to a common date index using an inner join,
# so that every row in the regression dataset has observations for all variables.
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
    """
    Apply Principal Component Analysis (PCA) to a DataFrame of yield changes.

    The function computes the sample covariance matrix, performs an eigen-decomposition,
    and projects the data onto the top `n_factors` eigenvectors (principal components).
    The resulting factors capture, in decreasing order, the directions of maximum
    variance in yield changes — conventionally interpreted as level, slope, and curvature.

    Parameters
    ----------
    X : pd.DataFrame
        Input data (e.g., daily yield changes) with observations as rows and
        maturities as columns.
    n_factors : int
        Number of principal components to retain.

    Returns
    -------
    factors_df : pd.DataFrame
        Time series of factor scores with shape (n_obs, n_factors).
    loadings_df : pd.DataFrame
        Eigenvector matrix (factor loadings) with shape (n_vars, n_factors).
    vals : np.ndarray
        Full array of eigenvalues in descending order, used to report explained variance.
    """
    # Demean the data so that PCA captures variance, not level
    Xc = X - X.mean(axis=0)
    cov = np.cov(Xc.T)

    # eigh assumes a symmetric matrix and returns eigenvalues in ascending order
    vals, vecs = eigh(cov)

    # Reorder to descending eigenvalue order (largest variance first)
    idx = np.argsort(vals)[::-1]
    vals = vals[idx]
    vecs = vecs[:, idx]

    # Project centred data onto the top n_factors eigenvectors
    factors = Xc.values.dot(vecs[:, :n_factors])
    factors_df = pd.DataFrame(factors, index=X.index, columns=[f"F{i+1}" for i in range(n_factors)])

    # Store the factor loading matrix for later use in the NS simulation mapping
    loadings_df = pd.DataFrame(vecs[:, :n_factors], index=X.columns, columns=[f"F{i+1}" for i in range(n_factors)])

    return factors_df, loadings_df, vals

# Extract three PCA factors from daily yield changes.
# F1 typically captures the parallel level shift, F2 the slope (short vs long end),
# and F3 the curvature (hump shape).
factors_ts, factor_loadings, eigvals = compute_pca_factors(yield_changes, n_factors=3)
print("PCA explained variance (desc):", (eigvals / eigvals.sum())[:3])

# ---------------------------
# 4. Regression of ETF returns on PCA factors + Spread
# ---------------------------
print("Regressing ETF returns on PCA factors + Spread...")

# Assemble the regressor matrix: PCA factors F1, F2, F3 plus the credit spread factor
X = pd.concat([factors_ts, spread], axis=1)
n_f = factors_ts.shape[1]
X.columns = [f"F{i+1}" for i in range(n_f)] + ["Spread"]

# Add an intercept column (constant) required by statsmodels OLS
X = sm.add_constant(X)

betas = []
tickers = etf_rets.columns
residuals = pd.DataFrame(index=X.index, columns=tickers)

# Run a separate OLS regression for each ETF.
# This decomposes each ETF's return into systematic (factor-driven) and
# idiosyncratic (residual) components.
for ticker in tickers:
    y = etf_rets[ticker].dropna()
    # Align y and X to the same dates to handle any remaining index mismatches
    y, X_aligned = y.align(X, join='inner', axis=0)
    model = sm.OLS(y, X_aligned).fit()
    betas.append(model.params.values)
    # Store residuals for later idiosyncratic covariance estimation
    residuals.loc[y.index, ticker] = model.resid

# Stack per-ETF parameter vectors into a (n_etf x n_regressors) matrix
betas = np.array(betas)
betas_df = pd.DataFrame(betas, index=tickers, columns=X.columns)

print("\nEstimated Factor Loadings (betas):")
print(betas_df.round(5))

# Estimate the residual covariance matrix (idiosyncratic risk).
# This is used in the Monte Carlo simulation to add uncorrelated noise to each ETF.
resid_cov = residuals.cov()

# Estimate the factor covariance matrix from the regression regressors (excluding the constant).
factor_cov = X.drop(columns="const").cov()

# Plot the estimated betas for visual inspection of each ETF's factor sensitivities
ax = betas_df.drop(columns="const").plot(kind="bar", figsize=(10,6))
ax.set_title("ETF Factor Loadings (Yield PCA + Credit Spread)")
ax.set_ylabel("Beta")
plt.tight_layout()
plt.show()

# ---------------------------
# 5. Nelson-Siegel monthly fits (fast)
# ---------------------------
print("Fitting Nelson-Siegel parameters (monthly sampling, warm-start)...")

# Build the tenor array (in years) that matches the yield curve column order.
# Supported labels: "3M" -> 0.25, "5Y" -> 5.0, "10Y" -> 10.0, "30Y" -> 30.0, etc.
tenor_map = {"3M": 0.25, "6M":0.5, "1Y":1.0, "2Y":2.0, "5Y":5.0, "10Y":10.0, "30Y":30.0}

def parse_tenor(col):
    """
    Convert a maturity label string to a numeric value in years.

    Handles month-denominated labels (e.g., '3M', '6M') by dividing by 12,
    and year-denominated labels (e.g., '1Y', '10Y') directly.
    Falls back to float conversion for plain numeric strings.

    Parameters
    ----------
    col : str
        Maturity label such as '3M', '5Y', '10Y'.

    Returns
    -------
    float
        Maturity expressed in years.
    """
    col = col.upper().strip()
    if col.endswith('M'):
        return float(col.replace('M', '')) / 12.0
    elif col.endswith('Y'):
        return float(col.replace('Y', ''))
    else:
        return float(col)

# Convert all yield column headers to numeric tenors in years
tenors = np.array([parse_tenor(c) for c in yields.columns])

# Downsample yields to monthly frequency (mean within each month).
# Monthly fitting is much faster than daily and captures the low-frequency
# dynamics of the yield curve shape adequately for simulation purposes.
yields_m = yields.resample("M").mean().dropna()

params = []
dates = []

# Use the first month's average yield as the initial parameter guess
p0 = [yields_m.iloc[0].mean(), -0.02, 0.02, 1.0]

iterable = yields_m.iterrows()
if use_tqdm:
    iterable = tqdm(yields_m.iterrows(), total=len(yields_m), desc="NS fit (monthly)")

for date, row in iterable:
    yvals = row.values
    # Fit NS parameters to this month's observed yield curve
    popt = fit_ns_curve(yvals, tenors, p0=p0, maxfev=3000)
    params.append(popt)
    dates.append(date)
    # Warm-start: use the current month's fitted parameters as the next month's initial guess.
    # This improves convergence speed and parameter continuity across time.
    if not np.isnan(popt).any():
        p0 = popt

# Collect all monthly NS parameter estimates into a single DataFrame
ns_params = pd.DataFrame(params, index=dates, columns=["beta0","beta1","beta2","lambd"])
print("Fitted NS params (monthly) shape:", ns_params.shape)

# Forward-fill any months where the optimiser failed (NaN parameters),
# then drop remaining NaNs at the start if no prior estimate is available.
ns_params = ns_params.fillna(method="ffill").dropna()

# Reindex the monthly NS parameters to the daily regression index by forward-filling.
# This gives a daily time series of NS parameters used to compute empirical daily deltas.
ns_daily = ns_params.reindex(pd.DatetimeIndex(X.index)).fillna(method="ffill").dropna()

# Compute daily first-differences of NS parameters to model day-to-day dynamics
ns_changes = ns_daily.diff().dropna()

# Estimate the empirical mean and covariance of daily NS parameter changes.
# These are used to parameterise the multivariate normal distribution from which
# NS shocks are drawn during Monte Carlo simulation.
ns_mean = ns_changes.mean().values
ns_cov = ns_changes.cov().values

# ---------------------------
# 6. Simulate NS-driven factor shocks
# ---------------------------
def simulate_ns_based_factor_shocks(n_sim, ns_mean, ns_cov, tenors, today_ns_params, pca_loadings_matrix):
    """
    Simulate PCA factor shocks derived from stochastic Nelson-Siegel parameter dynamics.

    The procedure is:
        1. Draw n_sim shocks to the NS parameter vector from a multivariate normal
           distribution parameterised by the empirical mean and covariance of
           historical daily NS parameter changes.
        2. For each simulated NS parameter vector, compute the implied yield curve
           and subtract today's NS-implied yield curve to obtain a yield change vector.
        3. Project the yield change vector onto the PCA loading matrix to convert
           it into PCA factor space, producing one row of factor shocks per simulation.

    This approach ensures that the simulated yield curve scenarios are consistent
    with the estimated yield curve dynamics captured by the NS model.

    Parameters
    ----------
    n_sim : int
        Number of simulation paths to generate.
    ns_mean : np.ndarray
        Empirical mean vector of daily NS parameter changes, shape (4,).
    ns_cov : np.ndarray
        Empirical covariance matrix of daily NS parameter changes, shape (4, 4).
    tenors : np.ndarray
        Maturities in years corresponding to the yield curve nodes.
    today_ns_params : np.ndarray
        NS parameters fit to the most recent observation, shape (4,).
    pca_loadings_matrix : np.ndarray
        PCA eigenvector matrix, shape (n_yield_vars, n_pca_factors).

    Returns
    -------
    np.ndarray
        Simulated PCA factor shocks, shape (n_sim, n_pca_factors).
    """
    # Compute the matrix square root of the NS parameter covariance for Cholesky-style sampling
    Lns = np.real(sqrtm(ns_cov))  # take only the real part to handle floating-point asymmetry

    # Draw standard normal samples and apply the linear transformation to match ns_cov
    z = np.random.normal(size=(n_sim, ns_mean.shape[0]))
    ns_deltas = z.dot(Lns.T) + ns_mean  # shape: n_sim x 4

    # Compute today's yield curve from the current NS parameters as the baseline
    today_yields = ns_yields_from_params(today_ns_params, tenors)

    # Allocate output array for PCA factor shocks
    factor_shocks = np.zeros((n_sim, pca_loadings_matrix.shape[1]))

    for i in range(n_sim):
        # Perturb today's NS parameters by the simulated shock
        sim_params = today_ns_params + ns_deltas[i]
        # Reconstruct the hypothetical yield curve under the perturbed parameters
        sim_yields = ns_yields_from_params(sim_params, tenors)
        # Yield change = simulated curve minus today's curve
        yield_change = (sim_yields - today_yields)  # shape: (n_yield_vars,)
        # Project the yield change into PCA factor space
        factor_shocks[i, :] = yield_change.dot(pca_loadings_matrix)

    return factor_shocks

# Use the last available NS parameter set as today's yield curve baseline
today_ns = ns_daily.iloc[-1].values

# Factor loading matrix maps yield space to PCA factor space: shape (n_yield_vars, n_pca_factors)
pca_loadings_matrix = factor_loadings.values

print("Simulating NS-driven PCA factor shocks...")
pca_factor_shocks_from_ns = simulate_ns_based_factor_shocks(
    N_SIM, ns_mean, ns_cov, tenors, today_ns, pca_loadings_matrix
)

# Simulate credit spread shocks independently from a normal distribution
# parameterised by the empirical mean and standard deviation of historical daily spread changes.
spread_changes = spread.diff().dropna()
spread_mean = spread_changes.mean()
spread_std = spread_changes.std()
sim_spread_shocks = np.random.normal(loc=spread_mean, scale=spread_std, size=(N_SIM,))

# Compose the full factor shock matrix in the regression order: [F1, F2, F3, Spread]
# Each row is one simulation path's set of factor realisations.
sim_full_factors = np.column_stack([
    pca_factor_shocks_from_ns[:, 0],  # Level factor shock
    pca_factor_shocks_from_ns[:, 1],  # Slope factor shock
    pca_factor_shocks_from_ns[:, 2],  # Curvature factor shock
    sim_spread_shocks                 # Credit spread shock
])

# ---------------------------
# 7. Map simulated factors -> ETF returns and compute VaR
# ---------------------------
# Extract the beta sub-matrix corresponding only to the systematic factors
# (excluding the intercept column), preserving the same factor order as sim_full_factors.
regressor_names = [f"F{i+1}" for i in range(factors_ts.shape[1])] + ["Spread"]
beta_matrix = betas_df.loc[:, regressor_names].values  # shape: (n_etf, n_factors)
consts = betas_df["const"].values  # per-ETF intercept terms, shape: (n_etf,)

# Simulate idiosyncratic (residual) return noise from the estimated residual covariance matrix.
# Using sqrtm (matrix square root) ensures the simulated residuals have the correct cross-sectional
# covariance structure, preserving correlations between ETFs that are not explained by the factors.
resid_L = np.real(sqrtm(resid_cov.fillna(0).values))
sim_idio = np.random.normal(size=(N_SIM, len(TICKERS))).dot(resid_L.T)

# Combine systematic factor returns and idiosyncratic noise to form full simulated ETF returns.
# Shape: (N_SIM, n_etf). The intercept term (consts) is added to each row.
r_sim_ns = sim_full_factors.dot(beta_matrix.T) + consts + sim_idio

# Aggregate simulated ETF returns to portfolio level using the fixed weight vector
port_returns_ns = r_sim_ns.dot(WEIGHTS)

# NS-driven Monte Carlo VaR: the ALPHA-quantile loss (sign-flipped so positive = loss)
var_mc_ns = -np.percentile(port_returns_ns, ALPHA * 100)

# NS-driven Monte Carlo CVaR (Expected Shortfall): average loss conditional on exceeding VaR
cvar_mc_ns = -port_returns_ns[port_returns_ns <= np.percentile(port_returns_ns, ALPHA * 100)].mean()

print(f"NS-driven Monte Carlo VaR (99%): {var_mc_ns:.6f}, CVaR (99%): {cvar_mc_ns:.6f}")

# ---------------------------
# 8. Parametric VaR (delta-normal) from simulated returns covariance
# ---------------------------
# Estimate the return covariance matrix from the simulated ETF returns.
# Using the simulated returns (rather than historical returns alone) ensures this
# covariance is consistent with the NS-driven factor structure.
cov_r_mat = np.cov(r_sim_ns.T)  # shape: (n_etf, n_etf)

# Portfolio variance = w' * Sigma * w
port_var = WEIGHTS.dot(cov_r_mat).dot(WEIGHTS)
port_sigma = np.sqrt(port_var)

# Delta-normal VaR uses the normal quantile at the given significance level.
# The portfolio mean is assumed to be zero for a short holding period (one day).
z = norm.ppf(ALPHA)
var_param = -(0 + z * port_sigma)  # z is negative, so negating gives a positive VaR

print(f"Parametric (delta-normal) VaR (approx): {var_param:.6f}")

# ---------------------------
# 9. Component / marginal VaR
# ---------------------------
# Parametric marginal VaR: approximates each asset's contribution to total portfolio VaR.
# The contribution of asset i is proportional to its weight and its covariance with
# the portfolio return, scaled by the inverse of portfolio standard deviation.
cov_with_port = cov_r_mat.dot(WEIGHTS)  # Cov(r_i, r_portfolio) for each i
# mVaR_i = -z * (w_i * Cov(r_i, r_port)) / sigma_port
mvar_param = -(z * (WEIGHTS * cov_with_port) / port_sigma)

# MC tail-based component VaR: identifies the average return of each ETF conditional
# on the portfolio falling into the tail (i.e., scenarios worse than VaR).
# The component contribution is the weight-scaled tail-conditional mean return, sign-flipped.
threshold = np.percentile(port_returns_ns, ALPHA*100)
tail_idx = port_returns_ns <= threshold

if isinstance(tail_idx, np.ndarray):
    tail_mean_asset_returns_in_tail = r_sim_ns[tail_idx].mean(axis=0)
else:
    # Handle pandas boolean index if port_returns_ns is a Series
    tail_mean_asset_returns_in_tail = r_sim_ns[tail_idx.values].mean(axis=0)

comp_var_mc = -WEIGHTS * tail_mean_asset_returns_in_tail

# Normalise both contribution vectors so that they sum to the NS-driven MC VaR.
# This ensures the components are additive and consistent with the headline risk measure.
sum_param = mvar_param.sum()
comp_param_norm = (mvar_param * (var_mc_ns / sum_param)) if (sum_param != 0) else mvar_param

sum_mc = comp_var_mc.sum()
comp_mc_norm = (comp_var_mc * (var_mc_ns / sum_mc)) if (sum_mc != 0) else comp_var_mc

# Summarise per-asset risk contributions alongside their portfolio weights
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
# Bar chart of parametric component VaR: shows which ETFs drive the most systematic risk
plt.figure(figsize=(8,4))
risk_contrib_df["Param_ComponentVaR"].plot(kind="bar")
plt.title("Parametric Component VaR (normalized)")
plt.ylabel("Contribution to VaR (notional=1)")
plt.tight_layout()
plt.show()

# Bar chart of MC tail-based component VaR: shows risk contributions in extreme loss scenarios
plt.figure(figsize=(8,4))
risk_contrib_df["MC_Tail_ComponentVaR"].plot(kind="bar")
plt.title("MC Tail-based Component VaR (normalized)")
plt.ylabel("Contribution to VaR (notional=1)")
plt.tight_layout()
plt.show()

# ---------------------------
# 11. Historical & Standard MC VaR (original pipeline)
# ---------------------------
# Historical VaR: computed directly from realised portfolio returns without any model assumptions.
# This serves as a benchmark for the model-based estimates.
hist_port_returns = etf_rets.dot(WEIGHTS)
hist_var = -np.percentile(hist_port_returns, ALPHA * 100)
hist_cvar = -hist_port_returns[hist_port_returns <= np.percentile(hist_port_returns, ALPHA * 100)].mean()

# Baseline PCA Monte Carlo VaR: uses only the empirical PCA factor covariance
# (without NS dynamics) as an alternative to the NS-driven approach.
# This highlights the additional information contributed by the NS term structure model.
factor_mean = factors_ts.mean().values
factor_cov_emp = factors_ts.cov().values

# Draw factor realisations from the empirical factor covariance using a matrix square root
L = sqrtm(factor_cov_emp)
sim_factors = np.random.normal(size=(N_SIM, factors_ts.shape[1])).dot(L.T) + factor_mean

# Extract only the PCA factor betas (excluding the spread and constant)
beta_matrix_pca = betas_df.loc[:, [f"F{i+1}" for i in range(factors_ts.shape[1])]].values
consts_pca = betas_df["const"].values

# Reuse the same idiosyncratic noise draws as the NS simulation for a fair comparison
r_sim_baseline = (sim_factors.dot(beta_matrix_pca.T)) + consts_pca + sim_idio

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
    # Subset the return history to the stress period
    sub = etf_rets.loc[sd:ed]
    if sub.empty:
        # Record NaN if the period falls outside the available data range
        stress_results[name] = {"cumulative_return": np.nan, "worst_day_loss": np.nan}
        continue
    # Cumulative portfolio return: exponentiate cumulative log returns to get gross return, then weight
    cum_ret = (np.exp(sub).prod() - 1).dot(WEIGHTS)
    # Worst single-day loss: maximum daily portfolio loss during the period
    worst_day_loss = -sub.dot(WEIGHTS).min()
    stress_results[name] = {"cumulative_return": cum_ret, "worst_day_loss": worst_day_loss}

print("\nHistorical stress results:")
for k, v in stress_results.items():
    print(k, v)

# Hypothetical stress scenarios: apply a deterministic parallel shift to NS parameter beta0,
# which controls the long-run yield level. A shift of +0.01 corresponds to +100 basis points.
# The shock is mapped through the NS model to PCA factor space and then through the
# beta matrix to estimate the portfolio P&L impact.
hyp_shocks = {
    "Parallel +100bps": np.array([0.01, 0.0, 0.0, 0.0]),   # Increase in beta0 (level factor)
    "Parallel -100bps": np.array([-0.01, 0.0, 0.0, 0.0])   # Decrease in beta0 (level factor)
}

def ns_param_to_pca_shock(delta_ns):
    """
    Map a deterministic NS parameter shock to a PCA factor shock vector.

    Computes the yield curve implied by (today's NS parameters + delta_ns),
    subtracts today's yield curve, and projects the resulting yield change
    onto the PCA loading matrix to obtain the equivalent factor space shock.

    Parameters
    ----------
    delta_ns : np.ndarray
        Additive shock to the NS parameter vector, shape (4,).

    Returns
    -------
    np.ndarray
        Equivalent PCA factor shock, shape (n_pca_factors,).
    """
    sim_params = today_ns + delta_ns
    sim_yields = ns_yields_from_params(sim_params, tenors)
    today_y = ns_yields_from_params(today_ns, tenors)
    yield_change = sim_yields - today_y
    # Project the yield change into PCA factor space
    return yield_change.dot(pca_loadings_matrix)

for name, delta_ns in hyp_shocks.items():
    # Pad the shock vector to match the full NS parameter length (4) if needed
    if len(delta_ns) < today_ns.shape[0]:
        delta_ns_full = np.zeros_like(today_ns)
        delta_ns_full[:len(delta_ns)] = delta_ns
    else:
        delta_ns_full = delta_ns

    # Convert the NS parameter shock to a PCA factor shock
    pca_sh = ns_param_to_pca_shock(delta_ns_full)

    # Append a zero spread shock: hypothetical rate scenarios are assumed to not affect
    # credit spreads directly in this simplified stress framework
    combined = np.concatenate([pca_sh, [0.0]])

    # Estimate per-ETF return impact by multiplying through the beta matrix
    delta_r = combined.dot(beta_matrix.T)  # shape: (n_etf,)

    # Aggregate to portfolio P&L using fixed weights
    pnl = delta_r.dot(WEIGHTS)
    print(f"Hypothetical {name}: estimated pnl (notional=1) = {pnl:.6f}")

# ---------------------------
# 13. Save summary
# ---------------------------
# Compile all headline VaR estimates into a single summary table and export to CSV
summary = pd.DataFrame({
    "Metric": [
        "Historical VaR (99%)",
        "Baseline PCA MC VaR (99%)",
        "NS-driven MC VaR (99%)",
        "NS-driven MC CVaR (99%)"
    ],
    "Value": [hist_var, var_mc_baseline, var_mc_ns, cvar_mc_ns]
})
summary.to_csv("fixed_income_risk_summary_with_ns.csv", index=False)
print("\nSaved summary to fixed_income_risk_summary_with_ns.csv")
