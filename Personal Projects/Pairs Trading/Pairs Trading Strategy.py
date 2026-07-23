"""
Pairs Trading: Discovery + Selection Framework Iteration
=========================================================================
Single-split (not rolling) discovery -> selection design, used to
iteratively refine the pair-selection framework before ever touching the
out-of-sample window.

HARD GUARDRAIL: this script downloads data only through SELECTION_END. The
out-of-sample window (OOS_START onward) is never requested from the data
provider here — not filtered out after the fact, never in memory at all.
It is pulled once, by a separate script, only after the framework below
is frozen.

Pipeline
--------
1. Universe construction and price download, capped at SELECTION_END.
2. Discovery window: distance pre-filtering, Engle-Granger cointegration
   testing, OLS hedge-ratio estimation, zero-crossing screening, and
   Ornstein-Uhlenbeck mean-reversion characterization. Hedge ratios and
   regression direction are frozen here and reused unchanged.
3. Selection window: the frozen candidates are evaluated with a rolling
   z-score signal; performance thresholds (Sharpe/drawdown/return) decide
   which pairs make the final list, and portfolio weights (equal,
   Sharpe-weighted, inverse-volatility, risk parity) are fit on the same
   window.
4. Every run appends a structured record (full config snapshot, funnel
   counts, candidate table, chosen pairs/weights) to ITERATION_LOG_PATH,
   and prints a compact history of every run so far — this is the record
   to draw on when writing up how the final framework was chosen.

NEW (score validation): a separate mode (RUN_SCORE_VALIDATION) tests,
honestly and out-of-sample, whether ANY Discovery/lookback-time
characteristic -- structural (ADF, OU R^2, half-life, variance ratio,
beta stability) or trade-level (win rate, expectancy, tail-risk,
recovery-time consistency) -- predicts genuinely unseen forward
profitability. Only features that clear that bar get combined into a
composite score, which Phase 2 can then optionally use (USE_COMPOSITE_SCORE)
to keep only the top-scoring pairs each cycle, instead of only checking
minimum-quality thresholds.

Requires a local Excel file listing S&P 500 tickers (EXCEL_PATH).
"""

import json
import datetime
import numpy as np
import pandas as pd
import yfinance as yf
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint, adfuller
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import pdist, squareform
from scipy.optimize import minimize
from concurrent.futures import ProcessPoolExecutor
import os
import sys
import ctypes
import tabulate

# ============================================================================
# CONFIG
# ============================================================================
EXCEL_PATH = r"C:\Users\Admin\OneDrive\Desktop\Personal Projects\Pairs Trading\Bloomberg Excel_2025_Sept_04.xlsx"
SHEET_NAME = "SPX"

DATA_START = "2015-01-01"
DISCOVERY_END = "2022-06-30"    # Discovery window: [DATA_START, DISCOVERY_END]
SELECTION_END = "2024-06-30"    # Selection window: (DISCOVERY_END, SELECTION_END]

# HARD GUARDRAIL: this is the only date value ever passed as `end` to the
# data download call for Discovery/Selection in this script. It is one day
# past SELECTION_END (yfinance's `end` is exclusive) and nothing later.
DOWNLOAD_END = "2024-07-01"

# OOS section (bottom of this file) is gated behind this flag so it cannot
# run by accident while you're still iterating on Discovery/Selection —
# flip it to True only once the framework is genuinely frozen and you're
# ready to see the OOS result. When True, a SEPARATE data download is made
# for OOS_START..OOS_END, using only the tickers in the final pair list —
# still no code path touches OOS data unless this is explicitly set.
RUN_OOS_EVALUATION = True

# Discovery/Selection is the expensive part of this script (the
# cointegration + regression stages). If you already have a
# frozen_model.json from a prior run and just want to (re-)run OOS against
# it, set this to True — it skips Discovery/Selection entirely and loads
# the saved pairs/betas/weights straight from disk instead of recomputing
# them (which would reproduce the identical result anyway, since nothing
# about Discovery/Selection is random). Only meaningful when
# RUN_OOS_EVALUATION is also True and output/frozen_model.json exists.
SKIP_DISCOVERY_SELECTION = False

# Runs the expensive stages (distance filter, cointegration, regression) ONCE
# and then sweeps threshold combinations against cached results — see the
# "6c. THRESHOLD SENSITIVITY SWEEP" section below for why this doesn't
# multiply runtime the way naively re-running the pipeline per combination
# would. Mutually exclusive with the normal Discovery/Selection/OOS flow
# above — if True, this runs instead of main().
RUN_THRESHOLD_SWEEP = False

# Number of sequential sub-windows to re-test cointegration on, within the
# already-filtered ~3,300-pair subset (see compute_coint_persistence). Kept
# small by default specifically to bound added runtime — each extra window
# adds roughly one more parallelized pass over that subset, not over the
# full ~10,557-pair pool.
N_PERSISTENCE_WINDOWS = 2
OOS_START = "2024-07-01"

# ---- PHASE 1: fold-based threshold RANGE validation ----
# Instead of one 9.5-year Discovery run producing single threshold values,
# split 2015-2023 into non-overlapping 2-year folds spanning genuinely
# different regimes (calm / vol-spike / COVID / rate-hike bear market).
# Within each fold, sweep every parameter and find its stable plateau range;
# a threshold only counts as VALIDATED if its plateau overlaps across ALL
# folds, not just one. Mutually exclusive with the other modes.
RUN_PHASE1_VALIDATION = False
# Two folds, explicitly split at COVID rather than uniform 2-year slices —
# COVID is the single largest regime break in the whole 9.5-year span, so
# this is a more meaningful split than an arbitrary 4-way division, and it
# halves Phase 1's cost (2 expensive fold-runs instead of 4).
# Each tuple is (discovery_start, eval_start, fold_end): sub-discovery runs
# from discovery_start to eval_start, then eval_start to fold_end is the
# held-out evaluation window within that fold.
FOLD_DEFINITIONS = [
    ("2015-01-01", "2019-07-01", "2020-01-01"),   # pre-COVID: calm-to-moderate regime
    ("2020-01-01", "2023-07-01", "2024-01-01"),   # post-COVID: crash, recovery, 2022 bear market, 2023 rebound
]
N_FOLDS = len(FOLD_DEFINITIONS)

# Distance is swept for free using the same trick as everything else: prefilter
# at a GENEROUS percentile once, then any narrower percentile is just a subset
# of what's already computed (percentile rank within the generous pool). Wider
# than the usual 10% specifically to leave room to test narrower values below
# it without re-running cointegration — this is the one deliberate added cost
# in Phase 1, paid once per fold, not once per parameter combination.
GENEROUS_DISTANCE_PERCENTILE = 30   # reduced from 100 for speed -- 50 was already too slow, 100 (full universe) roughly doubled that again

# NOT swept, and deliberately so: this is a standard statistical convention,
# not a tuned parameter the way the others are, and validating it properly
# would mean re-running the single most expensive stage multiple times per
# fold for comparatively little insight. Held fixed at the conventional 0.05
# throughout Phase 1.
FOLD_COINT_PVALUE = 0.05

PLATEAU_MIN_SHARPE = 0.0      # a threshold value only counts as "in the plateau" if portfolio_sharpe >= this AND n_passed > 0

# ---- PHASE 2: rolling re-qualification OOS ----
# No connection to Discovery/Selection data at all — at each rebalance, looks
# back only PHASE2_LOOKBACK_MONTHS (fixed size, never expanding), re-applies
# the Phase-1-validated ranges (never re-tuned here), and trades whatever
# currently qualifies until the next rebalance.
RUN_PHASE2_OOS = False
PHASE2_LOOKBACK_MONTHS = 6
PHASE2_USE_CONDITIONAL_RANGES = False  # the last Phase 1 run's conditional variance_ratio band (40-50
# percentile) collapsed the universe to ~25 pairs/cycle (<1 trade/day) -- too tight to hit a >=10
# trades/day target. Set False to use only the primary (isolated-sweep) validated ranges instead,
# which historically gave a much broader ~500-580 pairs/cycle universe. Set True to go back to using
# whatever conditional ranges Phase 1 most recently found, if you want the tighter, more selective set.

# ---- SCORE VALIDATION (NEW): does any Discovery-time characteristic
# actually predict forward profitability, tested honestly (lookback stats
# vs. a forward window never touched during characterization) rather than
# eyeballing a top-decile description? See run_score_validation() below. ----
RUN_SCORE_VALIDATION = False
# Minimum number of lookback trades before trusting a pair's trade-level
# stats (win rate, expectancy, tail loss, recovery consistency) at all -- a
# "100% win rate" from one lucky trade is noise, not a real estimate. Pairs
# below this get NaN features rather than a spuriously confident value.
MIN_LOOKBACK_TRADES_FOR_FEATURES = 3
# Once a score has been validated (score_weights.json exists) and you want
# Phase 2 to actually TRADE using it instead of (on top of) just the
# structural funnel:
USE_COMPOSITE_SCORE = True
SCORE_TOP_PERCENTILE = 30   # keep only the top X% by composite score, each cycle
SCORE_WEIGHTS_PATH = None   # set below, after OUTPUT_DIR is defined
# Minimum |mean Information Coefficient| (Spearman rank correlation between
# a lookback feature and forward outcome, averaged across cycles) for a
# feature to be included in the composite score at all. Deliberately low --
# this is meant to be a real bar, not a rubber stamp, but individual-cycle
# ICs are noisy with only ~6-10 cycles, so don't set this high.
MIN_IC_FOR_INCLUSION = 0.05
# A feature only counts as "stable" (not just lucky in aggregate) if its
# per-cycle IC has the same sign in at least this fraction of cycles.
MIN_IC_SIGN_CONSISTENCY = 0.65
# If two validated features are more correlated than this (Pearson, on the
# raw per-pair values pooled across all cycles), only the one with the
# stronger combined IC is kept -- prevents e.g. lb_sharpe/lb_ret/lb_n_trades
# (which are likely near-duplicates of the same "pair looked unusually
# strong recently" signal) from being triple-counted into the composite,
# which empirically made live performance WORSE than a 2-feature subset,
# not better (confirmed on a real Phase 2 run: Sharpe 0.887 -> 0.629 at
# 5bps once lb_sharpe/lb_ret were actually applied instead of silently
# dropped by a key-naming bug).
MAX_FEATURE_CORRELATION = 0.7
# Manual override for testing specific feature combinations directly,
# bypassing both the IC bar and the correlation pruning below. Set to a
# list of feature names (e.g. ["cointegration_p", "lb_n_trades"]) to force
# exactly that subset into the saved score, or leave as None to use the
# automatic (IC-bar + correlation-pruned) selection.
SCORE_FEATURE_SUBSET = None

EXCLUDED_TICKERS = {"TSLA", "ERIE", "LYV", "HUM", "MCK", "NEM", "AKAM", "SMCI", "AVGO", "FSLR"}
# Isolated single-variable experiment: these are the tickers with the highest "lift" (over-representation
# in worst-decile losing trades relative to their own overall trading frequency) from the repeat-loser-
# tickers diagnostic on the last run. Excluding them entirely, with no other change, tests whether the
# framework's current failure to characterize these names' risk (event risk, sector dynamics, whatever
# it is) is actually costing real performance -- vs. whether their apparent over-representation was noise
# that will simply reappear as a different set of names next run. Set to an empty set (set()) to disable.
# No single ticker allowed in more than this many traded pairs per cycle.
# Confirmed empirically: without this, a stock with a large idiosyncratic
# move (e.g. CRWD in one real run) can spuriously "cointegrate" with dozens
# of unrelated names simultaneously, since they're really all just tracking
# that one stock's move -- 26% of one real cycle's book was CRWD pairs alone,
# turning what looked like diversification into one concentrated bet.
MAX_PAIRS_PER_TICKER = 3
# Hard cap on how much capital any single position can absorb on a given
# day. Without this, days with very few concurrent positions open (e.g.
# right after a rebalance) implicitly lever the book 50-100% into just 1-2
# pairs -- which is what was producing the inflated Sharpe. 0.05 = no more
# than 5% of capital in any one position at a time (equivalent to a fully-
# diversified ~20-position book); tune this to your actual risk tolerance.
MAX_POSITION_WEIGHT = 0.05
PHASE2_REBALANCE_MONTHS = 6   # 6 rebalances across the 3-year OOS window
PHASE2_START = "2023-07-01"
PHASE2_END = "2026-07-01"   # extended from 2025-07-01 to actually include the 2025-2026 period,
# needed to test the specific hypothesis that performance weakened during a later AI-momentum regime
REGIME_MAP = {
    # cycle start date -> short descriptive label. These are rough, editable descriptive tags for
    # readability when comparing cycle-level performance -- NOT a rigorous, independently-derived
    # regime classification. The actual test of "did performance differ by regime" is the per-cycle
    # Sharpe/return table this produces, not the labels themselves. Edit freely to match your own view
    # of what each period represented; cycles past the last entry reuse the final label.
    "2023-07-01": "Post-COVID Normalization",
    "2024-01-01": "AI Theme Emerging",
    "2024-07-01": "AI Momentum Building",
    "2025-01-01": "AI Leadership Concentration",
    "2025-07-01": "AI-Led Momentum (extended)",
    "2026-01-01": "AI-Led Momentum (extended)",
}
OOS_END = "2026-07-01"

MIN_DATA_POINTS = 800              # ~3+ years minimum; full range is ~9.5 years
DISTANCE_SELECT_PERCENTILE = 10
COINT_PVALUE = 0.05
# A rate-per-year assumption (crossings scale linearly with elapsed time)
# turned out to be wrong: a real run on the actual 7.5-year Discovery window
# showed crossing counts topping out around ~170, while the rate-based
# threshold implied ~199 — above the maximum any pair achieved, filtering
# out 100% of pairs. The reason crossings don't scale linearly with time
# here: the Discovery-window regression fits a SINGLE static hedge ratio
# across the whole 7.5 years, spanning very different regimes (COVID crash,
# 2022 bear market, etc). A spread built on one fixed relationship across
# regime changes has more sustained excursions and fewer tight oscillations
# than the same relationship would over a short, calm window — so crossing
# frequency is not simply proportional to elapsed time.
# Fix: self-calibrate the same way distance_prefilter does — drop the
# least-oscillating tail of whatever the actual current population looks
# like, rather than assuming a rate in advance.
ZERO_CROSSING_MIN_PERCENTILE = 20   # drop the bottom 20% least-oscillating cointegrated pairs
OU_DRIFT_PCTL = 0.30

# Was 50, capping how many of the 2,669 zero-crossing survivors ever got
# OU-fitted at all regardless of how many might have been good candidates.
# Raised so more of the population gets a chance.
OU_TOP_N = 150

# Was [(10,20),(20,30),(30,40)] — calibrated for 9-12 month Discovery windows,
# never rescaled when Discovery was extended to 7.5 years. A real run's half-life
# histogram showed the actual fitted distribution clustering around 40-65 trading
# days (peak ~55-58), almost entirely ABOVE the old 40-day ceiling — the bucket
# filter was excluding most of the real distribution's mass, not just outliers.
# Widened to actually cover where the pairs are.
HALF_LIFE_BUCKETS = [(10, 30), (30, 60), (60, 90)]

# Loosened from 0.70 as a companion change — with more candidates now reaching
# this stage via the two changes above, this is the next lever if still too few
# candidates survive.
R2_THRESHOLD = 0.50

ROLLING_WINDOW = 60
# Previously the SAME window computed both the rolling mean and the rolling
# std. Problem: if volatility spikes during that window (a broad selloff, a
# company-specific shock), the std inflates, which shrinks the Z-score's
# magnitude even though the actual dollar-level spread hasn't moved back
# toward its true mean at all -- the exit condition (|Z| < Z_EXIT) can then
# trigger on a volatility artifact, not genuine reversion. This is the
# specific mechanism behind real trades that exited as "reversion" while
# still deeply negative (WDC-MDT -28%, CRWD-EQIX -25%). A longer, separate
# window for std specifically makes it less reactive to a short-lived spike,
# while the mean stays responsive at the original window length.
Z_STD_WINDOW = 120
Z_ENTRY = 2.0
WAIT_FOR_Z_PEAK = True   # require |Z| to have already started declining from the prior day before
# entering, instead of entering the moment |Z| crosses Z_ENTRY. Direct counterfactual test: on genuinely
# mean-reverting synthetic paths, this raised average trade return from 1.098 to 1.334 (+21%) and win
# rate from 81.7% to 87.7%, with no measurable added protection on trending/broken series (average
# return there was identical either way). REAL TRADE-OFF: this also rejected ~65-70% of ALL candidate
# entries in the same test -- a large volume cost, directly in tension with the >=10 trades/day target
# established earlier. Set False to revert to entering immediately on the crossing, if the volume loss
# turns out to outweigh the per-trade quality gain once tested on real data.
Z_EXIT = 0.5           # tightened from 1.0. Real-data finding: decomposing the last run's loss
# population showed 71% of ALL losing trades (172/242) were "reversion" exits -- Z genuinely came
# back toward the mean, a real success by the strategy's own definition -- that were STILL net
# negative, because Z_EXIT=1.0 let the position close before it had fully recovered an earlier
# adverse move. Confirming synthetic test: on genuinely mean-reverting paths, tightening Z_EXIT from
# 1.0 to 0.5 raised average trade return by ~29% (1.38 -> 1.78) and roughly halved the reversion-
# loser rate (3.4% -> 1.8%), with no measurable added cost on trending/broken relationships (which
# mostly never reach even the old 1.0 threshold anyway, so tightening it further doesn't change how
# they get handled). 0.25 tested slightly better on mean-reverting return but showed the first signs
# of degrading win rate on trending series -- 0.5 is the better-supported balance for now.
ENTRY_ADF_RECHECK_WINDOW = 60   # trailing days of spread history re-tested for stationarity at each
# potential entry, independent of the discovery-time cointegration test done months earlier
ENTRY_ADF_RECHECK_PVALUE_MAX = 0.30   # RECALIBRATED from 0.10 after direct empirical testing showed
# the original threshold was far too strict: on a synthetic OU process matching this project's own
# observed half-life (~3.6 days), it rejected 54% of genuine mean-reverting entry signals -- while
# correctly rejecting 93% of entries on a genuinely broken/trending series. That's real discriminative
# power, just badly calibrated. At 0.30: 25% false-rejection of good signals (down from 54%), 80% of
# broken-relationship signals still caught (down from 93% but still strong separation). This directly
# explains the ~4x trade-frequency collapse seen after the original threshold was introduced -- it
# wasn't "no edge," it was throwing away roughly half of genuinely good entries.
# Neither of these existed before -- the original rule had no protective exit
# at all: once in a trade, it stays open indefinitely until Z drifts back
# into [-Z_EXIT, Z_EXIT], no matter how far the spread diverges or how long
# that takes. That produces the classic mean-reversion failure shape: many
# small wins capped by the exit band, offset by rare but uncapped losses
# when a relationship genuinely breaks and never reverts.
Z_STOP = 3.0          # RECALIBRATED from 2.25 after a direct counterfactual test exposed the exact
# trade-off the earlier tightening's reasoning missed. On identical genuinely-mean-reverting synthetic
# paths, Z_STOP=2.25 nearly HALVED average return (0.71 vs 1.39 with no Z-stop at all) and crushed win
# rate from 92% to 63% -- not by controlling loss size, but by cutting off a huge fraction of trades
# that would have reverted profitably if simply given more room. A full sweep confirmed this: mean-
# reverting performance climbs steadily as the stop loosens (0.71 at 2.25 -> 1.10 at 3.0 -> 1.33 at
# 4.0), while the cost on genuinely broken/trending relationships grows only modestly across the same
# range (-0.036 -> -0.073 -> -0.103). The asymmetric-loss-size reasoning that motivated the original
# tightening was correct about size but missed that tightening also multiplies how OFTEN genuine
# reversions get cut short, and that frequency effect dominates. Z_STOP also doesn't need to carry
# catastrophic tail-risk protection alone -- MAX_LOSS_PCT_PER_TRADE exists specifically for that.
MAX_HOLDING_DAYS = 40  # force-exit if still in a trade this many trading days after entry, regardless of Z
MAX_LOSS_PCT_PER_TRADE = 0.03  # tightened again from 0.05, for stronger max-DD and tail-loss control
# regardless of Z. The Z-based stop can be fooled by a rolling std that's stale or too small (esp. with the
# adaptive short window) -- a genuinely large spread move can occur before Z even reaches Z_STOP. This is a
# direct backstop on realized P&L, independent of how Z is currently being estimated.

PERF_SHARPE_MIN = 0.5
# Loosened from 10% to 15% for the 24-month Selection window: drawdown is
# path-dependent and mechanically tends to grow with time horizon even at
# constant volatility (more sequential draws = more chances for a bad
# stretch), so a flat 10% bar was structurally harder to clear over 24
# months than it was over 3. A real run showed every candidate failing on
# this specific filter despite Sharpe/return profiles that looked fine
# (e.g. NXPI-FCX: Sharpe 0.97, +42.7% return, failed only on 12.33% DD).
PERF_MAXDD_MAX = 0.15
PERF_RETURN_MIN_ANNUALIZED = 0.0

TRANSACTION_COST_BPS_PER_LEG = 5

OUTPUT_DIR = r"C:\Users\Admin\OneDrive\Desktop\Personal Projects\Pairs Trading\output"; os.makedirs(OUTPUT_DIR, exist_ok=True)
VALIDATED_RANGES_PATH = os.path.join(OUTPUT_DIR, "validated_ranges.json")
CHART_DIR = os.path.join(OUTPUT_DIR, "charts")
ITERATION_LOG_PATH = os.path.join(OUTPUT_DIR, "iteration_log.jsonl")
SCORE_WEIGHTS_PATH = os.path.join(OUTPUT_DIR, "score_weights.json")  # NEW

N_JOBS = max(1, (os.cpu_count() or 2) - 1)

# Compound the annualized return target over the actual Selection window
# length, so the bar means the same thing regardless of how long that
# window is (see the note on this in the walk-forward script — a flat
# absolute-return threshold applied to a short window is much stricter
# than the same threshold applied to a long one).
_selection_months = (pd.Timestamp(SELECTION_END).year - pd.Timestamp(DISCOVERY_END).year) * 12 + \
                     (pd.Timestamp(SELECTION_END).month - pd.Timestamp(DISCOVERY_END).month)
PERF_RETURN_MIN = (1 + PERF_RETURN_MIN_ANNUALIZED) ** (_selection_months / 12)

RNG_SEED = 42
np.random.seed(RNG_SEED)


# ============================================================================
# 1. DATA LOADING
# ============================================================================
def get_sp500_tickers_from_excel(file_path=EXCEL_PATH, sheet_name=SHEET_NAME):
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    tickers = (
        df["Ticker"].dropna().astype(str).str.replace("/", "-", regex=False).str.strip().tolist()
    )
    return tickers


def download_universe(tickers, start=DATA_START, end=DOWNLOAD_END, min_points=None):
    if min_points is None:
        min_points = MIN_DATA_POINTS
    data = yf.download(tickers, start=start, end=end, interval="1d")["Close"]
    data = data.dropna(how="all")

    fetched = set(data.columns.levels[1]) if isinstance(data.columns, pd.MultiIndex) else set(data.columns)
    missing = [t for t in tickers if t not in fetched]
    print(f"Data downloaded for {len(fetched)} tickers ({start} to {end}, exclusive). Missing: {len(missing)}")

    entry_counts = data.count()
    valid_tickers = entry_counts[entry_counts >= min_points].index.tolist()
    data = data[valid_tickers]

    before_complete = data.shape[1]
    data = data.dropna(axis=1, how="any")
    dropped_incomplete = before_complete - data.shape[1]
    print(f"Usable tickers with complete history over the full {start}-{end} span: {data.shape[1]}  "
          f"({dropped_incomplete} dropped for having any gap over that span)")
    return data


# ============================================================================
# 2. PAIR DISCOVERY  (Discovery window only)
# ============================================================================
def distance_prefilter(window_data, percentile=DISTANCE_SELECT_PERCENTILE):
    cumret = window_data / window_data.iloc[0]
    norm = (cumret - cumret.min()) / (cumret.max() - cumret.min())
    norm = norm.loc[:, norm.notna().any()]
    norm = norm.loc[:, np.isfinite(norm).any()]
    norm = norm.loc[:, ~(norm == 0).all()]
    norm = norm.dropna(axis=0)
    norm = norm[np.isfinite(norm).all(axis=1)]

    if norm.shape[1] < 2:
        raise ValueError(f"Need at least 2 tickers after cleaning, found {norm.shape[1]}")

    columns = norm.columns
    n = len(columns)

    distances = pdist(norm.T, metric="sqeuclidean")
    i_idx, j_idx = np.triu_indices(n, k=1)

    cutoff = np.percentile(distances, percentile)
    mask = distances <= cutoff
    d_vals = distances[mask]
    order = np.argsort(d_vals)

    close_pairs = [
        (columns[i_idx[mask][k]], columns[j_idx[mask][k]], d_vals[k])
        for k in order
    ]
    print(f"Distance pre-filter: {len(close_pairs)} / {len(distances)} pairs "
          f"(closest {percentile}%, SSD <= {cutoff:.2f})")
    return close_pairs, norm, distances


def _coint_test_one(args):
    t1, t2, dist, s1, s2, pvalue = args
    _, p, _ = coint(s1, s2)
    return (t1, t2, dist, p) if p < pvalue else None


def cointegration_filter(norm_window, close_pairs, pvalue=COINT_PVALUE, n_jobs=N_JOBS):
    tasks = [
        (t1, t2, dist, norm_window[t1].values, norm_window[t2].values, pvalue)
        for t1, t2, dist in close_pairs
    ]

    cointegrated = []
    if n_jobs <= 1 or len(tasks) < 200:
        for task in tasks:
            result = _coint_test_one(task)
            if result is not None:
                cointegrated.append(result)
    else:
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            for result in executor.map(_coint_test_one, tasks, chunksize=200):
                if result is not None:
                    cointegrated.append(result)

    print(f"Cointegration filter: {len(cointegrated)} pairs with p < {pvalue}  "
          f"[{len(tasks)} tests across {n_jobs} process(es)]")
    return cointegrated


def _direction_fit_one(args):
    t1, t2, dist, s1_vals, s2_vals, pval, index = args
    x1 = sm.add_constant(s2_vals)
    m1 = sm.OLS(s1_vals, x1).fit()
    adf_result1 = adfuller(m1.resid)  # default autolag='AIC' already does automatic multi-lag selection
    adf1, lags1 = adf_result1[1], adf_result1[2]

    x2 = sm.add_constant(s1_vals)
    m2 = sm.OLS(s2_vals, x2).fit()
    adf_result2 = adfuller(m2.resid)
    adf2, lags2 = adf_result2[1], adf_result2[2]

    if (m1.rsquared > m2.rsquared and adf1 < adf2) or (adf1 < 0.05 and adf2 >= 0.05):
        dependent, independent, model, residuals = t1, t2, m1, pd.Series(m1.resid, index=index)
        direction, r2, adf_p, adf_lags = f"{t1} ~ {t2}", m1.rsquared, adf1, lags1
    else:
        dependent, independent, model, residuals = t2, t1, m2, pd.Series(m2.resid, index=index)
        direction, r2, adf_p, adf_lags = f"{t2} ~ {t1}", m2.rsquared, adf2, lags2

    return {
        "pair": (t1, t2), "dependent": dependent, "independent": independent, "direction": direction,
        "beta": float(model.params[1]), "intercept": float(model.params[0]),
        "R_squared": r2, "ADF_pvalue": adf_p, "ADF_lags_used": int(adf_lags), "disc_spread": residuals,
        "distance": dist, "cointegration_p": pval,
    }


def fit_direction_and_beta(norm_window, cointegrated_pairs, n_jobs=N_JOBS):
    index = norm_window.index
    tasks = [
        (t1, t2, dist, norm_window[t1].values, norm_window[t2].values, pval, index)
        for t1, t2, dist, pval in cointegrated_pairs
    ]
    if n_jobs <= 1 or len(tasks) < 200:
        results = [_direction_fit_one(task) for task in tasks]
    else:
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            results = list(executor.map(_direction_fit_one, tasks, chunksize=200))
    print(f"Regression/direction fit: {len(results)} pairs  [{len(tasks)} pairs across {n_jobs} process(es)]  "
          f"(parameter estimation only -- every input pair gets a fitted beta/direction, nothing is rejected here)")
    return results


def count_zero_crossings(series):
    sign = np.sign(series.values)
    return int(np.sum(sign[:-1] != sign[1:]))


def zero_crossing_filter(reg_results, min_percentile=ZERO_CROSSING_MIN_PERCENTILE):
    for r in reg_results:
        s = r["disc_spread"].dropna()
        r["zero_crossings"] = count_zero_crossings(s)
        r["disc_mean_spread"] = float(s.mean())
        r["disc_var_spread"] = float(np.var(s, ddof=0))

    counts = [r["zero_crossings"] for r in reg_results]
    threshold_used = np.percentile(counts, min_percentile) if counts else 0
    kept = [r for r in reg_results if r["zero_crossings"] >= threshold_used]

    print(f"Zero-crossing filter: {len(kept)} / {len(reg_results)} pairs with >= {threshold_used:.0f} crossings "
          f"(bottom {min_percentile}% of the observed distribution dropped)")
    return kept, threshold_used


def compute_variance_ratio(spread, q):
    """Lo-MacKinlay style variance ratio: for a true random walk, the
    variance of a q-period change is exactly q times the variance of a
    1-period change (VR=1). A genuinely mean-reverting series has VR<1 at
    the horizon it reverts over (large deviations partially cancel out
    before they compound); a trending series has VR>1. Unlike phi/R^2 from
    a single-lag AR(1) fit (which mostly just measures how close phi is to
    1, and turned out empirically not to separate good pairs from bad),
    this is a model-free, direct measurement of the specific behavior that
    actually matters for a mean-reversion trade."""
    s = spread.dropna().values
    n = len(s)
    if q < 2 or n < q * 4:  # need enough non-overlapping windows for a stable estimate
        return np.nan
    diffs_1 = np.diff(s)
    var_1 = np.var(diffs_1, ddof=1)
    if var_1 == 0 or np.isnan(var_1):
        return np.nan
    diffs_q = s[q:] - s[:-q]
    var_q = np.var(diffs_q, ddof=1)
    return float(var_q / (q * var_1))


def fit_ou_on_levels(spread, dt=1.0):
    s = spread.dropna()
    if len(s) < 50:
        return {"valid": False, "reason": "too few points"}
    y = s.iloc[1:].values
    X = sm.add_constant(s.iloc[:-1].values)
    ar1 = sm.OLS(y, X).fit()
    alpha, phi = float(ar1.params[0]), float(ar1.params[1])
    if not (0 < phi < 1):
        return {"valid": False, "phi": phi, "alpha": alpha, "r_squared": ar1.rsquared}
    k = -np.log(phi) / dt
    mu = alpha / (1.0 - phi)
    half_life = np.log(2.0) / k
    eps_var = float(ar1.mse_resid)
    sigma = np.sqrt(max(1e-12, eps_var) * 2.0 * k / (1.0 - phi ** 2))
    vr_q = max(2, min(round(2 * half_life), len(s) // 4))
    variance_ratio = compute_variance_ratio(s, vr_q)
    return {"valid": True, "alpha": alpha, "phi": phi, "k": k, "mu": mu, "sigma": sigma,
            "half_life": half_life, "r_squared": ar1.rsquared, "p_phi": float(ar1.pvalues[1]), "n": len(s),
            "variance_ratio": variance_ratio, "variance_ratio_q": vr_q}


def ou_selection(kept_pairs):
    sorted_by_drift = sorted(kept_pairs, key=lambda r: abs(r["disc_mean_spread"]))
    top_drift = sorted_by_drift[: max(1, int(OU_DRIFT_PCTL * len(sorted_by_drift)))]
    top_for_ou = sorted(top_drift, key=lambda r: r["disc_var_spread"], reverse=True)[:OU_TOP_N]

    ou_results = []
    for r in top_for_ou:
        res = fit_ou_on_levels(r["disc_spread"], dt=1.0)
        res.update(r)
        ou_results.append(res)

    stage1 = [r for r in ou_results if r.get("valid", False)]
    stage2 = [r for r in stage1 if r["ADF_pvalue"] < 0.05]
    stage3 = [r for r in stage2 if 0.5 < r["phi"] < 0.995]

    bucketed = {}
    for low, high in HALF_LIFE_BUCKETS:
        bucketed[(low, high)] = [r for r in stage3 if low <= r["half_life"] <= high]

    candidates = []
    for (low, high), pairs in bucketed.items():
        candidates.extend([r for r in pairs if r["r_squared"] > R2_THRESHOLD])

    print(f"OU + half-life + R^2 filter: {len(candidates)} candidate pairs (frozen beta/direction)")
    return candidates, stage3


# ============================================================================
# 3. SIGNAL GENERATION
# ============================================================================
def adaptive_window(half_life, max_window=ROLLING_WINDOW, max_std_window=Z_STD_WINDOW, min_window=10, min_std_window=20):
    """Derives a rolling window sized to THIS pair's own reversion speed,
    instead of applying a fixed global window regardless of how fast or
    slow the pair actually is. Motivated directly by evidence: once
    half-life's upper bound was dropped (validated as unnecessary), the
    traded population's median half-life fell to ~3-4 days -- a fixed
    60/120-day window is 15-30x longer than that, too slow to track a
    genuinely fast-reverting spread, and produced a high rate of
    near-immediate stop-losses (39% of trades, ~2.7 days to stop-out on
    average) consistent with the entry signal being computed against a
    stale, lagging baseline. Capped at the existing 60/120 as an upper
    bound, so slow pairs are completely unaffected -- this only shrinks
    the window for fast ones."""
    window = int(np.clip(round(3 * half_life), min_window, max_window))
    std_window = int(np.clip(round(6 * half_life), min_std_window, max_std_window))
    return window, std_window


def entry_still_stationary(spread_vals, i, window=ENTRY_ADF_RECHECK_WINDOW, pvalue_max=ENTRY_ADF_RECHECK_PVALUE_MAX):
    """Re-runs ADF on the trailing `window` days of spread history at a
    potential entry point, independent of the discovery-time cointegration
    test done up to 6 months earlier. Directly targets the temporary-
    deviation-vs-structural-break distinction: a pair can look cointegrated
    at discovery and have since drifted into a trending/broken relationship
    by the time a Z-based entry signal fires. Fails open (allows entry) on
    insufficient data or numerical errors -- this is a confirmation check
    layered on top of the existing entry logic, not a replacement for it."""
    start = max(0, i - window + 1)
    window_vals = spread_vals[start:i + 1]
    window_vals = window_vals[~np.isnan(window_vals)]
    if len(window_vals) < 20:
        return True
    try:
        pval = adfuller(window_vals, autolag="AIC")[1]
    except Exception:
        return True
    return pval <= pvalue_max


def build_spread_and_signals(price_data, dependent, independent, beta, intercept,
                              window=ROLLING_WINDOW, std_window=Z_STD_WINDOW, z_entry=Z_ENTRY, z_exit=Z_EXIT,
                              z_stop=Z_STOP, max_holding_days=MAX_HOLDING_DAYS, cost_bps=TRANSACTION_COST_BPS_PER_LEG):
    df = pd.DataFrame(index=price_data.index)
    df["dep"] = price_data[dependent]
    df["indep"] = price_data[independent]
    df = df.dropna()

    df["Spread"] = df["dep"] - beta * df["indep"] - intercept
    roll_mean = df["Spread"].rolling(window, min_periods=window).mean()
    roll_std = df["Spread"].rolling(std_window, min_periods=window).std()
    df["Z"] = (df["Spread"] - roll_mean) / roll_std

    # Computed here (not just later) so the stateful loop below can track
    # each open position's ACTUAL realized P&L day by day, independent of
    # what Z says -- this is what lets the hard percentage stop below work.
    spread_change_vals = df["Spread"].diff().values
    capital_deployed_vals = (df["dep"].shift(1) + abs(beta) * df["indep"].shift(1)).values

    # Stateful loop (not vectorized) since "days held" and "was this exit a
    # stop-loss vs. a normal reversion" are both path-dependent. Position[t]
    # is decided using Z[t] and is applied to returns realized over [t, t+1]
    # via the shift(1) below, same timing convention as before.
    z_vals = df["Z"].values
    spread_vals = df["Spread"].values
    n = len(z_vals)
    position = np.zeros(n)
    exit_reason = np.array([""] * n, dtype=object)
    entry_wait_days = np.full(n, np.nan)
    current_pos, days_held, trade_cum_return = 0, 0, 0.0
    pending_crossing_i = -1  # day of the first Z-crossing not yet entered on, so we can
    # measure how long WAIT_FOR_Z_PEAK / entry_still_stationary delayed the actual entry

    for i in range(n):
        z = z_vals[i]
        if current_pos != 0:
            # Realized P&L for the position held INTO today, regardless of
            # whether Z is currently readable -- a large price move can
            # still hurt the position even on a day Z itself is NaN.
            day_ret = capital_deployed_vals[i]
            if not np.isnan(spread_change_vals[i]) and day_ret and not np.isnan(day_ret) and day_ret != 0:
                day_ret = current_pos * spread_change_vals[i] / day_ret
                trade_cum_return = (1 + trade_cum_return) * (1 + day_ret) - 1
        if np.isnan(z):
            position[i] = current_pos
            continue
        if current_pos == 0:
            crossed = z > z_entry or z < -z_entry
            if crossed and pending_crossing_i == -1:
                pending_crossing_i = i
            elif not crossed:
                pending_crossing_i = -1  # the opportunity passed without ever entering
            if crossed and entry_still_stationary(spread_vals, i):
                peak_ok = True
                if WAIT_FOR_Z_PEAK:
                    prev_z = z_vals[i - 1] if i > 0 else np.nan
                    peak_ok = not np.isnan(prev_z) and abs(z) < abs(prev_z)
                if peak_ok:
                    current_pos, days_held, trade_cum_return = (-1, 0, 0.0) if z > z_entry else (1, 0, 0.0)
                    entry_wait_days[i] = i - pending_crossing_i
                    pending_crossing_i = -1
        else:
            days_held += 1
            if trade_cum_return <= -MAX_LOSS_PCT_PER_TRADE:
                exit_reason[i] = "hard_stop_pct"
                current_pos, days_held, trade_cum_return = 0, 0, 0.0
            elif -z_exit <= z <= z_exit:
                exit_reason[i] = "reversion"
                current_pos, days_held, trade_cum_return = 0, 0, 0.0
            elif abs(z) >= z_stop:
                exit_reason[i] = "stop_loss"
                current_pos, days_held, trade_cum_return = 0, 0, 0.0
            elif days_held >= max_holding_days:
                exit_reason[i] = "time_stop"
                current_pos, days_held, trade_cum_return = 0, 0, 0.0
        position[i] = current_pos

    df["Position"] = position
    df["Exit_Reason"] = exit_reason
    df["Entry_Wait_Days"] = entry_wait_days

    # Dollar-neutral P&L: mark-to-market change in the actual raw-price
    # spread, scaled by position, divided by the dollar capital required to
    # hold it (1 unit dependent + |beta| units independent, valued at the
    # PRIOR day's prices to avoid look-ahead). Previously this combined a
    # price-level beta with percentage returns (dep_ret - beta*indep_ret),
    # which isn't dollar-neutral and produces distorted Sharpe values.
    spread_change = df["Spread"].diff()
    capital_deployed = df["dep"].shift(1) + abs(beta) * df["indep"].shift(1)
    gross_ret = df["Position"].shift(1) * spread_change / capital_deployed

    turnover = df["Position"].diff().abs().fillna(0)
    cost = turnover * (cost_bps / 10000.0) * 2
    df["Strategy_Return"] = gross_ret - cost
    df["Cumulative_Return"] = (1 + df["Strategy_Return"].fillna(0)).cumprod()
    return df


def extract_trades(df, start=None, end=None, momentum_window=5):
    """Reconstructs individual round-trip trades from a signal dataframe's
    Position/Exit_Reason/Strategy_Return columns, for trade-level P&L
    diagnostics rather than only an aggregate portfolio Sharpe. Also tracks
    entry Z-score, maximum adverse excursion (MAE), maximum favorable
    excursion (MFE) -- the worst and best cumulative mark-to-market P&L
    reached DURING the trade, not just at exit -- and adverse_momentum_at_entry:
    how much Z had moved AGAINST the entry direction over the preceding
    `momentum_window` days. Positive = the spread was still actively moving
    away from the mean at entry ("catching a falling knife"); negative =
    the adverse move had already started reversing before entry."""
    sub = df.loc[start:end] if start is not None else df
    z_momentum = df["Z"] - df["Z"].shift(momentum_window)
    trades = []
    in_trade, entry_date, entry_dir, entry_z, entry_mom, entry_wait, trade_ret = \
        False, None, 0, np.nan, np.nan, np.nan, []
    for date, row in sub.iterrows():
        pos = row["Position"]
        if not in_trade and pos != 0:
            in_trade, entry_date, entry_dir, entry_z, trade_ret = True, date, pos, row["Z"], []
            entry_wait = row.get("Entry_Wait_Days", np.nan)
            raw_mom = z_momentum.loc[date] if date in z_momentum.index else np.nan
            # Normalize by entry direction: positive = Z still moving further
            # in the direction that triggered entry (adverse, still falling);
            # negative = already turning back before we entered.
            entry_mom = float(-pos * raw_mom) if not (isinstance(raw_mom, float) and np.isnan(raw_mom)) else np.nan
        if in_trade:
            if not np.isnan(row["Strategy_Return"]):
                trade_ret.append(row["Strategy_Return"])
            if row["Exit_Reason"] in ("reversion", "stop_loss", "time_stop", "hard_stop_pct"):
                cum_path = np.cumprod([1 + r for r in trade_ret]) - 1 if trade_ret else np.array([0.0])
                cum = float(cum_path[-1]) if len(cum_path) else 0.0
                mfe = float(cum_path.max()) if len(cum_path) else 0.0
                mae = float(cum_path.min()) if len(cum_path) else 0.0
                trades.append({"entry_date": entry_date, "exit_date": date, "direction": entry_dir,
                                "entry_z": round(float(entry_z), 3) if not np.isnan(entry_z) else np.nan,
                                "adverse_momentum_at_entry": round(entry_mom, 4) if not np.isnan(entry_mom) else np.nan,
                                "entry_wait_days": int(entry_wait) if not (isinstance(entry_wait, float) and np.isnan(entry_wait)) else np.nan,
                                "days_held": len(trade_ret), "return_%": round(cum * 100, 3),
                                "mfe_%": round(mfe * 100, 3), "mae_%": round(mae * 100, 3),
                                "exit_reason": row["Exit_Reason"]})
                in_trade = False
    return pd.DataFrame(trades)


def slice_metrics(df, start, end):
    sub = df.loc[start:end, "Strategy_Return"].dropna()
    position_sub = df.loc[start:end, "Position"]
    days_active = int((position_sub != 0).sum())
    pct_active = (days_active / len(position_sub) * 100) if len(position_sub) > 0 else 0.0
    n_trades = int(position_sub.diff().abs().fillna(0).sum() // 2)

    if len(sub) < 5 or sub.std() == 0:
        return {"sharpe_ratio": 0, "max_drawdown": np.nan, "final_return": np.nan, "n_trades": n_trades,
                "days_active": days_active, "pct_active": pct_active, "returns": sub}
    cum = (1 + sub).cumprod()
    dd = (cum.cummax() - cum) / cum.cummax()
    sharpe = sub.mean() / sub.std() * np.sqrt(252)
    return {
        "sharpe_ratio": sharpe, "max_drawdown": dd.max(), "final_return": cum.iloc[-1],
        "n_trades": n_trades, "days_active": days_active, "pct_active": pct_active, "returns": sub,
    }


# ============================================================================
# 4. PORTFOLIO CONSTRUCTION
# ============================================================================
def compute_weights(returns_df):
    n = returns_df.shape[1]
    daily_ret = returns_df.mean()
    daily_std = returns_df.std()
    sharpe = (daily_ret / daily_std * np.sqrt(252)).fillna(0)

    w_equal = np.ones(n) / n
    sharpe_vals = sharpe.values.copy()
    sharpe_vals[sharpe_vals < 0] = 0
    w_sharpe = sharpe_vals / sharpe_vals.sum() if sharpe_vals.sum() > 0 else w_equal

    vol_vals = daily_std.values
    w_invvol = (1 / vol_vals) / (1 / vol_vals).sum()

    cov = returns_df.cov() * 252
    def risk_parity_obj(w):
        port_vol = np.sqrt(w @ cov.values @ w)
        if port_vol == 0:
            return 1e6
        risk_contrib = w * (cov.values @ w) / port_vol
        target = port_vol / len(w)
        return np.sum((risk_contrib - target) ** 2)

    res = minimize(risk_parity_obj, np.ones(n) / n, method="SLSQP",
                    bounds=tuple((0, 1) for _ in range(n)),
                    constraints={"type": "eq", "fun": lambda x: np.sum(x) - 1})
    w_riskpar = res.x if res.success else w_equal

    return {"Equal Weight": w_equal, "Sharpe-Weighted": w_sharpe,
            "Inv Volatility": w_invvol, "Risk Parity": w_riskpar}


def portfolio_metrics(weights, returns_df):
    port_ret = returns_df.values @ weights
    port_ret = pd.Series(port_ret, index=returns_df.index)
    daily_ret, daily_std = port_ret.mean(), port_ret.std()
    ann_ret, ann_std = daily_ret * 252, daily_std * np.sqrt(252)
    sharpe = ann_ret / ann_std if ann_std > 0 else 0
    cum = (1 + port_ret).cumprod()
    dd = (cum.cummax() - cum) / cum.cummax()
    max_dd = dd.max()
    downside = port_ret[port_ret < 0]
    downside_std = downside.std() * np.sqrt(252) if len(downside) > 1 else np.nan
    sortino = ann_ret / downside_std if downside_std and downside_std > 0 else np.nan
    calmar = ann_ret / max_dd if max_dd > 0 else np.nan
    return {"Ann_Ret_%": ann_ret * 100, "Ann_Vol_%": ann_std * 100, "Sharpe": sharpe,
            "Sortino": sortino, "Max_DD_%": max_dd * 100, "Calmar": calmar, "cum_ret": cum}


# ============================================================================
# 5. ITERATION LOGGING
# ============================================================================
def save_frozen_model(candidates, passed, best_method, best_weights, path=os.path.join(OUTPUT_DIR, "frozen_model.json")):
    """Persists the exact frozen state needed to run OOS later: each passed
    pair's dependent/independent tickers, hedge ratio (beta), intercept,
    regression direction, and the chosen portfolio weights. This is the
    complete, unambiguous definition of the strategy at the moment the
    framework was frozen — the OOS script consumes this file and nothing
    else about how the pairs were chosen."""
    by_name = {f"{r['dependent']}-{r['independent']}": r for r in candidates}
    weights = dict(zip(passed, [float(w) for w in best_weights]))

    model = {
        "frozen_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "discovery_start": DATA_START,
        "discovery_end": DISCOVERY_END,
        "selection_end": SELECTION_END,
        "weighting_method": best_method,
        "rolling_window": ROLLING_WINDOW,
        "z_entry": Z_ENTRY,
        "z_exit": Z_EXIT,
        "transaction_cost_bps_per_leg": TRANSACTION_COST_BPS_PER_LEG,
        "pairs": [
            {
                "name": name,
                "dependent": by_name[name]["dependent"],
                "independent": by_name[name]["independent"],
                "beta": by_name[name]["beta"],
                "intercept": by_name[name]["intercept"],
                "weight": weights[name],
            }
            for name in passed
        ],
    }
    with open(path, "w") as f:
        json.dump(model, f, indent=2, default=str)
    print(f"\nFrozen model saved to {path} — this is the exact state the OOS script will load and trade.")


def current_config_snapshot():
    return {
        "DATA_START": DATA_START, "DISCOVERY_END": DISCOVERY_END, "SELECTION_END": SELECTION_END,
        "MIN_DATA_POINTS": MIN_DATA_POINTS, "DISTANCE_SELECT_PERCENTILE": DISTANCE_SELECT_PERCENTILE,
        "COINT_PVALUE": COINT_PVALUE, "ZERO_CROSSING_MIN_PERCENTILE": ZERO_CROSSING_MIN_PERCENTILE,
        "OU_DRIFT_PCTL": OU_DRIFT_PCTL, "OU_TOP_N": OU_TOP_N, "HALF_LIFE_BUCKETS": HALF_LIFE_BUCKETS,
        "R2_THRESHOLD": R2_THRESHOLD, "ROLLING_WINDOW": ROLLING_WINDOW, "Z_ENTRY": Z_ENTRY, "Z_EXIT": Z_EXIT,
        "PERF_SHARPE_MIN": PERF_SHARPE_MIN, "PERF_MAXDD_MAX": PERF_MAXDD_MAX,
        "PERF_RETURN_MIN_ANNUALIZED": PERF_RETURN_MIN_ANNUALIZED,
        "TRANSACTION_COST_BPS_PER_LEG": TRANSACTION_COST_BPS_PER_LEG,
    }


def log_iteration(record):
    record = dict(record)
    record["timestamp"] = datetime.datetime.now().isoformat(timespec="seconds")
    record["config"] = current_config_snapshot()
    with open(ITERATION_LOG_PATH, "a") as f:
        f.write(json.dumps(record, default=str) + "\n")
    print(f"\nLogged this run to {ITERATION_LOG_PATH}")


def print_iteration_history():
    if not os.path.exists(ITERATION_LOG_PATH):
        print("\nNo prior iterations logged yet — this is the first run.")
        return

    rows = []
    with open(ITERATION_LOG_PATH) as f:
        for i, line in enumerate(f):
            rec = json.loads(line)
            cfg = rec.get("config", {})
            rows.append({
                "Run": i + 1,
                "Timestamp": rec.get("timestamp", "")[:19],
                "Dist %ile": cfg.get("DISTANCE_SELECT_PERCENTILE"),
                "Coint p": cfg.get("COINT_PVALUE"),
                "ZeroCross %ile": cfg.get("ZERO_CROSSING_MIN_PERCENTILE"),
                "R2 Min": cfg.get("R2_THRESHOLD"),
                "Sharpe Min": cfg.get("PERF_SHARPE_MIN"),
                "Candidates": rec.get("n_ou_candidates"),
                "Passed": rec.get("n_passed_selection"),
                "Best Method": rec.get("weighting_method", "-"),
                "Sel. Sharpe": rec.get("selection_sharpe", "-"),
            })
    print("\n" + "=" * 80)
    print(f"ITERATION HISTORY  ({len(rows)} run(s) logged to {ITERATION_LOG_PATH})")
    print("=" * 80)
    print(tabulate.tabulate(pd.DataFrame(rows), headers="keys", tablefmt="pretty", showindex=False))


# ============================================================================
# 6. CHARTS
# ============================================================================
def save_and_show(name):
    os.makedirs(CHART_DIR, exist_ok=True)
    plt.tight_layout()
    plt.savefig(os.path.join(CHART_DIR, name), dpi=150, bbox_inches="tight")
    plt.show()


def plot_discovery_diagnostics(distances, zero_crossings, zc_threshold_used, half_lives):
    plt.figure(figsize=(10, 6))
    plt.hist(distances, bins=100, color="steelblue", edgecolor="black")
    cutoff = np.percentile(distances, DISTANCE_SELECT_PERCENTILE)
    plt.axvline(cutoff, color="red", linestyle="--",
                label=f"closest {DISTANCE_SELECT_PERCENTILE}% cutoff = {cutoff:.1f}")
    plt.title(f"Histogram of SSD Distances (Discovery: {DATA_START} to {DISCOVERY_END})")
    plt.xlabel("SSD Distance")
    plt.ylabel("Frequency")
    plt.legend()
    save_and_show("01_ssd_distance_histogram.png")

    plt.figure(figsize=(10, 6))
    plt.hist(zero_crossings, bins=30, color="skyblue", edgecolor="black")
    plt.axvline(zc_threshold_used, color="red", linestyle="--",
                label=f"threshold = {zc_threshold_used:.0f} (bottom {ZERO_CROSSING_MIN_PERCENTILE}% dropped)")
    plt.title("Histogram of Zero Crossings in Spread")
    plt.xlabel("Number of Zero Crossings")
    plt.ylabel("Frequency")
    plt.legend()
    save_and_show("02_zero_crossings_histogram.png")

    if half_lives:
        plt.figure(figsize=(10, 6))
        plt.hist(half_lives, bins=30, color="mediumseagreen", edgecolor="black")
        plt.title("Distribution of OU Half-Lives")
        plt.xlabel("Half-life (trading days)")
        plt.ylabel("Count")
        save_and_show("03_half_life_histogram.png")


def plot_pair_correlation(sel_returns_df):
    if sel_returns_df.shape[1] < 2:
        return
    corr = sel_returns_df.corr()
    plt.figure(figsize=(max(8, corr.shape[0]), max(6, corr.shape[0] * 0.8)))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdYlGn", center=0, square=True,
                linewidths=0.5, cbar_kws={"shrink": 0.8})
    plt.title("Candidate Pair Returns Correlation Matrix (Selection window)")
    save_and_show("04_pair_correlation_heatmap.png")


def plot_weight_comparison(weight_schemes, pair_names):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for idx, (wname, w) in enumerate(weight_schemes.items()):
        ax = axes[idx // 2, idx % 2]
        order = np.argsort(w)[::-1]
        ax.barh(range(len(w)), np.array(w)[order] * 100, color="steelblue")
        ax.set_yticks(range(len(w)))
        ax.set_yticklabels([pair_names[k] for k in order], fontsize=8)
        ax.set_xlabel("Weight (%)")
        ax.set_title(wname)
        ax.grid(alpha=0.3, axis="x")
    fig.suptitle("Portfolio Weighting Schemes (Selection window)", y=1.02)
    save_and_show("05_weight_comparison.png")


def plot_selection_results(port_metrics, best_method):
    cum = port_metrics["cum_ret"]
    dd = (cum.cummax() - cum) / cum.cummax()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    ax1.plot(cum.index, cum.values, linewidth=2, color="steelblue")
    ax1.set_title(f"Selection-Window Portfolio Cumulative Return ({best_method})")
    ax1.set_ylabel("Cumulative Return")
    ax1.grid(alpha=0.3)
    ax2.plot(dd.index, -dd.values * 100, linewidth=1.5, color="firebrick")
    ax2.set_title("Selection-Window Portfolio Drawdown")
    ax2.set_ylabel("Drawdown (%)")
    ax2.set_xlabel("Date")
    ax2.grid(alpha=0.3)
    save_and_show("06_selection_cumulative_return_and_drawdown.png")


# ============================================================================
# 6b. OUT-OF-SAMPLE EVALUATION  (only runs if RUN_OOS_EVALUATION = True)
# ============================================================================
def download_oos_data(tickers, start=OOS_START, end=OOS_END):
    """The only data download in this file that isn't capped at DOWNLOAD_END —
    reached only when RUN_OOS_EVALUATION is explicitly True, and only for the
    tickers in the already-frozen pair list, never the full universe."""
    data = yf.download(tickers, start=start, end=end, interval="1d")["Close"]
    data = data.dropna(how="all")
    fetched = set(data.columns) if not isinstance(data.columns, pd.MultiIndex) else set(data.columns.levels[1])
    missing = [t for t in tickers if t not in fetched]
    print(f"\nOOS data downloaded ({start} to {end}, exclusive) for {len(fetched)} / {len(tickers)} tickers. "
          f"Missing: {missing if missing else 'none'}")
    return data


def plot_oos_results(port_metrics):
    cum = port_metrics["cum_ret"]
    dd = (cum.cummax() - cum) / cum.cummax()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    ax1.plot(cum.index, cum.values, linewidth=2, color="darkgreen")
    ax1.axhline(1.0, color="gray", linestyle=":", alpha=0.7)
    ax1.set_title("OOS Portfolio Cumulative Return (frozen pairs/betas/weights)")
    ax1.set_ylabel("Cumulative Return")
    ax1.grid(alpha=0.3)
    ax2.plot(dd.index, -dd.values * 100, linewidth=1.5, color="firebrick")
    ax2.set_title("OOS Portfolio Drawdown")
    ax2.set_ylabel("Drawdown (%)")
    ax2.set_xlabel("Date")
    ax2.grid(alpha=0.3)
    save_and_show("oos_01_cumulative_return_and_drawdown.png")


def plot_oos_correlation(returns_df):
    if returns_df.shape[1] < 2:
        return
    corr = returns_df.corr()
    plt.figure(figsize=(max(8, corr.shape[0]), max(6, corr.shape[0] * 0.8)))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdYlGn", center=0, square=True,
                linewidths=0.5, cbar_kws={"shrink": 0.8})
    plt.title("OOS Pair Returns Correlation Matrix")
    save_and_show("oos_02_pair_correlation_heatmap.png")


def load_frozen_model(path=os.path.join(OUTPUT_DIR, "frozen_model.json")):
    """Counterpart to save_frozen_model — reconstructs the candidates/passed/
    weights structures run_oos_evaluation expects, straight from disk, with
    no Discovery/Selection computation at all."""
    with open(path) as f:
        model = json.load(f)
    print(f"Loaded frozen model from {path}, frozen at {model['frozen_at']}")
    print(f"Discovery: {model['discovery_start']} -> {model['discovery_end']}   "
          f"Selection: {model['discovery_end']} -> {model['selection_end']}")
    print(f"Weighting method: {model['weighting_method']}")

    candidates = [
        {"dependent": p["dependent"], "independent": p["independent"],
         "beta": p["beta"], "intercept": p["intercept"]}
        for p in model["pairs"]
    ]
    passed = [p["name"] for p in model["pairs"]]
    best_weights = np.array([p["weight"] for p in model["pairs"]])
    for p in model["pairs"]:
        print(f"  {p['name']:<20} beta={p['beta']:.4f}  intercept={p['intercept']:.4f}  weight={p['weight']:.4f}")
    return candidates, passed, best_weights


def run_oos_only():
    """Entry point for SKIP_DISCOVERY_SELECTION=True: loads the saved frozen
    model and runs OOS directly, without recomputing Discovery/Selection."""
    print("SKIP_DISCOVERY_SELECTION is True — loading saved frozen model, "
          "Discovery/Selection will not be recomputed.\n")
    candidates, passed, best_weights = load_frozen_model()
    return run_oos_evaluation(candidates, passed, best_weights)



def run_oos_evaluation(candidates, passed, best_weights):
    """Applies the frozen pairs/betas/weights, unchanged, to the OOS window.
    No fitting, filtering, or selection happens here — every beta, intercept,
    and weight comes from what Discovery/Selection already decided."""
    print("\n" + "=" * 80)
    print(f"OOS EVALUATION  ({OOS_START} to {OOS_END})  — RUN_OOS_EVALUATION is True")
    print("=" * 80)

    by_name = {f"{r['dependent']}-{r['independent']}": r for r in candidates}
    frozen_weights = dict(zip(passed, [float(w) for w in best_weights]))

    tickers = sorted({t for name in passed for t in (by_name[name]["dependent"], by_name[name]["independent"])})
    data = download_oos_data(tickers)

    pair_dfs, rows, usable = {}, [], []
    for name in passed:
        r = by_name[name]
        if r["dependent"] not in data.columns or r["independent"] not in data.columns:
            print(f"Skipping {name}: missing OOS price data for one or both legs.")
            continue
        df = build_spread_and_signals(data, r["dependent"], r["independent"], r["beta"], r["intercept"])
        pair_dfs[name] = df
        m = slice_metrics(df, data.index.min(), data.index.max())
        rows.append({
            "Pair": name, "Frozen_Weight": round(frozen_weights[name], 4),
            "Sharpe": round(m["sharpe_ratio"], 2),
            "Max_DD_%": round(m["max_drawdown"] * 100, 2) if not np.isnan(m["max_drawdown"]) else float("nan"),
            "Return_%": round((m["final_return"] - 1) * 100, 2) if not np.isnan(m["final_return"]) else float("nan"),
            "N_Trades": m["n_trades"], "Days_Active": m["days_active"], "%_Active": round(m["pct_active"], 1),
        })
        usable.append(name)

    pair_table = pd.DataFrame(rows).sort_values("Sharpe", ascending=False)
    print(f"\nPer-pair OOS performance ({len(usable)} / {len(passed)} pairs usable):")
    print(tabulate.tabulate(pair_table, headers="keys", tablefmt="pretty", showindex=False))

    if not usable:
        print("\nNo pairs had usable OOS data. Stopping.")
        return

    raw_weights = np.array([frozen_weights[name] for name in usable])
    weights = raw_weights / raw_weights.sum()
    if len(usable) < len(passed):
        print(f"\nNOTE: {len(passed) - len(usable)} pair(s) dropped for missing data; "
              f"remaining weights renormalized to sum to 1.")

    returns_df = pd.DataFrame({name: pair_dfs[name]["Strategy_Return"].fillna(0) for name in usable})
    port = portfolio_metrics(weights, returns_df)

    print("\n" + "=" * 80)
    print(f"OOS RESULT  ({OOS_START} to {OOS_END})")
    print("=" * 80)
    print(f"Ann. Return: {port['Ann_Ret_%']:.2f}%   Ann. Vol: {port['Ann_Vol_%']:.2f}%   "
          f"Sharpe: {port['Sharpe']:.3f}   Max DD: {port['Max_DD_%']:.2f}%   "
          f"Sortino: {port['Sortino']:.3f}   Calmar: {port['Calmar']:.3f}")
    print(f"\nFirst ~{ROLLING_WINDOW} trading days are a rolling-window warm-up with no trades — expected, not a bug.")
    print("This number is the honest answer: no OOS data was used to choose these pairs, hedge ratios, or "
          "weights. Whatever it says, it should not be re-tuned against.")

    plot_oos_results(port)
    plot_oos_correlation(returns_df)

    with open(os.path.join(OUTPUT_DIR, "oos_results.json"), "w") as f:
        json.dump({
            "oos_start": OOS_START, "oos_end": OOS_END, "n_pairs_usable": len(usable),
            "ann_return_%": round(port["Ann_Ret_%"], 2), "ann_vol_%": round(port["Ann_Vol_%"], 2),
            "sharpe": round(port["Sharpe"], 3), "max_dd_%": round(port["Max_DD_%"], 2),
            "sortino": round(port["Sortino"], 3), "calmar": round(port["Calmar"], 3),
            "pair_table": pair_table.to_dict(orient="records"),
        }, f, indent=2, default=str)
    print(f"\nSaved to {os.path.join(OUTPUT_DIR, 'oos_results.json')}")

    return {"port_metrics": port, "pair_table": pair_table}


# ============================================================================
# 7. MAIN
# ============================================================================
# ============================================================================
# 6c. THRESHOLD SENSITIVITY SWEEP  (only runs if RUN_THRESHOLD_SWEEP = True)
# ============================================================================
# The point of this section: every "quality" threshold we've been hand-tuning
# (zero-crossing cutoff, half-life range, R^2, the Selection performance
# filters) is applied DOWNSTREAM of the two expensive stages — cointegration
# testing and regression/ADF fitting. Those two stages are run exactly ONCE
# here, for every cointegrated pair, uncapped (no OU_TOP_N-style shortcut).
# Every downstream statistic (zero-crossings, OU fit, Selection-window
# Sharpe/DD/Return) is computed once per pair and cached. Sweeping hundreds
# of threshold combinations after that is pure pandas filtering over an
# already-computed table — milliseconds per combination, not another
# statsmodels call. This is what makes checking for a stable "plateau" of
# good threshold values (vs. a fragile single spike) tractable at all.
def compute_coint_persistence(norm_disc, cointegrated_pairs, n_windows=N_PERSISTENCE_WINDOWS, n_jobs=N_JOBS):
    """Cointegration is a statement about one specific window, not a fixed
    property of a pair — a real relationship tends to test significant across
    multiple sub-windows, while a spurious one (expected by chance at the
    ~5% rate, given how many pairs get tested) usually doesn't repeat. This
    checks that directly by re-testing cointegration on N_WINDOWS sequential
    slices of the Discovery period.

    COST CONTROL: only applied to the pairs that already passed the FULL-window
    cointegration test (~3,300 last run), never the full ~10,557-pair
    distance-filtered pool — that keeps the added cost proportional to a
    fraction of the main bottleneck rather than multiplying it. Each
    additional window adds roughly one more parallelized pass over that
    already-small subset."""
    if n_windows <= 1:
        return {(t1, t2): 1.0 for t1, t2, dist, pval in cointegrated_pairs}

    n = len(norm_disc)
    bounds = np.linspace(0, n, n_windows + 1).astype(int)

    tasks, task_keys = [], []
    for t1, t2, dist, pval in cointegrated_pairs:
        s1, s2 = norm_disc[t1].values, norm_disc[t2].values
        for w in range(n_windows):
            lo, hi = bounds[w], bounds[w + 1]
            tasks.append((t1, t2, dist, s1[lo:hi], s2[lo:hi], COINT_PVALUE))
            task_keys.append((t1, t2))

    if n_jobs <= 1 or len(tasks) < 200:
        results = [_coint_test_one(t) for t in tasks]
    else:
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            results = list(executor.map(_coint_test_one, tasks, chunksize=200))

    sig_count = {}
    for key, res in zip(task_keys, results):
        sig_count[key] = sig_count.get(key, 0) + (1 if res is not None else 0)

    persistence = {key: cnt / n_windows for key, cnt in sig_count.items()}
    print(f"Cointegration persistence: {len(cointegrated_pairs)} pairs re-tested across {n_windows} "
          f"sequential sub-windows of Discovery  [{len(tasks)} additional tests across {n_jobs} process(es)]")
    return persistence


def compute_full_pair_statistics(discovery_data, signal_data, selection_start, selection_end):
    close_pairs, norm_disc, distances = distance_prefilter(discovery_data)
    # BUG FIX: cointegration/regression previously ran on norm_disc (the
    # min-max [0,1] normalized series built only for the distance/SSD
    # calculation), while trading later applied the resulting beta/intercept
    # to RAW prices. Normalization is a per-ticker linear rescaling, so a
    # beta fit in that space is not the correct raw-price hedge ratio --
    # applying it directly to raw prices produces a distorted spread.
    # Cointegration/regression now use discovery_data (raw prices) directly;
    # norm_disc is used ONLY for distance, above.
    cointegrated = cointegration_filter(discovery_data, close_pairs)
    reg_results = fit_direction_and_beta(discovery_data, cointegrated)
    persistence = compute_coint_persistence(discovery_data, cointegrated)

    print(f"\nComputing full per-pair statistics for all {len(reg_results)} cointegrated pairs "
          f"(zero-crossings, OU fit, Selection-window performance)...")
    rows = []
    for r in reg_results:
        s = r["disc_spread"].dropna()
        zero_crossings = count_zero_crossings(s)
        ou = fit_ou_on_levels(r["disc_spread"], dt=1.0)
        if not ou.get("valid", False):
            continue

        name = f"{r['dependent']}-{r['independent']}"
        df = build_spread_and_signals(signal_data, r["dependent"], r["independent"], r["beta"], r["intercept"])
        m = slice_metrics(df, selection_start, selection_end)

        rows.append({
            "name": name, "dependent": r["dependent"], "independent": r["independent"],
            "beta": r["beta"], "intercept": r["intercept"], "distance": r["distance"],
            "cointegration_p": r["cointegration_p"], "zero_crossings": zero_crossings,
            "phi": ou["phi"], "half_life": ou["half_life"], "ou_r_squared": ou["r_squared"],
            "coint_adf_p": r["ADF_pvalue"], "coint_persistence": persistence.get(r["pair"], 0.0),
            "sel_sharpe": m["sharpe_ratio"], "sel_maxdd": m["max_drawdown"], "sel_return": m["final_return"],
            "sel_n_trades": m["n_trades"], "sel_days_active": m["days_active"], "sel_pct_active": m["pct_active"],
            "_sel_returns": df.loc[selection_start:selection_end, "Strategy_Return"].fillna(0),
        })
    master_df = pd.DataFrame(rows)
    print(f"Master statistics table built: {len(master_df)} pairs with valid OU fits.")
    return master_df


def apply_thresholds(master_df, zero_crossing_min_percentile, half_life_min, half_life_max,
                      r2_threshold, perf_sharpe_min, perf_maxdd_max, perf_return_min, min_persistence=0.0):
    if len(master_df) == 0:
        return {"n_candidates": 0, "n_passed": 0, "portfolio_sharpe": np.nan,
                "portfolio_return_%": np.nan, "portfolio_maxdd_%": np.nan}

    zc_cutoff = np.percentile(master_df["zero_crossings"], zero_crossing_min_percentile)
    candidates = master_df[
        (master_df["zero_crossings"] >= zc_cutoff)
        & (master_df["phi"] > 0.5) & (master_df["phi"] < 0.995)
        & (master_df["coint_adf_p"] < 0.05)
        & (master_df["half_life"] >= half_life_min) & (master_df["half_life"] <= half_life_max)
        & (master_df["ou_r_squared"] > r2_threshold)
        & (master_df["coint_persistence"] >= min_persistence)
    ]
    passed = candidates[
        (candidates["sel_sharpe"] > perf_sharpe_min)
        & (candidates["sel_maxdd"] < perf_maxdd_max)
        & (candidates["sel_return"] > perf_return_min)
    ]

    result = {"n_candidates": len(candidates), "n_passed": len(passed),
              "portfolio_sharpe": np.nan, "portfolio_return_%": np.nan, "portfolio_maxdd_%": np.nan}
    if len(passed) == 0:
        return result

    returns_df = pd.DataFrame({row["name"]: row["_sel_returns"] for _, row in passed.iterrows()})
    equal_weights = np.ones(len(passed)) / len(passed)
    port = portfolio_metrics(equal_weights, returns_df)
    result.update({"portfolio_sharpe": round(port["Sharpe"], 3),
                    "portfolio_return_%": round(port["Ann_Ret_%"], 2),
                    "portfolio_maxdd_%": round(port["Max_DD_%"], 2)})
    return result


def run_threshold_sweep(master_df):
    """1-D sweeps: vary one parameter across a range while holding the rest at
    the current CONFIG defaults, so you can see whether each threshold has a
    stable plateau of good values or a fragile spike at one number."""
    baseline = dict(
        zero_crossing_min_percentile=ZERO_CROSSING_MIN_PERCENTILE,
        half_life_min=HALF_LIFE_BUCKETS[0][0], half_life_max=HALF_LIFE_BUCKETS[-1][1],
        r2_threshold=R2_THRESHOLD, perf_sharpe_min=PERF_SHARPE_MIN,
        perf_maxdd_max=PERF_MAXDD_MAX, perf_return_min=PERF_RETURN_MIN,
        min_persistence=0.0,
    )
    sweeps = {
        "zero_crossing_min_percentile": [5, 10, 15, 20, 25, 30, 35, 40],
        "r2_threshold": [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
        "half_life_max": [30, 40, 50, 60, 70, 80, 90, 100],
        "perf_sharpe_min": [0.0, 0.25, 0.5, 0.75, 1.0, 1.25],
        "perf_maxdd_max": [0.10, 0.15, 0.20, 0.25, 0.30],
        "perf_return_min": [1.00, 1.05, 1.10, 1.15, 1.20, 1.25],
        "min_persistence": [0.0, 0.5, 1.0][:N_PERSISTENCE_WINDOWS + 1],
    }

    print("\n" + "=" * 80)
    print(f"THRESHOLD SENSITIVITY SWEEP  (baseline: {baseline})")
    print("=" * 80)
    all_sweep_results = {}
    for param, values in sweeps.items():
        rows = []
        for v in values:
            params = dict(baseline)
            params[param] = v
            rows.append({param: v, **apply_thresholds(master_df, **params)})
        sweep_df = pd.DataFrame(rows)
        all_sweep_results[param] = sweep_df
        print(f"\n--- Sweep: {param} (all other thresholds held at baseline) ---")
        print(tabulate.tabulate(sweep_df, headers="keys", tablefmt="pretty", showindex=False))

    print(
        "\nHow to read this: look for a PLATEAU of stable portfolio_sharpe across several adjacent\n"
        "values, not a single spike. A threshold whose neighboring values swing wildly is more likely\n"
        "fitting noise than a real effect — prefer picking a value from the middle of a stable range\n"
        "over whichever single value scored highest."
    )

    with open(os.path.join(OUTPUT_DIR, "threshold_sweep.json"), "w") as f:
        json.dump({p: df.to_dict(orient="records") for p, df in all_sweep_results.items()}, f, indent=2, default=str)
    print(f"\nSaved to {os.path.join(OUTPUT_DIR, 'threshold_sweep.json')}")

    return all_sweep_results



def run_sweep_mode():
    """Entry point for RUN_THRESHOLD_SWEEP=True: downloads Discovery/Selection
    data exactly like main() does, runs the expensive stages once, then
    sweeps threshold combinations against the cached results."""
    print(f"Discovery window:  {DATA_START} -> {DISCOVERY_END}")
    print(f"Selection window:  {DISCOVERY_END} -> {SELECTION_END}")
    print("OOS window is NOT downloaded by this mode either.\n")

    tickers = get_sp500_tickers_from_excel()
    print(f"Loaded {len(tickers)} tickers from Excel.")
    data = download_universe(tickers)

    discovery_data = data.loc[DATA_START:DISCOVERY_END]
    selection_start = pd.Timestamp(DISCOVERY_END) + pd.Timedelta(days=1)
    selection_end = pd.Timestamp(SELECTION_END)

    master_df = compute_full_pair_statistics(discovery_data, data, selection_start, selection_end)
    return run_threshold_sweep(master_df)


# ============================================================================
# PHASE 1: FOLD-BASED THRESHOLD RANGE VALIDATION
# ============================================================================
def print_pair_diagnostic_table(df, name_col, p_col, beta_col, sharpe_col, n_trades_col, avg_trade_col,
                                 stop_loss_col, time_stop_col, label, top_n=30, csv_path=None):
    """Requested diagnostic format: Pair | Discovery p-value | Beta | Selection
    Sharpe | Trades | Avg Trade | Stop Loss % | Time Stop %. Full table saved
    to CSV (can run to thousands of rows); console shows the top N by Sharpe
    to stay readable."""
    if len(df) == 0:
        print(f"\n{label}: no pairs to report.")
        return
    table = pd.DataFrame({
        "Pair": df[name_col], "Discovery_p": df[p_col].round(4), "Beta": df[beta_col].round(4),
        "Sel_Sharpe": df[sharpe_col].round(3), "Trades": df[n_trades_col],
        "Avg_Trade_%": df[avg_trade_col].round(2), "Stop_Loss_%": df[stop_loss_col].round(1),
        "Time_Stop_%": df[time_stop_col].round(1),
    }).sort_values("Sel_Sharpe", ascending=False)
    print(f"\n{label}  (showing top {min(top_n, len(table))} of {len(table)} by Selection Sharpe)")
    print(tabulate.tabulate(table.head(top_n), headers="keys", tablefmt="pretty", showindex=False))
    if csv_path:
        table.to_csv(csv_path, index=False)
        print(f"Full table ({len(table)} pairs) saved to {csv_path}")


def compute_fold_master_table(data, discovery_start, eval_start, fold_end):
    """Same idea as compute_full_pair_statistics, but: (a) scoped to one fold's
    short sub-discovery window instead of the full 9.5 years, and (b) distance
    is prefiltered at GENEROUS_DISTANCE_PERCENTILE with each pair's percentile
    rank stored, so narrower distance percentiles can be swept for free
    downstream exactly like the other parameters."""
    discovery_start, eval_start, fold_end = pd.Timestamp(discovery_start), pd.Timestamp(eval_start), pd.Timestamp(fold_end)
    discovery_data = data.loc[discovery_start:eval_start]
    signal_data = data.loc[discovery_start:fold_end]

    close_pairs, norm_disc, distances = distance_prefilter(discovery_data, percentile=GENEROUS_DISTANCE_PERCENTILE)
    # BUG FIX: see compute_full_pair_statistics -- cointegration/regression
    # now use raw discovery_data, not the normalized norm_disc (which is
    # kept strictly for the distance/SSD calculation above).
    cointegrated = cointegration_filter(discovery_data, close_pairs, pvalue=FOLD_COINT_PVALUE)
    reg_results = fit_direction_and_beta(discovery_data, cointegrated)
    persistence = compute_coint_persistence(discovery_data, cointegrated)

    rows = []
    for r in reg_results:
        s = r["disc_spread"].dropna()
        zero_crossings = count_zero_crossings(s)
        ou = fit_ou_on_levels(r["disc_spread"], dt=1.0)
        if not ou.get("valid", False):
            continue
        name = f"{r['dependent']}-{r['independent']}"
        win, std_win = adaptive_window(ou["half_life"])
        df = build_spread_and_signals(signal_data, r["dependent"], r["independent"], r["beta"], r["intercept"],
                                       window=win, std_window=std_win)
        m = slice_metrics(df, eval_start, fold_end)
        pair_trades = extract_trades(df, start=eval_start, end=fold_end)  # reuses the df already built above
        n_trades = len(pair_trades)
        avg_trade_pct = pair_trades["return_%"].mean() if n_trades else np.nan
        stop_loss_pct = (pair_trades["exit_reason"] == "stop_loss").mean() * 100 if n_trades else np.nan
        time_stop_pct = (pair_trades["exit_reason"] == "time_stop").mean() * 100 if n_trades else np.nan
        rows.append({
            "name": name, "dependent": r["dependent"], "independent": r["independent"],
            "beta": r["beta"], "intercept": r["intercept"], "distance": r["distance"],
            "cointegration_p": r["cointegration_p"],
            "coint_adf_p": r["ADF_pvalue"], "coint_adf_lags": r["ADF_lags_used"], "zero_crossings": zero_crossings,
            "phi": ou["phi"], "half_life": ou["half_life"], "ou_r_squared": ou["r_squared"],
            "variance_ratio": ou["variance_ratio"], "variance_ratio_q": ou["variance_ratio_q"],
            "coint_persistence": persistence.get(r["pair"], 0.0),
            "sel_sharpe": m["sharpe_ratio"], "sel_maxdd": m["max_drawdown"], "sel_return": m["final_return"],
            "n_trades": n_trades, "avg_trade_pct": avg_trade_pct,
            "stop_loss_pct": stop_loss_pct, "time_stop_pct": time_stop_pct,
            "_sel_returns": df.loc[eval_start:fold_end, "Strategy_Return"].fillna(0),
        })
    master_df = pd.DataFrame(rows)
    if len(master_df):
        master_df["distance_pctile"] = master_df["distance"].rank(pct=True) * 100
    print(f"Fold master table: {len(master_df)} pairs with valid OU fits "
          f"(discovery {discovery_start.date()}-{eval_start.date()}, eval to {fold_end.date()})")

    if len(master_df):
        fold_tag = f"{discovery_start.date()}_{fold_end.date()}"
        print_pair_diagnostic_table(
            master_df, "name", "cointegration_p", "beta", "sel_sharpe", "n_trades", "avg_trade_pct",
            "stop_loss_pct", "time_stop_pct",
            label=f"PHASE 1 PER-PAIR DIAGNOSTIC ({discovery_start.date()} -> {fold_end.date()})",
            csv_path=os.path.join(OUTPUT_DIR, f"phase1_pair_diagnostic_{fold_tag}.csv"),
        )

    # Diagnostic: FOLD_SWEEP_GRID's tested ranges were carried over from
    # earlier design work and never checked against what this fold's actual
    # candidates look like. If a real run shows a parameter's observed range
    # sitting mostly outside its grid's tested values, that's the grid being
    # miscalibrated for this fold, not a genuine absence of a stable
    # threshold -- print the real distribution so this is visible directly
    # instead of inferred from an all-None sweep result.
    if len(master_df):
        vals = master_df["half_life"]
        print(f"  Observed half_life: min={vals.min():.2f} p10={vals.quantile(0.1):.2f} "
              f"median={vals.median():.2f} p90={vals.quantile(0.9):.2f} max={vals.max():.2f}  "
              f"[half_life_pctile_lo/hi grid now self-calibrates to this distribution directly]")

        # r2_min_percentile's grid tests PERCENTILE CUTOFFS, not raw R2 values
        # -- show what raw R2 value each percentile boundary actually maps to.
        r2 = master_df["ou_r_squared"]
        r2_grid_lo, r2_grid_hi = min(FOLD_SWEEP_GRID["r2_min_percentile"]), max(FOLD_SWEEP_GRID["r2_min_percentile"])
        print(f"  Observed ou_r_squared: min={r2.min():.3f} median={r2.median():.3f} max={r2.max():.3f}  "
              f"[r2_min_percentile grid tests {r2_grid_lo}-{r2_grid_hi} percentile, "
              f"i.e. R2 cutoffs {r2.quantile(r2_grid_lo/100):.3f}-{r2.quantile(r2_grid_hi/100):.3f}]")

        # zero_crossing_min_percentile's grid tests PERCENTILE CUTOFFS (5-40),
        # not raw crossing counts -- show what raw count each grid boundary
        # actually corresponds to, so it's comparable to the other two.
        zc = master_df["zero_crossings"]
        grid_lo, grid_hi = min(FOLD_SWEEP_GRID["zero_crossing_min_percentile"]), max(FOLD_SWEEP_GRID["zero_crossing_min_percentile"])
        print(f"  Observed zero_crossings: min={zc.min():.0f} median={zc.median():.0f} max={zc.max():.0f}  "
              f"[zero_crossing_min_percentile grid tests {grid_lo}-{grid_hi} percentile, "
              f"i.e. count cutoffs {zc.quantile(grid_lo/100):.0f}-{zc.quantile(grid_hi/100):.0f}]")

        # phi (0.5 < phi < 0.995) and coint_adf_p (< 0.05) are HARDCODED
        # filters in apply_thresholds_v2 -- not part of the sweep or the
        # permissive baseline at all. If a fold's candidates sit mostly
        # outside these bounds, nothing will ever pass at ANY tested value
        # of ANY swept parameter, which looks identical to "no plateau
        # exists" but is actually this unrelated, always-on filter.
        phi = master_df["phi"]
        in_band = ((phi > 0.5) & (phi < 0.995)).mean() * 100
        print(f"  Observed phi: min={phi.min():.3f} median={phi.median():.3f} max={phi.max():.3f}  "
              f"[hardcoded band is 0.5-0.995, {in_band:.0f}% of pairs currently fall inside it]")
        adf = master_df["coint_adf_p"]
        below_05 = (adf < 0.05).mean() * 100
        print(f"  Observed coint_adf_p: min={adf.min():.4f} median={adf.median():.4f} max={adf.max():.4f}  "
              f"[hardcoded requirement is < 0.05, {below_05:.0f}% of pairs currently satisfy it]")

        # Candidates that never traded at all in the eval window (Z never
        # crossed the entry threshold, or too few days to compute a Sharpe)
        # get NaN for sel_maxdd/sel_return. NaN < X and NaN > X are always
        # False in pandas, so those candidates fail the performance filter
        # regardless of how loose the threshold is -- this can silently
        # zero out a fold's results even when the "quality" filters
        # (phi/ADF/R2/half-life) all look fine, and differs fold-to-fold
        # if one eval window happens to be quieter than the other.
        n_inactive = master_df["sel_maxdd"].isna().sum()
        print(f"  Candidates with NO trades in the eval window (NaN Sharpe/DD/Return, always fail "
              f"the performance filter regardless of threshold): {n_inactive} / {len(master_df)} "
              f"({n_inactive / len(master_df) * 100:.0f}%)")

        # New: variance ratio (model-free mean-reversion test, VR<1 = reverting,
        # VR=1 = random walk, VR>1 = trending) and the ADF test's automatically
        # selected lag count (already computed via autolag='AIC', previously discarded).
        vr = master_df["variance_ratio"].dropna()
        n_vr_nan = master_df["variance_ratio"].isna().sum()
        if len(vr):
            print(f"  Observed variance_ratio: min={vr.min():.3f} p10={vr.quantile(0.1):.3f} "
                  f"median={vr.median():.3f} p90={vr.quantile(0.9):.3f} max={vr.max():.3f}  "
                  f"({(vr < 1).mean()*100:.0f}% below 1.0, i.e. reverting faster than a random walk; "
                  f"{n_vr_nan} pairs had insufficient data for a stable estimate)")
        lags = master_df["coint_adf_lags"]
        print(f"  Observed ADF lags used (autolag='AIC'): min={lags.min()} median={lags.median():.0f} "
              f"max={lags.max()}  (0 lags = plain Dickey-Fuller was already sufficient; higher means the "
              f"spread needed more lagged terms to properly test for a unit root)")

    return master_df


def apply_thresholds_v2(master_df, distance_pctile_max, zero_crossing_min_percentile, half_life_pctile_lo,
                         half_life_pctile_hi, r2_min_percentile, variance_ratio_max_percentile, perf_sharpe_min,
                         perf_maxdd_max, perf_return_min, min_persistence=0.0):
    if len(master_df) == 0:
        return {"n_candidates": 0, "n_passed": 0, "portfolio_sharpe": np.nan}
    zc_cutoff = np.percentile(master_df["zero_crossings"], zero_crossing_min_percentile)
    # R2 was a structural no-op as a fixed absolute threshold: an AR(1) fit's
    # R2 is largely just re-expressing how close phi is to 1 (already
    # filtered separately below), so with median phi ~0.97 the observed R2
    # floor sits at 0.79-0.87 -- above the entire 0.2-0.8 grid that used to
    # be tested here. Percentile-based, like zero-crossing, self-calibrates
    # to wherever the real distribution actually sits each cycle.
    r2_cutoff = np.percentile(master_df["ou_r_squared"], r2_min_percentile)
    # Variance ratio: KEEP the lowest (most mean-reverting) X percentile.
    # NaN (insufficient data for a stable VR estimate) fails this filter by
    # default, since pandas NaN comparisons are always False -- correct
    # behavior here, we can't confirm reversion for those pairs.
    vr_valid = master_df["variance_ratio"].dropna()
    vr_cutoff = np.percentile(vr_valid, variance_ratio_max_percentile) if len(vr_valid) else np.inf
    # BUG FIX: half-life used to be a fixed absolute day-count band (e.g.
    # 10-90), which was the one parameter never given the same
    # self-calibrating treatment as everything else. Confirmed directly:
    # Phase 1's multi-year Discovery windows produce half-life medians
    # around 24-31 days, but Phase 2's 6-month lookback produces medians
    # around 3.6-3.9 days -- an absolute band calibrated on one estimation
    # window length is meaningless applied to a very different one (0-2% of
    # candidates ever fell inside it). Percentile-based, like the others,
    # self-calibrates to whatever this cycle's own distribution looks like.
    hl_lo_cutoff = np.percentile(master_df["half_life"], half_life_pctile_lo)
    hl_hi_cutoff = np.percentile(master_df["half_life"], half_life_pctile_hi)
    candidates = master_df[
        (master_df["distance_pctile"] <= distance_pctile_max)
        & (master_df["zero_crossings"] >= zc_cutoff)
        & (master_df["phi"] > 0.5) & (master_df["phi"] < 0.995)
        & (master_df["coint_adf_p"] < 0.05)
        # Hardcoded, not swept -- like phi/ADF, this is a structural sanity
        # bound, not a percentile question. |beta| near 0 or very large means
        # the regression is dominated by one leg (very different price
        # scales, or one series barely contributing to the spread) -- the
        # relationship can be statistically cointegrated while being
        # economically fragile. Directly evidenced in our own diagnostic
        # tables (NVR-CBOE beta=33.5, GWW-BSX beta=19.2, PPL-SYK beta=0.03).
        & (master_df["beta"].abs() > 0.2) & (master_df["beta"].abs() < 5)
        & (master_df["half_life"] >= hl_lo_cutoff) & (master_df["half_life"] <= hl_hi_cutoff)
        & (master_df["ou_r_squared"] >= r2_cutoff)
        & (master_df["variance_ratio"] <= vr_cutoff)
        & (master_df["coint_persistence"] >= min_persistence)
    ]
    passed = candidates[
        (candidates["sel_sharpe"] > perf_sharpe_min)
        & (candidates["sel_maxdd"] < perf_maxdd_max)
        & (candidates["sel_return"] > perf_return_min)
    ]
    result = {"n_candidates": len(candidates), "n_passed": len(passed), "portfolio_sharpe": np.nan}
    if len(passed) == 0:
        return result
    returns_df = pd.DataFrame({row["name"]: row["_sel_returns"] for _, row in passed.iterrows()})
    port = portfolio_metrics(np.ones(len(passed)) / len(passed), returns_df)
    result["portfolio_sharpe"] = round(port["Sharpe"], 3)
    return result


def find_plateau(sweep_df, param_col, min_sharpe=PLATEAU_MIN_SHARPE, min_width=2):
    """Widest contiguous run of tested values where n_passed > 0 and
    portfolio_sharpe >= min_sharpe. Returns (low, high) or None if no such
    run exists at all in this fold.

    MIN_WIDTH: a run must span at least this many consecutive grid points to
    count as a genuine plateau. Without this, an isolated single-point spike
    (one value passes, both neighbors fail) could still be returned as "the
    widest run" whenever every other passing point was equally isolated --
    exactly the "single spike, not a stable plateau" pattern this whole
    design was meant to reject. Confirmed causing a real failure: two
    isolated points (5th and 20th percentile) each passed alone in one real
    fold, and simple last-wins tie-breaking on equal width picked one of them
    as if it were a validated range, producing a degenerate single-point
    band that then filtered an entire Phase 2 run down to zero pairs.

    EDGE HANDLING: if the plateau's low/high edge coincides with the
    smallest/largest value actually swept, that's not a genuine discovered
    limit — it's just where testing stopped. That side is returned as None
    (open-ended) rather than as a hard number, so downstream code doesn't
    mistake "we didn't test past here" for "here is where it starts failing."""
    has_candidates = sweep_df["n_passed"] > 0
    clears_sharpe = sweep_df["portfolio_sharpe"] >= min_sharpe
    ok = has_candidates & clears_sharpe
    # Diagnostic: these are two separate conditions. If candidates pass the
    # loose baseline filters at some grid points but the resulting
    # equal-weighted portfolio's Sharpe is negative at every one of them,
    # that's a different failure than "no candidate ever qualifies" and
    # looks identical in the final (None) result without this breakdown.
    print(f"    [{param_col}: {has_candidates.sum()}/{len(sweep_df)} grid points had n_passed>0, "
          f"{clears_sharpe.sum()}/{len(sweep_df)} had portfolio_sharpe>={min_sharpe}, "
          f"{ok.sum()}/{len(sweep_df)} had both]")
    print(f"    [{param_col}: full sweep, every tested value]")
    print(tabulate.tabulate(sweep_df, headers="keys", tablefmt="pretty", showindex=False))
    if not ok.any():
        return None
    best_run, best_run_points, run_start = None, 0, None
    for i, is_ok in enumerate(ok.tolist() + [False]):
        if is_ok and run_start is None:
            run_start = i
        elif not is_ok and run_start is not None:
            run_points = i - run_start
            if run_points >= min_width and run_points > best_run_points:
                best_run = (sweep_df[param_col].iloc[run_start], sweep_df[param_col].iloc[i - 1])
                best_run_points = run_points
            run_start = None

    if best_run is None:
        print(f"    [{param_col}: only isolated single-point spikes passed (no run of {min_width}+ "
              f"consecutive grid points) -- treating as no stable plateau, not a validated range]")
        return None

    grid_lo, grid_hi = sweep_df[param_col].min(), sweep_df[param_col].max()
    lo, hi = best_run
    lo = None if lo <= grid_lo else lo
    hi = None if hi >= grid_hi else hi
    return (lo, hi)


FOLD_SWEEP_GRID = {
    # NOTE: this is a percentile of the pool GENEROUS_DISTANCE_PERCENTILE already
    # prefiltered to, not of the original universe -- testing up to 100 here
    # means "the full prefiltered pool", which corresponds to
    # GENEROUS_DISTANCE_PERCENTILE's actual boundary (currently 50% of the
    # true universe). Stopping this grid at 50 (an earlier version) only
    # reached ~25% of the true universe, not 50% as intended.
    "distance_pctile_max": [5, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100],
    "zero_crossing_min_percentile": [5, 10, 15, 20, 25, 30, 35, 40],
    "r2_min_percentile": [0, 10, 20, 30, 40, 50, 60, 70, 80, 90],
    "variance_ratio_max_percentile": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
    # BUG FIX: was a fixed absolute day-count band (10-90), the one parameter
    # never given the same self-calibrating treatment as everything else.
    # Confirmed: Phase 1's multi-year Discovery windows produce half-life
    # medians ~24-31 days, but Phase 2's 6-month lookback produces medians
    # ~3.6-3.9 days -- an absolute band from one window length is meaningless
    # applied to a very different one (0-2% of candidates ever passed).
    # Percentile-based now, self-calibrates to each cycle's own distribution.
    "half_life_pctile_lo": [0, 5, 10, 15, 20, 25, 30],
    "half_life_pctile_hi": [70, 75, 80, 85, 90, 95, 100],
    # Previously computed (real cost -- N_PERSISTENCE_WINDOWS sub-window
    # re-tests per pair) but never actually used as a constraint: neither
    # FOLD_SWEEP_GRID nor the permissive baseline ever set min_persistence,
    # so apply_thresholds_v2's parameter silently fell back to its function
    # default of 0.0, and since persistence can't be negative, ">= 0.0" was
    # always trivially true. With N_PERSISTENCE_WINDOWS=2, persistence only
    # takes the discrete values {0.0, 0.5, 1.0} -- swept directly, not as a
    # percentile like the others.
    "min_persistence": [0.0, 0.5, 1.0],
    "perf_sharpe_min": [0.0, 0.25, 0.5, 0.75, 1.0, 1.25],
    "perf_maxdd_max": [0.10, 0.15, 0.20, 0.25, 0.30],
    "perf_return_min": [1.00, 1.05, 1.10, 1.15, 1.20, 1.25],
}


def sweep_fold(master_df, eval_months, condition_on=None):
    # DEEPER FIX: a 1-D sweep only produces useful signal if the "everything
    # else held fixed" baseline can pass at all. The previous baseline held
    # R2>0.5, Sharpe>0.5, and MaxDD<15% all fixed simultaneously while
    # sweeping any other parameter -- if that specific 3-way combination
    # never has a single passing pair in a fold's data (very plausible over
    # a 6-month window), EVERY sweep shows empty results regardless of the
    # swept parameter's value, which is what real runs were showing. Fixing
    # perf_return_min's window-scaling (kept below) was necessary but not
    # sufficient on its own. Non-swept parameters are now held at genuinely
    # permissive values -- not the pipeline's real operating defaults -- so
    # each parameter's true marginal effect is isolated rather than
    # confounded with several other strict requirements at once.
    #
    # CONDITIONAL SWEEP (condition_on): an isolated sweep tests "does this
    # parameter help against the ~near-unfiltered pool" -- a different,
    # weaker question than "does this parameter help ON TOP OF a filter
    # that's already known to matter (distance)". condition_on lets a
    # caller override specific baseline values (typically
    # distance_pctile_max, fixed at its validated boundary) so every other
    # parameter's sweep is evaluated against the realistic operating
    # scenario instead of the raw pool.
    fold_perf_return_min = (1 + PERF_RETURN_MIN_ANNUALIZED) ** (eval_months / 12)

    permissive_baseline = dict(
        distance_pctile_max=100, zero_crossing_min_percentile=0,
        half_life_pctile_lo=0, half_life_pctile_hi=100,
        r2_min_percentile=0, variance_ratio_max_percentile=100, min_persistence=0.0,
        perf_sharpe_min=-10.0, perf_maxdd_max=5.0, perf_return_min=0.0,
    )
    if condition_on:
        permissive_baseline.update(condition_on)
    # For the sweep OF perf_return_min specifically, its own tested grid
    # values [1.00..1.25] are absolute cumulative multipliers already
    # calibrated as reasonable 6-month figures — no fold-scaling needed there.
    # fold_perf_return_min is only used as a sanity floor when reporting.

    results = {}
    for param, values in FOLD_SWEEP_GRID.items():
        if condition_on and param in condition_on:
            continue  # don't sweep a parameter that's being held fixed as the condition
        rows = []
        for v in values:
            params = dict(permissive_baseline)
            params[param] = v
            rows.append({param: v, **apply_thresholds_v2(master_df, **params)})
        results[param] = pd.DataFrame(rows)
    return results


def aggregate_fold_plateaus(fold_plateaus, header):
    """Cross-fold intersection logic, extracted so it can be reused for both
    the primary (isolated) sweep pass and the conditional (distance-fixed)
    pass without duplicating the aggregation rules."""
    print("\n" + "=" * 80)
    print(header)
    print("=" * 80)
    n_folds = len(next(iter(fold_plateaus.values()))) if fold_plateaus else 0
    validated_ranges = {}
    for param, plateaus in fold_plateaus.items():
        valid = [p for p in plateaus if p is not None]
        if len(valid) < n_folds:
            print(f"  {param}: NOT robust — only {len(valid)}/{n_folds} folds had a stable plateau at all.")
            validated_ranges[param] = None
            continue

        # None on either side means that fold's plateau ran to the edge of what
        # was tested -- treat as "this fold found no evidence of a limit here"
        # rather than a hard number, so it doesn't wrongly win the max()/min().
        lo_candidates = [p[0] for p in valid if p[0] is not None]
        hi_candidates = [p[1] for p in valid if p[1] is not None]
        lo = float(max(lo_candidates)) if lo_candidates else None
        hi = float(min(hi_candidates)) if hi_candidates else None

        if lo is not None and hi is not None and lo > hi:
            print(f"  {param}: NOT robust — per-fold plateaus don't overlap ({valid}).")
            validated_ranges[param] = None
        else:
            print(f"  {param}: VALIDATED range = [{lo}, {hi}]  (per-fold: {valid})")
            validated_ranges[param] = [lo, hi]
    return validated_ranges


def run_phase1_validation():
    print(f"PHASE 1: fold-based threshold validation across {N_FOLDS} folds "
          f"(pre/post-COVID split)\n")
    tickers = get_sp500_tickers_from_excel()
    print(f"Loaded {len(tickers)} tickers from Excel.")
    fold_end_dates = [f[2] for f in FOLD_DEFINITIONS]
    data = download_universe(tickers, start=FOLD_DEFINITIONS[0][0], end=max(fold_end_dates))

    fold_plateaus = {param: [] for param in FOLD_SWEEP_GRID}
    fold_master_tables = {}
    fold_eval_months = {}
    for i, (d_start, e_start, f_end) in enumerate(FOLD_DEFINITIONS):
        print("\n" + "=" * 80)
        print(f"FOLD {i + 1}/{N_FOLDS}:  discovery {d_start} -> {e_start}   eval {e_start} -> {f_end}")
        print("=" * 80)
        master_df = compute_fold_master_table(data, d_start, e_start, f_end)
        eval_months = ((pd.Timestamp(f_end).year - pd.Timestamp(e_start).year) * 12
                        + (pd.Timestamp(f_end).month - pd.Timestamp(e_start).month))
        fold_master_tables[i] = master_df   # kept in memory for the conditional pass below -- no re-running Discovery
        fold_eval_months[i] = eval_months
        sweeps = sweep_fold(master_df, eval_months)
        for param, sweep_df in sweeps.items():
            plateau = find_plateau(sweep_df, param)
            fold_plateaus[param].append(plateau)
            print(f"  {param}: plateau in this fold = {plateau}")

    validated_ranges = aggregate_fold_plateaus(fold_plateaus, "CROSS-FOLD VALIDATED RANGES  (isolated sweeps, "
                                                               "each parameter tested against the ~unfiltered pool)")

    with open(VALIDATED_RANGES_PATH, "w") as f:
        json.dump(validated_ranges, f, indent=2)
    print(f"\nSaved to {VALIDATED_RANGES_PATH}")

    # CONDITIONAL PASS: an isolated sweep only tells you whether a parameter
    # helps against the near-unfiltered pool -- a different, weaker question
    # than "does it help ON TOP OF the filter we already know matters
    # (distance)". If distance validated a real boundary, re-sweep the other
    # four parameters with distance held fixed at that boundary instead of
    # permissive, reusing the master tables already computed above (no
    # re-running Discovery/cointegration).
    dist_range = validated_ranges.get("distance_pctile_max")
    if dist_range and dist_range[1] is not None:
        condition_value = dist_range[1]
        print("\n" + "=" * 80)
        print(f"CONDITIONAL SWEEP: does each parameter add value ON TOP OF distance_pctile_max <= "
              f"{condition_value}, instead of against the near-unfiltered pool?")
        print("=" * 80)
        cond_fold_plateaus = {p: [] for p in FOLD_SWEEP_GRID if p != "distance_pctile_max"}
        for i in fold_master_tables:
            print(f"\n--- Fold {i + 1}/{N_FOLDS} (conditional on distance_pctile_max <= {condition_value}) ---")
            sweeps = sweep_fold(fold_master_tables[i], fold_eval_months[i],
                                 condition_on={"distance_pctile_max": condition_value})
            for param, sweep_df in sweeps.items():
                plateau = find_plateau(sweep_df, param)
                cond_fold_plateaus[param].append(plateau)
                print(f"  {param}: plateau in this fold = {plateau}")

        conditional_validated_ranges = aggregate_fold_plateaus(
            cond_fold_plateaus,
            f"CONDITIONAL VALIDATED RANGES  (given distance_pctile_max <= {condition_value})"
        )
        cond_path = os.path.join(OUTPUT_DIR, "validated_ranges_conditional.json")
        with open(cond_path, "w") as f:
            json.dump({"condition": {"distance_pctile_max": condition_value},
                       "ranges": conditional_validated_ranges}, f, indent=2)
        print(f"\nSaved to {cond_path}  (diagnostic only -- Phase 2 still reads {VALIDATED_RANGES_PATH} "
              f"unless you decide to adopt these and update that file)")
    else:
        print("\nNo validated distance boundary to condition on -- skipping the conditional sweep pass.")

    return validated_ranges


# ============================================================================
# PHASE 2: ROLLING RE-QUALIFICATION OOS
# ============================================================================
def load_validated_ranges(path=VALIDATED_RANGES_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"\n\n'{path}' not found. This file is produced by Phase 1 and normally persists in your "
            f"output/ folder between runs -- but it's a local file, not something that survives closing "
            f"your terminal/IDE if the folder itself gets cleared. Set RUN_PHASE1_VALIDATION = True and "
            f"RUN_PHASE2_OOS = False, and rerun once to regenerate it before running Phase 2 again.\n"
        )
    with open(path) as f:
        ranges = json.load(f)

    # Merge in the conditional sweep's results where available: a parameter
    # validated conditionally (with distance already held tight) was tested
    # in the scenario that actually matches how Phase 2 applies it, so it
    # takes precedence over the same parameter's isolated/primary result.
    cond_path = os.path.join(OUTPUT_DIR, "validated_ranges_conditional.json")
    if PHASE2_USE_CONDITIONAL_RANGES and os.path.exists(cond_path):
        with open(cond_path) as f:
            cond_data = json.load(f)
        cond_ranges = cond_data.get("ranges", {})
        upgraded = []
        for param, cond_val in cond_ranges.items():
            # A fully open [None, None] conditional result means "this
            # parameter didn't discriminate anything once distance was
            # already tight" -- that's a real, useful finding on its own,
            # but it is NOT evidence that a real constraint the primary/
            # isolated sweep found (e.g. min_persistence >= 1.0) should be
            # discarded. Only override when the conditional itself found an
            # actual bound.
            if cond_val is not None and any(v is not None for v in cond_val):
                ranges[param] = cond_val
                upgraded.append(param)
        if upgraded:
            print(f"Using CONDITIONAL validated ranges (more realistic -- tested with distance already "
                  f"filtering) for: {upgraded}")

    missing = [p for p, r in ranges.items() if r is None]
    if missing:
        print(f"WARNING: {missing} had no validated range from Phase 1 (isolated or conditional) — "
              f"falling back to this script's CONFIG defaults for those.")
    return ranges


def operating_thresholds_from_ranges(ranges):
    """Every validated parameter is now a genuine two-sided band, matching how
    half-life already worked: Phase 1 proved performance is stable for values
    WITHIN [lo, hi], so a pair must fall within that band, not just clear one
    edge of it. This also screens out suspiciously too-good values (a
    near-perfect R^2 or an outsized Sharpe) as readily as too-weak ones —
    both are more likely to be noise/overfitting than genuine signal.

    EDGE-ARTIFACT GUARD: a validated boundary sitting exactly at the edge of
    what FOLD_SWEEP_GRID actually tested isn't a genuine discovered
    ceiling/floor — it's just where the sweep stopped testing. That side
    falls back to a value here.

    PERMISSIVE FALLBACKS: that fallback must be genuinely "no constraint",
    not the single-window pipeline's own operating threshold (PERF_SHARPE_MIN,
    R2_THRESHOLD, etc.) — those are real, meaningful bars for THAT pipeline,
    but can collide with or exceed a genuinely discovered opposite bound here,
    collapsing the two-sided band to a single degenerate point (e.g. a
    discovered Sharpe ceiling of 0.5 combined with PERF_SHARPE_MIN=0.5 as the
    "open" floor previously produced the unusable range (0.5, 0.5), where only
    a pair with Sharpe of exactly 0.5 could ever qualify)."""
    PERMISSIVE = {
        "distance_pctile_max": (0.0, 100.0),
        "zero_crossing_min_percentile": (0.0, 100.0),
        "r2_min_percentile": (0.0, 100.0),
        "variance_ratio_max_percentile": (0.0, 100.0),
        "min_persistence": (0.0, 1.0),
        "perf_sharpe_min": (-10.0, 10.0),
        "perf_maxdd_max": (0.0, 5.0),
        "perf_return_min": (0.0, 10.0),
        "half_life_pctile_lo": (0.0, 100.0),
        "half_life_pctile_hi": (0.0, 100.0),
    }

    def rng(key):
        default_lo, default_hi = PERMISSIVE[key]
        r = ranges.get(key)
        if not r:
            return (default_lo, default_hi)
        # New Phase 1 format marks open-ended edges explicitly as null --
        # honor that directly.
        lo = default_lo if r[0] is None else float(r[0])
        hi = default_hi if r[1] is None else float(r[1])
        # Defensive fallback for older validated_ranges.json files saved before
        # Phase 1 started marking open edges as null: a numeric value sitting
        # exactly at the tested grid's edge gets the same treatment.
        grid = FOLD_SWEEP_GRID.get(key)
        if grid:
            grid_lo, grid_hi = min(grid), max(grid)
            if r[1] is not None and hi >= grid_hi:
                hi = default_hi
            if r[0] is not None and lo <= grid_lo:
                lo = default_lo
        return (lo, hi)

    return dict(
        distance_pctile=rng("distance_pctile_max"),
        zero_crossing_pctile=rng("zero_crossing_min_percentile"),
        half_life_pctile=(rng("half_life_pctile_lo")[0], rng("half_life_pctile_hi")[1]),
        r2_pctile=rng("r2_min_percentile"),
        variance_ratio_pctile=rng("variance_ratio_max_percentile"),
        min_persistence=rng("min_persistence"),
        perf_sharpe=rng("perf_sharpe_min"),
        perf_maxdd=rng("perf_maxdd_max"),
        perf_return=rng("perf_return_min"),
    )


def build_phase2_rebalance_dates():
    dates, current = [], pd.Timestamp(PHASE2_START)
    end = pd.Timestamp(PHASE2_END)
    while current < end:
        dates.append(current)
        current = current + pd.DateOffset(months=PHASE2_REBALANCE_MONTHS)
    return dates


def compute_beta_stability(df, lookback_start, rebal_date):
    """Splits the lookback window in half and refits the hedge ratio
    separately on each half. Tests a genuinely different hypothesis than
    |beta| itself (already tested and ruled out by the loser analysis): a
    pair whose relationship is shifting mid-window -- beta materially
    different in the first half vs the second -- may be a relationship in
    the process of breaking, even if its FULL-window beta looks reasonable.
    Returned as a relative measure (0 = identical, larger = more unstable)
    so pairs of very different price scales remain comparable."""
    lb = df.loc[lookback_start:rebal_date].dropna(subset=["dep", "indep"])
    if len(lb) < 20:
        return np.nan
    mid = len(lb) // 2
    first, second = lb.iloc[:mid], lb.iloc[mid:]
    try:
        b1 = sm.OLS(first["dep"].values, sm.add_constant(first["indep"].values)).fit().params[1]
        b2 = sm.OLS(second["dep"].values, sm.add_constant(second["indep"].values)).fit().params[1]
    except Exception:
        return np.nan
    denom = abs(b1) + abs(b2)
    return float(abs(b1 - b2) / denom) if denom > 1e-9 else np.nan


# ============================================================================
# NEW: TRADE-LEVEL FEATURES + COMPOSITE SCORE (feedback-driven)
# ============================================================================
def compute_trade_level_features(df, start, end, min_trades=MIN_LOOKBACK_TRADES_FOR_FEATURES):
    """Computes expectancy, tail-risk, and recovery-consistency features from
    ACTUAL simulated round-trip trades within [start, end] -- a different
    statistic than portfolio Sharpe or the OU fit. A pair can have a fine
    Sharpe while individual trades carry a fat left tail (rare blowups) or
    wildly inconsistent reversion times (3, 30, 2, 25, 40 days -- same mean
    half-life as 3, 4, 5, but much harder to trade reliably). Returns NaN
    for every feature if there are too few trades to estimate anything
    meaningfully, rather than a spuriously confident value from 1-2 trades.

    expectancy_pct is computed on return_% (which already nets against
    TRANSACTION_COST_BPS_PER_LEG inside build_spread_and_signals) -- this
    targets "will this pair actually pay after costs", not "is this
    statistically mean-reverting", which is the gap the feedback identified."""
    trades = extract_trades(df, start=start, end=end)
    n = len(trades)
    out = {
        "n_trades": n, "win_rate": np.nan, "avg_win_pct": np.nan, "avg_loss_pct": np.nan,
        "expectancy_pct": np.nan, "p95_loss_pct": np.nan, "p99_loss_pct": np.nan,
        "days_held_mean": np.nan, "days_held_cv": np.nan,
    }
    if n < min_trades:
        return out

    rets = trades["return_%"]
    wins, losses = rets[rets > 0], rets[rets <= 0]
    win_rate = len(wins) / n
    avg_win = float(wins.mean()) if len(wins) else 0.0
    avg_loss = float(losses.mean()) if len(losses) else 0.0  # <= 0
    expectancy = win_rate * avg_win + (1 - win_rate) * avg_loss

    days_held = trades["days_held"]
    dh_mean, dh_std = days_held.mean(), days_held.std()
    dh_cv = float(dh_std / dh_mean) if dh_mean and not np.isnan(dh_std) else np.nan

    out.update({
        "win_rate": float(win_rate), "avg_win_pct": avg_win, "avg_loss_pct": avg_loss,
        "expectancy_pct": float(expectancy),
        # np.percentile(rets, 5) = the 5th percentile of the return
        # distribution = the 95th-percentile LOSS (worse than 95% of trades).
        # A pair can look clean on Sharpe/win-rate and still have an ugly tail here.
        "p95_loss_pct": float(np.percentile(rets, 5)),
        "p99_loss_pct": float(np.percentile(rets, 1)) if n >= 10 else float(rets.min()),
        "days_held_mean": float(dh_mean),
        "days_held_cv": dh_cv,  # LOWER = more consistent reversion speed = better
    })
    return out


def apply_composite_score_filter(scored, weights_path=SCORE_WEIGHTS_PATH, top_pct=SCORE_TOP_PERCENTILE):
    """Ranks the qualified population by the validated composite score
    (built by build_and_save_composite_score, from ONLY features that
    cleared an honest predictive-power bar) and keeps only the top X% --
    'is this pair among the historically BEST', not just 'does it clear a
    minimum quality bar'. If no score has been validated yet
    (score_weights.json missing or empty), returns the input UNCHANGED
    with a warning -- this must never silently no-op filter into an empty
    or wrongly-shrunk population."""
    if not os.path.exists(weights_path):
        print("WARNING: USE_COMPOSITE_SCORE=True but no score_weights.json found. "
              "Run RUN_SCORE_VALIDATION=True first. Skipping score filter this cycle.")
        return scored
    with open(weights_path) as f:
        spec = json.load(f)
    weights = spec.get("weights", {})
    if not weights:
        print("WARNING: score_weights.json has no validated weights (no feature cleared the "
              "predictive-power bar last time this was built). Skipping score filter this cycle.")
        return scored
    if not scored:
        return scored

    lower_is_better = set(spec.get("lower_is_better", []))
    df = pd.DataFrame(scored)
    score = pd.Series(0.0, index=df.index)
    used = []
    for feat, w in weights.items():
        if feat not in df.columns:
            continue
        col = df[feat]
        z = (col - col.mean()) / col.std() if col.std() > 0 else pd.Series(0.0, index=col.index)
        if feat in lower_is_better:
            z = -z
        score += w * z.fillna(0)
        used.append(feat)
    df["_composite_score"] = score
    cutoff = np.percentile(score.dropna(), 100 - top_pct) if score.notna().any() else np.inf
    kept = df[df["_composite_score"] >= cutoff]
    print(f"  Composite score filter: kept top {top_pct}% -> {len(kept)} / {len(scored)} pairs "
          f"(score cutoff = {cutoff:.3f}, features used: {used})")
    return kept.to_dict(orient="records")


def run_phase2_cycle(data, rebal_date, next_rebal_date, thresholds):
    lookback_start = rebal_date - pd.DateOffset(months=PHASE2_LOOKBACK_MONTHS)
    discovery_data = data.loc[lookback_start:rebal_date]
    signal_data = data.loc[lookback_start:next_rebal_date]

    close_pairs, norm_disc, distances = distance_prefilter(discovery_data, percentile=GENEROUS_DISTANCE_PERCENTILE)
    # BUG FIX: see compute_full_pair_statistics -- cointegration/regression
    # now use raw discovery_data, not the normalized norm_disc (which is
    # kept strictly for the distance/SSD calculation above).
    cointegrated = cointegration_filter(discovery_data, close_pairs, pvalue=FOLD_COINT_PVALUE)
    reg_results = fit_direction_and_beta(discovery_data, cointegrated)
    persistence = compute_coint_persistence(discovery_data, cointegrated)

    zc_counts = []
    r2_vals = []
    vr_vals = []
    dist_vals = []
    hl_vals = []
    prelim = []
    for r in reg_results:
        s = r["disc_spread"].dropna()
        zc = count_zero_crossings(s)
        ou = fit_ou_on_levels(r["disc_spread"], dt=1.0)
        if not ou.get("valid", False):
            continue
        prelim.append((r, zc, ou))
        zc_counts.append(zc)
        r2_vals.append(ou["r_squared"])
        dist_vals.append(r["distance"])
        hl_vals.append(ou["half_life"])
        if not np.isnan(ou["variance_ratio"]):
            vr_vals.append(ou["variance_ratio"])
    if not prelim:
        return None, [], pd.DataFrame(), [], []

    # BUG FIX: previously computed against `distances`, the FULL unfiltered
    # universe returned by distance_prefilter -- a completely different,
    # much wider population than what Phase 1 validated this percentile
    # range against (master_df["distance"], ranked only among cointegrated
    # survivors). That mismatch made the effective cutoff far stricter than
    # the validated range intended. Now computed against dist_vals (this
    # cycle's own cointegrated survivors), matching Phase 1's reference set.
    dist_cutoff_lo = np.percentile(dist_vals, thresholds["distance_pctile"][0])
    dist_cutoff_hi = np.percentile(dist_vals, thresholds["distance_pctile"][1])
    hl_pctile_lo, hl_pctile_hi = thresholds["half_life_pctile"]
    hl_arr = np.array(hl_vals)
    hl_lo = np.percentile(hl_arr, hl_pctile_lo)
    print(f"  Observed half_life this cycle: min={hl_arr.min():.2f} median={np.median(hl_arr):.2f} "
          f"max={hl_arr.max():.2f}  [applied lower bound: percentile {hl_pctile_lo} of THIS cycle's own "
          f"distribution = {hl_lo:.2f} days -- only the lower bound validated (excludes noisiest very-short "
          f"half-lives); the upper bound never showed real signal, so it's no longer applied]")
    if vr_vals:
        vr_cutoff_lo = np.percentile(vr_vals, thresholds["variance_ratio_pctile"][0])
        vr_cutoff_hi = np.percentile(vr_vals, thresholds["variance_ratio_pctile"][1])
    else:
        vr_cutoff_lo, vr_cutoff_hi = -np.inf, np.inf

    # SIMPLIFIED FUNNEL: zero-crossing, R2, and Sharpe removed -- none ever
    # validated real cross-fold signal (isolated or conditional), across
    # every run so far. Distance, half-life's lower bound, variance ratio,
    # and persistence are kept -- all validated genuine constraints.
    funnel = [("Initial candidates (cointegrated, valid OU fit)", len(prelim))]
    stage = prelim
    stage = [t for t in stage if dist_cutoff_lo <= t[0]["distance"] <= dist_cutoff_hi]
    funnel.append(("Distance", len(stage)))
    stage = [t for t in stage if 0.5 < t[2]["phi"] < 0.995 and t[0]["ADF_pvalue"] < 0.05]
    funnel.append(("Phi/ADF (hardcoded)", len(stage)))
    stage = [t for t in stage if 0.2 < abs(t[0]["beta"]) < 5]
    funnel.append(("Hedge ratio (hardcoded, 0.2<|beta|<5)", len(stage)))
    stage = [t for t in stage if t[2]["half_life"] >= hl_lo]
    funnel.append(("Half-life (lower bound only)", len(stage)))
    stage = [t for t in stage
             if not np.isnan(t[2]["variance_ratio"]) and vr_cutoff_lo <= t[2]["variance_ratio"] <= vr_cutoff_hi]
    funnel.append(("Variance Ratio", len(stage)))
    persist_lo, persist_hi = thresholds["min_persistence"]
    stage = [t for t in stage if persist_lo <= persistence.get(t[0]["pair"], 0.0) <= persist_hi]
    funnel.append(("Persistence", len(stage)))
    prelim_survivors = stage

    qualified = []
    for r, zc, ou in prelim_survivors:
        qualified.append({"dependent": r["dependent"], "independent": r["independent"],
                           "beta": r["beta"], "intercept": r["intercept"], "cointegration_p": r["cointegration_p"],
                           "half_life": ou["half_life"], "variance_ratio": ou["variance_ratio"],
                           "name": f"{r['dependent']}-{r['independent']}"})

    if not qualified:
        print(tabulate.tabulate(funnel, headers=["Stage", "Pairs Remaining"], tablefmt="pretty"))
        return None, [], pd.DataFrame(), [], []

    maxdd_lo, maxdd_hi = thresholds["perf_maxdd"]
    return_lo, return_hi = thresholds["perf_return"]

    # NEW: for every qualified pair, compute lookback trade-level features
    # (expectancy, tail-risk, recovery consistency -- net of transaction
    # cost) and beta stability, alongside the existing Sharpe/DD/Return.
    # These feed both the existing diagnostics AND the new score-validation
    # pipeline below.
    scored = []
    for q in qualified:
        win, std_win = adaptive_window(q["half_life"])
        df = build_spread_and_signals(signal_data, q["dependent"], q["independent"], q["beta"], q["intercept"],
                                       window=win, std_window=std_win)
        m = slice_metrics(df, lookback_start, rebal_date)
        lb_feat = compute_trade_level_features(df, lookback_start, rebal_date)
        beta_stab = compute_beta_stability(df, lookback_start, rebal_date)
        scored.append({**q, "df": df, "sharpe": m["sharpe_ratio"], "maxdd": m["max_drawdown"], "ret": m["final_return"],
                "beta_stability": beta_stab,
                "lb_sharpe": m["sharpe_ratio"], "lb_ret": m["final_return"], "lb_maxdd": m["max_drawdown"],
                **{f"lb_{k}": v for k, v in lb_feat.items()}})

    stage = [p for p in scored if not np.isnan(p["maxdd"]) and not np.isnan(p["ret"])]
    stage = [p for p in stage if maxdd_lo <= p["maxdd"] <= maxdd_hi]
    funnel.append(("Drawdown", len(stage)))
    stage = [p for p in stage if return_lo <= p["ret"] <= return_hi]
    funnel.append(("Return", len(stage)))
    passed_pairs = stage

    if not passed_pairs:
        print(tabulate.tabulate(funnel, headers=["Stage", "Pairs Remaining"], tablefmt="pretty"))
        return None, [], pd.DataFrame(), [], []

    # NEW: forward outcomes computed for the FULL passed population (before
    # any composite-score filter and before the concentration cap), so the
    # score-validation analysis has more statistical power than just the
    # smaller traded/capped set. Still zero look-ahead: the forward window
    # [rebal_date, next_rebal_date] was never used to compute any lb_*
    # feature or beta_stability above.
    all_pair_specs = []
    for p in passed_pairs:
        fwd_m = slice_metrics(p["df"], rebal_date, next_rebal_date)
        fwd_feat = compute_trade_level_features(p["df"], rebal_date, next_rebal_date)
        all_pair_specs.append({
            "name": p["name"], "dependent": p["dependent"], "independent": p["independent"],
            "beta": p["beta"], "half_life": p["half_life"], "variance_ratio": p["variance_ratio"],
            "cointegration_p": p["cointegration_p"],
            "lb_sharpe": p["sharpe"], "lb_maxdd": p["maxdd"], "lb_ret": p["ret"],
            "lb_win_rate": p["lb_win_rate"], "lb_expectancy_pct": p["lb_expectancy_pct"],
            "lb_p95_loss_pct": p["lb_p95_loss_pct"], "lb_p99_loss_pct": p["lb_p99_loss_pct"],
            "lb_days_held_cv": p["lb_days_held_cv"], "lb_n_trades": p["lb_n_trades"],
            "beta_stability": p["beta_stability"],
            "fwd_sharpe": fwd_m["sharpe_ratio"], "fwd_ret": fwd_m["final_return"],
            "fwd_expectancy_pct": fwd_feat["expectancy_pct"], "fwd_win_rate": fwd_feat["win_rate"],
            "fwd_n_trades": fwd_feat["n_trades"],
        })

    # NEW: optionally filter to only the top-scoring pairs by the validated
    # composite score, BEFORE the concentration cap -- so the cap is applied
    # to an already-quality-ranked set, not the other way around.
    if USE_COMPOSITE_SCORE:
        passed_pairs = apply_composite_score_filter(passed_pairs)
        funnel.append(("Composite Score", len(passed_pairs)))
        if not passed_pairs:
            print(tabulate.tabulate(funnel, headers=["Stage", "Pairs Remaining"], tablefmt="pretty"))
            return None, [], pd.DataFrame(), [], all_pair_specs

    # Concentration cap: no single ticker allowed in more than
    # MAX_PAIRS_PER_TICKER traded pairs this cycle. Without this, one stock's
    # large idiosyncratic move can spuriously "pair" with dozens of unrelated
    # names -- confirmed empirically (CRWD alone was 26% of one real cycle's
    # book), turning what looks like a diversified portfolio into a single
    # concentrated bet dressed up as many. Greedily keeps the highest-Sharpe
    # pairs first, skipping any pair where either leg has already hit the cap.
    ticker_counts = {}
    capped_pairs = []
    for p in sorted(passed_pairs, key=lambda x: x["sharpe"], reverse=True):
        dep, indep = p["dependent"], p["independent"]
        if ticker_counts.get(dep, 0) >= MAX_PAIRS_PER_TICKER or ticker_counts.get(indep, 0) >= MAX_PAIRS_PER_TICKER:
            continue
        capped_pairs.append(p)
        ticker_counts[dep] = ticker_counts.get(dep, 0) + 1
        ticker_counts[indep] = ticker_counts.get(indep, 0) + 1
    funnel.append(("Ticker cap", len(capped_pairs)))
    n_dropped = len(passed_pairs) - len(capped_pairs)
    if n_dropped:
        print(f"  Concentration cap: dropped {n_dropped} / {len(passed_pairs)} pairs "
              f"(no ticker in more than {MAX_PAIRS_PER_TICKER} pairs this cycle)")

    pair_returns, pair_positions, perf_rows, all_trades, beta_stability_by_name, forward_sharpe_by_name, \
        pair_vol_by_name = {}, {}, [], [], {}, {}, {}
    diag_rows = []
    for p in capped_pairs:
        sub = p["df"].loc[rebal_date:next_rebal_date]
        pair_returns[p["name"]] = sub["Strategy_Return"].fillna(0)
        pair_positions[p["name"]] = (sub["Position"].shift(1) != 0).astype(int)
        perf_rows.append({"Pair": p["name"], "Sharpe": round(p["sharpe"], 2)})
        pair_trades = extract_trades(p["df"], start=rebal_date, end=next_rebal_date)
        if len(pair_trades):
            pair_trades.insert(0, "pair", p["name"])
            all_trades.append(pair_trades)
        n_trades = len(pair_trades)
        # Reuse the beta_stability already computed above (in `scored`)
        # instead of recomputing it -- kept as a separate dict for the
        # existing downstream analysis functions that expect it.
        beta_stability = p["beta_stability"]
        beta_stability_by_name[p["name"]] = beta_stability
        # Lookback-only, so it's usable for actual capital allocation (not
        # just validation) -- how volatile has this pair's OWN strategy
        # return been historically, used below to size positions inversely
        # to volatility instead of giving every active pair equal weight.
        lookback_vol = p["df"].loc[lookback_start:rebal_date, "Strategy_Return"].std()
        pair_vol_by_name[p["name"]] = float(lookback_vol) if not np.isnan(lookback_vol) and lookback_vol > 0 else np.nan
        # VALIDATION ONLY -- never fed back into filtering or ranking, which
        # both already happened above using p["sharpe"] (lookback-only).
        # Computed purely to test whether that lookback ranking has any
        # real predictive relationship with what actually happened forward.
        forward_sharpe = slice_metrics(p["df"], rebal_date, next_rebal_date)["sharpe_ratio"]
        forward_sharpe_by_name[p["name"]] = forward_sharpe
        diag_rows.append({
            "name": p["name"], "cointegration_p": p["cointegration_p"], "beta": p["beta"], "sharpe": p["sharpe"],
            "half_life": p["half_life"], "variance_ratio": p["variance_ratio"], "beta_stability": beta_stability,
            "n_trades": n_trades,
            "avg_trade_pct": pair_trades["return_%"].mean() if n_trades else np.nan,
            "stop_loss_pct": (pair_trades["exit_reason"] == "stop_loss").mean() * 100 if n_trades else np.nan,
            "time_stop_pct": (pair_trades["exit_reason"] == "time_stop").mean() * 100 if n_trades else np.nan,
        })
    print("\n  Filter funnel this cycle:")
    print(tabulate.tabulate(funnel, headers=["Stage", "Pairs Remaining"], tablefmt="pretty"))

    print_pair_diagnostic_table(
        pd.DataFrame(diag_rows), "name", "cointegration_p", "beta", "sharpe", "n_trades", "avg_trade_pct",
        "stop_loss_pct", "time_stop_pct",
        label=f"PHASE 2 PER-PAIR DIAGNOSTIC (cycle {rebal_date.date()} -> {next_rebal_date.date()})",
        csv_path=os.path.join(OUTPUT_DIR, f"phase2_pair_diagnostic_{rebal_date.date()}.csv"),
    )

    if not pair_returns:
        return None, [], pd.DataFrame(), [], all_pair_specs

    returns_df = pd.DataFrame(pair_returns)
    active_df = pd.DataFrame(pair_positions)
    n_active = active_df.sum(axis=1)

    # Inverse-volatility position sizing: a pair with 2x the historical
    # (lookback-only) volatility of another gets roughly half the capital
    # weight when both are active the same day. Pairs with missing/zero
    # lookback vol (too little data) fall back to the cross-sectional
    # median vol this cycle, rather than getting an arbitrary weight.
    vol_series = pd.Series({c: pair_vol_by_name.get(c, np.nan) for c in returns_df.columns})
    median_vol = vol_series.median()
    fallback_vol = median_vol if not np.isnan(median_vol) and median_vol > 0 else 1.0
    vol_series = vol_series.fillna(fallback_vol).replace(0, fallback_vol)
    inv_vol = 1.0 / vol_series

    # Each day, weight is proportional to inv_vol among pairs ACTIVE that
    # day only (inactive pairs contribute 0 automatically), normalized to
    # sum to 1, then capped per-position at MAX_POSITION_WEIGHT -- same as
    # before, any weight given up to the cap simply isn't reallocated
    # rather than being force-concentrated elsewhere.
    raw_weight = active_df.mul(inv_vol, axis=1)
    raw_weight_sum = raw_weight.sum(axis=1)
    weight_df = raw_weight.div(raw_weight_sum.replace(0, np.nan), axis=0).fillna(0)
    weight_df = weight_df.clip(upper=MAX_POSITION_WEIGHT)
    forward_return = (returns_df * weight_df).sum(axis=1)
    trades_df = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()

    # DIAGNOSTIC: a very high portfolio Sharpe from averaging many small edges
    # is mathematically real IF those edges are genuinely independent (Sharpe
    # scales ~sqrt(N) for N independent bets) -- but real pairs books rarely
    # see Sharpe this high, because "dollar-neutral per pair" doesn't
    # guarantee independence ACROSS pairs (shared sector/factor exposure).
    # Measure it directly instead of assuming either way.
    if len(returns_df.columns) > 1:
        active_returns = returns_df.where(active_df.astype(bool))
        corr = active_returns.corr()
        avg_pairwise_corr = corr.values[np.triu_indices_from(corr.values, k=1)]
        avg_pairwise_corr = np.nanmean(avg_pairwise_corr) if len(avg_pairwise_corr) else np.nan
        print(f"  Concurrency/correlation diagnostic: avg {n_active.mean():.1f} positions open per day "
              f"(max {int(n_active.max())})  |  avg pairwise correlation among active pairs' daily "
              f"returns = {avg_pairwise_corr:.4f}  (near 0 = genuinely independent bets, consistent with "
              f"the high Sharpe being real diversification; notably positive = hidden shared risk the "
              f"aggregation is NOT capturing, meaning Sharpe is still overstated)")

    n_trades_this_cycle = sum(len(t) for t in all_trades) if all_trades else 0
    trades_per_pair = n_trades_this_cycle / len(capped_pairs) if capped_pairs else float("nan")
    cycle_days = (next_rebal_date - rebal_date).days
    days_per_trade_str = f"{cycle_days / trades_per_pair:.0f}" if trades_per_pair else "inf (no trades)"
    print(f"  Trade frequency diagnostic: {n_trades_this_cycle} trades across {len(capped_pairs)} pairs "
          f"this cycle = {trades_per_pair:.2f} trades/pair over ~{cycle_days} days (a full round trip every "
          f"~{days_per_trade_str} days per pair, on average). The adaptive window shrinks each "
          f"pair's rolling std to match its own fast half-life -- a std estimated over only ~10-20 days is "
          f"itself a noisy, low-degrees-of-freedom estimate, which can cross Z_ENTRY on its own estimation "
          f"noise rather than a genuine repeated reversion. High trades/pair here is a warning sign worth "
          f"weighing against that mechanism, not just evidence of a strong edge.")

    pair_specs = [{"name": p["name"], "dependent": p["dependent"], "independent": p["independent"],
                    "beta": p["beta"], "intercept": p["intercept"], "half_life": p["half_life"],
                    "variance_ratio": p["variance_ratio"], "cointegration_p": p["cointegration_p"],
                    "beta_stability": beta_stability_by_name.get(p["name"]),
                    "lookback_sharpe": p["sharpe"], "forward_sharpe": forward_sharpe_by_name.get(p["name"]),
                    "vol": pair_vol_by_name.get(p["name"])}
                   for p in capped_pairs]
    return forward_return, perf_rows, trades_df, pair_specs, all_pair_specs


def feature_predictive_power_analysis(all_pair_specs_cache, outcome_col="fwd_sharpe"):
    """THE core test the feedback is asking for, done honestly: for every
    candidate feature computed ONLY from the lookback window, does it
    actually correlate with an outcome computed ONLY from the forward
    window it never saw? Uses Spearman rank correlation (the 'Information
    Coefficient' in quant-finance terms) per cycle, then averages across
    cycles and checks sign consistency -- a feature that's positively
    correlated in 2 cycles and negatively in the other 4 is noise, even if
    the pooled correlation looks nonzero.

    This directly replaces "describe the top decile of Discovery Sharpe" --
    which risks re-discovering in-sample lucky-trade artifacts -- with
    "does this lookback statistic predict a genuinely unseen forward
    outcome", generalizing what ranking_validation_analysis already did for
    lookback_sharpe alone to every candidate feature at once."""
    rows = []
    for cycle_num, rebal_date, next_rebal, specs in all_pair_specs_cache:
        if len(specs) < 20:
            continue
        df = pd.DataFrame(specs)
        rows.append((cycle_num, df))

    candidate_features = [
        "lb_sharpe", "lb_maxdd", "lb_ret", "lb_win_rate", "lb_expectancy_pct",
        "lb_p95_loss_pct", "lb_p99_loss_pct", "lb_days_held_cv", "lb_n_trades",
        "half_life", "variance_ratio", "cointegration_p", "beta_stability",
    ]
    # Features where LOWER is structurally "better" get flipped so a positive
    # IC always means "higher value of this transformed feature -> better
    # forward outcome", making signs directly comparable across features.
    lower_is_better = {"lb_maxdd", "lb_p95_loss_pct", "lb_p99_loss_pct", "lb_days_held_cv",
                        "half_life", "variance_ratio", "cointegration_p"}

    results = {}
    for feat in candidate_features:
        per_cycle_ic = []
        for cycle_num, df in rows:
            if feat not in df.columns or outcome_col not in df.columns:
                continue
            sub = df[[feat, outcome_col]].dropna()
            # LOWERED from 15 to 8: trade-count-gated features (lb_win_rate,
            # lb_expectancy_pct, lb_p95/p99_loss_pct, lb_days_held_cv) are
            # NaN for most pairs in BOTH the lookback and forward windows
            # simultaneously at ~0.5-0.7 trades/pair per 6-month cycle -- a
            # strict n>=15 was silently DROPPING these features from the
            # results dict entirely (never reaching any cycle's IC) rather
            # than honestly evaluating and failing them. 8 is still enough
            # for a Spearman correlation to mean something, not a rubber stamp.
            if len(sub) < 8:
                continue
            x = -sub[feat] if feat in lower_is_better else sub[feat]
            ic = x.corr(sub[outcome_col], method="spearman")
            if not np.isnan(ic):
                per_cycle_ic.append(ic)
        if not per_cycle_ic:
            continue
        mean_ic = float(np.mean(per_cycle_ic))
        sign_consistency = float(np.mean([np.sign(v) == np.sign(mean_ic) for v in per_cycle_ic])) if mean_ic != 0 else 0.0
        results[feat] = {"mean_ic": round(mean_ic, 4), "n_cycles": len(per_cycle_ic),
                          "sign_consistency": round(sign_consistency, 2),
                          "per_cycle_ic": [round(v, 3) for v in per_cycle_ic]}

    print("\n" + "=" * 80)
    print(f"FEATURE PREDICTIVE POWER  (Spearman IC of lookback feature vs. {outcome_col}, "
          f"a window the feature never saw -- 'lower is better' features flipped so + IC always = good)")
    print("=" * 80)
    if not results:
        print("No feature had enough overlapping data to compute an IC at all this run.")
        return results
    table = pd.DataFrame(results).T.sort_values("mean_ic", key=abs, ascending=False)
    print(tabulate.tabulate(table[["mean_ic", "sign_consistency", "n_cycles"]], headers="keys", tablefmt="pretty"))
    print(
        f"\nmean_ic near 0 (regardless of sign) = no real predictive power, matching what the\n"
        f"existing Ranking Validation table already showed for lookback_sharpe specifically.\n"
        f"sign_consistency < ~0.65 means the correlation flips sign cycle to cycle -- likely noise\n"
        f"even if the pooled mean_ic looks nonzero. A feature worth trusting needs BOTH a\n"
        f"meaningfully nonzero mean_ic AND consistent sign across most cycles."
    )
    return results


def _pooled_feature_table(all_pair_specs_cache, features):
    """Pools every cycle's all_pair_specs into one long DataFrame, restricted
    to the requested feature columns, for a correlation check across the
    FULL sample rather than per-cycle. Correlation here is about redundancy
    between features (does lb_sharpe just re-express lb_ret?), not about
    predictive power, so pooling across cycles is the right scope -- unlike
    the IC calculation, which must stay per-cycle to avoid conflating
    cross-sectional relationships with time-varying regime effects."""
    frames = []
    for cycle_num, rebal_date, next_rebal, specs in all_pair_specs_cache:
        if not specs:
            continue
        df = pd.DataFrame(specs)
        cols = [c for c in features if c in df.columns]
        if cols:
            frames.append(df[cols])
    if not frames:
        return pd.DataFrame(columns=features)
    return pd.concat(frames, ignore_index=True)


def _prune_correlated_features(selected, all_pair_specs_cache, max_corr=MAX_FEATURE_CORRELATION):
    """Greedily drops features that are too correlated with an
    already-kept, stronger feature -- prevents e.g. lb_sharpe, lb_ret, and
    lb_n_trades (plausibly near-duplicates of the same 'pair looked
    unusually strong recently' signal) from all entering the composite and
    getting triple-counted. Confirmed empirically to matter: applying all
    4 originally-validated features live performed WORSE (Sharpe 0.629 at
    5bps) than a 2-feature subset that happened to avoid this redundancy
    (Sharpe 0.887) -- so this isn't a theoretical concern, it's a measured
    failure mode.

    Keeps the feature with the larger |mean_ic| whenever two features
    exceed max_corr; ties are broken by whichever was considered first
    (features are processed in descending |mean_ic| order, so the stronger
    one is always kept)."""
    if len(selected) < 2:
        return selected, pd.DataFrame()

    table = _pooled_feature_table(all_pair_specs_cache, list(selected.keys()))
    if table.empty or table.shape[1] < 2:
        return selected, pd.DataFrame()

    corr = table.corr(method="pearson")
    print("\n" + "-" * 80)
    print("PAIRWISE FEATURE CORRELATION  (pooled across all cycles, before pruning)")
    print("-" * 80)
    print(tabulate.tabulate(corr.round(3), headers="keys", tablefmt="pretty"))

    ordered = sorted(selected.keys(), key=lambda f: abs(selected[f]), reverse=True)
    kept, dropped = [], []
    for feat in ordered:
        too_correlated_with = None
        for kept_feat in kept:
            if feat in corr.index and kept_feat in corr.columns:
                c = corr.loc[feat, kept_feat]
                if not np.isnan(c) and abs(c) > max_corr:
                    too_correlated_with = (kept_feat, c)
                    break
        if too_correlated_with is None:
            kept.append(feat)
        else:
            dropped.append((feat, too_correlated_with[0], too_correlated_with[1]))

    if dropped:
        print(f"\nDropped for redundancy (|corr| > {max_corr} with an already-kept, stronger feature):")
        for feat, kept_feat, c in dropped:
            print(f"  {feat}  (corr={c:.3f} with {kept_feat}, IC={selected[feat]:.4f} vs "
                  f"{kept_feat}'s IC={selected[kept_feat]:.4f} -- weaker one dropped)")
    else:
        print(f"\nNo pair exceeded |corr| > {max_corr} -- no redundancy pruning needed.")

    pruned = {f: selected[f] for f in kept}
    return pruned, corr


def build_and_save_composite_score(all_pair_specs_cache, min_ic=MIN_IC_FOR_INCLUSION,
                                    min_consistency=MIN_IC_SIGN_CONSISTENCY, path=SCORE_WEIGHTS_PATH,
                                    feature_subset=SCORE_FEATURE_SUBSET):
    """Builds a score ONLY from features that pass an honest predictive-power
    bar (see feature_predictive_power_analysis) against BOTH a risk-adjusted
    outcome (fwd_sharpe) and a raw net-of-cost outcome (fwd_expectancy_pct)
    -- deliberately not an unregularized hand-picked weighted formula, since
    with a dozen correlated candidate features and only a handful of cycles,
    an unconstrained composite risks re-fitting noise the same way lookback
    Sharpe alone did. Weight = the feature's own mean IC (sign-adjusted, so
    'lower is better' features are already flipped by the analysis above) --
    a simple, transparent, IC-weighted composite, not a fitted regression.

    NEW: after the IC bar, features that survive are checked pairwise for
    redundancy (see _prune_correlated_features) -- validating on the IC bar
    alone doesn't catch two features that are really the same signal
    measured twice, which empirically made a real Phase 2 run WORSE, not
    better, once fully applied.

    feature_subset: if given, bypasses BOTH the IC bar and correlation
    pruning and forces exactly this list of features into the score
    (still requires each to have been computed by feature_predictive_power_
    analysis so a sign-appropriate IC weight exists) -- for directly testing
    a specific hand-picked combination, e.g. to replicate or compare against
    a subset found informally during debugging.

    If NOTHING clears the bar (or feature_subset is empty/invalid), this
    saves an explicitly unvalidated (empty-weights) file rather than
    forcing a score -- "none of these Discovery-time characteristics
    predict forward profitability" is itself a real, useful finding, and
    apply_composite_score_filter treats an empty weights file as a no-op
    with a warning, never a silent wrong filter."""
    fwd_sharpe_results = feature_predictive_power_analysis(all_pair_specs_cache, "fwd_sharpe")
    fwd_expectancy_results = feature_predictive_power_analysis(all_pair_specs_cache, "fwd_expectancy_pct")

    if feature_subset is not None:
        print("\n" + "=" * 80)
        print(f"COMPOSITE SCORE CONSTRUCTION  (MANUAL OVERRIDE -- forcing feature_subset={feature_subset}, "
              f"bypassing IC bar and correlation pruning)")
        print("=" * 80)
        selected = {}
        for feat in feature_subset:
            r2 = fwd_expectancy_results.get(feat)
            if r2 is None:
                print(f"  {feat}: no IC available on fwd_expectancy_pct (too few overlapping cycles) -- "
                      f"cannot assign a weight, skipping.")
                continue
            selected[feat] = r2["mean_ic"]
        if not selected:
            print("None of the requested features had a usable IC. Saving empty/unvalidated score.")
            with open(path, "w") as f:
                json.dump({"weights": {}, "validated": False}, f, indent=2)
            return {}
        total = sum(abs(v) for v in selected.values())
        weights = {k: round(v / total, 4) for k, v in selected.items()}
        print(f"Forced features and weights: {weights}")
        with open(path, "w") as f:
            json.dump({"weights": weights, "validated": True, "manual_override": True,
                        "lower_is_better": ["lb_maxdd", "lb_p95_loss_pct", "lb_p99_loss_pct", "lb_days_held_cv",
                                             "half_life", "variance_ratio", "cointegration_p"]}, f, indent=2)
        print(f"Saved to {path}")
        return weights

    # PRIMARY outcome is fwd_expectancy_pct (net-of-cost, the thing that
    # actually matters), with a minimum-cycle-count requirement so a feature
    # that was only evaluable in 1-2 cycles doesn't get treated the same as
    # one tested and consistently rejected across all 6. fwd_sharpe is used
    # only as a CONFIRMING check -- it must not actively contradict the
    # expectancy result, but a feature that's simply data-starved on
    # fwd_sharpe (too few cycles) isn't penalized for that.
    min_cycles_required = 4
    selected = {}
    for feat in set(fwd_sharpe_results) | set(fwd_expectancy_results):
        r1, r2 = fwd_sharpe_results.get(feat), fwd_expectancy_results.get(feat)
        if r2 is None or r2["n_cycles"] < min_cycles_required:
            continue
        if not (abs(r2["mean_ic"]) >= min_ic and r2["sign_consistency"] >= min_consistency):
            continue
        if r1 is not None and r1["n_cycles"] >= min_cycles_required:
            if np.sign(r1["mean_ic"]) != np.sign(r2["mean_ic"]) and r1["sign_consistency"] >= min_consistency:
                print(f"  {feat}: passed on fwd_expectancy_pct but CONTRADICTED by fwd_sharpe "
                      f"(IC={r1['mean_ic']}, consistency={r1['sign_consistency']}) -- excluding.")
                continue
        selected[feat] = r2["mean_ic"]

    print("\n" + "=" * 80)
    print(f"COMPOSITE SCORE CONSTRUCTION  (features must clear |IC|>={min_ic}, "
          f"sign_consistency>={min_consistency}, on BOTH fwd_sharpe and fwd_expectancy_pct)")
    print("=" * 80)
    if not selected:
        print("NO feature cleared the bar on both outcomes. The honest answer: none of the tested\n"
              "Discovery-time characteristics predict forward profitability at all -- a real, useful\n"
              "(if unwelcome) finding, not a bug. Not forcing a score in this case.")
        with open(path, "w") as f:
            json.dump({"weights": {}, "validated": False}, f, indent=2)
        return {}

    print(f"\nPassed the IC bar (pre-pruning): "
          f"{ {k: round(v, 4) for k, v in selected.items()} }")

    selected, _ = _prune_correlated_features(selected, all_pair_specs_cache)

    total = sum(abs(v) for v in selected.values())
    weights = {k: round(v / total, 4) for k, v in selected.items()}
    print(f"\nFinal validated features and weights (post-pruning): {weights}")

    with open(path, "w") as f:
        json.dump({"weights": weights, "validated": True, "manual_override": False,
                    "lower_is_better": ["lb_maxdd", "lb_p95_loss_pct", "lb_p99_loss_pct", "lb_days_held_cv",
                                         "half_life", "variance_ratio", "cointegration_p"]}, f, indent=2)
    print(f"Saved to {path}")
    return weights


def good_trades_entry_exit_analysis(trades_df):
    """Focused specifically on WINNING trades (return_% > 0) -- separate from
    every other diagnostic here, which mostly focuses on what makes trades
    go wrong. Two questions:
    (1) Entry timing: does waiting longer for confirmation (WAIT_FOR_Z_PEAK /
        entry_still_stationary delaying entry past the first Z-crossing)
        cost return even on trades that DO eventually win? If winners with
        a longer entry_wait_days have notably worse returns than winners
        that entered immediately, that's evidence we're entering too late
        even on the trades that work out.
    (2) Exit timing: for winners specifically, how much of the peak
        available profit (MFE) is actually captured at exit? A large gap
        means winners are being cut short before the move finishes."""
    if "return_%" not in trades_df.columns or len(trades_df) == 0:
        return
    winners = trades_df[trades_df["return_%"] > 0].copy()
    if len(winners) == 0:
        return

    print("\n" + "=" * 80)
    print(f"GOOD TRADES: ENTRY TIMING  (winners only, n={len(winners)}) -- does waiting longer for "
          f"confirmation cost return even on trades that end up winning?")
    print("=" * 80)
    if "entry_wait_days" in winners.columns and winners["entry_wait_days"].notna().any():
        winners["wait_bucket"] = pd.cut(
            winners["entry_wait_days"], bins=[-0.5, 0.5, 1.5, 3.5, 100],
            labels=["0 (entered immediately)", "1 day wait", "2-3 day wait", "4+ day wait"])
        wait_summary = winners.groupby("wait_bucket", observed=True).agg(
            n=("return_%", "size"), avg_return_pct=("return_%", "mean"),
            avg_mfe_pct=("mfe_%", "mean"), avg_days_held=("days_held", "mean"),
        ).round(3)
        print(tabulate.tabulate(wait_summary, headers="keys", tablefmt="pretty"))
        print(
            "\nIf avg_return_pct drops as wait increases, entries ARE happening too late -- confirmation\n"
            "is costing return even on trades that still end up winning, meaning some of the available\n"
            "move is being missed before we ever get in. If it's flat or improves with longer waits, the\n"
            "confirmation delay isn't the bottleneck on this side."
        )
    else:
        print("No entry_wait_days data available (WAIT_FOR_Z_PEAK/entry_still_stationary may be disabled).")

    print("\n" + "=" * 80)
    print(f"GOOD TRADES: EXIT TIMING  (winners only, n={len(winners)}) -- how much of the peak "
          f"available profit is actually captured at exit?")
    print("=" * 80)
    winners["capture_pct_of_mfe"] = np.where(winners["mfe_%"] > 0, winners["return_%"] / winners["mfe_%"] * 100, np.nan)
    print(f"Avg final return: {winners['return_%'].mean():.3f}%   Avg MFE: {winners['mfe_%'].mean():.3f}%   "
          f"Avg %% of MFE captured at exit: {winners['capture_pct_of_mfe'].mean():.1f}%")
    by_exit = winners.groupby("exit_reason").agg(
        n=("return_%", "size"), avg_return_pct=("return_%", "mean"), avg_mfe_pct=("mfe_%", "mean"),
        avg_capture_pct=("capture_pct_of_mfe", "mean"),
    ).round(3)
    print(tabulate.tabulate(by_exit, headers="keys", tablefmt="pretty"))
    print(
        "\ncapture_pct close to 100% means winners are riding close to their own best point before\n"
        "exiting -- exits are efficient. Notably below 100% means real profit is being given back\n"
        "before the exit triggers -- a genuine 'closing early' problem worth addressing directly\n"
        "(e.g. a trailing-stop-style exit instead of waiting for Z to reach a fixed target)."
    )


def entry_signal_and_excursion_analysis(trades_df):
    """Two diagnostics using the entry_z/mfe_%/mae_% columns extract_trades
    now captures:
    (1) Does a larger entry |Z| actually predict a larger subsequent
        reversion? Buckets trades by |entry_z| and compares average return.
    (2) MAE/MFE: for each trade, the best (MFE) and worst (MAE) cumulative
        P&L reached DURING the trade, not just at exit. If MFE is notably
        above the final return, exits are cutting winners short before
        they capture the move already available. If the final return sits
        close to MAE, the trade rode most of its own worst excursion --
        exits aren't the problem, the entry/relationship itself was weak."""
    if "entry_z" not in trades_df.columns or trades_df["entry_z"].isna().all():
        return
    print("\n" + "=" * 80)
    print("ENTRY SIGNAL QUALITY  (does a larger |entry Z| predict a larger reversion?)")
    print("=" * 80)
    df = trades_df.copy()
    df["abs_entry_z"] = df["entry_z"].abs()
    df["z_bucket"] = pd.cut(df["abs_entry_z"], bins=[2.0, 2.25, 2.5, 2.75, 3.0, 3.5, 10.0],
                             labels=["2.0-2.25", "2.25-2.5", "2.5-2.75", "2.75-3.0", "3.0-3.5", "3.5+"])
    z_summary = df.groupby("z_bucket", observed=True).agg(
        n=("return_%", "size"), avg_return_pct=("return_%", "mean"),
        win_rate_pct=("return_%", lambda s: (s > 0).mean() * 100),
    ).round(3)
    print(tabulate.tabulate(z_summary, headers="keys", tablefmt="pretty"))
    print(
        "\nIf avg_return_pct/win_rate rises with entry |Z|, more extreme entries genuinely predict\n"
        "stronger reversions -- raising Z_ENTRY would be a real, evidence-backed lever. If it's flat\n"
        "or non-monotonic, the entry threshold isn't discriminating trade quality at all."
    )

    if "adverse_momentum_at_entry" in df.columns and not df["adverse_momentum_at_entry"].isna().all():
        print("\n" + "=" * 80)
        print("ADVERSE MOMENTUM AT ENTRY  (was the spread still moving away from the mean at entry, "
              "or already turning back?)")
        print("=" * 80)
        df["mom_bucket"] = pd.cut(
            df["adverse_momentum_at_entry"],
            bins=[-np.inf, -0.5, 0.0, 0.5, 1.0, np.inf],
            labels=["<-0.5 (already reversing)", "-0.5-0.0", "0.0-0.5", "0.5-1.0", ">1.0 (still accelerating away)"])
        mom_summary = df.groupby("mom_bucket", observed=True).agg(
            n=("return_%", "size"), avg_return_pct=("return_%", "mean"),
            win_rate_pct=("return_%", lambda s: (s > 0).mean() * 100),
            avg_mfe_pct=("mfe_%", "mean"),
        ).round(3)
        print(tabulate.tabulate(mom_summary, headers="keys", tablefmt="pretty"))
        print(
            "\nPositive adverse_momentum_at_entry means Z was still moving further in the adverse\n"
            "direction over the prior few days when the trade was entered -- 'catching a falling knife.'\n"
            "If return/win-rate is notably worse in the positive buckets than the negative ones, this is\n"
            "a real, concrete filter: only enter once the adverse move has already started slowing, not\n"
            "while it's still accelerating -- directly testable before adding it as a hard requirement."
        )

    print("\n" + "=" * 80)
    print("MAE / MFE  (best and worst mark-to-market P&L reached DURING each trade)")
    print("=" * 80)
    print(f"Avg final return: {df['return_%'].mean():.3f}%   Avg MFE: {df['mfe_%'].mean():.3f}%   "
          f"Avg MAE: {df['mae_%'].mean():.3f}%")
    winners, losers = df[df["return_%"] > 0], df[df["return_%"] <= 0]
    if len(winners):
        print(f"Winning trades: avg final {winners['return_%'].mean():.3f}%  vs  avg MFE "
              f"{winners['mfe_%'].mean():.3f}%  (gap = upside left on the table by the exit rule)")
    if len(losers):
        print(f"Losing trades:  avg final {losers['return_%'].mean():.3f}%  vs  avg MAE "
              f"{losers['mae_%'].mean():.3f}%  (final close to MAE = the trade rode most of its own "
              f"drawdown; final well above MAE = the stop/exit already recovered ground)")
    print(
        "\nHow to read this: a large gap between avg MFE and avg final return on winners means real\n"
        "profit was building and then given back -- an exit-timing problem worth fixing directly.\n"
        "If final losing returns sit close to their own MAE, the exit rules aren't the main issue --\n"
        "these trades just moved against the position and stayed there."
    )


def repeat_loser_tickers_analysis(trades_df, n_worst_pct=10, min_overall_trades=5):
    """Checks whether the worst-performing trades keep involving the same
    individual tickers, disproportionate to how often that ticker trades
    AT ALL -- raw counts alone are misleading here, since MAX_PAIRS_PER_TICKER
    means popular tickers appear in many pairs and thus many trades
    regardless of quality. Computes a "lift" ratio: a ticker's share of
    worst-decile trades divided by its share of all trades. Lift > 1 means
    genuinely over-represented in losses, not just frequently traded."""
    if "pair" not in trades_df.columns or len(trades_df) == 0:
        return
    n_worst = max(1, int(len(trades_df) * n_worst_pct / 100))
    worst = trades_df.nsmallest(n_worst, "return_%")

    def ticker_counts_from(frame):
        tickers = []
        for pair in frame["pair"]:
            tickers.extend(pair.split("-"))
        return pd.Series(tickers).value_counts()

    overall_counts = ticker_counts_from(trades_df)
    worst_counts = ticker_counts_from(worst)
    overall_legs, worst_legs = overall_counts.sum(), worst_counts.sum()

    lift_df = pd.DataFrame({"overall_count": overall_counts}).join(
        pd.DataFrame({"worst_count": worst_counts}), how="left").fillna({"worst_count": 0})
    lift_df = lift_df[lift_df["overall_count"] >= min_overall_trades].copy()
    lift_df["overall_rate_%"] = (lift_df["overall_count"] / overall_legs * 100).round(3)
    lift_df["worst_rate_%"] = (lift_df["worst_count"] / worst_legs * 100).round(3)
    lift_df["lift"] = (lift_df["worst_rate_%"] / lift_df["overall_rate_%"]).round(2)
    lift_df = lift_df.sort_values("lift", ascending=False)

    print("\n" + "=" * 80)
    print(f"REPEAT LOSER TICKERS  (worst {n_worst_pct}% of trades = {n_worst} trades; lift = a "
          f"ticker's share of worst-decile trades vs. its share of all trades; only tickers with "
          f"{min_overall_trades}+ overall trades shown, to avoid noise from rarely-traded names)")
    print("=" * 80)
    top_lift = lift_df[lift_df["lift"] > 1.5].head(20)
    if len(top_lift):
        print("Tickers most OVER-represented in worst-decile losses relative to their own trading frequency:")
        print(tabulate.tabulate(top_lift[["overall_count", "worst_count", "lift"]].reset_index().rename(
            columns={"index": "ticker"}), headers="keys", tablefmt="pretty", showindex=False))
        print(
            "\nlift > 1 means that ticker shows up in worst-decile losses more than its overall trading\n"
            "frequency would predict -- a real, normalized candidate for exclusion, not just a name that\n"
            "happens to trade often. lift near 1 across the board would mean losses are genuinely spread\n"
            "proportionally, not concentrated in specific names."
        )
    else:
        print("No ticker shows meaningfully elevated lift in worst-decile losses -- once normalized by\n"
              "how often each ticker trades at all, losses look proportionally spread rather than\n"
              "concentrated in specific structurally-bad names.")


def ranking_validation_analysis(cycle_pair_cache):
    """Compares top-ranked vs bottom-ranked pairs (by lookback_sharpe, the
    same statistic used to rank pairs for the concentration cap) against
    their ACTUAL forward performance. If the ranking adds no real
    predictive value, forward performance should look similar across
    deciles regardless of lookback rank. Note: since this uses capped_pairs
    (the traded set, after the concentration cap), there's a mild selection
    bias -- a low-ranked pair sharing a ticker with a high-ranked one is
    more likely excluded -- so treat this as directionally informative,
    not a perfectly clean experiment."""
    rows = []
    for cycle_num, rebal_date, next_rebal, pair_specs in cycle_pair_cache:
        for spec in pair_specs:
            if spec.get("forward_sharpe") is not None and not (isinstance(spec["forward_sharpe"], float)
                                                                 and np.isnan(spec["forward_sharpe"])):
                rows.append({"lookback_sharpe": spec["lookback_sharpe"], "forward_sharpe": spec["forward_sharpe"]})
    if len(rows) < 20:
        return
    df = pd.DataFrame(rows).sort_values("lookback_sharpe").reset_index(drop=True)
    df["rank_decile"] = pd.qcut(df["lookback_sharpe"], 10, labels=False, duplicates="drop")
    summary = df.groupby("rank_decile").agg(
        n=("forward_sharpe", "size"), avg_lookback_sharpe=("lookback_sharpe", "mean"),
        avg_forward_sharpe=("forward_sharpe", "mean"),
    ).round(3)
    print("\n" + "=" * 80)
    print("RANKING VALIDATION  (decile 0 = lowest lookback_sharpe, decile 9 = highest)")
    print("=" * 80)
    print(tabulate.tabulate(summary, headers="keys", tablefmt="pretty"))
    print(
        "\nIf avg_forward_sharpe rises from decile 0 to decile 9 alongside avg_lookback_sharpe, the\n"
        "ranking has real predictive value -- worth keeping and possibly leaning on more. If forward\n"
        "Sharpe looks flat or unrelated to the lookback rank, the ranking isn't adding anything: a\n"
        "pair looking good in its own lookback window doesn't mean it'll perform well going forward."
    )


def _merged_trades_with_pair_characteristics(trades_df, cycle_pair_cache):
    """Shared helper: joins each trade back to the pair characteristics known
    at decision time (beta, half-life, variance ratio, discovery p-value,
    beta stability -- all from the lookback window, never the forward window
    being traded, to avoid the same look-ahead problem fixed elsewhere)."""
    char_rows = []
    for cycle_num, rebal_date, next_rebal, pair_specs in cycle_pair_cache:
        for spec in pair_specs:
            char_rows.append({"cycle": cycle_num, "pair": spec["name"], "abs_beta": abs(spec["beta"]),
                               "half_life": spec["half_life"], "variance_ratio": spec["variance_ratio"],
                               "cointegration_p": spec["cointegration_p"],
                               "beta_stability": spec.get("beta_stability")})
    if not char_rows:
        return None
    char_df = pd.DataFrame(char_rows)
    merged = trades_df.merge(char_df, on=["cycle", "pair"], how="left")
    return merged if not merged["abs_beta"].isna().all() else None


def winners_vs_losers_comparison(trades_df, cycle_pair_cache):
    """Direct P(X | winner) vs P(X | loser) comparison across every known-
    at-entry characteristic tracked, instead of eyeballing the worst 10
    trades against no baseline (the base-rate fallacy: a characteristic that
    is the MODAL value across ~90% of ALL trades will trivially show up in
    10/10 worst trades too, without that meaning it discriminates anything).
    Reports mean, median, and (where a threshold is meaningful) the
    fraction crossing it, for both groups side by side."""
    merged = _merged_trades_with_pair_characteristics(trades_df, cycle_pair_cache)
    if merged is None:
        return
    merged = merged.copy()
    merged["abs_entry_z"] = merged["entry_z"].abs() if "entry_z" in merged.columns else np.nan
    winners = merged[merged["return_%"] > 0]
    losers = merged[merged["return_%"] <= 0]
    if len(winners) == 0 or len(losers) == 0:
        return

    print("\n" + "=" * 80)
    print(f"WINNERS vs LOSERS COMPARISON  (winners n={len(winners)}, losers n={len(losers)}) -- "
          f"P(X | winner) vs P(X | loser) directly, not P(X | worst 10) with no baseline")
    print("=" * 80)
    rows = []
    specs = [
        ("adverse_momentum_at_entry", "adverse_momentum", [1.0, 2.0]),
        ("entry_wait_days", "entry_wait_days", None),
        ("abs_entry_z", "abs_entry_z", [2.5, 3.0]),
        ("abs_beta", "abs_beta", None),
        ("half_life", "half_life", None),
        ("variance_ratio", "variance_ratio", None),
        ("beta_stability", "beta_stability", None),
        ("cointegration_p", "discovery_p", None),
    ]
    for col, label, thresholds in specs:
        if col not in merged.columns or merged[col].isna().all():
            continue
        w, l = winners[col].dropna(), losers[col].dropna()
        if len(w) == 0 or len(l) == 0:
            continue
        row = {"feature": label, "winner_mean": w.mean(), "loser_mean": l.mean(),
               "winner_median": w.median(), "loser_median": l.median()}
        if thresholds:
            for th in thresholds:
                row[f"win_%>{th}"] = (w > th).mean() * 100
                row[f"lose_%>{th}"] = (l > th).mean() * 100
        rows.append(row)
    comp_df = pd.DataFrame(rows).set_index("feature").round(3)
    print(tabulate.tabulate(comp_df, headers="keys", tablefmt="pretty"))
    print(
        "\nA feature actually matters if its winner vs loser values (mean, median, and especially the\n"
        "%%>threshold columns) are MATERIALLY different -- not just present in both groups. If win_%%\n"
        "and lose_%% for a threshold are close (e.g. both ~85-90%%), that characteristic is common to\n"
        "most trades regardless of outcome and isn't a real discriminator, even if it shows up in\n"
        "every one of the worst 10 trades. A real signal shows a clear, sizeable gap between the two\n"
        "columns, not just presence in the loser group alone."
    )


def loser_characteristics_analysis(trades_df, cycle_pair_cache):
    """Joins each trade back to the pair characteristics that were known at
    decision time (beta, half-life, variance ratio, discovery p-value --
    NOT anything from the forward window, to avoid the same look-ahead
    problem already fixed elsewhere), then stratifies by return decile.
    Directly tests the hypothesis that large losses cluster around specific
    characteristics (e.g. extreme |beta|, very short half-life, weak
    variance ratio) rather than being evenly spread across all trades --
    turning "some pairs undergo structural breakdowns" into something
    checkable instead of a guess from eyeballing the worst 10 trades."""
    merged = _merged_trades_with_pair_characteristics(trades_df, cycle_pair_cache)
    if merged is None:
        return

    merged = merged.sort_values("return_%").reset_index(drop=True)
    merged["decile"] = pd.qcut(merged["return_%"], 10, labels=False, duplicates="drop")
    agg_kwargs = dict(
        n=("return_%", "size"), avg_return_pct=("return_%", "mean"),
        avg_abs_beta=("abs_beta", "mean"), avg_half_life=("half_life", "mean"),
        avg_variance_ratio=("variance_ratio", "mean"), avg_discovery_p=("cointegration_p", "mean"),
        avg_beta_stability=("beta_stability", "mean"),
    )
    # These two are already columns on trades_df (from extract_trades), added here so the worst
    # decile can be compared against the FULL population's distribution -- not just eyeballed from
    # the worst 10 rows, which can look distinctive purely because a characteristic is the MODAL
    # value across almost all trades (good and bad alike), not because it's actually discriminating.
    if "adverse_momentum_at_entry" in merged.columns:
        agg_kwargs["avg_adverse_momentum"] = ("adverse_momentum_at_entry", "mean")
        agg_kwargs["median_adverse_momentum"] = ("adverse_momentum_at_entry", "median")
        agg_kwargs["pct_mom_gt_1"] = ("adverse_momentum_at_entry", lambda s: (s > 1).mean() * 100)
        agg_kwargs["pct_mom_gt_2"] = ("adverse_momentum_at_entry", lambda s: (s > 2).mean() * 100)
    if "entry_wait_days" in merged.columns:
        agg_kwargs["avg_entry_wait_days"] = ("entry_wait_days", "mean")
    summary = merged.groupby("decile").agg(**agg_kwargs).round(3)
    print("\n" + "=" * 80)
    print("LOSER CHARACTERISTICS ANALYSIS  (trades stratified by return decile, decile 0 = worst)")
    print("=" * 80)
    print(tabulate.tabulate(summary, headers="keys", tablefmt="pretty"))
    print(
        "\nHow to read this: compare decile 0 (worst trades) against the middle deciles on each\n"
        "characteristic. If avg_abs_beta, avg_half_life, or avg_discovery_p is notably higher (or\n"
        "avg_variance_ratio notably different) in decile 0 than elsewhere, that characteristic is a\n"
        "real, checkable predictor of large losses -- a concrete filter to tighten. If the worst\n"
        "decile looks statistically similar to the rest, large losses aren't explained by these\n"
        "known-at-entry characteristics and are more likely genuine tail risk (regime shifts,\n"
        "relationships breaking) that filtering on these particular stats won't catch.\n"
        "Specifically for avg_adverse_momentum and avg_entry_wait_days: if decile 0 isn't clearly\n"
        "higher than the middle deciles, the worst-10-trades table showing 'high momentum, 1-day wait'\n"
        "was likely reflecting the MODAL value across almost all trades, not a real differentiator --\n"
        "check the full population's rate (printed in the ADVERSE MOMENTUM AT ENTRY section above)\n"
        "before concluding this characteristic explains the losses."
    )


def regime_label_for(date):
    """Looks up the regime label whose start date is the latest one on or
    before `date`, so cycles past the last explicit entry in REGIME_MAP
    reuse the final label rather than erroring."""
    date = pd.Timestamp(date)
    applicable = [(pd.Timestamp(k), v) for k, v in REGIME_MAP.items() if pd.Timestamp(k) <= date]
    return max(applicable, key=lambda x: x[0])[1] if applicable else "Unlabeled"


def run_phase2_oos():
    print(f"PHASE 2: rolling re-qualification OOS, {PHASE2_START} -> {PHASE2_END}")
    print(f"Lookback: {PHASE2_LOOKBACK_MONTHS} months, fixed size, never expanding back to Discovery/Selection data.")
    print(f"Rebalance every {PHASE2_REBALANCE_MONTHS} months.\n")

    ranges = load_validated_ranges()
    thresholds = operating_thresholds_from_ranges(ranges)
    print("Operating bands (full validated range, applied as [lo, hi] on each pair's own statistic):")
    for k, v in thresholds.items():
        print(f"  {k}: {v}")
    if USE_COMPOSITE_SCORE:
        print(f"\nUSE_COMPOSITE_SCORE is True: after the structural funnel above, pairs will additionally "
              f"be filtered to the top {SCORE_TOP_PERCENTILE}% by the composite score in {SCORE_WEIGHTS_PATH} "
              f"(if it exists and has validated weights).")

    tickers = get_sp500_tickers_from_excel()
    if EXCLUDED_TICKERS:
        before = len(tickers)
        excluded_found = sorted(EXCLUDED_TICKERS & set(tickers))
        tickers = [t for t in tickers if t not in EXCLUDED_TICKERS]
        print(f"EXCLUDED_TICKERS active: removed {before - len(tickers)} tickers from the universe "
              f"before any pair discovery ({excluded_found}).\n")
    lookback_start = pd.Timestamp(PHASE2_START) - pd.DateOffset(months=PHASE2_LOOKBACK_MONTHS)
    span_days = (pd.Timestamp(PHASE2_END) - lookback_start).days
    # MIN_DATA_POINTS (800) is calibrated for the 9.5-year Discovery span — applied
    # unchanged here, every ticker would be filtered out before anything else could
    # run, since a ~2.5 year window simply doesn't contain 800 trading days. Scale
    # to ~85% of this window's expected trading days instead; the strict
    # dropna(how="any") completeness check right after this still does the real
    # work of requiring genuinely gap-free data, this is just a sane pre-filter.
    phase2_min_points = int(span_days / 365 * 252 * 0.85)
    data = download_universe(tickers, start=lookback_start, end=PHASE2_END, min_points=phase2_min_points)

    rebal_dates = build_phase2_rebalance_dates()
    segments, cycle_summaries, all_trades = [], [], []
    cycle_pair_cache = []      # (rebal_date, next_rebal, pair_specs) -- reused for cheap cost resweeping, no re-discovery
    all_pair_specs_cache = []  # NEW: (rebal_date, next_rebal, all_pair_specs) -- full passed population, every cycle,
                                # used by feature_predictive_power_analysis / build_and_save_composite_score
    for i, rebal_date in enumerate(rebal_dates):
        next_rebal = rebal_dates[i + 1] if i + 1 < len(rebal_dates) else pd.Timestamp(PHASE2_END)
        print("\n" + "=" * 80)
        print(f"CYCLE {i + 1}/{len(rebal_dates)}  rebalance {rebal_date.date()} -> {next_rebal.date()}  "
              f"(lookback from {(rebal_date - pd.DateOffset(months=PHASE2_LOOKBACK_MONTHS)).date()})")
        print("=" * 80)
        forward_return, perf_rows, trades_df, pair_specs, all_pair_specs = run_phase2_cycle(
            data, rebal_date, next_rebal, thresholds)
        regime = regime_label_for(rebal_date)
        all_pair_specs_cache.append((i + 1, rebal_date, next_rebal, all_pair_specs))  # NEW: always cache, even on a sit-out
        if forward_return is None:
            print("No pairs qualified this cycle; sitting out.")
            cycle_summaries.append({"rebalance": rebal_date.date(), "regime": regime, "n_pairs": 0,
                                     "ann_return_%": np.nan, "sharpe": np.nan, "max_dd_%": np.nan})
            continue
        print(f"-> {len(perf_rows)} pairs qualified and traded: {[r['Pair'] for r in perf_rows]}")
        segments.append(forward_return)
        cycle_metrics = portfolio_metrics(np.ones(1), forward_return.to_frame())
        cycle_summaries.append({"rebalance": rebal_date.date(), "regime": regime, "n_pairs": len(perf_rows),
                                 "ann_return_%": cycle_metrics["Ann_Ret_%"], "sharpe": cycle_metrics["Sharpe"],
                                 "max_dd_%": cycle_metrics["Max_DD_%"]})
        cycle_pair_cache.append((i + 1, rebal_date, next_rebal, pair_specs))
        if len(trades_df):
            trades_df.insert(0, "cycle", i + 1)
            all_trades.append(trades_df)

    if not segments:
        print("\nNo cycle produced a tradeable portfolio across the whole Phase 2 window.")
        return None

    combined = pd.concat(segments).sort_index()
    combined = combined[~combined.index.duplicated(keep="first")]
    overall = portfolio_metrics(np.ones(1), combined.to_frame())

    print("\n" + "=" * 80)
    print("PHASE 2 RESULT (rolling re-qualification, all cycles stitched together)")
    print("=" * 80)
    print(f"Cycles with trades: {sum(1 for c in cycle_summaries if c['n_pairs'] > 0)} / {len(rebal_dates)}")
    print(f"Ann. Return: {overall['Ann_Ret_%']:.2f}%   Ann. Vol: {overall['Ann_Vol_%']:.2f}%   "
          f"Sharpe: {overall['Sharpe']:.3f}   Max DD: {overall['Max_DD_%']:.2f}%   "
          f"Sortino: {overall['Sortino']:.3f}   Calmar: {overall['Calmar']:.3f}")

    print("\n" + "=" * 80)
    print("PERFORMANCE BY REGIME  (per-cycle, not the stitched aggregate above)")
    print("=" * 80)
    regime_df = pd.DataFrame(cycle_summaries).round(3)
    print(tabulate.tabulate(regime_df, headers="keys", tablefmt="pretty", showindex=False))
    regime_grouped = pd.DataFrame(cycle_summaries).dropna(subset=["sharpe"]).groupby("regime").agg(
        n_cycles=("sharpe", "size"), avg_ann_return_pct=("ann_return_%", "mean"),
        avg_sharpe=("sharpe", "mean"), worst_max_dd_pct=("max_dd_%", "max"),
    ).round(3)
    if len(regime_grouped) > 1:
        print("\nGrouped by regime label:")
        print(tabulate.tabulate(regime_grouped, headers="keys", tablefmt="pretty"))
    print(
        "\nThis is the actual test of a regime-sensitivity hypothesis: compare avg_sharpe and\n"
        "avg_ann_return_pct across regime labels directly. A real, evidence-backed claim needs this\n"
        "table to show a genuine, sizeable gap between regimes -- not just one weak cycle at the end\n"
        "of the window, which could as easily be noise from a single 6-month period as a real regime\n"
        "effect. Edit REGIME_MAP's labels/boundaries at the top of the file to match your own view of\n"
        "what each period represented, then re-run to see how sensitive this comparison is to the\n"
        "exact boundaries chosen."
    )

    trades_df = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()
    if len(trades_df):
        wins = trades_df[trades_df["return_%"] > 0]
        losses = trades_df[trades_df["return_%"] <= 0]
        print("\n" + "=" * 80)
        print(f"TRADE-LEVEL P&L  ({len(trades_df)} round-trip trades across all cycles)")
        print("=" * 80)
        print(f"Win rate: {len(wins)} / {len(trades_df)} ({len(wins) / len(trades_df) * 100:.1f}%)")
        print(f"Avg win: {wins['return_%'].mean():.2f}%   Avg loss: {losses['return_%'].mean():.2f}%   "
              f"(if avg loss magnitude >> avg win, losses are the problem even with a high win rate)")
        print(f"Worst trade: {trades_df['return_%'].min():.2f}%   Best trade: {trades_df['return_%'].max():.2f}%")
        print("\nExit reason breakdown:")
        exit_summary = trades_df.groupby("exit_reason").agg(
            n=("return_%", "size"), avg_return_pct=("return_%", "mean"),
            avg_days_held=("days_held", "mean"),
        ).round(2)
        print(tabulate.tabulate(exit_summary, headers="keys", tablefmt="pretty"))
        print(
            "\nreversion = normal exit, spread came back as expected.\n"
            "stop_loss = Z kept moving against the position past Z_STOP -- capped what would otherwise\n"
            "  have been a larger, uncapped loss under the old no-stop rule.\n"
            "time_stop = position never reverted OR stopped out within MAX_HOLDING_DAYS -- forced flat rather\n"
            "  than left open indefinitely. If this row's avg_return_pct is strongly negative, that's\n"
            "  evidence of relationships that broke and never came back -- exactly the failure mode the\n"
            "  stop-loss was added to cap, not eliminate; it still costs money, just a bounded amount."
        )
        print("\n10 worst trades:")
        print(tabulate.tabulate(trades_df.nsmallest(10, "return_%"), headers="keys", tablefmt="pretty", showindex=False))

        good_trades_entry_exit_analysis(trades_df)
        entry_signal_and_excursion_analysis(trades_df)
        repeat_loser_tickers_analysis(trades_df)
        loser_characteristics_analysis(trades_df, cycle_pair_cache)
        winners_vs_losers_comparison(trades_df, cycle_pair_cache)
        ranking_validation_analysis(cycle_pair_cache)

        trades_df.to_csv(os.path.join(OUTPUT_DIR, "phase2_trades.csv"), index=False)
        print(f"\nFull trade log saved to {os.path.join(OUTPUT_DIR, 'phase2_trades.csv')}")

    with open(os.path.join(OUTPUT_DIR, "phase2_results.json"), "w") as f:
        json.dump({"cycle_summaries": [dict(c) for c in cycle_summaries],
                    "ann_return_%": round(overall["Ann_Ret_%"], 2), "sharpe": round(overall["Sharpe"], 3),
                    "max_dd_%": round(overall["Max_DD_%"], 2)}, f, indent=2, default=str)
    print(f"\nSaved to {os.path.join(OUTPUT_DIR, 'phase2_results.json')}")

    cost_results = cost_sensitivity_analysis(data, cycle_pair_cache, cost_levels=[0, 5, 10])

    return {"combined_returns": combined, "overall_metrics": overall, "cycle_summaries": cycle_summaries,
            "trades_df": trades_df, "cost_sensitivity": cost_results,
            "all_pair_specs_cache": all_pair_specs_cache}  # NEW: available for ad-hoc score analysis after a normal run too


def cost_sensitivity_analysis(data, cycle_pair_cache, cost_levels=(0, 5, 10)):
    """Reruns ONLY the cheap signal-generation/P&L step (build_spread_and_signals)
    at each cost level, reusing the pairs already discovered in the real run
    above -- no re-running cointegration/regression, which is the actual
    expensive part. Answers directly: is the near-breakeven result being
    eaten by transaction costs, or is the gross edge itself just small?"""
    print("\n" + "=" * 80)
    print(f"COST SENSITIVITY ANALYSIS  (bps per leg: {list(cost_levels)})")
    print("=" * 80)
    rows = []
    for cost_bps in cost_levels:
        segments = []
        all_trade_returns = []
        for cycle_num, rebal_date, next_rebal, pair_specs in cycle_pair_cache:
            if not pair_specs:
                continue
            signal_data = data.loc[(rebal_date - pd.DateOffset(months=PHASE2_LOOKBACK_MONTHS)):next_rebal]
            pair_returns, pair_positions = {}, {}
            for spec in pair_specs:
                win, std_win = adaptive_window(spec["half_life"])
                df = build_spread_and_signals(signal_data, spec["dependent"], spec["independent"],
                                               spec["beta"], spec["intercept"], cost_bps=cost_bps,
                                               window=win, std_window=std_win)
                sub = df.loc[rebal_date:next_rebal]
                pair_returns[spec["name"]] = sub["Strategy_Return"].fillna(0)
                pair_positions[spec["name"]] = (sub["Position"].shift(1) != 0).astype(int)
                trades = extract_trades(df, start=rebal_date, end=next_rebal)
                if len(trades):
                    all_trade_returns.extend(trades["return_%"].tolist())
            if pair_returns:
                returns_df = pd.DataFrame(pair_returns)
                active_df = pd.DataFrame(pair_positions)
                vol_series = pd.Series({spec["name"]: spec.get("vol") for spec in pair_specs}).reindex(returns_df.columns)
                median_vol = vol_series.median()
                fallback_vol = median_vol if not np.isnan(median_vol) and median_vol > 0 else 1.0
                vol_series = vol_series.fillna(fallback_vol).replace(0, fallback_vol)
                inv_vol = 1.0 / vol_series
                raw_weight = active_df.mul(inv_vol, axis=1)
                raw_weight_sum = raw_weight.sum(axis=1)
                weight_df = raw_weight.div(raw_weight_sum.replace(0, np.nan), axis=0).fillna(0)
                weight_df = weight_df.clip(upper=MAX_POSITION_WEIGHT)
                seg = (returns_df * weight_df).sum(axis=1)
                segments.append(seg)
        if not segments:
            rows.append({"cost_bps_per_leg": cost_bps, "ann_return_%": np.nan, "sharpe": np.nan,
                         "max_dd_%": np.nan, "n_trades": 0, "avg_trade_%": np.nan})
            continue
        combined = pd.concat(segments).sort_index()
        combined = combined[~combined.index.duplicated(keep="first")]
        port = portfolio_metrics(np.ones(1), combined.to_frame())
        rows.append({
            "cost_bps_per_leg": cost_bps, "ann_return_%": round(port["Ann_Ret_%"], 2),
            "sharpe": round(port["Sharpe"], 3), "max_dd_%": round(port["Max_DD_%"], 2),
            "n_trades": len(all_trade_returns),
            "avg_trade_%": round(float(np.mean(all_trade_returns)), 3) if all_trade_returns else np.nan,
        })
    result_df = pd.DataFrame(rows)
    print(tabulate.tabulate(result_df, headers="keys", tablefmt="pretty", showindex=False))
    print(
        "\nHow to read this: if Sharpe/return improve sharply as cost drops toward 0, the near-breakeven\n"
        "result at the real cost level is COST-DOMINATED -- the gross edge is real but too small relative\n"
        "to trading costs at this trade frequency. If Sharpe stays weak even at 0bps, costs aren't the\n"
        "main issue -- the entry/exit/hedge-ratio mechanics themselves aren't capturing a real edge."
    )
    result_df.to_csv(os.path.join(OUTPUT_DIR, "cost_sensitivity.csv"), index=False)
    print(f"Saved to {os.path.join(OUTPUT_DIR, 'cost_sensitivity.csv')}")
    return result_df


def run_score_validation():
    """NEW entry point for RUN_SCORE_VALIDATION=True. Runs the same rolling-
    cycle machinery as Phase 2, but with PERMISSIVE structural thresholds
    (only the hardcoded sanity bounds -- phi/ADF, |beta| range -- are
    applied) so the candidate population being scored is as broad as
    possible, not already pre-shrunk by Phase-1-chosen thresholds. Trades
    nothing and never applies the composite score while building it (that
    would be circular); purely collects lookback features + forward
    outcomes across cycles, then runs the honest predictive-power test and
    (if anything clears the bar) builds and saves a composite score."""
    print("SCORE VALIDATION: broad lookback-feature vs. forward-outcome test across "
          f"{PHASE2_START} -> {PHASE2_END}\n")
    permissive_thresholds = dict(
        distance_pctile=(0.0, 100.0), zero_crossing_pctile=(0.0, 100.0),
        half_life_pctile=(0.0, 100.0), r2_pctile=(0.0, 100.0),
        variance_ratio_pctile=(0.0, 100.0), min_persistence=(0.0, 1.0),
        perf_sharpe=(-10.0, 10.0), perf_maxdd=(0.0, 5.0), perf_return=(0.0, 10.0),
    )
    tickers = get_sp500_tickers_from_excel()
    if EXCLUDED_TICKERS:
        tickers = [t for t in tickers if t not in EXCLUDED_TICKERS]
    lookback_start = pd.Timestamp(PHASE2_START) - pd.DateOffset(months=PHASE2_LOOKBACK_MONTHS)
    span_days = (pd.Timestamp(PHASE2_END) - lookback_start).days
    data = download_universe(tickers, start=lookback_start, end=PHASE2_END,
                              min_points=int(span_days / 365 * 252 * 0.85))

    all_pair_specs_cache = []
    rebal_dates = build_phase2_rebalance_dates()
    for i, rebal_date in enumerate(rebal_dates):
        next_rebal = rebal_dates[i + 1] if i + 1 < len(rebal_dates) else pd.Timestamp(PHASE2_END)
        print(f"\nCYCLE {i + 1}/{len(rebal_dates)}: {rebal_date.date()} -> {next_rebal.date()} "
              f"(permissive structural thresholds -- scoring the broad population, not filtering)")
        # Never let a stale composite score filter the population WHILE we're
        # trying to build a fresh one -- that would be circular.
        prior_score_flag = globals()["USE_COMPOSITE_SCORE"]
        globals()["USE_COMPOSITE_SCORE"] = False
        _, _, _, _, all_specs = run_phase2_cycle(data, rebal_date, next_rebal, permissive_thresholds)
        globals()["USE_COMPOSITE_SCORE"] = prior_score_flag
        all_pair_specs_cache.append((i + 1, rebal_date, next_rebal, all_specs))

    weights = build_and_save_composite_score(all_pair_specs_cache)
    if weights:
        print(f"\nDone. Set USE_COMPOSITE_SCORE=True and RUN_PHASE2_OOS=True, then re-run to trade the "
              f"top {SCORE_TOP_PERCENTILE}% by this validated score.")
    else:
        print("\nDone. No feature validated -- USE_COMPOSITE_SCORE would be a no-op (with a warning) "
              "if enabled now. Consider the honest possibility that none of the tested characteristics "
              "predict forward profitability, before trying more features.")
    return all_pair_specs_cache


def main():
    print(f"Discovery window:  {DATA_START} -> {DISCOVERY_END}")
    print(f"Selection window:  {DISCOVERY_END} -> {SELECTION_END}")
    print(f"OOS window (2024-07-01 onward) is NOT downloaded by this script.\n")

    tickers = get_sp500_tickers_from_excel()
    print(f"Loaded {len(tickers)} tickers from Excel.")
    data = download_universe(tickers)  # capped at DOWNLOAD_END, never touches OOS

    discovery_data = data.loc[DATA_START:DISCOVERY_END]
    selection_start = (pd.Timestamp(DISCOVERY_END) + pd.Timedelta(days=1))
    selection_end = pd.Timestamp(SELECTION_END)

    # ---- Discovery ----
    close_pairs, norm_disc, distances = distance_prefilter(discovery_data)
    # BUG FIX: see compute_full_pair_statistics -- cointegration/regression
    # now use raw discovery_data, not the normalized norm_disc (which is
    # kept strictly for the distance/SSD calculation above).
    cointegrated = cointegration_filter(discovery_data, close_pairs)
    reg_results = fit_direction_and_beta(discovery_data, cointegrated)
    zc_filtered, zc_threshold_used = zero_crossing_filter(reg_results)
    candidates, half_life_population = ou_selection(zc_filtered)

    plot_discovery_diagnostics(
        distances,
        [r["zero_crossings"] for r in reg_results],
        zc_threshold_used,
        [r["half_life"] for r in half_life_population],
    )

    funnel = {
        "n_close_pairs": len(close_pairs), "n_coint": len(cointegrated),
        "n_zero_crossing": len(zc_filtered), "n_ou_candidates": len(candidates),
        "n_passed_selection": 0,
    }

    if not candidates:
        print("\nNo candidates survived Discovery filtering.")
        log_iteration(funnel)
        print_iteration_history()
        return

    # ---- Selection ----
    candidate_series = {
        f"{r['dependent']}-{r['independent']}": build_spread_and_signals(
            data, r["dependent"], r["independent"], r["beta"], r["intercept"]
        )
        for r in candidates
    }

    sel_metrics = {name: slice_metrics(df, selection_start, selection_end) for name, df in candidate_series.items()}
    passed = [
        name for name, m in sel_metrics.items()
        if m["sharpe_ratio"] > PERF_SHARPE_MIN
        and (not np.isnan(m["max_drawdown"])) and m["max_drawdown"] < PERF_MAXDD_MAX
        and (not np.isnan(m["final_return"])) and m["final_return"] > PERF_RETURN_MIN
    ]

    candidate_table = pd.DataFrame([
        {
            "Pair": name,
            "Sharpe": round(m["sharpe_ratio"], 2),
            "Max_DD_%": round(m["max_drawdown"] * 100, 2) if not np.isnan(m["max_drawdown"]) else float("nan"),
            "Return_%": round((m["final_return"] - 1) * 100, 2) if not np.isnan(m["final_return"]) else float("nan"),
            "N_Trades": m["n_trades"],
            "Days_Active": m["days_active"],
            "%_Active": round(m["pct_active"], 1),
            "Passed": name in passed,
        }
        for name, m in sel_metrics.items()
    ]).sort_values("Sharpe", ascending=False)
    print(f"\nCandidate pairs evaluated in Selection window "
          f"(need Sharpe>{PERF_SHARPE_MIN}, MaxDD<{PERF_MAXDD_MAX*100:.0f}%, "
          f"Return>{(PERF_RETURN_MIN-1)*100:.1f}% over {_selection_months} months):")
    print(tabulate.tabulate(candidate_table, headers="keys", tablefmt="pretty", showindex=False))
    print("Days_Active / %_Active show how much of the window the pair actually held a position — "
          "Sharpe/Return above are averaged over the FULL window including flat days, so a pair active "
          "only a small % of the time will look weaker here than its performance while actually in a trade.")

    funnel["n_passed_selection"] = len(passed)
    funnel["candidate_table"] = candidate_table.to_dict(orient="records")

    if not passed:
        print("\nNo candidates passed the Selection performance filters.")
        log_iteration(funnel)
        print_iteration_history()
        return

    sel_returns_df = pd.DataFrame({name: candidate_series[name].loc[selection_start:selection_end, "Strategy_Return"].fillna(0)
                                    for name in passed})
    weight_schemes = compute_weights(sel_returns_df)
    sel_port_metrics = {wname: portfolio_metrics(w, sel_returns_df) for wname, w in weight_schemes.items()}
    best_method = max(sel_port_metrics, key=lambda k: sel_port_metrics[k]["Sharpe"])
    best_weights = weight_schemes[best_method]

    print("\n" + "=" * 80)
    print(f"SELECTION RESULT: {len(passed)} pairs, best method = {best_method}")
    print("=" * 80)
    m = sel_port_metrics[best_method]
    print(f"Ann. Return: {m['Ann_Ret_%']:.2f}%   Ann. Vol: {m['Ann_Vol_%']:.2f}%   Sharpe: {m['Sharpe']:.3f}   "
          f"Max DD: {m['Max_DD_%']:.2f}%   Sortino: {m['Sortino']:.3f}   Calmar: {m['Calmar']:.3f}")
    print(f"Pairs and weights: {dict(zip(passed, [round(w, 3) for w in best_weights]))}")

    plot_pair_correlation(sel_returns_df)
    plot_weight_comparison(weight_schemes, passed)
    plot_selection_results(m, best_method)

    save_frozen_model(candidates, passed, best_method, best_weights)

    log_iteration({
        **funnel,
        "passed_pairs": passed,
        "weighting_method": best_method,
        "weights": dict(zip(passed, [round(float(w), 4) for w in best_weights])),
        "selection_sharpe": round(m["Sharpe"], 3),
        "selection_ann_return_%": round(m["Ann_Ret_%"], 2),
        "selection_max_dd_%": round(m["Max_DD_%"], 2),
    })
    print_iteration_history()

    oos_result = None
    if RUN_OOS_EVALUATION:
        oos_result = run_oos_evaluation(candidates, passed, best_weights)
    else:
        print("\nRUN_OOS_EVALUATION is False — OOS was not touched. Set it to True once the "
              "Discovery/Selection framework above is genuinely frozen and you're ready to test it.")

    return {
        "candidates": candidates, "passed_pairs": passed, "weight_schemes": weight_schemes,
        "best_method": best_method, "selection_metrics": sel_port_metrics, "oos_result": oos_result,
    }


def _prevent_sleep():
    """Windows equivalent of macOS 'caffeinate' -- tells Windows this process
    needs the system to stay awake, including keeping execution going with
    the display off/lid closed. No-op on non-Windows platforms."""
    if sys.platform != "win32":
        return
    ES_CONTINUOUS = 0x80000000
    ES_SYSTEM_REQUIRED = 0x00000001
    ES_AWAYMODE_REQUIRED = 0x00000040
    ctypes.windll.kernel32.SetThreadExecutionState(
        ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_AWAYMODE_REQUIRED
    )


def _allow_sleep():
    """Releases the sleep-prevention state set by _prevent_sleep(). Always
    called in a finally block so a normal sleep policy resumes once the
    script finishes or errors out -- this should never permanently disable
    sleep on the machine."""
    if sys.platform != "win32":
        return
    ES_CONTINUOUS = 0x80000000
    ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS)


if __name__ == "__main__":
    _prevent_sleep()
    try:
        if RUN_SCORE_VALIDATION:
            run_score_validation()
        elif RUN_PHASE1_VALIDATION:
            run_phase1_validation()
        elif RUN_PHASE2_OOS:
            run_phase2_oos()
        elif RUN_THRESHOLD_SWEEP:
            run_sweep_mode()
        elif SKIP_DISCOVERY_SELECTION:
            if not RUN_OOS_EVALUATION:
                print("SKIP_DISCOVERY_SELECTION is True but RUN_OOS_EVALUATION is False — "
                      "there's nothing to do (Discovery/Selection is skipped and OOS is off). "
                      "Set RUN_OOS_EVALUATION = True as well.")
            else:
                run_oos_only()
        else:
            main()
    finally:
        _allow_sleep()