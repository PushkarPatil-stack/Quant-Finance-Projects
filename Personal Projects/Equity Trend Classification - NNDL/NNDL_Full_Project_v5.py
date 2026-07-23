"""
Low-Frequency Signal Decomposition for Equity Trend Analysis
Using CNN-GRU Fusion  ——  VERSION 3
=============================================================
Project:  NNDL Final Project
Data:     NNDL_-_Final_Dataset.xlsx

Architecture:
  Preprocessing : causal dual-EWM decomposition (trend + cycle + noise)
  CNN-Short     : 1D conv on 20-day window  → inflection detection
  CNN-Medium    : 1D conv on 60-day window  → trend structure
  GRU           : GRU on 60-day window      → regime persistence
  Fusion head   : concat(CNN-S, CNN-M, GRU, regime_embed) → dense → 4-class softmax

Target:   SPY trend state: Trending-Up(3) / Transition-Up(2) /
                           Transition-Down(1) / Trending-Down(0)
          (Choppy merged into nearest Transition class)
Eval:     Chronological 50/25/25 train/val/test split, no lookahead at any stage
          Val set used exclusively for early stopping / model selection.
          Test set touched once for final reported metrics.
          Metrics: Directional Accuracy, Macro F1-Score
          Ablation: CNN-only, GRU-only, CNN-GRU fusion
          Regime analysis: trending vs mean-reverting periods
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
from scipy.signal import butter, filtfilt
from scipy.signal import argrelextrema
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score,
    accuracy_score, ConfusionMatrixDisplay
)
from sklearn.preprocessing import StandardScaler
from numpy.linalg import lstsq
import pickle, os, time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════
DATA_PATH  = r"C:\Users\Admin\OneDrive\Desktop\NCSU Semester 2\Neural Networks\NNDL - Final Dataset.xlsx"
SEQ_LEN_M  = 60    # medium-term lookback (CNN-medium + GRU)
SEQ_LEN_S  = 20    # short-term lookback  (CNN-short, inflection)
TRAIN_FRAC = 0.50   # 50 / 25 / 25 chronological split
VAL_FRAC   = 0.25   # val used ONLY for early stopping / model selection
BATCH_SIZE = 64
EPOCHS     = 60
LR         = 1e-3
PATIENCE   = 10

OUT_DIR = r"C:\Users\Admin\OneDrive\Desktop\NCSU Semester 2\Neural Networks\Results_v3"
os.makedirs(OUT_DIR, exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1: DATA LOADING  (unchanged from v1)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  SECTION 1: LOADING DATA")
print("=" * 70)

def load_ohlcv(sheet, close_col="Close", date_col="Date"):
    df = pd.read_excel(DATA_PATH, sheet_name=sheet, parse_dates=[date_col])
    df = df.rename(columns={date_col: "Date"})
    df = df.dropna(subset=["Date"]).set_index("Date").sort_index()
    return df

def load_price_only(sheet, price_col="Last Price"):
    df = pd.read_excel(DATA_PATH, sheet_name=sheet)
    df.columns = [str(c).strip() for c in df.columns]
    df = df.rename(columns={df.columns[0]: "Date", price_col: "Price"})
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).set_index("Date").sort_index()
    return df["Price"]

spy_df    = load_ohlcv("SPY")
spy_close = spy_df["Close"].dropna()
spy_vol   = spy_df["Volume"].dropna()
spy_high  = spy_df["High"].dropna()
spy_low   = spy_df["Low"].dropna()
print(f"  SPY   : {len(spy_close)} rows  {spy_close.index[0].date()} → {spy_close.index[-1].date()}")

qqq_close = load_ohlcv("QQQ")["Close"].dropna()
tlt_close = load_ohlcv("TLT")["Close"].dropna()

xle_df    = pd.read_excel(DATA_PATH, sheet_name="XLE")
xle_s     = pd.Series(xle_df.iloc[:, 1].values,
                       index=pd.to_datetime(xle_df.iloc[:, 0], errors="coerce"),
                       name="XLE").dropna()
xle_s     = xle_s[xle_s.index.notna()].sort_index()

xlf_df    = pd.read_excel(DATA_PATH, sheet_name="XLF")
xlf_price = xlf_df["Last Price"] if "Last Price" in xlf_df.columns else xlf_df.iloc[:, -1]
xlf_s     = pd.Series(xlf_price.values,
                       index=pd.to_datetime(xlf_df.iloc[:, 0], errors="coerce"),
                       name="XLF").dropna()
xlf_s     = xlf_s[xlf_s.index.notna()].sort_index()

gold_s    = load_price_only("Gold").rename("Gold")
move_s    = load_price_only("MOVE").rename("MOVE")

macro_daily_df = pd.read_excel(DATA_PATH, sheet_name="Macro_Daily", header=0)
vix_df  = macro_daily_df.iloc[:, [0, 1]].copy()
yr10_df = macro_daily_df.iloc[:, [3, 4]].copy()
yr2_df  = macro_daily_df.iloc[:, [6, 7]].copy()

def clean_macro_pair(df):
    df.columns = ["Date", "Value"]
    df["Date"]  = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).set_index("Date").sort_index()
    return df["Value"].dropna()

vix_s  = clean_macro_pair(vix_df).rename("VIX")
yr10_s = clean_macro_pair(yr10_df).rename("Yield10yr")
yr2_s  = clean_macro_pair(yr2_df).rename("Yield2yr")

cpi_df = pd.read_excel(DATA_PATH, sheet_name="Macro_Monthly")
cpi_df.columns = ["Date", "CPI_YoY"]
cpi_df["Date"] = pd.to_datetime(cpi_df["Date"], errors="coerce")
cpi_s  = cpi_df.dropna(subset=["Date"]).set_index("Date").sort_index()["CPI_YoY"]

print(f"  QQQ / TLT / XLE / XLF / Gold / MOVE loaded")
print(f"  Macro: VIX, 10yr, 2yr, CPI loaded")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2: IMPROVED SIGNAL DECOMPOSITION  [CHANGES 1 + 5]
#   - Dual-EWM trend (fast + slow) for inflection detection
#   - Cycle component: residual split into cycle + noise via 2nd EWM pass
#   - Butterworth kept for visualization only
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  SECTION 2: DUAL-SCALE CAUSAL SIGNAL DECOMPOSITION  [v2: +cycle]")
print("=" * 70)

def butterworth_noncausal(arr, cutoff=0.04, order=4):
    """Non-causal Butterworth for visualization only."""
    b, a = butter(order, cutoff, btype="low", analog=False)
    return filtfilt(b, a, arr)

def dual_ewm_decomposition(series):
    """
    Causal decomposition into 3 components:
      trend_fast  : EWM(span=20)  — tracks medium-term trend
      trend_slow  : EWM(span=60)  — tracks long-term structural trend
      cycle       : trend_fast - trend_slow  — medium-freq oscillation
      noise       : price - trend_fast       — high-freq residual
    All strictly causal (ewm adjust=False).
    """
    arr   = pd.Series(series.values.astype(float), index=series.index)
    t_fast = arr.ewm(span=20, adjust=False).mean()
    t_slow = arr.ewm(span=60, adjust=False).mean()
    cycle  = t_fast - t_slow
    noise  = arr - t_fast
    return t_fast, t_slow, cycle, noise

spy_arr    = spy_close.values.astype(float)
trend_viz  = butterworth_noncausal(spy_arr)          # viz only

trend_fast, trend_slow, cycle_comp, noise_comp = dual_ewm_decomposition(spy_close)

print(f"  trend_fast (EWM-20) σ  : ${trend_fast.std():.2f}")
print(f"  trend_slow (EWM-60) σ  : ${trend_slow.std():.2f}")
print(f"  cycle component σ      : ${cycle_comp.std():.2f}")
print(f"  noise component σ      : ${noise_comp.std():.2f}")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3: COMPREHENSIVE FEATURE ENGINEERING  [CHANGES 2 + 3 + 6]
#   [2] ADX, MA stack, 200d filter, S/R proximity, swing highs/lows
#   [3] Delta features (1-step changes) — added to master feature matrix
#   [6] Risk-on/off composite, trend concordance across indices
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  SECTION 3: FEATURE ENGINEERING  [v2: +ADX +S/R +delta +composite]")
print("=" * 70)

idx = spy_close.index

def align(s, idx):
    return s.reindex(idx).ffill().bfill()

# ── Helper: ADX (Average Directional Index) ───────────────────────────────────
def compute_adx(high, low, close, period=14):
    """
    Causal ADX — quantifies trend strength regardless of direction.
    Returns: adx, plus_di, minus_di as pd.Series aligned to close.index
    """
    h = high.reindex(close.index).ffill()
    l = low.reindex(close.index).ffill()
    c = close

    tr   = pd.concat([h - l,
                      (h - c.shift(1)).abs(),
                      (l - c.shift(1)).abs()], axis=1).max(axis=1)
    dm_p = (h - h.shift(1)).clip(lower=0).where((h - h.shift(1)) > (l.shift(1) - l), 0)
    dm_m = (l.shift(1) - l).clip(lower=0).where((l.shift(1) - l) > (h - h.shift(1)), 0)

    atr   = tr.ewm(span=period, adjust=False).mean()
    di_p  = 100 * dm_p.ewm(span=period, adjust=False).mean() / (atr + 1e-9)
    di_m  = 100 * dm_m.ewm(span=period, adjust=False).mean() / (atr + 1e-9)
    dx    = 100 * (di_p - di_m).abs() / (di_p + di_m + 1e-9)
    adx   = dx.ewm(span=period, adjust=False).mean()
    return adx, di_p, di_m

# ── Helper: swing high/low proximity ─────────────────────────────────────────
def swing_proximity(series, order=10, window=252):
    """
    Returns distance from recent swing high and swing low as % of price.
    Uses argrelextrema on a rolling basis (causal approximation).
    """
    arr = series.values
    n   = len(arr)
    dist_high = np.full(n, np.nan)
    dist_low  = np.full(n, np.nan)
    for i in range(window, n):
        seg = arr[max(0, i - window): i]
        local_max = argrelextrema(seg, np.greater, order=order)[0]
        local_min = argrelextrema(seg, np.less,    order=order)[0]
        if len(local_max):
            dist_high[i] = (arr[i] - seg[local_max[-1]]) / (seg[local_max[-1]] + 1e-9)
        if len(local_min):
            dist_low[i]  = (arr[i] - seg[local_min[-1]]) / (seg[local_min[-1]] + 1e-9)
    return (pd.Series(dist_high, index=series.index),
            pd.Series(dist_low,  index=series.index))

# ── Build master feature DataFrame ────────────────────────────────────────────
feat = pd.DataFrame(index=idx)

p   = spy_close
vol = spy_vol.reindex(idx).ffill()
tf  = trend_fast.reindex(idx)
ts  = trend_slow.reindex(idx)
cy  = cycle_comp.reindex(idx)
ns  = noise_comp.reindex(idx)

# ── [2a] Trend structure features ─────────────────────────────────────────────
feat["trend_fast_slope_5"]  = tf.diff(5)  / (tf.shift(5)  + 1e-9)
feat["trend_fast_slope_20"] = tf.diff(20) / (tf.shift(20) + 1e-9)
feat["trend_slow_slope_20"] = ts.diff(20) / (ts.shift(20) + 1e-9)
feat["trend_slow_slope_60"] = ts.diff(60) / (ts.shift(60) + 1e-9)
feat["trend_accel"]         = tf.diff().diff()   # 2nd derivative of fast trend
feat["trend_accel_slow"]    = ts.diff().diff()

# Fast vs slow trend relationship (inflection signal)
feat["trend_fast_vs_slow"]  = (tf - ts) / (ts + 1e-9)   # + = fast above slow = bullish
feat["trend_cross_signal"]  = np.sign(tf - ts)           # direction of fast-slow spread
feat["trend_cross_change"]  = feat["trend_cross_signal"].diff()  # when this ≠ 0 = crossover

feat["price_vs_fast"]       = (p - tf) / (tf + 1e-9)
feat["price_vs_slow"]       = (p - ts) / (ts + 1e-9)

# ── [5] Cycle component features ──────────────────────────────────────────────
feat["cycle_level"]         = cy / (p + 1e-9)   # cycle as % of price
feat["cycle_slope"]         = cy.diff(5)
feat["cycle_zscore"]        = (cy - cy.rolling(60).mean()) / (cy.rolling(60).std() + 1e-9)
feat["noise_abs_20"]        = ns.abs().rolling(20).mean() / (p + 1e-9)  # relative noise

# ── [2b] ADX — trend strength ─────────────────────────────────────────────────
adx, di_p, di_m = compute_adx(spy_high, spy_low, p)
feat["adx_14"]              = adx
feat["adx_zscore_60"]       = (adx - adx.rolling(60).mean()) / (adx.rolling(60).std() + 1e-9)
feat["di_diff"]             = di_p - di_m          # + = bullish directional pressure
feat["adx_trend_strength"]  = adx / 25.0           # normalised (ADX>25 = trending)

# ── [2c] MA stack — multi-timeframe alignment ──────────────────────────────────
ma5   = p.rolling(5).mean()
ma20  = p.rolling(20).mean()
ma50  = p.rolling(50).mean()
ma200 = p.rolling(200).mean()

feat["ma_cross_5_20"]       = ma5  / (ma20  + 1e-9) - 1
feat["ma_cross_20_50"]      = ma20 / (ma50  + 1e-9) - 1
feat["ma_cross_50_200"]     = ma50 / (ma200 + 1e-9) - 1
feat["price_vs_200d"]       = (p - ma200) / (ma200 + 1e-9)  # key trend filter
feat["ma_stack_score"]      = (                              # +1 per aligned condition
    (ma5 > ma20).astype(int) +
    (ma20 > ma50).astype(int) +
    (ma50 > ma200).astype(int)
).astype(float) / 3.0   # 0=fully bearish stack, 1=fully bullish stack

# ── [2d] Support/Resistance proximity ─────────────────────────────────────────
dist_swing_high, dist_swing_low = swing_proximity(p, order=10, window=252)
feat["dist_swing_high"]     = dist_swing_high
feat["dist_swing_low"]      = dist_swing_low
feat["dist_52w_high"]       = (p - p.rolling(252).max()) / (p.rolling(252).max() + 1e-9)
feat["dist_52w_low"]        = (p - p.rolling(252).min()) / (p.rolling(252).min() + 1e-9)

# Bollinger band position (dynamic S/R)
bb_mid = p.rolling(20).mean()
bb_std = p.rolling(20).std()
feat["bb_position"]         = (p - bb_mid) / (2 * bb_std + 1e-9)
feat["bb_width"]            = (2 * bb_std) / (bb_mid + 1e-9)  # volatility of range

# ── [2e] Standard price/vol features (kept from v1) ───────────────────────────
for w in [5, 10, 20, 60]:
    feat[f"spy_ret_{w}d"]  = p.pct_change(w)
    feat[f"spy_vol_{w}d"]  = p.pct_change().rolling(w).std()

delta = p.diff()
gain  = delta.clip(lower=0).rolling(14).mean()
loss  = (-delta.clip(upper=0)).rolling(14).mean()
feat["rsi_14"]              = 100 - (100 / (1 + gain / (loss + 1e-9)))
feat["vol_ratio_20"]        = vol / (vol.rolling(20).mean() + 1e-9)

# ── [6a] Cross-asset features ─────────────────────────────────────────────────
qqq = align(qqq_close, idx)
tlt = align(tlt_close, idx)
xle = align(xle_s,     idx)
xlf = align(xlf_s,     idx)

feat["qqq_ret_20d"]         = qqq.pct_change(20)
feat["tlt_ret_20d"]         = tlt.pct_change(20)
feat["xle_ret_20d"]         = xle.pct_change(20)
feat["xlf_ret_20d"]         = xlf.pct_change(20)
feat["spy_qqq_spread"]      = p.pct_change(20) - qqq.pct_change(20)
feat["spy_tlt_spread"]      = p.pct_change(20) - tlt.pct_change(20)
feat["xlf_xle_rel"]         = xlf.pct_change(20) - xle.pct_change(20)

# ── [6b] Trend concordance — are all indices trending together? ───────────────
def ma_trend_signal(s, fast=20, slow=60):
    """+1 if fast MA > slow MA (uptrend), -1 otherwise."""
    return np.sign(s.rolling(fast).mean() - s.rolling(slow).mean())

tc_spy = ma_trend_signal(p)
tc_qqq = ma_trend_signal(qqq)
tc_tlt = ma_trend_signal(tlt)
tc_xle = ma_trend_signal(xle)
tc_xlf = ma_trend_signal(xlf)

feat["trend_concordance"]   = (tc_spy + tc_qqq + tc_xlf + tc_xle) / 4.0  # -1 to +1
feat["equity_bond_diverge"] = tc_spy - tc_tlt   # equity and bond trend divergence

# ── [6c] Risk-on/off composite ────────────────────────────────────────────────
vix   = align(vix_s,  idx)
yr10  = align(yr10_s, idx)
yr2   = align(yr2_s,  idx)
gold  = align(gold_s, idx)
move  = align(move_s, idx)
cpi   = cpi_s.resample("D").ffill().reindex(idx).ffill().bfill()

vix_z  = (vix  - vix.rolling(60).mean())  / (vix.rolling(60).std()  + 1e-9)
move_z = (move - move.rolling(60).mean()) / (move.rolling(60).std() + 1e-9)
tlt_z  = (tlt  - tlt.rolling(60).mean())  / (tlt.rolling(60).std()  + 1e-9)
# Risk-off when VIX elevated + MOVE elevated + TLT rallying
feat["risk_off_composite"]  = (vix_z + move_z - tlt_z) / 3.0   # + = risk-off
feat["risk_off_direction"]  = feat["risk_off_composite"].diff(5)

# ── [6d] Macro features ───────────────────────────────────────────────────────
feat["vix_level"]           = vix
feat["vix_change_5d"]       = vix.pct_change(5)
feat["vix_zscore_60d"]      = vix_z
feat["yield_spread"]        = yr10 - yr2
feat["yield_spread_ch5"]    = (yr10 - yr2).diff(5)
feat["yield_10yr"]          = yr10
feat["gold_ret_20d"]        = gold.pct_change(20)
feat["gold_spy_ratio"]      = gold / (p + 1e-9)
feat["move_level"]          = move
feat["move_zscore_60d"]     = move_z
feat["cpi_yoy"]             = cpi
feat["cpi_change_3m"]       = cpi - cpi.shift(63)

n_features_base = feat.shape[1]
feat_names_base  = list(feat.columns)
print(f"  Base features: {n_features_base}")

# ── [3] DELTA FEATURES — 1-step changes at each timestep ──────────────────────
# These are critical for the GRU to detect dynamics rather than just levels.
# We compute deltas for the most informative features and store them separately;
# they will be stacked onto the sequence at sequence-build time.
DELTA_COLS = [
    "trend_fast_slope_5", "trend_fast_vs_slow", "trend_cross_signal",
    "adx_14", "di_diff", "ma_stack_score", "bb_position",
    "price_vs_fast", "price_vs_slow", "cycle_level", "cycle_zscore",
    "vix_level", "yield_spread", "risk_off_composite",
    "trend_concordance", "spy_ret_5d", "rsi_14"
]
DELTA_COLS = [c for c in DELTA_COLS if c in feat.columns]

delta_feat = feat[DELTA_COLS].diff(1)
delta_feat.columns = [f"Δ{c}" for c in DELTA_COLS]
feat = pd.concat([feat, delta_feat], axis=1)

n_features = feat.shape[1]
feat_names  = list(feat.columns)
print(f"  Total features (base + deltas): {n_features}")
print(f"  Delta features added: {len(DELTA_COLS)}")

# ── [7] REGIME FEATURE — ADX regime for conditioning the fusion head ──────────
# Discretised into 3 bins: Weak(0) / Moderate(1) / Strong(2) trend
adx_aligned = adx.reindex(idx).ffill().bfill()
adx_regime  = pd.cut(adx_aligned, bins=[-np.inf, 20, 35, np.inf],
                     labels=[0, 1, 2]).astype(float)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4: TREND-STATE LABELS  [CHANGE 1]
#
#   Instead of 20-day forward return buckets, label each day with its
#   CURRENT trend state based on the decomposed signal:
#
#   State 3 — Trending Up:    fast > slow, ADX > 20, DI+ > DI-
#   State 2 — Transition Up:  fast > slow but ADX weak OR recent crossover up
#   State 1 — Transition Down: fast < slow but ADX weak OR recent crossover dn
#   State 0 — Trending Down:  fast < slow, ADX > 20, DI- > DI+
#
#   This is strictly causal — all inputs are computed from historical data only.
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  SECTION 4: TREND-STATE LABELS  [v2: current regime, not fwd return]")
print("=" * 70)

fast_above_slow = (trend_fast > trend_slow).reindex(idx).astype(int)
adx_strong      = (adx_aligned > 20).astype(int)
di_bull         = (di_p.reindex(idx).ffill() > di_m.reindex(idx).ffill()).astype(int)

# Combine into 4-class trend state
def build_trend_state(fast_above, adx_str, di_b):
    state = np.zeros(len(fast_above), dtype=int)
    for i in range(len(fast_above)):
        fa = fast_above.iloc[i]
        ad = adx_str.iloc[i]
        db = di_b.iloc[i]
        if fa == 1 and ad == 1 and db == 1:
            state[i] = 3   # Trending Up
        elif fa == 1:
            state[i] = 2   # Transition Up (fast above slow but weak/mixed)
        elif fa == 0 and ad == 1 and db == 0:
            state[i] = 0   # Trending Down
        else:
            state[i] = 1   # Transition Down
    return pd.Series(state, index=fast_above.index)

labels_raw = build_trend_state(fast_above_slow, adx_strong, di_bull)

# ── Label smoothing: require a state to persist ≥5 days before recording ──────
# A rolling mode with window=5 suppresses single-day flickers where the trend
# state oscillates due to borderline ADX / DI readings.  This does NOT look
# forward — pd.Series.rolling is applied left-to-right and mode() only sees the
# current window of *past* labels (including the current day).
labels = (
    labels_raw
    .rolling(10, min_periods=1)
    .apply(lambda x: pd.Series(x).mode()[0], raw=False)
    .astype(int)
)
n_changed = (labels != labels_raw).sum()
print(f"  Label smoothing (window=10): {n_changed} days changed "
      f"({n_changed/len(labels)*100:.1f}% of samples)")

STATE_NAMES = {0: "Trending-Down", 1: "Trans-Down", 2: "Trans-Up", 3: "Trending-Up"}

valid_idx    = feat.dropna().index.intersection(labels.dropna().index)
feat_clean   = feat.loc[valid_idx]
labels_clean = labels.loc[valid_idx]
regime_clean = adx_regime.loc[valid_idx]

split_tr  = int(len(valid_idx) * TRAIN_FRAC)
split_val = int(len(valid_idx) * (TRAIN_FRAC + VAL_FRAC))
train_idx = valid_idx[:split_tr]
val_idx   = valid_idx[split_tr:split_val]
test_idx  = valid_idx[split_val:]

X_tr  = feat_clean.loc[train_idx].values
X_val = feat_clean.loc[val_idx].values
X_te  = feat_clean.loc[test_idx].values
y_tr  = labels_clean.loc[train_idx].values
y_val = labels_clean.loc[val_idx].values
y_te  = labels_clean.loc[test_idx].values

scaler    = StandardScaler()
X_tr_s    = scaler.fit_transform(X_tr)   # fit on train only
X_val_s   = scaler.transform(X_val)
X_te_s    = scaler.transform(X_te)

vc    = labels_clean.value_counts().sort_index()
naive = float(vc.max() / vc.sum())

print(f"  Total valid samples : {len(valid_idx)}")
print(f"  Train (50%)         : {len(train_idx)}  ({train_idx[0].date()} → {train_idx[-1].date()})")
print(f"  Val   (25%)         : {len(val_idx)}  ({val_idx[0].date()} → {val_idx[-1].date()})")
print(f"  Test  (25%)         : {len(test_idx)}  ({test_idx[0].date()} → {test_idx[-1].date()})")
for lbl, cnt in vc.items():
    print(f"    {STATE_NAMES[lbl]:15s} ({lbl}): {cnt:5d}  ({cnt/len(labels_clean)*100:.1f}%)")
print(f"  Naive baseline      : {naive:.4f}")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5: BASELINE MODELS
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  SECTION 5: BASELINE MODELS")
print("=" * 70)

# ── 5A: MA Stack Heuristic (maps directly to trend states) ────────────────────
print("\n── Baseline A: MA Stack Heuristic ──")
ma_stack_raw = feat_clean.loc[test_idx, "ma_stack_score"].values
y_ma = np.where(ma_stack_raw > 0.8,  3,
       np.where(ma_stack_raw > 0.5,  2,
       np.where(ma_stack_raw > 0.2,  1, 0))).astype(int)
acc_ma = accuracy_score(y_te, y_ma)
f1_ma  = f1_score(y_te, y_ma, average="macro", zero_division=0)
print(f"  Accuracy: {acc_ma:.4f}   F1 Macro: {f1_ma:.4f}")
print(classification_report(y_te, y_ma,
      target_names=[STATE_NAMES[i] for i in range(4)], zero_division=0))

# ── 5B: ADX + DI Rule-Based  (FIXED: uses t-1 values to eliminate lookahead) ──
# The label at time t is built from adx_t and di_t, so reading those same
# features at t to *predict* the label would be tautological and inflate scores
# to ~1.000.  Shifting by one day ensures the baseline only sees information
# that was available *before* the label was assigned.
print("\n── Baseline B: ADX + DI Rule-Based (lagged t-1) ──")
adx_te_lag  = feat_clean.loc[test_idx, "adx_14"].shift(1).fillna(method="bfill").values
di_te_lag   = feat_clean.loc[test_idx, "di_diff"].shift(1).fillna(method="bfill").values
fast_te_lag = feat_clean.loc[test_idx, "trend_fast_vs_slow"].shift(1).fillna(method="bfill").values
y_adx = np.where((fast_te_lag > 0) & (adx_te_lag > 20) & (di_te_lag > 0), 3,
        np.where((fast_te_lag > 0), 2,
        np.where((fast_te_lag < 0) & (adx_te_lag > 20) & (di_te_lag < 0), 0, 1))).astype(int)
acc_adx = accuracy_score(y_te, y_adx)
f1_adx  = f1_score(y_te, y_adx, average="macro", zero_division=0)
print(f"  Accuracy: {acc_adx:.4f}   F1 Macro: {f1_adx:.4f}")
print(classification_report(y_te, y_adx,
      target_names=[STATE_NAMES[i] for i in range(4)], zero_division=0))

# ── 5C: Random Forest ─────────────────────────────────────────────────────────
print("\n── Baseline C: Random Forest (300 trees) ──")
rf = RandomForestClassifier(
    n_estimators=300, max_depth=10, min_samples_leaf=15,
    class_weight="balanced", random_state=SEED, n_jobs=-1
)
rf.fit(X_tr_s, y_tr)
y_rf   = rf.predict(X_te_s)
acc_rf = accuracy_score(y_te, y_rf)
f1_rf  = f1_score(y_te, y_rf, average="macro", zero_division=0)
print(f"  Accuracy: {acc_rf:.4f}   F1 Macro: {f1_rf:.4f}")
print(classification_report(y_te, y_rf,
      target_names=[STATE_NAMES[i] for i in range(4)], zero_division=0))

importances = pd.Series(rf.feature_importances_, index=feat_names).sort_values(ascending=False)
print("\n  Top-10 Features:")
for i, (nm, v) in enumerate(importances.head(10).items()):
    print(f"    {i+1:2d}. {nm:35s}  {v:.4f}")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6: SEQUENCE DATASET — DUAL-SCALE  [CHANGE 4]
#   Each sample contains TWO sequence windows:
#     X_short  : (batch, SEQ_LEN_S=20, n_features)  — inflection detection
#     X_medium : (batch, SEQ_LEN_M=60, n_features)  — trend structure
#   Plus a scalar regime code per sample.
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  SECTION 6: DUAL-SCALE SEQUENCE DATASET  [v2: 20-day + 60-day]")
print("=" * 70)

feat_scaled_all   = scaler.transform(feat_clean.values)
label_arr         = labels_clean.values
regime_arr        = regime_clean.values.astype(np.int64)

def build_dual_sequences(X, y, regime, seq_short, seq_long):
    """
    Returns X_short (N, seq_short, F), X_medium (N, seq_long, F),
            y (N,), regime (N,)
    Indexed from seq_long onward so both windows are always valid.
    """
    Xs_s, Xs_m, ys, rs = [], [], [], []
    for i in range(seq_long, len(X)):
        Xs_s.append(X[i - seq_short: i])
        Xs_m.append(X[i - seq_long:  i])
        ys.append(y[i])
        rs.append(regime[i])
    return (np.array(Xs_s, dtype=np.float32),
            np.array(Xs_m, dtype=np.float32),
            np.array(ys,   dtype=np.int64),
            np.array(rs,   dtype=np.int64))

Xs_s, Xs_m, y_seq, r_seq = build_dual_sequences(
    feat_scaled_all, label_arr, regime_arr, SEQ_LEN_S, SEQ_LEN_M)

# Chronological boundaries in sequence space (offset by SEQ_LEN_M warmup)
seq_split_tr  = split_tr  - SEQ_LEN_M
seq_split_val = split_val - SEQ_LEN_M

Xs_s_tr,  Xs_s_val,  Xs_s_te  = Xs_s[:seq_split_tr],  Xs_s[seq_split_tr:seq_split_val],  Xs_s[seq_split_val:]
Xs_m_tr,  Xs_m_val,  Xs_m_te  = Xs_m[:seq_split_tr],  Xs_m[seq_split_tr:seq_split_val],  Xs_m[seq_split_val:]
y_seq_tr, y_seq_val, y_seq_te  = y_seq[:seq_split_tr], y_seq[seq_split_tr:seq_split_val], y_seq[seq_split_val:]
r_seq_tr, r_seq_val, r_seq_te  = r_seq[:seq_split_tr], r_seq[seq_split_tr:seq_split_val], r_seq[seq_split_val:]

print(f"  Short sequences  : {Xs_s.shape}  (samples × {SEQ_LEN_S} × features)")
print(f"  Medium sequences : {Xs_m.shape}  (samples × {SEQ_LEN_M} × features)")
print(f"  Train / Val / Test : {len(y_seq_tr)} / {len(y_seq_val)} / {len(y_seq_te)}")

class DualSeqDataset(Dataset):
    def __init__(self, Xs, Xm, y, r):
        self.Xs = torch.tensor(Xs, dtype=torch.float32)
        self.Xm = torch.tensor(Xm, dtype=torch.float32)
        self.y  = torch.tensor(y,  dtype=torch.long)
        self.r  = torch.tensor(r,  dtype=torch.long)
    def __len__(self):  return len(self.y)
    def __getitem__(self, i): return self.Xs[i], self.Xm[i], self.y[i], self.r[i]

train_ds = DualSeqDataset(Xs_s_tr,  Xs_m_tr,  y_seq_tr,  r_seq_tr)
val_ds   = DualSeqDataset(Xs_s_val, Xs_m_val, y_seq_val, r_seq_val)
test_ds  = DualSeqDataset(Xs_s_te,  Xs_m_te,  y_seq_te,  r_seq_te)
train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=False)
val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)
test_dl  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False)

class_counts  = np.bincount(y_seq_tr, minlength=4)
class_weights = torch.tensor(
    1.0 / (class_counts + 1e-6), dtype=torch.float32
).to(device)
class_weights = class_weights / class_weights.sum() * 4

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7: MODEL ARCHITECTURES  [CHANGES 4 + 7]
#   [4] Dual-scale CNN: CNNShort (20d) + CNNMedium (60d)
#   [7] Regime-conditioned fusion: ADX regime embedded and fed to dense head
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  SECTION 7: DUAL-SCALE CNN-GRU + REGIME CONDITIONING  [v2]")
print("=" * 70)

class CNNShortBranch(nn.Module):
    """
    Short-window CNN (20-day): learns inflection patterns and crossover shapes.
    Lighter architecture for the shorter window.
    """
    def __init__(self, in_features):
        super().__init__()
        self.conv1   = nn.Conv1d(in_features, 64, kernel_size=3, padding=1)
        self.conv2   = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.pool    = nn.AdaptiveAvgPool1d(4)
        self.dropout = nn.Dropout(0.3)
        self.out_dim = 128 * 4

    def forward(self, x):
        # x: (B, seq_short, F) → (B, F, seq_short)
        x = x.permute(0, 2, 1)
        x = F.relu(self.conv1(x));  x = self.dropout(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        return x.flatten(1)   # (B, 128*4=512)

class CNNMediumBranch(nn.Module):
    """
    Medium-window CNN (60-day): learns trend structure and morphological patterns.
    Deeper architecture for the longer window.
    """
    def __init__(self, in_features):
        super().__init__()
        self.conv1   = nn.Conv1d(in_features, 64,  kernel_size=5, padding=2)
        self.conv2   = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.conv3   = nn.Conv1d(128, 64, kernel_size=3, padding=1)
        self.pool    = nn.AdaptiveAvgPool1d(8)
        self.dropout = nn.Dropout(0.3)
        self.out_dim = 64 * 8

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = F.relu(self.conv1(x));  x = self.dropout(x)
        x = F.relu(self.conv2(x));  x = self.dropout(x)
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        return x.flatten(1)   # (B, 64*8=512)

class GRUBranch(nn.Module):
    """
    GRU (60-day): captures regime persistence and long-horizon dependencies.
    """
    def __init__(self, in_features, hidden=128, n_layers=2):
        super().__init__()
        self.gru     = nn.GRU(in_features, hidden, num_layers=n_layers,
                              batch_first=True,
                              dropout=0.3 if n_layers > 1 else 0)
        self.dropout = nn.Dropout(0.3)
        self.out_dim = hidden

    def forward(self, x):
        out, _ = self.gru(x)
        return self.dropout(out[:, -1, :])   # last hidden state

class DualCNNGRUFusion(nn.Module):
    """
    Full dual-scale CNN-GRU fusion with regime conditioning.  [v2]

    Components:
      cnn_short  : CNN on 20-day window  → inflection signals
      cnn_medium : CNN on 60-day window  → trend structure
      gru        : GRU on 60-day window  → regime persistence
      regime_emb : embedding of ADX regime (3 states)  [CHANGE 7]
      fusion head: concat all → BN → dense → 4-class softmax

    Ablation modes:
      'cnn_only'    : uses only cnn_medium (as in v1)
      'gru_only'    : uses only gru
      'dual_cnn'    : cnn_short + cnn_medium (no GRU)
      'fusion'      : all three + regime embedding  [full model]
    """
    def __init__(self, in_features, n_classes=4, mode="fusion"):
        super().__init__()
        self.mode       = mode
        self.cnn_short  = CNNShortBranch(in_features)
        self.cnn_medium = CNNMediumBranch(in_features)
        self.gru        = GRUBranch(in_features)
        self.regime_emb = nn.Embedding(3, 8)   # 3 ADX regimes → 8-dim embedding

        if mode == "fusion":
            fc_in = self.cnn_short.out_dim + self.cnn_medium.out_dim + self.gru.out_dim + 8
        elif mode == "dual_cnn":
            fc_in = self.cnn_short.out_dim + self.cnn_medium.out_dim
        elif mode == "cnn_only":
            fc_in = self.cnn_medium.out_dim
        elif mode == "gru_only":
            fc_in = self.gru.out_dim

        self.fc1     = nn.Linear(fc_in, 256)
        self.fc2     = nn.Linear(256, 128)
        self.fc3     = nn.Linear(128, 64)
        self.out     = nn.Linear(64, n_classes)
        self.dropout = nn.Dropout(0.3)
        self.bn1     = nn.BatchNorm1d(256)
        self.bn2     = nn.BatchNorm1d(128)
        self.bn3     = nn.BatchNorm1d(64)

    def forward(self, xs, xm, regime):
        if self.mode == "fusion":
            cs  = self.cnn_short(xs)
            cm  = self.cnn_medium(xm)
            g   = self.gru(xm)
            re  = self.regime_emb(regime)
            z   = torch.cat([cs, cm, g, re], dim=1)
        elif self.mode == "dual_cnn":
            cs  = self.cnn_short(xs)
            cm  = self.cnn_medium(xm)
            z   = torch.cat([cs, cm], dim=1)
        elif self.mode == "cnn_only":
            z   = self.cnn_medium(xm)
        elif self.mode == "gru_only":
            z   = self.gru(xm)

        z = F.relu(self.bn1(self.fc1(z)));  z = self.dropout(z)
        z = F.relu(self.bn2(self.fc2(z)));  z = self.dropout(z)
        z = F.relu(self.bn3(self.fc3(z)));  z = self.dropout(z)
        return self.out(z)

n_feat = Xs_s.shape[2]
for mode in ["cnn_only", "gru_only", "dual_cnn", "fusion"]:
    m       = DualCNNGRUFusion(n_feat, mode=mode)
    n_params = sum(p.numel() for p in m.parameters() if p.requires_grad)
    print(f"  {mode:12s}: {n_params:,} trainable parameters")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 8: TRAINING LOOP
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  SECTION 8: TRAINING MODELS")
print("=" * 70)

def train_model(mode, epochs=EPOCHS, patience=PATIENCE, lr=LR):
    print(f"\n  Training: {mode.upper()}")
    model     = DualCNNGRUFusion(n_feat, mode=mode).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=4, factor=0.5, min_lr=1e-5)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    best_val_f1  = -1
    best_weights = None
    no_improve   = 0
    history      = {"train_loss": [], "val_loss": [], "val_acc": [], "val_f1": []}

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0
        for xs, xm, yb, rb in train_dl:
            xs, xm, yb, rb = xs.to(device), xm.to(device), yb.to(device), rb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xs, xm, rb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item() * len(yb)
        train_loss /= len(train_ds)

        # ── Validation (early stopping only — never touches test set) ────────
        model.eval()
        val_loss, all_preds, all_true = 0, [], []
        with torch.no_grad():
            for xs, xm, yb, rb in val_dl:
                xs, xm, yb, rb = xs.to(device), xm.to(device), yb.to(device), rb.to(device)
                logits   = model(xs, xm, rb)
                val_loss += criterion(logits, yb).item() * len(yb)
                all_preds.extend(logits.argmax(1).cpu().numpy())
                all_true.extend(yb.cpu().numpy())
        val_loss /= len(val_ds)
        val_acc   = accuracy_score(all_true, all_preds)
        val_f1    = f1_score(all_true, all_preds, average="macro", zero_division=0)
        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_f1"].append(val_f1)

        if val_f1 > best_val_f1:
            best_val_f1  = val_f1
            best_weights = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve   = 0
        else:
            no_improve += 1

        if epoch % 10 == 0 or epoch == 1:
            print(f"    Epoch {epoch:3d} | TrLoss {train_loss:.4f} | ValLoss {val_loss:.4f} | "
                  f"ValAcc {val_acc:.4f} | ValF1 {val_f1:.4f}")

        if no_improve >= patience:
            print(f"    Early stop at epoch {epoch}  (best ValF1={best_val_f1:.4f})")
            break

    # ── Final evaluation on held-out TEST set (best weights, touched once) ───
    model.load_state_dict(best_weights)
    model.eval()
    all_preds, all_true = [], []
    with torch.no_grad():
        for xs, xm, yb, rb in test_dl:
            xs, xm = xs.to(device), xm.to(device)
            rb = rb.to(device)
            all_preds.extend(model(xs, xm, rb).argmax(1).cpu().numpy())
            all_true.extend(yb.numpy())

    acc = accuracy_score(all_true, all_preds)
    f1  = f1_score(all_true, all_preds, average="macro", zero_division=0)
    print(f"\n  ── Final TEST: {mode.upper()} ──")
    print(f"  Accuracy : {acc:.4f}   F1 Macro : {f1:.4f}")
    print(classification_report(all_true, all_preds,
                                target_names=[STATE_NAMES[i] for i in range(4)],
                                zero_division=0))
    return model, acc, f1, np.array(all_preds), np.array(all_true), history

results_nn = {}
for mode in ["cnn_only", "gru_only", "dual_cnn", "fusion"]:
    model, acc, f1, preds, trues, history = train_model(mode)
    results_nn[mode] = {
        "model": model, "acc": acc, "f1": f1,
        "preds": preds, "trues": trues, "history": history
    }

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 9: REGIME ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  SECTION 9: REGIME ANALYSIS")
print("=" * 70)

n_test_seq      = len(Xs_s_te)
test_dates_seq  = valid_idx[split_val + SEQ_LEN_M: split_val + SEQ_LEN_M + n_test_seq]

adx_test        = feat_clean.loc[test_dates_seq[:n_test_seq], "adx_14"].values if len(test_dates_seq) >= n_test_seq else feat_clean.loc[test_dates_seq, "adx_14"].values
trending_mask   = adx_test > 25
meanrev_mask    = ~trending_mask

print(f"  Test samples     : {n_test_seq}")
print(f"  Strong trend (ADX>25): {trending_mask.sum()} ({trending_mask.mean()*100:.1f}%)")
print(f"  Weak/choppy           : {meanrev_mask.sum()}  ({meanrev_mask.mean()*100:.1f}%)")

fusion_preds = results_nn["fusion"]["preds"]
fusion_true  = results_nn["fusion"]["trues"]
rf_preds_seq = y_rf[SEQ_LEN_M:]
y_te_seq     = y_te[SEQ_LEN_M:]

n = min(len(fusion_preds), len(trending_mask), len(rf_preds_seq))
trending_mask = trending_mask[:n]
meanrev_mask  = meanrev_mask[:n]

print("\n  ── Fusion model by regime ──")
for regime_name, mask in [("Strong Trend (ADX>25)", trending_mask),
                           ("Weak/Choppy (ADX≤25)",  meanrev_mask)]:
    if mask.sum() == 0:
        continue
    fp = fusion_preds[:n][mask];  ft = fusion_true[:n][mask]
    rp = rf_preds_seq[:n][mask];  rt = y_te_seq[:n][mask]
    print(f"\n  {regime_name} ({mask.sum()} samples):")
    print(f"    Fusion  → Acc={accuracy_score(ft,fp):.4f}  "
          f"F1={f1_score(ft,fp,average='macro',zero_division=0):.4f}")
    print(f"    RF      → Acc={accuracy_score(rt,rp):.4f}  "
          f"F1={f1_score(rt,rp,average='macro',zero_division=0):.4f}")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 9B: FORWARD TREND ANALYSIS — INFLECTION FINGERPRINTING
#
#  The question: does what the backward-looking signal tells us at time T
#  correctly foreshadow what the signal does over the next 20 and 60 days?
#
#  Three-stage process mirroring the train/val/test split:
#    Stage 1 (Train) — identify pre-inflection signal fingerprints with hindsight
#    Stage 2 (Val)   — validate fingerprints predict real forward inflections
#    Stage 3 (Test)  — apply blind: report continuation vs transition rates
#                      and at what point the signal first flagged a change
#
#  All analysis uses signal (EWM/cycle/ADX/MA stack), not price returns.
#  The forward window is signal-driven: a change is recorded when the
#  decomposition shifts state, not at a fixed calendar horizon.
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  SECTION 9B: FORWARD TREND ANALYSIS — INFLECTION FINGERPRINTING")
print("=" * 70)

# ── Core signals aligned to valid_idx ─────────────────────────────────────────
sig = pd.DataFrame(index=valid_idx)
sig["label"]        = labels_clean                        # smoothed trend state
sig["cycle"]        = cy.reindex(valid_idx)               # fast−slow spread ($)
sig["cycle_slope"]  = sig["cycle"].diff(5)                # 5-day momentum of cycle
sig["fast_vs_slow"] = feat_clean["trend_fast_vs_slow"]    # % spread fast/slow EWM
sig["adx"]          = feat_clean["adx_14"]
sig["di_diff"]      = feat_clean["di_diff"]               # DI+ − DI−
sig["ma_stack"]     = feat_clean["ma_stack_score"]        # 0–1 bull alignment
sig["price_vs_fast"]= feat_clean["price_vs_fast"]         # price vs EWM-20

# ── Helper: detect first signal-driven state change within a forward window ───
def forward_inflection(sig_df, start_pos, window, current_state):
    """
    From start_pos, scan up to `window` days forward in sig_df.
    Returns (changed: bool, day_of_change: int or None, new_state: int or None)
    A change is confirmed when the label shifts away from current_state
    and holds for at least 3 consecutive days (avoids noise flickers).
    """
    end = min(start_pos + window, len(sig_df) - 1)
    labels_fwd = sig_df["label"].iloc[start_pos: end].values
    for j in range(len(labels_fwd) - 2):
        if (labels_fwd[j]     != current_state and
            labels_fwd[j + 1] != current_state and
            labels_fwd[j + 2] != current_state):
            return True, j + 1, int(labels_fwd[j])
    return False, None, None

# ── Helper: extract pre-inflection signal fingerprint ─────────────────────────
def pre_inflection_fingerprint(sig_df, inflection_pos, lookback):
    """
    From an inflection point, look back `lookback` days and return
    mean signal levels over that window — the 'fingerprint' of approach.
    """
    start = max(0, inflection_pos - lookback)
    window = sig_df.iloc[start: inflection_pos]
    if len(window) == 0:
        return None
    return {
        "cycle_mean":       window["cycle"].mean(),
        "cycle_slope_mean": window["cycle_slope"].mean(),
        "fast_vs_slow_mean":window["fast_vs_slow"].mean(),
        "adx_mean":         window["adx"].mean(),
        "di_diff_mean":     window["di_diff"].mean(),
        "ma_stack_mean":    window["ma_stack"].mean(),
        "cycle_trend":      window["cycle"].iloc[-1] - window["cycle"].iloc[0],
        "adx_trend":        window["adx"].iloc[-1]   - window["adx"].iloc[0],
        "di_trend":         window["di_diff"].iloc[-1]- window["di_diff"].iloc[0],
    }

# ══════════════════════════════════════════════════════════════════════════════
# STAGE 1: Learn inflection fingerprints from TRAIN period
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n  ── Stage 1: Learning inflection fingerprints from TRAIN ──")
print(f"     Period: {train_idx[0].date()} → {train_idx[-1].date()}")

sig_tr = sig.loc[train_idx].reset_index(drop=False).rename(columns={sig.index.name or "index": "date"})
train_inflections = []   # list of dicts: {pos, from_state, to_state, fingerprint_20, fingerprint_60}

i = 1
while i < len(sig_tr) - 3:
    prev_state = int(sig_tr["label"].iloc[i - 1])
    curr_state = int(sig_tr["label"].iloc[i])
    if curr_state != prev_state:
        # Confirmed inflection — collect fingerprints at both lookback horizons
        fp20 = pre_inflection_fingerprint(sig_tr, i, 20)
        fp60 = pre_inflection_fingerprint(sig_tr, i, 60)
        if fp20 and fp60:
            train_inflections.append({
                "pos":        i,
                "date":       sig_tr["date"].iloc[i],
                "from_state": prev_state,
                "to_state":   curr_state,
                "fp20":       fp20,
                "fp60":       fp60,
            })
        i += 3   # skip ahead to avoid double-counting the same inflection
    else:
        i += 1

print(f"     Inflection points found: {len(train_inflections)}")

# Summarise fingerprints by transition type
transitions = {}
for inf in train_inflections:
    key = (inf["from_state"], inf["to_state"])
    if key not in transitions:
        transitions[key] = []
    transitions[key].append(inf["fp20"])

print(f"\n     Pre-inflection signal fingerprints (20-day lookback, mean across events):")
print(f"     {'Transition':<28} {'N':>4}  {'CycleSlope':>10}  {'FastVsSlow':>10}  "
      f"{'ADX Δ':>8}  {'DI Δ':>8}  {'MAStack':>8}")
for (fs, ts_), fps in sorted(transitions.items()):
    if len(fps) < 2:
        continue
    cs_mean  = np.mean([f["cycle_slope_mean"] for f in fps])
    fvs_mean = np.mean([f["fast_vs_slow_mean"] for f in fps])
    adx_tr   = np.mean([f["adx_trend"]         for f in fps])
    di_tr    = np.mean([f["di_trend"]           for f in fps])
    ma_mean  = np.mean([f["ma_stack_mean"]      for f in fps])
    label_str = f"{STATE_NAMES[fs]} → {STATE_NAMES[ts_]}"
    print(f"     {label_str:<28} {len(fps):>4}  {cs_mean:>10.4f}  {fvs_mean:>10.4f}  "
          f"{adx_tr:>8.2f}  {di_tr:>8.2f}  {ma_mean:>8.3f}")

# ── Aggregate fingerprint thresholds for bullish→bearish and bearish→bullish ──
# These are the "warning sign" thresholds learned from train
bull_to_bear = [(fs, ts_) for (fs, ts_) in transitions
                if fs in [2, 3] and ts_ in [0, 1]]
bear_to_bull = [(fs, ts_) for (fs, ts_) in transitions
                if fs in [0, 1] and ts_ in [2, 3]]

def avg_fingerprint(transition_keys, transitions_dict, fp_key):
    all_fps = []
    for k in transition_keys:
        if k in transitions_dict:
            all_fps.extend(transitions_dict[k])
    if not all_fps:
        return {}
    return {sig_key: np.mean([f[sig_key] for f in all_fps]) for sig_key in all_fps[0]}

btb_fp = avg_fingerprint(bull_to_bear, transitions, "fp20")
btb_fp60 = avg_fingerprint(bull_to_bear,
    {k: [inf["fp60"] for inf in train_inflections if (inf["from_state"], inf["to_state"]) == k]
     for k in bull_to_bear}, "fp60")
bbb_fp  = avg_fingerprint(bear_to_bull, transitions, "fp20")

print(f"\n     Composite Bull→Bear fingerprint (20d lookback):")
if btb_fp:
    for k, v in btb_fp.items():
        print(f"       {k:<22}: {v:>9.4f}")

print(f"\n     Composite Bear→Bull fingerprint (20d lookback):")
if bbb_fp:
    for k, v in bbb_fp.items():
        print(f"       {k:<22}: {v:>9.4f}")

# ══════════════════════════════════════════════════════════════════════════════
# STAGE 2: Validate fingerprints on VAL period
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n  ── Stage 2: Validating fingerprints on VAL ──")
print(f"     Period: {val_idx[0].date()} → {val_idx[-1].date()}")

sig_val = sig.loc[val_idx].reset_index(drop=False).rename(columns={sig.index.name or "index": "date"})

def score_fingerprint_match(current_fp, reference_fp, direction="bull_to_bear"):
    """
    Score how closely current signal matches the reference fingerprint.
    Returns a score 0–1; higher = stronger match to learned pattern.
    Direction-specific: bull_to_bear checks for deteriorating signals.
    """
    if not reference_fp or not current_fp:
        return 0.0
    score = 0.0
    checks = 0
    if direction == "bull_to_bear":
        # Warning signs: cycle slope falling, fast_vs_slow narrowing,
        # ADX weakening, DI turning negative, MA stack deteriorating
        if current_fp.get("cycle_slope_mean", 0) < reference_fp.get("cycle_slope_mean", 0):
            score += 1
        if current_fp.get("fast_vs_slow_mean", 1) < reference_fp.get("fast_vs_slow_mean", 1):
            score += 1
        if current_fp.get("adx_trend", 0) < 0:   # ADX declining
            score += 1
        if current_fp.get("di_trend", 0) < 0:     # DI+ losing to DI−
            score += 1
        if current_fp.get("ma_stack_mean", 1) < reference_fp.get("ma_stack_mean", 1):
            score += 1
        checks = 5
    else:  # bear_to_bull: improving signals
        if current_fp.get("cycle_slope_mean", 0) > reference_fp.get("cycle_slope_mean", 0):
            score += 1
        if current_fp.get("fast_vs_slow_mean", -1) > reference_fp.get("fast_vs_slow_mean", -1):
            score += 1
        if current_fp.get("adx_trend", 0) > 0:
            score += 1
        if current_fp.get("di_trend", 0) > 0:
            score += 1
        if current_fp.get("ma_stack_mean", 0) > reference_fp.get("ma_stack_mean", 0):
            score += 1
        checks = 5
    return score / checks

def run_forward_analysis(sig_df, period_name, reference_btb, reference_bbb,
                         windows=(20, 60)):
    """
    For each day in sig_df:
      - Extract current backward fingerprint (20d lookback)
      - Score against learned fingerprints from train
      - Check whether a real state change occurred in each forward window
      - Record: predicted_continuation, actual_continuation, early_warning_day
    Returns a summary dict.
    """
    results = {w: {
        "total": 0, "continued": 0, "changed": 0,
        "warned_and_changed": 0, "warned_not_changed": 0,
        "change_day_sum": 0, "change_day_n": 0,
        "by_state": {s: {"continued": 0, "changed": 0} for s in range(4)}
    } for w in windows}

    warning_threshold = 0.6   # 3/5 signals must match fingerprint

    for i in range(20, len(sig_df) - max(windows)):
        current_state = int(sig_df["label"].iloc[i])
        current_fp    = pre_inflection_fingerprint(sig_df, i, 20)
        if current_fp is None:
            continue

        # Score warning signal
        if current_state in [2, 3]:   # bullish — watch for deterioration
            warn_score = score_fingerprint_match(current_fp, reference_btb, "bull_to_bear")
        else:                          # bearish — watch for recovery
            warn_score = score_fingerprint_match(current_fp, reference_bbb, "bear_to_bull")
        warning_fired = warn_score >= warning_threshold

        for w in windows:
            changed, change_day, new_state = forward_inflection(sig_df, i, w, current_state)
            results[w]["total"] += 1
            results[w]["by_state"][current_state]["continued" if not changed else "changed"] += 1
            if changed:
                results[w]["changed"] += 1
                results[w]["change_day_sum"] += (change_day or 0)
                results[w]["change_day_n"]   += 1
                if warning_fired:
                    results[w]["warned_and_changed"] += 1
            else:
                results[w]["continued"] += 1
                if warning_fired:
                    results[w]["warned_not_changed"] += 1

    print(f"\n  {period_name} Forward Analysis:")
    for w in windows:
        r  = results[w]
        n  = r["total"]
        if n == 0:
            continue
        cont_rate = r["continued"] / n
        chng_rate = r["changed"]   / n
        avg_day   = (r["change_day_sum"] / r["change_day_n"]) if r["change_day_n"] > 0 else 0
        # Warning precision: of days where warning fired + change happened vs all warning days
        warned_total = r["warned_and_changed"] + r["warned_not_changed"]
        warn_prec = r["warned_and_changed"] / warned_total if warned_total > 0 else 0
        # Warning recall: of all actual changes, how many were warned
        warn_rec  = r["warned_and_changed"] / r["changed"] if r["changed"] > 0 else 0

        print(f"\n    {w}-day forward window  (n={n}):")
        print(f"      Continuation rate : {cont_rate:.1%}  ({r['continued']} days)")
        print(f"      Transition rate   : {chng_rate:.1%}  ({r['changed']} days)")
        print(f"      Avg day of change : {avg_day:.1f} days into window")
        print(f"      Warning precision : {warn_prec:.1%}  "
              f"(warned→changed / all warned)")
        print(f"      Warning recall    : {warn_rec:.1%}  "
              f"(warned→changed / all changes)")
        print(f"      By starting state:")
        for s in range(4):
            sc = r["by_state"][s]
            tot = sc["continued"] + sc["changed"]
            if tot == 0:
                continue
            print(f"        {STATE_NAMES[s]:15s}: "
                  f"continued {sc['continued']:3d} ({sc['continued']/tot:.0%})  "
                  f"changed {sc['changed']:3d} ({sc['changed']/tot:.0%})")
    return results

# Run Stage 2 — Val
val_results = run_forward_analysis(
    sig_val, "VAL", btb_fp, bbb_fp, windows=(20, 60))

# ══════════════════════════════════════════════════════════════════════════════
# STAGE 3: Apply blind on TEST period
# ══════════════════════════════════════════════════════════════════════════════
print(f"\n  ── Stage 3: Blind application on TEST ──")
print(f"     Period: {test_idx[0].date()} → {test_idx[-1].date()}")

sig_te = sig.loc[test_idx].reset_index(drop=False).rename(columns={sig.index.name or "index": "date"})

test_results = run_forward_analysis(
    sig_te, "TEST", btb_fp, bbb_fp, windows=(20, 60))

# ── Foresight vs Hindsight summary ────────────────────────────────────────────
print(f"\n  ── Foresight vs Hindsight Summary ──")
print(f"  {'Metric':<35} {'Train':>10}  {'Val':>10}  {'Test':>10}")
print(f"  {'─'*35} {'─'*10}  {'─'*10}  {'─'*10}")

# Recompute train forward stats for comparison
sig_tr_reset = sig.loc[train_idx].reset_index(drop=False).rename(columns={sig.index.name or "index": "date"})
train_fwd = run_forward_analysis(
    sig_tr_reset, "TRAIN (hindsight check)", btb_fp, bbb_fp, windows=(20, 60))

for w in (20, 60):
    tr = train_fwd[w]; vl = val_results[w]; te = test_results[w]
    n_tr = tr["total"]; n_vl = vl["total"]; n_te = te["total"]
    if n_tr == 0 or n_vl == 0 or n_te == 0:
        continue
    print(f"\n  {w}-day window:")
    print(f"  {'Continuation rate':<35} "
          f"{tr['continued']/n_tr:>10.1%}  "
          f"{vl['continued']/n_vl:>10.1%}  "
          f"{te['continued']/n_te:>10.1%}")
    print(f"  {'Transition rate':<35} "
          f"{tr['changed']/n_tr:>10.1%}  "
          f"{vl['changed']/n_vl:>10.1%}  "
          f"{te['changed']/n_te:>10.1%}")
    wt_tr = tr['warned_and_changed'] + tr['warned_not_changed']
    wt_vl = vl['warned_and_changed'] + vl['warned_not_changed']
    wt_te = te['warned_and_changed'] + te['warned_not_changed']
    wp_tr = tr['warned_and_changed'] / wt_tr if wt_tr > 0 else 0
    wp_vl = vl['warned_and_changed'] / wt_vl if wt_vl > 0 else 0
    wp_te = te['warned_and_changed'] / wt_te if wt_te > 0 else 0
    print(f"  {'Warning precision':<35} "
          f"{wp_tr:>10.1%}  {wp_vl:>10.1%}  {wp_te:>10.1%}")
    wr_tr = tr['warned_and_changed'] / tr['changed'] if tr['changed'] > 0 else 0
    wr_vl = vl['warned_and_changed'] / vl['changed'] if vl['changed'] > 0 else 0
    wr_te = te['warned_and_changed'] / te['changed'] if te['changed'] > 0 else 0
    print(f"  {'Warning recall':<35} "
          f"{wr_tr:>10.1%}  {wr_vl:>10.1%}  {wr_te:>10.1%}")
    print(f"  {'Avg day of change detected':<35} "
          f"{tr['change_day_sum']/tr['change_day_n'] if tr['change_day_n']>0 else 0:>10.1f}  "
          f"{vl['change_day_sum']/vl['change_day_n'] if vl['change_day_n']>0 else 0:>10.1f}  "
          f"{te['change_day_sum']/te['change_day_n'] if te['change_day_n']>0 else 0:>10.1f}")

print(f"""
  Interpretation:
  - Continuation rate: how often the current trend state persisted through
    the full forward window. High = regime persistence is real.
  - Transition rate: how often the signal shifted state within the window.
    Consistent across train/val/test = the inflection patterns are structural.
  - Warning precision: of days where pre-inflection fingerprint fired,
    how many actually led to a transition. Measures signal quality.
  - Warning recall: of all real transitions, how many were flagged in advance.
    Measures coverage — did the fingerprint catch the inflections that happened?
  - Avg day of change: how early within the window the signal shifted.
    Early detection (< 10 days) = the backward-look pattern has genuine
    foresight; late detection (> 40 days) = barely within the window.
  If train ≈ val ≈ test across all metrics, the backward-to-forward mapping
  is structurally consistent — what hindsight reveals, foresight can detect.
""")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 10: VISUALIZATIONS
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  SECTION 10: GENERATING FIGURES")
print("=" * 70)

PALETTE = {
    "navy":   "#0D1B2A", "teal":   "#0D9488", "gold":   "#D4A017",
    "gray":   "#64748B", "red":    "#C0392B", "green":  "#27AE60",
    "orange": "#E67E22", "blue":   "#2E86C1",
}

STATE_COLORS = {
    3: "#27AE60",   # Trending Up  — green
    2: "#2E86C1",   # Trans-Up     — blue
    1: "#E67E22",   # Trans-Down   — orange
    0: "#C0392B",   # Trending Down — red
}

# ── Figure 1: Dual-scale decomposition ───────────────────────────────────────
fig1, axes = plt.subplots(4, 1, figsize=(16, 12), sharex=True)
fig1.suptitle("SPY Dual-Scale Signal Decomposition\n"
              "(Fast EWM-20 + Slow EWM-60 + Cycle + Noise  [all causal])",
              fontsize=13, fontweight="bold", y=0.98)
axes[0].plot(spy_close.index, spy_arr,            color=PALETTE["navy"],  lw=0.8,  label="Raw Close")
axes[0].plot(trend_fast.index, trend_fast.values, color=PALETTE["teal"],  lw=1.5,  label="Trend Fast (EWM-20)")
axes[0].plot(trend_slow.index, trend_slow.values, color=PALETTE["red"],   lw=1.5,  label="Trend Slow (EWM-60)", ls="--")
axes[0].set_ylabel("Price ($)");  axes[0].legend(fontsize=9);  axes[0].grid(alpha=0.3)
axes[1].plot(trend_fast.index, trend_fast.values, color=PALETTE["teal"],  lw=1.5,  label="Trend Fast")
axes[1].plot(trend_slow.index, trend_slow.values, color=PALETTE["red"],   lw=1.5,  label="Trend Slow", ls="--")
axes[1].fill_between(trend_fast.index,
                     trend_fast.values, trend_slow.values,
                     where=(trend_fast.values > trend_slow.values),
                     alpha=0.25, color=PALETTE["green"], label="Fast > Slow (bullish)")
axes[1].fill_between(trend_fast.index,
                     trend_fast.values, trend_slow.values,
                     where=(trend_fast.values <= trend_slow.values),
                     alpha=0.25, color=PALETTE["red"], label="Fast < Slow (bearish)")
axes[1].set_ylabel("Trend ($)");  axes[1].legend(fontsize=9);  axes[1].grid(alpha=0.3)
axes[2].plot(cycle_comp.index, cycle_comp.values, color=PALETTE["gold"],  lw=1.0,  label="Cycle (Fast − Slow)")
axes[2].axhline(0, color="gray", ls="--", lw=0.8)
axes[2].fill_between(cycle_comp.index, cycle_comp.values, 0,
                     where=(cycle_comp.values > 0), alpha=0.25, color=PALETTE["green"])
axes[2].fill_between(cycle_comp.index, cycle_comp.values, 0,
                     where=(cycle_comp.values < 0), alpha=0.25, color=PALETTE["red"])
axes[2].set_ylabel("Cycle ($)");  axes[2].legend(fontsize=9);  axes[2].grid(alpha=0.3)
axes[3].plot(noise_comp.index, noise_comp.values, color=PALETTE["blue"],  lw=0.5,  label="Noise (Price − Fast)")
axes[3].axhline(0, color="gray", ls="--", lw=0.8)
axes[3].set_ylabel("Noise ($)");  axes[3].set_xlabel("Date")
axes[3].legend(fontsize=9);  axes[3].grid(alpha=0.3)
plt.tight_layout(rect=[0, 0, 1, 0.96])
fig1.savefig(f"{OUT_DIR}/fig1_dual_decomp.png", dpi=150, bbox_inches="tight");  plt.close()

# ── Figure 2: Trend state labels on price chart ───────────────────────────────
fig2, ax2 = plt.subplots(figsize=(16, 6))
ax2.plot(spy_close.index, spy_arr, color=PALETTE["navy"], lw=0.8, label="SPY Close", zorder=3)
for state, col in STATE_COLORS.items():
    mask = labels == state
    ax2.fill_between(spy_close.index, 0, spy_arr, where=mask.values,
                     alpha=0.3, color=col, label=STATE_NAMES[state])
ax2.set_ylabel("Price ($)");  ax2.set_xlabel("Date")
ax2.set_title("SPY Price with Trend State Labels\n"
              "(Green=Trending-Up, Blue=Trans-Up, Orange=Trans-Down, Red=Trending-Down)",
              fontsize=13, fontweight="bold")
ax2.legend(fontsize=9, loc="upper left");  ax2.grid(alpha=0.3)
plt.tight_layout()
fig2.savefig(f"{OUT_DIR}/fig2_trend_state_labels.png", dpi=150, bbox_inches="tight");  plt.close()

# ── Figure 3: ADX + DI overlay ────────────────────────────────────────────────
fig3, axes3 = plt.subplots(3, 1, figsize=(16, 10), sharex=True)
fig3.suptitle("ADX Trend Strength + Directional Indicators", fontsize=13, fontweight="bold")
axes3[0].plot(spy_close.index, spy_arr, color=PALETTE["navy"], lw=0.8)
axes3[0].set_ylabel("Price ($)");  axes3[0].grid(alpha=0.3)
axes3[1].plot(adx.index, adx.values, color=PALETTE["teal"], lw=1.5, label="ADX-14")
axes3[1].axhline(20, color="gray",  ls="--", lw=1, label="ADX=20 (trend threshold)")
axes3[1].axhline(35, color="orange", ls="--", lw=1, label="ADX=35 (strong trend)")
axes3[1].set_ylabel("ADX");  axes3[1].legend(fontsize=9);  axes3[1].grid(alpha=0.3)
axes3[2].plot(di_p.index, di_p.values, color=PALETTE["green"],  lw=1.2, label="DI+")
axes3[2].plot(di_m.index, di_m.values, color=PALETTE["red"],    lw=1.2, label="DI−")
axes3[2].fill_between(di_p.index, di_p.values, di_m.values,
                      where=(di_p.values > di_m.values), alpha=0.2, color=PALETTE["green"])
axes3[2].fill_between(di_p.index, di_p.values, di_m.values,
                      where=(di_p.values <= di_m.values), alpha=0.2, color=PALETTE["red"])
axes3[2].set_ylabel("DI+/−");  axes3[2].set_xlabel("Date")
axes3[2].legend(fontsize=9);  axes3[2].grid(alpha=0.3)
plt.tight_layout()
fig3.savefig(f"{OUT_DIR}/fig3_adx_di.png", dpi=150, bbox_inches="tight");  plt.close()

# ── Figure 4: MA Stack & S/R ─────────────────────────────────────────────────
fig4, axes4 = plt.subplots(2, 1, figsize=(16, 10), sharex=True)
fig4.suptitle("MA Stack & Support/Resistance Proximity", fontsize=13, fontweight="bold")
axes4[0].plot(spy_close.index, spy_arr, color=PALETTE["navy"], lw=0.7, label="SPY")
axes4[0].plot(ma20.index,  ma20.values,  color=PALETTE["blue"],   lw=1.2, label="MA-20",  ls="--")
axes4[0].plot(ma50.index,  ma50.values,  color=PALETTE["gold"],   lw=1.2, label="MA-50",  ls="--")
axes4[0].plot(ma200.index, ma200.values, color=PALETTE["red"],    lw=1.5, label="MA-200", ls="-")
axes4[0].set_ylabel("Price ($)");  axes4[0].legend(fontsize=9);  axes4[0].grid(alpha=0.3)
ma_stack_s = feat["ma_stack_score"].reindex(spy_close.index)
axes4[1].plot(ma_stack_s.index, ma_stack_s.values, color=PALETTE["teal"], lw=1.2,
              label="MA Stack Score (0=full bear, 1=full bull)")
axes4[1].axhline(0.5, color="gray", ls="--", lw=0.8)
axes4[1].fill_between(ma_stack_s.index, ma_stack_s.values, 0.5,
                      where=(ma_stack_s.values > 0.5), alpha=0.25, color=PALETTE["green"])
axes4[1].fill_between(ma_stack_s.index, ma_stack_s.values, 0.5,
                      where=(ma_stack_s.values <= 0.5), alpha=0.25, color=PALETTE["red"])
axes4[1].set_ylabel("Stack Score");  axes4[1].set_xlabel("Date")
axes4[1].legend(fontsize=9);  axes4[1].grid(alpha=0.3)
plt.tight_layout()
fig4.savefig(f"{OUT_DIR}/fig4_ma_stack.png", dpi=150, bbox_inches="tight");  plt.close()

# ── Figure 5: Risk-on/off composite ──────────────────────────────────────────
fig5, axes5 = plt.subplots(3, 1, figsize=(16, 10), sharex=True)
fig5.suptitle("Risk-On / Risk-Off Composite Signal", fontsize=13, fontweight="bold")
axes5[0].plot(spy_close.index, spy_arr, color=PALETTE["navy"], lw=0.7, label="SPY")
axes5[0].set_ylabel("Price ($)");  axes5[0].grid(alpha=0.3)
roc = feat["risk_off_composite"].reindex(spy_close.index)
axes5[1].plot(roc.index, roc.values, color=PALETTE["orange"], lw=1.2, label="Risk-Off Composite")
axes5[1].axhline(0, color="gray", ls="--", lw=0.8)
axes5[1].fill_between(roc.index, roc.values, 0,
                      where=(roc.values > 0), alpha=0.3, color=PALETTE["red"],   label="Risk-Off")
axes5[1].fill_between(roc.index, roc.values, 0,
                      where=(roc.values < 0), alpha=0.3, color=PALETTE["green"], label="Risk-On")
axes5[1].set_ylabel("Composite");  axes5[1].legend(fontsize=9);  axes5[1].grid(alpha=0.3)
tc = feat["trend_concordance"].reindex(spy_close.index)
axes5[2].plot(tc.index, tc.values, color=PALETTE["teal"], lw=1.2,
              label="Trend Concordance (SPY/QQQ/XLF/XLE)")
axes5[2].axhline(0, color="gray", ls="--", lw=0.8)
axes5[2].fill_between(tc.index, tc.values, 0,
                      where=(tc.values > 0), alpha=0.2, color=PALETTE["green"])
axes5[2].fill_between(tc.index, tc.values, 0,
                      where=(tc.values < 0), alpha=0.2, color=PALETTE["red"])
axes5[2].set_ylabel("Concordance");  axes5[2].set_xlabel("Date")
axes5[2].legend(fontsize=9);  axes5[2].grid(alpha=0.3)
plt.tight_layout()
fig5.savefig(f"{OUT_DIR}/fig5_risk_composite.png", dpi=150, bbox_inches="tight");  plt.close()

# ── Figure 6: Baseline confusion matrices ─────────────────────────────────────
fig6, axes6 = plt.subplots(1, 3, figsize=(18, 5))
fig6.suptitle("Baseline Confusion Matrices — Trend State (Test Set)",
              fontsize=13, fontweight="bold")
tnames = [STATE_NAMES[i] for i in range(4)]
for ax, (name, yt, yp) in zip(axes6, [
    ("MA Stack Heuristic",  y_te, y_ma),
    ("ADX+DI Rule-Based",   y_te, y_adx),
    ("Random Forest",       y_te, y_rf),
]):
    cm   = confusion_matrix(yt, yp, labels=[0, 1, 2, 3])
    disp = ConfusionMatrixDisplay(cm, display_labels=tnames)
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(f"{name}\nAcc={accuracy_score(yt,yp):.3f}  F1={f1_score(yt,yp,average='macro',zero_division=0):.3f}")
    ax.tick_params(axis="x", labelrotation=30)
plt.tight_layout()
fig6.savefig(f"{OUT_DIR}/fig6_confusion_baselines.png", dpi=150, bbox_inches="tight");  plt.close()

# ── Figure 7: Model performance comparison ────────────────────────────────────
fig7, axes7 = plt.subplots(1, 2, figsize=(14, 5))
fig7.suptitle("Model Performance Comparison — Trend State Classification",
              fontsize=13, fontweight="bold")
all_results = [
    ("MA Stack",        acc_ma,  f1_ma),
    ("ADX+DI",          acc_adx, f1_adx),
    ("Rand.Forest",     acc_rf,  f1_rf),
    ("CNN-only",        results_nn["cnn_only"]["acc"],  results_nn["cnn_only"]["f1"]),
    ("GRU-only",        results_nn["gru_only"]["acc"],  results_nn["gru_only"]["f1"]),
    ("Dual-CNN",        results_nn["dual_cnn"]["acc"],  results_nn["dual_cnn"]["f1"]),
    ("CNN-GRU\nFusion", results_nn["fusion"]["acc"],    results_nn["fusion"]["f1"]),
]
names  = [r[0] for r in all_results]
accs   = [r[1] for r in all_results]
f1s    = [r[2] for r in all_results]
pals   = [PALETTE["gray"], PALETTE["orange"], PALETTE["blue"],
          "#8E44AD", "#1ABC9C", "#E91E8C", PALETTE["teal"]]
for ax, vals, ylabel, title in [
    (axes7[0], accs, "Accuracy",   "Directional Accuracy"),
    (axes7[1], f1s,  "F1 (Macro)", "F1 Score (Macro)"),
]:
    bars = ax.bar(names, vals, color=pals, edgecolor="white", width=0.6)
    ax.axhline(naive, ls="--", color=PALETTE["red"], lw=1.4, label=f"Naive ({naive:.3f})")
    ax.set_ylabel(ylabel, fontsize=11);  ax.set_title(title, fontsize=12);  ax.set_ylim(0, 1)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width()/2, v + 0.012, f"{v:.3f}",
                ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.legend(fontsize=9);  ax.grid(axis="y", alpha=0.3)
    ax.tick_params(axis="x", labelsize=8)
plt.tight_layout()
fig7.savefig(f"{OUT_DIR}/fig7_model_comparison.png", dpi=150, bbox_inches="tight");  plt.close()

# ── Figure 8: Ablation confusion matrices ─────────────────────────────────────
fig8, axes8 = plt.subplots(1, 4, figsize=(22, 5))
fig8.suptitle("CNN-GRU Ablation — Trend State Confusion Matrices",
              fontsize=13, fontweight="bold")
for ax, mode in zip(axes8, ["cnn_only", "gru_only", "dual_cnn", "fusion"]):
    yt   = results_nn[mode]["trues"]
    yp   = results_nn[mode]["preds"]
    cm   = confusion_matrix(yt, yp, labels=[0, 1, 2, 3])
    disp = ConfusionMatrixDisplay(cm, display_labels=tnames)
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(f"{mode.replace('_',' ').title()}\n"
                 f"Acc={results_nn[mode]['acc']:.3f}  F1={results_nn[mode]['f1']:.3f}")
    ax.tick_params(axis="x", labelrotation=30)
plt.tight_layout()
fig8.savefig(f"{OUT_DIR}/fig8_ablation_confusion.png", dpi=150, bbox_inches="tight");  plt.close()

# ── Figure 9: Training history ────────────────────────────────────────────────
fig9, axes9 = plt.subplots(1, 4, figsize=(20, 4))
fig9.suptitle("Training History — Ablation Models", fontsize=13, fontweight="bold")
cols9 = {"cnn_only": "#8E44AD", "gru_only": "#1ABC9C",
          "dual_cnn": "#E91E8C", "fusion": PALETTE["teal"]}
for ax, mode in zip(axes9, ["cnn_only", "gru_only", "dual_cnn", "fusion"]):
    h  = results_nn[mode]["history"]
    ep = range(1, len(h["train_loss"]) + 1)
    ax.plot(ep, h["val_f1"],   color=cols9[mode], lw=2,   label="Val F1")
    ax.plot(ep, h["val_acc"],  color=cols9[mode], lw=1.5, ls="--", label="Val Acc", alpha=0.7)
    ax.plot(ep, h["train_loss"], color="gray", lw=1, ls=":", label="Train Loss", alpha=0.5)
    ax.set_title(mode.replace("_", " ").title());  ax.set_xlabel("Epoch")
    ax.set_ylim(0, 1);  ax.legend(fontsize=8);  ax.grid(alpha=0.3)
plt.tight_layout()
fig9.savefig(f"{OUT_DIR}/fig9_training_history.png", dpi=150, bbox_inches="tight");  plt.close()

# ── Figure 10: RF feature importance (v2) ─────────────────────────────────────
fig10, ax10 = plt.subplots(figsize=(12, 8))
top20 = importances.head(20)
def feat_color(n):
    if any(x in n for x in ["trend_fast", "trend_slow", "trend_accel", "trend_cross",
                              "price_vs_fast", "price_vs_slow", "cycle"]):
        return "#1A5276"   # trend/decomp — dark blue
    if any(x in n for x in ["adx", "di_diff", "ma_cross", "ma_stack", "price_vs_200"]):
        return "#0D9488"   # trend-strength indicators — teal
    if any(x in n for x in ["dist_swing", "dist_52", "bb_"]):
        return "#8E44AD"   # S/R — purple
    if any(x in n for x in ["vix","yield","cpi","gold","move","risk_off","cpi"]):
        return "#E67E22"   # macro — orange
    if any(x in n for x in ["qqq","tlt","xle","xlf","concordance","equity_bond"]):
        return "#2E86C1"   # cross-asset — blue
    return "#76B5DB"        # other/momentum
colors_bar = [feat_color(n) for n in top20.index]
ax10.barh(range(len(top20)), top20.values[::-1], color=colors_bar[::-1])
ax10.set_yticks(range(len(top20))); ax10.set_yticklabels(top20.index[::-1], fontsize=9)
ax10.set_xlabel("Feature Importance (Gini)", fontsize=11)
ax10.set_title("Random Forest — Top 20 Features  [v2 Feature Set]",
               fontsize=13, fontweight="bold")
ax10.grid(axis="x", alpha=0.3)
ax10.legend(handles=[
    Patch(color="#1A5276", label="Trend decomposition (fast/slow/cycle)"),
    Patch(color="#0D9488", label="Trend strength (ADX/DI/MA stack)"),
    Patch(color="#8E44AD", label="Support/Resistance proximity"),
    Patch(color="#E67E22", label="Macro (VIX/Yields/CPI/Gold/MOVE)"),
    Patch(color="#2E86C1", label="Cross-asset + concordance"),
    Patch(color="#76B5DB", label="Price momentum / other"),
], fontsize=9, loc="lower right")
plt.tight_layout()
fig10.savefig(f"{OUT_DIR}/fig10_feature_importance_v2.png", dpi=150, bbox_inches="tight");  plt.close()

print("  Saved: fig1–fig10")

# ── Figure 11: Forward Trend Analysis — Foresight vs Hindsight ───────────────
fig11, axes11 = plt.subplots(2, 3, figsize=(20, 10))
fig11.suptitle("Forward Trend Analysis — Foresight vs Hindsight\n"
               "(Signal-driven: state change detected when decomposition shifts, not at fixed date)",
               fontsize=13, fontweight="bold")

periods      = ["Train\n(Hindsight)", "Val\n(Validation)", "Test\n(Blind)"]
period_res   = [train_fwd, val_results, test_results]
period_cols  = [PALETTE["blue"], PALETTE["teal"], PALETTE["green"]]

for row, w in enumerate([20, 60]):
    for col, (pname, pres, pcol) in enumerate(zip(periods, period_res, period_cols)):
        ax  = axes11[row, col]
        r   = pres[w]
        n   = r["total"]
        if n == 0:
            ax.text(0.5, 0.5, "No data", ha="center", va="center")
            continue

        # Stacked bar: by starting state — continuation vs transition
        states      = [s for s in range(4) if (r["by_state"][s]["continued"] +
                                                r["by_state"][s]["changed"]) > 0]
        cont_vals   = [r["by_state"][s]["continued"] for s in states]
        chng_vals   = [r["by_state"][s]["changed"]   for s in states]
        s_labels    = [STATE_NAMES[s] for s in states]
        x           = np.arange(len(states))

        bars1 = ax.bar(x, cont_vals, color=[STATE_COLORS[s] for s in states],
                       alpha=0.8, label="Continued")
        bars2 = ax.bar(x, chng_vals, bottom=cont_vals,
                       color=[STATE_COLORS[s] for s in states],
                       alpha=0.3, hatch="///", label="Changed")

        ax.set_xticks(x)
        ax.set_xticklabels(s_labels, fontsize=8, rotation=20)
        ax.set_title(f"{pname}  —  {w}-day window", fontsize=10, fontweight="bold")
        ax.set_ylabel("Days")
        ax.grid(axis="y", alpha=0.3)

        # Annotate continuation rates
        totals = [c + g for c, g in zip(cont_vals, chng_vals)]
        for xi, (c, t) in enumerate(zip(cont_vals, totals)):
            if t > 0:
                ax.text(xi, t + 2, f"{c/t:.0%}", ha="center", fontsize=8,
                        fontweight="bold", color="black")

        # Warning stats as text box
        wt = r["warned_and_changed"] + r["warned_not_changed"]
        wp = r["warned_and_changed"] / wt if wt > 0 else 0
        wr = r["warned_and_changed"] / r["changed"] if r["changed"] > 0 else 0
        ad = r["change_day_sum"] / r["change_day_n"] if r["change_day_n"] > 0 else 0
        ax.text(0.98, 0.97,
                f"Warn precision: {wp:.0%}\nWarn recall: {wr:.0%}\nAvg change day: {ad:.1f}",
                transform=ax.transAxes, fontsize=8, va="top", ha="right",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

        if col == 0:
            ax.legend(fontsize=8, loc="upper left")

plt.tight_layout()
fig11.savefig(f"{OUT_DIR}/fig11_forward_analysis.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: fig11_forward_analysis.png")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 11: SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  FINAL SUMMARY  [v3]")
print("=" * 70)
print(f"  {'Model':<22} {'Accuracy':>10}  {'F1-Macro':>10}")
print(f"  {'─'*22} {'─'*10}  {'─'*10}")
for name, acc, f1 in [
    ("MA Stack Heuristic",    acc_ma,  f1_ma),
    ("ADX+DI Rule-Based",     acc_adx, f1_adx),
    ("Random Forest",         acc_rf,  f1_rf),
    ("CNN-only",              results_nn["cnn_only"]["acc"],  results_nn["cnn_only"]["f1"]),
    ("GRU-only",              results_nn["gru_only"]["acc"],  results_nn["gru_only"]["f1"]),
    ("Dual-CNN",              results_nn["dual_cnn"]["acc"],  results_nn["dual_cnn"]["f1"]),
    ("CNN-GRU Fusion",        results_nn["fusion"]["acc"],    results_nn["fusion"]["f1"]),
]:
    print(f"  {name:<22} {acc:>10.4f}  {f1:>10.4f}")
print(f"\n  Naive baseline (majority class): {naive:.4f}")
print(f"  Features: {n_features}  (base={n_features_base} + deltas={len(DELTA_COLS)*2})")
print(f"  Seq lengths: short={SEQ_LEN_S}d / medium={SEQ_LEN_M}d")
print(f"  Target: 4-class trend state (not fwd return)")

# ── Honest interpretation of model scores ─────────────────────────────────────
print("\n  ── Interpretation Note: Feature-Label Correlation ──")
print("""
  The trend-state labels are constructed from three signals: fast EWM trend,
  slow EWM trend (via fast_above_slow), ADX, and DI+/DI-.  Several features
  in the RF feature matrix — including 'adx_14', 'di_diff', 'trend_fast_vs_slow',
  and 'trend_cross_signal' — are derived from those same underlying signals.

  This overlap between feature construction and label construction inflates
  the Random Forest accuracy (~0.988): the RF is partly recovering the label
  formula from its own inputs rather than learning a generalizable pattern.
  The ADX+DI rule-based baseline is affected by the same issue (now corrected
  by using t-1 lag, which reduced the score from ~1.000 to a more plausible
  level).

  The neural sequence models (~0.85 F1) are the more honest comparison point.
  They operate on 60-day rolling windows of *all* features and must learn
  temporal dynamics across time, rather than reading the instantaneous label
  ingredients.  Their scores are therefore less inflated by feature-label
  overlap and better reflect genuine out-of-sample predictive ability.

  For future work, a cleaner label design would use entirely separate signals
  for labeling (e.g., realized Sharpe over a forward window) vs. the technical
  indicators retained as features.
  """)


all_res = {
    "baselines": {
        "ma_stack": {"acc": acc_ma,  "f1": f1_ma},
        "adx_di":   {"acc": acc_adx, "f1": f1_adx},
        "rf":       {"acc": acc_rf,  "f1": f1_rf},
    },
    "nn_models":   {k: {"acc": v["acc"], "f1": v["f1"]} for k, v in results_nn.items()},
    "naive":       naive,
    "n_features":  n_features,
    "n_train":     len(train_idx),
    "n_val":       len(val_idx),
    "n_test":      len(test_idx),
    "top10_features": list(importances.head(10).index),
    "label_type":  "trend_state_4class",
}
with open(f"{OUT_DIR}/all_results_v3.pkl", "wb") as f:
    pickle.dump(all_res, f)
print(f"\n  Results saved to {OUT_DIR}/all_results_v3.pkl")
print("=" * 70)
