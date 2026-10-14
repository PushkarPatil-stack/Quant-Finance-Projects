"""
build_calibration_panel.py
---------------------------
Merges three quarterly sources into one calibration panel and finds the date
range available for every series:

  1. Fed 2026 supervisory macro history   (--fed-dir)
  2. FRED loss series, from fetch_fred_losses.py   (--fred)
  3. BAC quarterly table, from fetch_bac_xbrl.py   (--bac)

It also converts the Fed Baseline and Severely Adverse paths to the same
column names (scenarios_long.csv), so history and forecast line up.

    py build_calibration_panel.py
    py build_calibration_panel.py --exclude bac_nco_total      # drop a series before
                                                               # computing the window

Outputs (in --outdir, default data/panel):
    calibration_panel_full.csv    every quarter, every column, blanks where missing
    calibration_panel_common.csv  only the quarters where EVERY required series exists
    scenarios_long.csv            Baseline + Severely Adverse, same macro columns
    window_report.txt             coverage of each series, which series limits the
                                  window, and what the window would be without it

"Required" = every series except those listed in OPTIONAL below. If the
quarters with complete data are not one unbroken run, the longest run ending
latest is used and the gap is reported.
"""

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

MACRO_RENAME = {
    "Real GDP growth": "macro_real_gdp_growth",
    "Nominal GDP growth": "macro_nominal_gdp_growth",
    "Real disposable income growth": "macro_real_dpi_growth",
    "Nominal disposable income growth": "macro_nominal_dpi_growth",
    "Unemployment rate": "macro_unemployment",
    "CPI inflation rate": "macro_cpi_inflation",
    "3-month Treasury rate": "macro_tbill_3m",
    "5-year Treasury yield": "macro_ust_5y",
    "10-year Treasury yield": "macro_ust_10y",
    "BBB corporate yield": "macro_bbb_yield",
    "Mortgage rate": "macro_mortgage_rate",
    "Prime rate": "macro_prime_rate",
    "Dow Jones Total Stock Market Index (Level)": "macro_equity_index",
    "House Price Index (Level)": "macro_hpi",
    "Commercial Real Estate Price Index (Level)": "macro_cre_index",
    "Market Volatility Index (Level)": "macro_vix",
}
# Present in the full panel but not required for the common window
OPTIONAL = {
    "bac_nco_reported", "bac_provision_unfunded", "bac_provision_total",
    "bac_chargeoffs", "bac_recoveries",          # same coverage as bac_nco_total
    "fred_bal_ci", "fred_bal_real_estate", "fred_bal_cards",
}
LEVEL_INDICES = {"macro_equity_index", "macro_hpi", "macro_cre_index", "macro_vix"}
BLOCKS = {
    "macro only": lambda c: c.startswith("macro_"),
    "macro + FRED loss rates": lambda c: c.startswith("macro_") or (
        c.startswith("fred_") and c not in OPTIONAL),
    "macro + BAC income/PPNR": lambda c: c.startswith("macro_") or c in {
        "bac_nii", "bac_nonint_income", "bac_nonint_expense", "bac_ppnr",
        "bac_net_income", "bac_provision_loans"},
    "macro + BAC credit (NCO, allowance)": lambda c: c.startswith("macro_") or c in {
        "bac_nco_total", "bac_allowance"},
}


def find_one(folder, suffix):
    hits = sorted(glob.glob(os.path.join(folder, f"*{suffix}")))
    if not hits:
        sys.exit(f"No file ending in '{suffix}' found in {folder}")
    if len(hits) > 1:
        print(f"  note: several files match *{suffix}; using {os.path.basename(hits[-1])}")
    return hits[-1]


def load_fed(path):
    df = pd.read_csv(path)
    idx = pd.PeriodIndex(df["Date"].str.replace(" ", ""), freq="Q")
    out = df.drop(columns=["Scenario Name", "Date"]).rename(columns=MACRO_RENAME)
    out.index = idx
    out["macro_bbb_spread"] = out["macro_bbb_yield"] - out["macro_ust_10y"]
    return out


def load_bac(path):
    df = pd.read_csv(path, parse_dates=["quarter_end"])
    out = df.drop(columns=["quarter_end"]).add_prefix("bac_")
    out.index = df["quarter_end"].dt.to_period("Q")
    return out


def load_fred(path):
    df = pd.read_csv(path)
    out = df.drop(columns=["quarter"])
    out.index = pd.PeriodIndex(df["quarter"], freq="Q")
    return out


def longest_window(df):
    """Longest unbroken run of quarters where every column is present (latest wins ties)."""
    if df.empty or df.shape[1] == 0:
        return None, None, 0
    full = pd.period_range(df.index.min(), df.index.max(), freq="Q")
    ok = df.reindex(full).notna().all(axis=1).to_numpy()
    ords = full.asi8
    best = (0, -1, None, None)            # (length, last ordinal, first ordinal, last ordinal)
    i = 0
    while i < len(ok):
        if not ok[i]:
            i += 1
            continue
        j = i
        while j + 1 < len(ok) and ok[j + 1]:
            j += 1
        cand = (j - i + 1, ords[j], ords[i], ords[j])
        if cand[:2] > best[:2]:
            best = cand
        i = j + 1
    if best[2] is None:
        return None, None, 0
    return (pd.Period(ordinal=int(best[2]), freq="Q"),
            pd.Period(ordinal=int(best[3]), freq="Q"), int(best[0]))


def fmt(w):
    return f"{w[0]} to {w[1]} ({w[2]} quarters)" if w[2] else "no overlapping quarters"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fed-dir", default="data/raw/fed")
    ap.add_argument("--bac", default="data/xbrl/bac_quarterly_xbrl.csv")
    ap.add_argument("--fred", default="data/fred/fred_losses_quarterly.csv")
    ap.add_argument("--outdir", default="data/panel")
    ap.add_argument("--exclude", default="", help="comma-separated columns to drop from the window test")
    ap.add_argument("--adverse-weight", type=float, default=0.5,
                    help="Adverse = Baseline + w x (Severely Adverse - Baseline); the Fed publishes no Adverse path")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    lines = []

    hist = load_fed(find_one(args.fed_dir, "Historic_Domestic.csv"))
    parts = [hist]
    if os.path.exists(args.bac):
        parts.append(load_bac(args.bac))
    else:
        lines.append(f"WARNING: BAC table not found at {args.bac} (run fetch_bac_xbrl.py)")
    if os.path.exists(args.fred):
        parts.append(load_fred(args.fred))
    else:
        lines.append(f"WARNING: FRED losses not found at {args.fred} (run fetch_fred_losses.py)")

    panel = pd.concat(parts, axis=1).sort_index()
    panel.index.name = "quarter"
    exclude = {c.strip() for c in args.exclude.split(",") if c.strip()}
    unknown = exclude - set(panel.columns)
    if unknown:
        sys.exit(f"--exclude names not in the panel: {sorted(unknown)}")
    required = [c for c in panel.columns if c not in OPTIONAL and c not in exclude]

    # coverage of every series
    rows = []
    for c in panel.columns:
        s = panel[c].dropna()
        inside = panel.loc[s.index.min():s.index.max(), c].isna().sum() if len(s) else 0
        rows.append({"series": c, "required": c in required, "first": s.index.min(),
                     "last": s.index.max(), "n_valid": len(s), "gaps_inside": int(inside)})
    cov = pd.DataFrame(rows)

    win = longest_window(panel[required])
    lines.append(f"STRICT COMMON WINDOW (all {len(required)} required series): {fmt(win)}")
    if win[2]:
        start, end, _ = win
        def names(lst):
            return ", ".join(lst) if len(lst) <= 4 else f"{len(lst)} series ({', '.join(lst[:3])}, ...)"
        binding_start = cov[cov.required & (cov["first"] == start)]["series"].tolist()
        if not binding_start:   # window starts right after a gap inside a series
            before = start - 1
            binding_start = [f"{c} (gap at {before})" for c in required
                             if before in panel.index and pd.isna(panel.loc[before, c])]
        binding_end = cov[cov.required & (cov["last"] == end)]["series"].tolist()
        lines.append(f"  start set by: {names(binding_start) or '(unknown)'}")
        lines.append(f"  end set by:   {names(binding_end) or '(unknown)'}")
        later = cov[cov.required & (cov["last"] > end)]
        if len(later):
            for last, grp in later.groupby("last"):
                names = grp["series"].tolist()
                shown = ", ".join(names) if len(names) <= 4 else f"{len(names)} series ({names[0]} ...)"
                lines.append(f"  data continues after the window to {last}: {shown}")
        common = panel.loc[start:end, required]
        common.to_csv(os.path.join(args.outdir, "calibration_panel_common.csv"))
    else:
        lines.append("  no quarter has every required series; see the table below")

    lines.append("\nWINDOW BY BLOCK (what each model block could use on its own):")
    for name, pick in BLOCKS.items():
        cols = [c for c in panel.columns if pick(c) and c not in exclude]
        if not any(not c.startswith("macro_") for c in cols) and name != "macro only":
            lines.append(f"  {name:38s} (series not loaded)")
            continue
        lines.append(f"  {name:38s} {fmt(longest_window(panel[cols]))}")

    lines.append("\nWHAT IF A SERIES WERE DROPPED (largest gains in window length):")
    base_n = win[2]
    gains = []
    for c in required:
        w = longest_window(panel[[x for x in required if x != c]])
        if w[2] > base_n:
            gains.append((w[2] - base_n, c, w))
    for g, c, w in sorted(gains, reverse=True)[:8]:
        lines.append(f"  drop {c:26s} -> {fmt(w)} (+{g}q)")
    if not gains:
        lines.append("  no single series is limiting the window")

    lines.append("\nSERIES COVERAGE:")
    lines.append(cov.to_string(index=False))

    panel_out = panel.copy()
    panel_out.to_csv(os.path.join(args.outdir, "calibration_panel_full.csv"))

    # scenarios in the same column names
    sc = []
    for suffix in ("Baseline_Domestic.csv", "Severely_Adverse_Domestic.csv"):
        df = pd.read_csv(find_one(args.fed_dir, suffix))
        names = df["Scenario Name"]
        d = df.drop(columns=["Scenario Name", "Date"]).rename(columns=MACRO_RENAME)
        d["macro_bbb_spread"] = d["macro_bbb_yield"] - d["macro_ust_10y"]
        d.insert(0, "quarter", df["Date"].str.replace(" ", ""))
        d.insert(0, "scenario", names)
        sc.append(d)
    base_df, sev_df = sc[0].reset_index(drop=True), sc[1].reset_index(drop=True)
    adv = base_df.copy()
    adv["scenario"] = "Constructed Adverse"
    w = args.adverse_weight
    for c in [c for c in base_df.columns if c.startswith("macro_")]:
        if c in LEVEL_INDICES:                     # price indices: interpolate in logs
            adv[c] = np.exp(np.log(base_df[c]) + w * (np.log(sev_df[c]) - np.log(base_df[c])))
        else:
            adv[c] = base_df[c] + w * (sev_df[c] - base_df[c])
    lines.append(f"Constructed Adverse scenario = Baseline + {w:.2f} x (Severely Adverse - Baseline); "
                 "price indices interpolated in logs. This is an assumption, not a Fed scenario.")
    pd.concat(sc + [adv]).to_csv(os.path.join(args.outdir, "scenarios_long.csv"), index=False)

    text = "\n".join(lines)
    with open(os.path.join(args.outdir, "window_report.txt"), "w") as f:
        f.write(text + "\n")
    print(text)
    print(f"\nWrote calibration_panel_full.csv ({len(panel)} quarters), "
          f"calibration_panel_common.csv, scenarios_long.csv, window_report.txt in {args.outdir}")


if __name__ == "__main__":
    main()
