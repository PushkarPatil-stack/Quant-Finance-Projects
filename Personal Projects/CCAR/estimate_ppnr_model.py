"""
estimate_ppnr_model.py
-----------------------
Pre-provision net revenue (PPNR) model for Bank of America, estimated on BAC's own reported income
(SEC XBRL) and projected under the Fed Baseline and Severely Adverse paths.

    py estimate_ppnr_model.py

Inputs : data/panel/calibration_panel_full.csv, data/panel/scenarios_long.csv, data/rate_risk/nii_scenario_effect.csv (run build_nii_eve_model.py first)
Outputs (data/models): ppnr_coefficients.csv, ppnr_fit.csv, ppnr_projections.csv, ppnr_report.txt

PPNR = net interest income + noninterest income - noninterest expense. Three equations, each a
one-step error-correction / ARDL(1,1) form (same as the loss models), estimated 2014Q1-2025Q4:
  NIM : NII x4 / total assets (%)          drivers: 3-month T-bill rate, yield-curve slope (10y - 3m)
  FEE : noninterest income x4 / assets (%) drivers: 4-quarter equity-index change, log VIX
  EXP : log noninterest expense            driver : log revenue (NII + noninterest income)
  plus a COVID dummy 2020Q2-2021Q4 in each.
Balance sheet: total assets held flat at the latest level (simplification).

WHY 2014Q1 AND NOT 2008Q2: BAC's 2008-2013 income statement is dominated by items that are not macro
driven - the Countrywide/Merrill integrations, a $10B goodwill impairment (2010), mortgage
representation-and-warranty and litigation charges (2010-2013). The NIM equation fitted from 2008Q4
has R2 0.47 against 0.87 from 2014Q1, and the 2008-2013 fee and expense lines contain one-offs of
$10-20B in single quarters. 2008-2013 data are kept in the panel and used for the descriptive check below.

STRESS OVERLAYS (see OVERLAYS below) are applied to the severely adverse path only.
CAUTION: the sample (2014-2025) has no deep recession - only COVID (policy-supported) and the 2022
equity/rate shock. The severe-scenario response (equity -58%, VIX 72) is therefore an extrapolation.
The Fed's PPNR also includes operational-risk losses (booked in noninterest expense) that this model
has no separate term for. Trading and counterparty losses are NOT part of Fed PPNR; they are a separate
line in the capital waterfall.
"""
import argparse
import os

import numpy as np
import pandas as pd

FED_BAC_PPNR_9Q = 66.2      # Fed 2026 DFAST results, Bank of America, severely adverse, 9-quarter PPNR ($B)
FED_BAC_COMPONENTS = {"nii": 151.1, "nonint_income": 89.1, "nonint_expense": 174.0}   # same source, benchmark only

# ---- STRESS OVERLAYS (severely adverse only). The 2014-2025 sample has no deep recession, so the stress
# response is not estimable from it; these are explicit parameters. Each is sourced or documented in its note.
# NII stress is NOT an overlay parameter any more: it comes from build_nii_eve_model.py (nii_scenario_effect.csv), a model fitted
# to BAC's disclosed NII sensitivities (10-K Tables 44/45: static and dynamic deposits, parallel, flattener, steepener, 2023-2025).
OVERLAYS = [
    ("fee_share_equity_linked", 17.6 / 53.0, "BAC 4Q25 deck: asset management 15.3 + brokerage 2.3 = $17.6B of $53.0B FY2025 noninterest income"),
    ("fee_share_investment_banking", 6.6 / 53.0, "BAC 4Q25 deck: investment banking fees $6.6B of $53.0B"),
    ("fee_share_sales_trading", 20.9 / 53.0, "BAC 4Q25 deck: sales and trading $20.9B of $53.0B"),
    ("equity_linked_elasticity", 0.6, "SOURCED RANGE 0.5-0.7: BAC 10-K GWIM 2007-2009. Columbia fees $1,435M (2007) -> $1,090M (2009) = -24% vs Fed equity index -34% (0.7); "
     "U.S. Trust 2008->2009 -10% vs -16% (0.6); GWIM AUM -19% in 2008 vs equity -39% year-end (0.5). Central value 0.6"),
    ("investment_banking_haircut_at_peak_stress", 0.40, "SOURCED RANGE 0.24-0.54 (central 0.40): BAC 10-K/earnings, total Corporation IB fees 4Q22 $1,071M vs 4Q21 $2,351M = -54% (boom base) "
     "or vs 2019 run-rate $5,642M/4 = -24%. Falls only in 2022; 2008 +7% ($2,537M->$2,708M) and 2020 +27% ($5,642M->$7,180M). Scaled by equity gap"),
    ("sales_trading_haircut_at_peak_stress", 0.0, "SET TO ZERO: no observed fall in client sales and trading revenue in stress (4Q22 $3,525M vs 4Q21 $2,936M; 2009 up). "
     "Losses seen in 2007-08 (-$2.1B, -$5.8B) were mark-to-market on legacy structured products, which the Fed treats as trading and counterparty loss, not PPNR. See sensitivity"),
    ("oprisk_pct_of_assets_9q", 211.0 / (719.0 / 0.03), "Fed 2026 DFAST results: aggregate operational-risk losses $211B; aggregate PPNR $719B = 3% of average assets "
                                                         "-> 0.88% of assets over 9 quarters, applied pro rata to BAC assets"),
]


def covid_dummy(idx):
    c = pd.Series(0.0, index=idx)
    c.loc["2020Q2":"2021Q4"] = 1.0
    return c


def macro_features(m):
    return pd.DataFrame({"t3m": m["macro_tbill_3m"], "slope": m["macro_ust_10y"] - m["macro_tbill_3m"],
                         "eq4": np.log(m["macro_equity_index"]).diff(4), "lvix": np.log(m["macro_vix"])})


def nw_ols(A, y, lags=4):
    n, k = A.shape
    XtXi = np.linalg.inv(A.T @ A)
    b = XtXi @ A.T @ y
    e = y - A @ b
    S = (A * e[:, None]).T @ (A * e[:, None])
    for L in range(1, lags + 1):
        w = 1 - L / (lags + 1)
        G = (A[L:] * e[L:, None]).T @ (A[:-L] * e[:-L, None])
        S += w * (G + G.T)
    se = np.sqrt(np.diag(XtXi @ S @ XtXi * n / (n - k)))
    return b, se, e, 1 - e @ e / ((y - y.mean()) ** 2).sum()


def build(y, drivers, idx):
    Z = pd.concat([y.rename("y"), y.shift(1).rename("ylag")] + [s.rename(n) for n, s in drivers.items()] +
                  [s.shift(1).rename(n + "_l1") for n, s in drivers.items()] +
                  [covid_dummy(idx).rename("covid")], axis=1)
    return Z


def fit(y, drivers, idx, lo, hi):
    Z = build(y, drivers, idx).loc[lo:hi].dropna()
    no_covid = Z["covid"].sum() == 0                  # dummy not identifiable in a pre-2020 sample
    Zf = Z.drop(columns="covid") if no_covid else Z
    A = np.column_stack([np.ones(len(Zf)), Zf.drop(columns="y").values])
    b, se, e, r2 = nw_ols(A, Zf["y"].values)
    if no_covid:
        b, se = np.append(b, 0.0), np.append(se, np.nan)
    return dict(names=["const"] + list(Z.columns[1:]), b=b, se=se, r2=r2, n=len(Z),
                ar1=float(np.corrcoef(e[1:], e[:-1])[0, 1]), sd=float(e.std(ddof=len(b))),
                dnames=list(drivers))


def predict(m, ylag, d_t, d_l1):
    v = [1.0, ylag] + list(d_t) + list(d_l1) + [0.0]
    return float(np.dot(m["b"], v))


def simulate(models, F, assets, start, qs, hist):
    nim, fee, ex = hist["nim"], hist["fee"], hist["lexp"]
    out = []
    prev_rev = None
    for q in qs:
        nim = predict(models["NIM"], nim, F.loc[q, ["t3m", "slope"]], F.loc[q - 1, ["t3m", "slope"]])
        fee = predict(models["FEE"], fee, F.loc[q, ["eq4", "lvix"]], F.loc[q - 1, ["eq4", "lvix"]])
        nii, nonii = nim / 100 * assets / 4, fee / 100 * assets / 4
        rev = nii + nonii
        lrev_prev = hist["lrev"] if prev_rev is None else np.log(prev_rev)
        ex = predict(models["EXP"], ex, [np.log(rev)], [lrev_prev])
        prev_rev = rev
        out.append({"quarter": str(q), "nii": nii, "nonint_income": nonii,
                    "nonint_expense": float(np.exp(ex)), "ppnr": nii + nonii - float(np.exp(ex))})
    return pd.DataFrame(out)


def apply_overlays(base, sev, Mb, Ms, exp_beta, assets, ov, nii_eff, scale_nii=1.0, scale_fee=1.0, peak_ref=None, opr_scale=1.0):
    """Stressed PPNR = baseline model path + NII effect from the rate-risk model + sourced fee/op-risk overlays (see OVERLAYS).
    nii_eff: quarterly NII change vs Baseline ($B) from nii_scenario_effect.csv for this scenario."""
    b, s_ = base.reset_index(drop=True).copy(), sev.reset_index(drop=True).copy()
    eq_gap = (1 - Ms["macro_equity_index"].values / Mb["macro_equity_index"].values).clip(0, None)
    peak = max(peak_ref if peak_ref is not None else eq_gap.max(), 1e-9)   # severe peak is the reference for all scenarios
    nii = b.nii + scale_nii * np.asarray(nii_eff, dtype=float)
    fee_loss = b.nonint_income * scale_fee * (
        ov["fee_share_equity_linked"] * ov["equity_linked_elasticity"] * eq_gap +
        (ov["fee_share_investment_banking"] * ov["investment_banking_haircut_at_peak_stress"] +
         ov["fee_share_sales_trading"] * ov["sales_trading_haircut_at_peak_stress"]) * eq_gap / peak)
    fee = b.nonint_income - fee_loss
    rev_b, rev = b.nii + b.nonint_income, nii + fee
    exp = b.nonint_expense * (rev / rev_b) ** exp_beta
    opr = np.zeros(len(b))
    opr[:9] = opr_scale * ov["oprisk_pct_of_assets_9q"] * assets / 9
    out = s_[["scenario", "quarter"]].copy()
    out["nii"], out["nonint_income"], out["nonint_expense"] = nii, fee, exp + opr
    out["oprisk_loss"] = opr
    out["ppnr"] = nii + fee - (exp + opr)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", default="data/panel/calibration_panel_full.csv")
    ap.add_argument("--scen", default="data/panel/scenarios_long.csv")
    ap.add_argument("--nii-effect", default="data/rate_risk/nii_scenario_effect.csv", help="output of build_nii_eve_model.py")
    ap.add_argument("--outdir", default="data/models")
    ap.add_argument("--start", default="2014Q1")
    ap.add_argument("--end", default="2025Q4")
    ap.add_argument("--test-from", default="2020Q1")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)

    P = pd.read_csv(a.panel)
    P.index = pd.PeriodIndex(P.pop("quarter"), freq="Q")
    S = pd.read_csv(a.scen)
    ta = P["bac_total_assets"].copy()
    ta.loc["2009Q1"] = np.nan                               # Merrill-quarter gap in XBRL
    ta = ta.interpolate(limit_area="inside")
    nim = P["bac_nii"] * 4 / ta * 100
    fee = P["bac_nonint_income"] * 4 / ta * 100
    rev = P["bac_nii"] + P["bac_nonint_income"]
    lexp = np.log(P["bac_nonint_expense"])
    mc = [c for c in P.columns if c.startswith("macro_")]
    F_hist = macro_features(P[mc])
    idx = P.index

    spec = {"NIM": (nim, {n: F_hist[n] for n in ["t3m", "slope"]}),
            "FEE": (fee, {n: F_hist[n] for n in ["eq4", "lvix"]}),
            "EXP": (lexp, {"lrev": np.log(rev)})}
    models = {k: fit(y, d, idx, a.start, a.end) for k, (y, d) in spec.items()}
    coef = [(k, n, b, s, b / s) for k, m in models.items() for n, b, s in zip(m["names"], m["b"], m["se"])]
    fits = [(k, m["n"], m["r2"], m["ar1"], m["b"][1], 1 - m["b"][1], m["sd"]) for k, m in models.items()]

    # out-of-sample check: refit on start..test_from-1, one-step-ahead fit on test period
    oos = []
    for k, (y, d) in spec.items():
        mtr = fit(y, d, idx, a.start, str(pd.Period(a.test_from, "Q") - 1))
        Z = build(y, d, idx).loc[a.test_from:a.end].dropna()
        pred = np.column_stack([np.ones(len(Z)), Z.drop(columns="y").values]) @ mtr["b"]
        err = Z["y"].values - pred
        oos.append((k, len(Z), float(np.sqrt((err ** 2).mean())), float(np.abs(err).mean()),
                    float(1 - (err ** 2).sum() / ((Z.y - Z.y.mean()) ** 2).sum())))

    last = pd.Period(a.end, "Q")
    assets = float(ta.loc[last])
    projs = []
    for sname, g in S.groupby("scenario"):
        g = g.set_index(pd.PeriodIndex(g["quarter"], freq="Q"))
        M = pd.concat([P[mc].loc[:last], g[mc]])
        F_all = macro_features(M)
        hist = {"nim": float(nim.loc[last]), "fee": float(fee.loc[last]),
                "lexp": float(lexp.loc[last]), "lrev": float(np.log(rev.loc[last]))}
        df = simulate(models, F_all, assets, last, list(g.index), hist)
        df.insert(0, "scenario", sname)
        projs.append(df)
    proj = pd.concat(projs)

    # ---- stress overlays on the severely adverse path
    ov = {k: v for k, v, _ in OVERLAYS}
    mE = models["EXP"]
    exp_beta = (mE["b"][mE["names"].index("lrev")] + mE["b"][mE["names"].index("lrev_l1")]) / (1 - mE["b"][1])
    sc = {n: g.set_index(pd.PeriodIndex(g["quarter"], freq="Q")) for n, g in S.groupby("scenario")}
    nb = next(n for n in sc if "Baseline" in n)
    ns = next(n for n in sc if "Severely" in n)
    base = proj[proj.scenario == nb]
    sev_model = proj[proj.scenario == ns]
    Mb, Ms = sc[nb][mc].reset_index(drop=True), sc[ns][mc].reset_index(drop=True)
    if not os.path.exists(a.nii_effect):
        raise SystemExit(f"{a.nii_effect} not found - run build_nii_eve_model.py first")
    NE = pd.read_csv(a.nii_effect)
    nii_eff = lambda name, g: NE[NE.scenario == name].set_index("quarter").loc[[str(q) for q in g.index], "nii_effect_vs_baseline_b"].values
    sev = apply_overlays(base, sev_model, Mb, Ms, exp_beta, assets, ov, nii_eff(ns, sc[ns]))
    sens = []
    for lab, sn, sf in [("base overlays", 1.0, 1.0), ("NII sensitivity x0.5", 0.5, 1.0), ("NII sensitivity x1.5", 1.5, 1.0),
                        ("fee stress x0.5", 1.0, 0.5), ("fee stress x1.5", 1.0, 1.5)]:
        r = apply_overlays(base, sev_model, Mb, Ms, exp_beta, assets, ov, nii_eff(ns, sc[ns]), sn, sf).head(9)
        sens.append((lab, r.ppnr.sum()))
    for lab, k, v in [("IB haircut 0.00 (2008 and 2020 behaviour)", "investment_banking_haircut_at_peak_stress", 0.0), ("IB haircut 0.24", "investment_banking_haircut_at_peak_stress", 0.24), ("IB haircut 0.54", "investment_banking_haircut_at_peak_stress", 0.54),
                      ("elasticity 0.5", "equity_linked_elasticity", 0.5), ("elasticity 0.7", "equity_linked_elasticity", 0.7),
                      ("S&T haircut 0.15", "sales_trading_haircut_at_peak_stress", 0.15)]:
        r = apply_overlays(base, sev_model, Mb, Ms, exp_beta, assets, {**ov, k: v}, nii_eff(ns, sc[ns])).head(9)
        sens.append((lab, r.ppnr.sum()))
    sev_model = sev_model.assign(scenario=ns + " (data-driven model only)")
    sev_peak = float((1 - Ms["macro_equity_index"].values / Mb["macro_equity_index"].values).clip(0, None).max())
    na = next((n for n in sc if "Adverse" in n and "Severely" not in n), None)
    parts = [base, sev.assign(scenario=ns + " (with overlays)"), sev_model]
    if na:       # Adverse: same overlays, scaled by its own (smaller) equity and rate gaps; op-risk at the same weight
        Ma = sc[na][mc].reset_index(drop=True)
        adv_model = proj[proj.scenario == na]
        w_adv = float(sev_peak and (1 - Ma["macro_equity_index"].values / Mb["macro_equity_index"].values).clip(0, None).max() / sev_peak)
        adv = apply_overlays(base, adv_model, Mb, Ma, exp_beta, assets, ov, nii_eff(na, sc[na]), peak_ref=sev_peak, opr_scale=w_adv)
        parts.append(adv.assign(scenario=na + " (with overlays)"))
    proj = pd.concat(parts)
    pd.DataFrame([("nii_effect (from file)", np.nan, "build_nii_eve_model.py: NII model fitted to BAC 10-K Tables 44/45; quarterly effect vs Baseline from the 3-month-rate gap, dynamic deposits")] +
                 [(n, v, d) for n, v, d in OVERLAYS], columns=["parameter", "value", "source"]).to_csv(
        os.path.join(a.outdir, "ppnr_overlays.csv"), index=False)

    pd.DataFrame(coef, columns=["equation", "term", "coef", "se_NW", "t"]).round(4).to_csv(
        os.path.join(a.outdir, "ppnr_coefficients.csv"), index=False)
    fit_df = pd.DataFrame(fits, columns=["equation", "n", "R2", "resid_AR1", "rho", "speed_1-rho", "resid_sd"])
    oos_df = pd.DataFrame(oos, columns=["equation", "n_test", "rmse", "mae", "R2_oos"])
    fit_df.merge(oos_df, on="equation").round(3).to_csv(os.path.join(a.outdir, "ppnr_fit.csv"), index=False)
    proj.round(3).to_csv(os.path.join(a.outdir, "ppnr_projections.csv"), index=False)

    run_rate = float(rev.loc[last] - np.exp(lexp.loc[last]))
    L = [f"PPNR model, fit {a.start}-{a.end}; assets held flat at ${assets:,.0f}B",
         "", "FIT (in sample) and out-of-sample check (refit before " + a.test_from + ")",
         fit_df.round(3).to_string(index=False), "", oos_df.round(3).to_string(index=False), "",
         f"Latest actual quarterly PPNR {run_rate:.1f}B; x9 = {run_rate*9:.0f}B (flat run-rate benchmark).", ""]
    for s, g in proj.groupby("scenario"):
        g9 = g.head(9)
        L.append(f"{s}: 9-quarter PPNR {g9.ppnr.sum():.1f}B | NII {g9.nii.sum():.1f} fees {g9.nonint_income.sum():.1f} "
                 f"expense {g9.nonint_expense.sum():.1f} | lowest quarter PPNR {g9.ppnr.min():.1f}B")
    L += ["", f"Expense elasticity to revenue used in the overlay (long-run, from the EXP equation): {exp_beta:.2f}",
          "Sensitivity of the with-overlay severe 9Q PPNR (NII effect scaled x0.5 / x1.5 or fee stress scaled): " + "; ".join(f"{n} {v:.1f}B" for n, v in sens),
          "Fed components, BAC severe 9Q (benchmark): " + ", ".join(f"{k} {v}" for k, v in FED_BAC_COMPONENTS.items()),
          "Overlay parameters and sources are in ppnr_overlays.csv; ranges and sources for each parameter are in the notes."]
    L += ["", f"Fed published BAC severely adverse 9-quarter PPNR: {FED_BAC_PPNR_9Q}B "
              "(benchmark only; nothing here is fitted to it)."]
    txt = "\n".join(L)
    open(os.path.join(a.outdir, "ppnr_report.txt"), "w").write(txt + "\n")
    print(txt)


if __name__ == "__main__":
    main()
