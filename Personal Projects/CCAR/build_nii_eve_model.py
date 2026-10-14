"""
build_nii_eve_model.py
-----------------------
Simplified banking-book NII sensitivity model for Bank of America (12-month, instantaneous shocks), fitted to the
eight Table 45 disclosures in BAC's FY2025 and FY2024 10-Ks. EVE is in build_basel_eve.py (Basel standardised framework).

    py build_nii_eve_model.py

Inputs : data/y9c/FRY9C_1073757_20251231.csv and FRY9C_1073757_20250930.csv (Q3 is used to isolate 4Q25 flows),
         data/panel/scenarios_long.csv (Fed paths, for the scenario NII effect)
Outputs (data/rate_risk): nii_eve_balance_sheet.csv, nii_eve_assumptions.csv, nii_eve_results.csv, nii_fit_vs_disclosed.csv, nii_eve_sensitivity.csv,
         nii_scenario_effect.csv, nii_eve_report.txt

WHAT IS DATA AND WHAT IS ASSUMED - every parameter is in nii_eve_assumptions.csv with source = DATA or ASSUMED.
  DATA   : balances by category, 4Q25 interest income and expense by category, securities maturity buckets,
           BAC FY2025 10-K Table 45 (eight NII sensitivities: parallel, flattener, steepener).
  ASSUMED: loan floating-rate share and fixed-rate life, MBS prepayment, how much of each funding source reprices,
           deposit floors, durations used for EVE. Deposit beta, deposit floor tier, long-end repricing and long-debt floating share are CALIBRATED by least squares to the eight disclosed points; they are not observed. Previously: the
           disclosed -$2.0B; it is not observed.
This is not BAC's internal model. Treat EVE in particular as illustrative: no BAC EVE figure was used.
"""
import argparse
import os

import numpy as np
import pandas as pd


# BAC 10-K FY2025 Table 45, 12-month NII change ($B): (long-end bp, short-end bp, disclosed)
# Table 44 (FY2024 10-K) STATIC-deposit columns: no balance change. 2024 and 2023.
TARGETS_STATIC_2024 = [(100, 100, 3.1), (-100, -100, -3.3), (200, 200, 6.2), (-200, -200, -6.9),
                       (0, 100, 2.8), (-100, 0, -0.4), (0, -100, -2.9), (100, 0, 0.3)]
TARGETS_STATIC_2023 = [(100, 100, 3.5), (-100, -100, -3.1), (0, 100, 3.2), (-100, 0, -0.3), (0, -100, -2.8), (100, 0, 0.3)]
LABELS_2023 = ["+100 parallel", "-100 parallel", "Flattener: short +100", "Flattener: long -100",
               "Steepener: short -100", "Steepener: long +100"]
# DYNAMIC-deposit columns (Table 45 FY2025 10-K shows 2025 and 2024 on this basis)
TARGETS_2024 = [(100, 100, 1.1), (-100, -100, -2.3), (200, 200, 2.0), (-200, -200, -5.4),
                (0, 100, 1.1), (-100, 0, -0.1), (0, -100, -2.1), (100, 0, 0.1)]
TARGETS = [(100, 100, 0.7), (-100, -100, -2.0), (200, 200, 0.8), (-200, -200, -4.9),
           (0, 100, 0.5), (-100, 0, -0.3), (0, -100, -1.7), (100, 0, 0.3)]
TARGET_LABELS = ["+100 parallel", "-100 parallel", "+200 parallel", "-200 parallel",
                 "Flattener: short +100", "Flattener: long -100", "Steepener: short -100", "Steepener: long +100"]

# Y-9C items ($ thousands). Flows are calendar year-to-date.
ITEMS = {
    "loans": "BHCK2122", "afs": "BHCK1773", "htm": "BHCK1754", "trading_assets": "BHCK3545",
    "reverse_repo": "BHCKB989", "bal_dom": "BHCK0395", "bal_for": "BHCK0397", "total_assets": "BHCK2170",
    "dep_nib_dom": "BHDM6631", "dep_ib_dom": "BHDM6636", "dep_nib_for": "BHFN6631", "dep_ib_for": "BHFN6636",
    "repo": "BHCKB995", "trading_liab": "BHCK3548", "other_borrowed": "BHCK3190", "sub_debt": "BHCK4062",
    "ob_le1y": "BHCK2332", "ob_gt1y": "BHCK2333", "equity": "BHCK3210", "total_liab": "BHCK2948",
    "sec_le1y": "BHCK0383", "sec_1_5y": "BHCK0384", "sec_gt5y": "BHCK0387",
    "loans_ci": "BHCK1763", "loans_ci_nonus": "BHCK1764", "cards": "BHCKB538", "other_revolving": "BHCKB539",
    "nonbank_fin": "BHCKJ454", "other_loans": "BHCKJ451", "auto": "BHCKK137", "other_cons": "BHCKK207",
    "re_total": "BHCK1410", "mort_first": "BHDM5367", "mort_junior": "BHDM5368", "heloc": "BHDM1797", "tier1": "BHCA8274", "cre_c1": "BHCKF158", "cre_c2": "BHCKF159", "cre_o": "BHCKF160", "cre_n": "BHCKF161",
}
FLOWS = {   # YTD interest, $ thousands
    "ii_loans": ["BHCK4435", "BHCK4436", "BHCKF821", "BHCK4059", "BHCK4065"],
    "ii_securities": ["BHCKB488", "BHCKB489", "BHCK4060"], "ii_trading": ["BHCK4069"],
    "ii_reverse_repo": ["BHCK4020"], "ii_balances": ["BHCK4115"], "ii_other": ["BHCK4518"],
    "ie_deposits": ["BHCK6761", "BHCKHK03", "BHCKHK04", "BHCK4172"], "ie_repo": ["BHCK4180"],
    "ie_other_borrowed": ["BHCK4185"], "ie_sub_debt": ["BHCK4397"],
    "ii_total": ["BHCK4107"], "ie_total": ["BHCK4073"], "nii_total": ["BHCK4074"],
}


def read(path):
    d = pd.read_csv(path, dtype=str).drop_duplicates("ItemName").set_index("ItemName")
    return pd.to_numeric(d["Value"], errors="coerce") / 1e6        # $ billions


def load(y4, y3):
    v4, v3 = read(y4), read(y3)
    bs4 = {k: float(v4.get(c, np.nan)) for k, c in ITEMS.items()}
    bs3 = {k: float(v3.get(c, np.nan)) for k, c in ITEMS.items()}
    q4 = {}
    for k, codes in FLOWS.items():
        q4[k] = float(sum(v4.get(c, 0.0) for c in codes) - sum(v3.get(c, 0.0) for c in codes))
    return bs4, bs3, q4


def build_frame(bs4, bs3, q4):
    avg = lambda k: (bs4[k] + bs3[k]) / 2
    bal = {
        "Cash and interest-bearing balances": ("A", avg("bal_dom") + avg("bal_for"), q4["ii_balances"]),
        "Reverse repo": ("A", avg("reverse_repo"), q4["ii_reverse_repo"]),
        "Trading assets": ("A", avg("trading_assets"), q4["ii_trading"]),
        "Debt securities (AFS + HTM)": ("A", avg("afs") + avg("htm"), q4["ii_securities"]),
        "Loans and leases": ("A", avg("loans"), q4["ii_loans"]),
        "Interest-bearing deposits": ("L", avg("dep_ib_dom") + avg("dep_ib_for"), q4["ie_deposits"]),
        "Repo": ("L", avg("repo"), q4["ie_repo"]),
        "Other borrowed money and trading liabilities": ("L", avg("other_borrowed") + avg("trading_liab"), q4["ie_other_borrowed"]),
        "Subordinated debt": ("L", avg("sub_debt"), q4["ie_sub_debt"]),
    }
    rows = [dict(category=k, side=s, avg_balance_b=b, q4_interest_b=i, annualized_rate_pct=i * 4 / b * 100)
            for k, (s, b, i) in bal.items()]
    return pd.DataFrame(rows)


def loan_split(bs4):
    float_bal = (bs4["loans_ci"] + bs4["loans_ci_nonus"] + bs4["cards"] + bs4["other_revolving"] + bs4["nonbank_fin"] +
                 bs4["other_loans"] + bs4["heloc"] + bs4["cre_c1"] + bs4["cre_c2"] + bs4["cre_o"] + bs4["cre_n"])
    return float_bal / bs4["loans"]


class Params:
    pass


def make_params(bs4, beta, **kw):
    p = Params()
    p.beta = beta
    p.loan_float = loan_split(bs4)               # ASSUMED to reprice with policy rates; balances are DATA
    p.loan_fixed_life = 5.0                       # ASSUMED years (mortgages, auto, term loans)
    p.sec_prepay = 0.10                           # ASSUMED annual repricing of >5y bucket from MBS prepayments
    p.sec_mid = (0.5, 3.0, 9.0)                   # ASSUMED bucket midpoints (years) for <=1y, 1-5y, >5y
    p.floor_tier_share = 0.30                     # ASSUMED share of interest-bearing deposits in a low-rate tier that floors at zero
    p.floor_tier_cost = 0.0015                    # ASSUMED cost of that tier (0.15%)
    p.ltd_float = 0.50                            # ASSUMED share of >1y borrowings effectively floating (swapped)
    p.long_scale = 1.0                            # CALIBRATED: scales repricing of fixed-rate assets (net of hedges)
    for k, v in kw.items():
        setattr(p, k, v)
    return p


def dnii(p, bs4, bs3, frame, shock_bp, short_bp=None):
    # shock_bp = long-end shift; short_bp = short-end shift (defaults to shock_bp = parallel)
    """12-month change in NII ($B) from an instantaneous shift. shock_bp applies to the whole curve; short_bp (optional)
    applies to funding and floating items (a steepener/flattener moves the short end differently from the long end)."""
    s = shock_bp / 1e4
    sh = (short_bp if short_bp is not None else shock_bp) / 1e4
    avgb = lambda k: (bs4[k] + bs3[k]) / 2
    g = lambda cat: float(frame.loc[frame.category == cat, "avg_balance_b"].iloc[0])
    # assets
    short_assets = g("Cash and interest-bearing balances") + g("Reverse repo") + g("Trading assets")
    d_assets = short_assets * sh
    L = g("Loans and leases")
    d_assets += L * p.loan_float * sh
    fixed_loans = L * (1 - p.loan_float)
    d_assets += fixed_loans * min(1.0, 1.0 / p.loan_fixed_life) * 0.5 * s * p.long_scale
    b1, b2, b3 = bs4["sec_le1y"], bs4["sec_1_5y"], bs4["sec_gt5y"]
    m = p.sec_mid
    roll = b1 * 1.0 + b2 * min(1.0, 1 / m[1] / 1.0) + b3 * (1 / m[2] + p.sec_prepay)
    d_assets += roll * 0.5 * s * p.long_scale
    # liabilities
    ib = g("Interest-bearing deposits")
    rate_ib = float(frame.loc[frame.category == "Interest-bearing deposits", "annualized_rate_pct"].iloc[0]) / 100
    a_share = p.floor_tier_share
    cost_a = p.floor_tier_cost
    cost_b = max((rate_ib - a_share * cost_a) / (1 - a_share), 0.0)
    dep_a = ib * a_share * (max(cost_a + p.beta * sh, 0) - cost_a)
    dep_b = ib * (1 - a_share) * (max(cost_b + p.beta * sh, 0) - cost_b)
    d_liab = dep_a + dep_b
    d_liab += g("Repo") * sh
    ob_total = g("Other borrowed money and trading liabilities")
    short_ob = bs4["ob_le1y"] + bs4["trading_liab"]
    long_ob = bs4["ob_gt1y"]
    d_liab += short_ob * sh + long_ob * p.ltd_float * sh
    return d_assets - d_liab


def mig(m, bs, short_bp):
    """Dynamic-deposit effect ($B, 12m): balances leave non-interest-bearing deposits when short rates rise (cost) and the
    reverse when they fall. Scales with NIB balances; driven by the short-end shock only (long-end moves add ~0.2-0.3)."""
    nib = bs["dep_nib_dom"] + bs["dep_nib_for"]
    x = short_bp / 100.0
    return m[0] * nib * x ** m[3] if x >= 0 else m[1] * nib * abs(x) ** m[2]


def predict(p, m, d, l, sh):
    v = dnii(p, d["bs4"], d["bs3"], d["frame"], l, sh)
    if d["basis"] == "dynamic" and m is not None:
        v += mig(m, d["bs4"], sh)
    return v


def fit(DS, fixed=None, per_year_long_scale=False):
    """Least squares over every disclosed point. Behavioural parameters (deposit beta, low-rate tier, floating long debt)
    are shared across years; the long-end repricing scale (hedge book) is shared unless per_year_long_scale. If any
    dataset is DYNAMIC, three deposit-migration parameters are fitted as well."""
    from scipy.optimize import least_squares
    fixed = fixed or {}
    shared = [n for n in ("beta", "floor_tier_share", "ltd_float") if n not in fixed]
    years = sorted({d["year"] for d in DS})
    nls = len(years) if per_year_long_scale else 1
    has_dyn = any(d["basis"] == "dynamic" for d in DS)
    lb0 = {"beta": 0.0, "floor_tier_share": 0.0, "ltd_float": 0.0}
    ub0 = {"beta": 1.0, "floor_tier_share": 0.9, "ltd_float": 1.0}
    x0 = [{"beta": 0.5, "floor_tier_share": 0.3, "ltd_float": 0.5}[n] for n in shared] + [0.3] * nls
    lb = [lb0[n] for n in shared] + [0.0] * nls
    ub = [ub0[n] for n in shared] + [1.5] * nls
    if has_dyn:
        x0 += [-4.0, 2.0, 0.7, 1.0]
        lb += [-30.0, 0.0, 0.2, 0.5]
        ub += [0.0, 30.0, 2.0, 2.0]
    ns = len(shared)

    def unpack(x, d):
        kw = dict(fixed)
        kw.update(dict(zip(shared, x[:ns])))
        beta = kw.pop("beta", 0.5)
        kw["long_scale"] = x[ns + (years.index(d["year"]) if per_year_long_scale else 0)]
        p = make_params(d["bs4"], beta, **kw)
        m = (x[-4] * 1e-3, x[-3] * 1e-3, x[-2], x[-1]) if has_dyn else None
        return p, m

    def resid(x):
        r = []
        for d in DS:
            p, m = unpack(x, d)
            r += [predict(p, m, d, l, sh) - t for l, sh, t in d["targets"]]
        return r
    r = least_squares(resid, x0, bounds=(lb, ub))
    out = []
    for d in DS:
        p, m = unpack(r.x, d)
        out.append((p, [predict(p, m, d, l, sh) for l, sh, t in d["targets"]]))
    return out, (unpack(r.x, DS[0])[1])


def make_datasets(paths):
    """paths: {year: (q4_file, q3_file)} -> dataset list (2023 static, 2024 static, 2024 dynamic, 2025 dynamic)."""
    L = {}
    for yr, (f4, f3) in paths.items():
        b4, b3, q = load(f4, f3)
        L[yr] = (b4, b3, build_frame(b4, b3, q), q)
    spec = [("2023", "static", TARGETS_STATIC_2023, LABELS_2023), ("2024", "static", TARGETS_STATIC_2024, TARGET_LABELS),
            ("2024", "dynamic", TARGETS_2024, TARGET_LABELS), ("2025", "dynamic", TARGETS, TARGET_LABELS)]
    return [dict(year=y, basis=b, bs4=L[y][0], bs3=L[y][1], frame=L[y][2], q=L[y][3], targets=tg, labels=lb) for y, b, tg, lb in spec]


def main():
    ap = argparse.ArgumentParser()
    for yr in ("2025", "2024", "2023"):
        ap.add_argument(f"--y9c-q4-{yr}", default=f"data/y9c/FRY9C_1073757_{yr}1231.csv")
        ap.add_argument(f"--y9c-q3-{yr}", default=f"data/y9c/FRY9C_1073757_{yr}0930.csv")
    ap.add_argument("--scen", default="data/panel/scenarios_long.csv")
    ap.add_argument("--outdir", default="data/rate_risk")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    A_ = vars(a)
    DS = make_datasets({y: (A_[f"y9c_q4_{y}"], A_[f"y9c_q3_{y}"]) for y in ("2025", "2024", "2023")})
    d25 = DS[3]
    bs4, bs3, frame, q4 = d25["bs4"], d25["bs3"], d25["frame"], d25["q"]
    asset_int = frame[frame.side == "A"].q4_interest_b.sum() + q4["ii_other"]
    liab_int = frame[frame.side == "L"].q4_interest_b.sum()
    check = dict(nii_actual_4q25=q4["nii_total"], nii_from_categories=asset_int - liab_int,
                 interest_income_total=q4["ii_total"], interest_expense_total=q4["ie_total"])
    frame.round(4).to_csv(os.path.join(a.outdir, "nii_eve_balance_sheet.csv"), index=False)

    res, m = fit(DS)
    p = res[3][0]
    FIT = pd.DataFrame([dict(year=d["year"], basis=d["basis"], shock=lab, disclosed_b=t, model_b=f, diff_b=f - t)
                        for d, (pp, fv) in zip(DS, res) for lab, (l, sh, t), f in zip(d["labels"], d["targets"], fv)])
    FIT.round(3).to_csv(os.path.join(a.outdir, "nii_fit_vs_disclosed.csv"), index=False)
    rm = lambda df: float(np.sqrt((df["diff_b"] ** 2).mean()))
    rm_static, rm_dyn = rm(FIT[FIT.basis == "static"]), rm(FIT[FIT.basis == "dynamic"])

    # out-of-sample test: fit WITHOUT 2025, predict the disclosed 2025 dynamic sensitivities
    DSo = DS[:3]
    reso, mo = fit(DSo)
    po = make_params(d25["bs4"], reso[2][0].beta, floor_tier_share=reso[2][0].floor_tier_share, ltd_float=reso[2][0].ltd_float,
                     long_scale=reso[2][0].long_scale)
    OOS = pd.DataFrame([dict(shock=lab, disclosed_b=t, predicted_b=predict(po, mo, d25, l, sh))
                        for lab, (l, sh, t) in zip(d25["labels"], d25["targets"])])
    OOS["diff_b"] = OOS.predicted_b - OOS.disclosed_b
    OOS.round(3).to_csv(os.path.join(a.outdir, "nii_out_of_sample_2025.csv"), index=False)
    rm_oos = float(np.sqrt((OOS.diff_b ** 2).mean()))

    rows = []
    for lab, sh in [("Parallel +200bp", 200), ("Parallel +100bp", 100), ("Parallel -100bp", -100),
                    ("Parallel -200bp", -200), ("Parallel -300bp (extrapolated)", -300)]:
        st = dnii(p, bs4, bs3, frame, sh)
        rows.append(dict(shock=lab, static_deposits_b=st, dynamic_deposits_b=st + mig(m, bs4, sh)))
    R = pd.DataFrame(rows)
    R["dynamic_pct_of_run_rate"] = R.dynamic_deposits_b / (q4["nii_total"] * 4) * 100
    R.round(3).to_csv(os.path.join(a.outdir, "nii_eve_results.csv"), index=False)

    # scenario effect: quarterly NII change vs Baseline from the gap in the 3-month rate (dynamic deposits)
    sc = pd.read_csv(a.scen)
    base = sc[sc.scenario.str.contains("Baseline")].reset_index(drop=True)
    out = []
    for name, g in sc.groupby("scenario"):
        g = g.reset_index(drop=True)
        gap_bp = (g["macro_tbill_3m"] - base["macro_tbill_3m"]).values * 100
        eff = np.array([(dnii(p, bs4, bs3, frame, x) + mig(m, bs4, x)) / 4 for x in gap_bp])
        for q, gp, e in zip(g.quarter, gap_bp, eff):
            out.append(dict(scenario=name, quarter=q, short_rate_gap_vs_baseline_bp=gp, nii_effect_vs_baseline_b=e))
    pd.DataFrame(out).round(3).to_csv(os.path.join(a.outdir, "nii_scenario_effect.csv"), index=False)

    nib = bs4["dep_nib_dom"] + bs4["dep_nib_for"]
    A = [("disclosed NII sensitivities", "static 2023 (6) + static 2024 (8) + dynamic 2024 (8) + dynamic 2025 (8)", "DATA", "BAC 10-K Tables 44 and 45"),
         ("deposit beta on interest-bearing deposits", p.beta, "CALIBRATED", "least squares, static and dynamic points"),
         ("share of interest-bearing deposits in low-rate tier", p.floor_tier_share, "CALIBRATED", "asymmetry between up and down shocks"),
         ("long-end repricing scale on fixed-rate assets (net of hedges)", p.long_scale, "CALIBRATED", "captures swaps/hedges not visible in Y-9C"),
         ("share of >1y borrowings effectively floating", p.ltd_float, "CALIBRATED", "10-K Note 11: contractual floating is only ~12% before swaps"),
         ("deposit migration, rates up: coefficient / exponent", f"{m[0]:.5f} / {m[3]:.2f}", "CALIBRATED", f"x NIB ${nib:.0f}B = ${m[0]*nib:.2f}B per +100bp"),
         ("deposit migration, rates down: coefficient / exponent", f"{m[1]:.4f} / {m[2]:.2f}", "CALIBRATED", "inflow benefit, concave in the size of the cut"),
         ("floating share of loans", p.loan_float, "ASSUMED mapping, DATA balances", "C&I, cards, other revolving, non-bank financial, other, HELOC, CRE floating; mortgages, auto fixed"),
         ("fixed-loan average life (years)", p.loan_fixed_life, "ASSUMED", ""),
         ("MBS prepayment: share of >5y bucket repricing per year", p.sec_prepay, "ASSUMED", ""),
         ("low-rate tier deposit cost", p.floor_tier_cost, "ASSUMED", ""),
         ("securities maturity buckets <=1y / 1-5y / >5y ($B)", f"{bs4['sec_le1y']:.0f} / {bs4['sec_1_5y']:.0f} / {bs4['sec_gt5y']:.0f}", "DATA", "Y-9C 2025Q4"),
         ("management action", "none", "per BAC disclosure", "instantaneous shock, no action")]
    pd.DataFrame(A, columns=["parameter", "value", "source", "note"]).to_csv(os.path.join(a.outdir, "nii_eve_assumptions.csv"), index=False)

    sens = []
    for lab, fx in [("base fit", {}), ("floating loan share -10pp", {"loan_float": p.loan_float - 0.10}),
                    ("floating loan share +10pp", {"loan_float": p.loan_float + 0.10}),
                    ("fixed-loan life 3y", {"loan_fixed_life": 3.0}), ("fixed-loan life 8y", {"loan_fixed_life": 8.0}),
                    ("low-tier cost 0%", {"floor_tier_cost": 0.0}), ("MBS prepayment 5%", {"sec_prepay": 0.05}),
                    ("long debt 12% floating (contractual)", {"ltd_float": 0.12})]:
        r_, m_ = fit(DS, fixed=fx)
        rmse_ = float(np.sqrt(np.mean([(f - t[2]) ** 2 for d, (pp, fv) in zip(DS, r_) for f, t in zip(fv, d["targets"])])))
        q_ = r_[3][0]
        sens.append((lab, q_.beta, q_.floor_tier_share, q_.long_scale, q_.ltd_float, rmse_,
                     dnii(q_, bs4, bs3, frame, -100) + mig(m_, bs4, -100), dnii(q_, bs4, bs3, frame, 100) + mig(m_, bs4, 100)))
    S = pd.DataFrame(sens, columns=["assumption_changed", "beta", "floor_share", "long_scale", "ltd_float", "fit_rmse_b",
                                    "dynamic_minus100_b", "dynamic_plus100_b"])
    S.round(3).to_csv(os.path.join(a.outdir, "nii_eve_sensitivity.csv"), index=False)

    L = ["NII: simplified banking-book model fitted to BAC Tables 44/45 (static and dynamic deposits), FR Y-9C balances",
         f"Check - 4Q25 NII from Y-9C {check['nii_actual_4q25']:.2f}B; rebuilt from categories {check['nii_from_categories']:.2f}B "
         f"(income {check['interest_income_total']:.1f}B, expense {check['interest_expense_total']:.1f}B)", "",
         "BALANCES AND 4Q25 RATES (average of Q3 and Q4 balances):", frame.round(2).to_string(index=False), "",
         "FIT TO DISCLOSED SENSITIVITIES, $B 12-month NII:", FIT.round(2).to_string(index=False),
         f"RMSE static {rm_static:.2f}B, dynamic {rm_dyn:.2f}B. Calibrated: deposit beta {p.beta:.2f}, low-rate tier {p.floor_tier_share:.0%}, "
         f"long-end scale {p.long_scale:.2f}, floating long debt {p.ltd_float:.0%}; deposit migration {m[0]*nib:.2f}B per +100bp.", "",
         f"OUT-OF-SAMPLE: parameters fitted WITHOUT 2025 predict the disclosed 2025 dynamic sensitivities with RMSE {rm_oos:.2f}B:",
         OOS.round(2).to_string(index=False), "",
         "RESULTS (2025 balance sheet, 12 months, instantaneous parallel shift):", R.round(2).to_string(index=False), "",
         "ROBUSTNESS: refit (all four datasets) after changing an assumed input:", S.round(2).to_string(index=False), "",
         "Parameters and sources: nii_eve_assumptions.csv. Everything marked ASSUMED is a modelling choice, not BAC data."]
    txt = "\n".join(L)
    open(os.path.join(a.outdir, "nii_eve_report.txt"), "w").write(txt + "\n")
    print(txt)


if __name__ == "__main__":
    main()
