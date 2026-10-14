"""
build_basel_eve.py
-------------------
Basel standardised EVE (BCBS d368, April 2016, Section IV and Annex 2) for Bank of America, from FR Y-9C 2025Q4.

    py build_basel_eve.py

Inputs : data/y9c/FRY9C_1073757_20251231.csv, ..._20250930.csv, data/panel/calibration_panel_full.csv (2025Q4 yield curve),
         build_nii_eve_model.py (same folder; supplies the balance-sheet loader and the NII-fit parameters)
Outputs (data/rate_risk): basel_eve_results.csv, basel_eve_by_position.csv, basel_eve_sensitivity.csv,
         basel_eve_assumptions.csv, basel_eve_report.txt

FROM THE BASEL TEXT (not assumptions): six scenarios; USD shocks parallel 200bp, short 300bp, long 150bp; short shock
  S(t)=exp(-t/4), long = 1-S; steepener = -0.65|short|+0.9|long|, flattener = +0.8|short|-0.6|long|; 19 time buckets and
  midpoints; NMD core-proportion caps 90/70/50% and average-maturity caps 5/4.5/4 years (retail transactional /
  retail non-transactional / wholesale); non-core NMDs overnight; prepayment multipliers 0.8 (rates up) / 1.2 (rates down);
  banking book only (trading excluded); equity excluded; EVE = PV(assets) - PV(liabilities) with continuous discounting;
  result = worst of six scenarios; outlier test 15% of Tier 1.
FROM Y-9C: balances, deposit split NIB/interest-bearing/foreign, securities maturity buckets and fair value (HTM and AFS),
  Tier 1 capital, interest rates earned/paid in 4Q25.
ASSUMED (listed in basel_eve_assumptions.csv): retail share of deposits, core share and maturity within the Basel caps,
  prepayment speeds, coupons and amortisation of fixed loans, floating-rate reset period, floating share of long debt
  (10-K Note 11 contractual share, before any swaps), the 2025Q4 base curve (Fed series), rate floor at zero after the shock, no derivatives.
  Hedging swaps are NOT visible in Y-9C, so the headline is for cash instruments only; hedge offsets are in the sensitivity.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy.optimize import brentq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import build_nii_eve_model as nii

UPPER = np.array([1 / 365, 1 / 12, 3 / 12, 6 / 12, 9 / 12, 1, 1.5, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, np.inf])
MID = np.array([0.0028, 0.0417, 0.1667, 0.375, 0.625, 0.875, 1.25, 1.75, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 12.5, 17.5, 25.0])
SHORT, LONG, PAR = 300.0, 150.0, 200.0                      # USD, Annex 2 Table 1 (bp)
# BAC FY2025 10-K Note 11 (long-term debt, $B): contractual floating-rate notes and the maturity ladder, excluding BofA Finance
# structured liabilities (which sit in trading liabilities in the Y-9C). Floating = senior 11.5 + sub 4.6 + BANA senior 11.2 + FHLB 4.1.
LTD_FLOAT = 31.4 / 263.3                      # contractual floating share, before swaps
# FY2025 10-K "Designated Fair Value Hedged Assets and Liabilities", December 31, 2025: hedged long-term debt carrying value
#   $175.7B and hedged AFS debt securities $236.3B (carrying value = amortised cost). Open hedges convert the hedged items to variable rate.
#   Effective floating share of LTD = (contractual floating 31.4 + hedged fixed 175.7) / LTD excluding structured liabilities (317.8 - 53.8).
LTD_FLOAT_HEDGED = (31.4 + 175.694) / (317.816 - 53.803)
AFS_HEDGED_AMT = 236.303
LTD_LADDER = [(0, 1, 29.8), (1, 2, 32.5), (2, 3, 34.4), (3, 4, 28.4), (4, 5, 10.0), (5, 20, 128.4)]   # 10-K ladder: maturing 2026..2030, thereafter; excl. structured liabilities
LOAN_FV_DISCOUNT = 1149.093 - 1136.439       # 10-K fair value footnote: loans carrying less fair value, $B (assumed all in fixed loans)
SCEN = ["Parallel up", "Parallel down", "Steepener", "Flattener", "Short rate up", "Short rate down"]
GAMMA = [0.8, 1.2, 0.8, 1.2, 0.8, 1.2]                       # Table 3 prepayment multipliers


def shocks():
    S = np.exp(-MID / 4.0)
    sh, lg = SHORT * S, LONG * (1 - S)
    return [np.full(19, PAR), np.full(19, -PAR), -0.65 * sh + 0.9 * lg, 0.8 * sh - 0.6 * lg, sh, -sh]   # bp, by bucket


def base_curve(panel):
    r = pd.read_csv(panel).set_index("quarter").loc["2025Q4"]
    t = np.array([0.0, 0.25, 5.0, 10.0, 30.0])
    y = np.array([r["macro_tbill_3m"], r["macro_tbill_3m"], r["macro_ust_5y"], r["macro_ust_10y"], r["macro_ust_10y"]]) / 100
    return np.interp(MID, t, y), r


def bucket(t_years):
    return np.minimum(np.searchsorted(UPPER, t_years, side="left"), 18)


def cf_vector(prin, intr, months):
    out = np.zeros(19)
    np.add.at(out, bucket(months / 12.0), prin + intr)
    return out


def level_pay(B, c, n_m, cpr):
    """Monthly principal and interest of a level-pay pool with annual CPR."""
    smm = 1 - (1 - cpr) ** (1 / 12)
    bal, P, I = B, np.zeros(n_m), np.zeros(n_m)
    i = c / 12
    for m in range(n_m):
        rem = n_m - m
        pay = bal * i / (1 - (1 + i) ** -rem) if i > 0 else bal / rem
        interest = bal * i
        sched = min(pay - interest, bal)
        prep = (bal - sched) * smm
        P[m], I[m] = sched + prep, interest
        bal -= sched + prep
    return P, I, np.arange(1, n_m + 1)


def uniform(B, c, t_lo, t_hi):
    lo, hi = max(int(round(t_lo * 12)), 0), max(int(round(t_hi * 12)), 1)
    n = max(hi - lo, 1)
    months = np.arange(lo + 1, lo + n + 1)
    P = np.full(n, B / n)
    bal = B - np.concatenate([[0], np.cumsum(P)[:-1]])
    I = bal * c / 12
    return P, I, months


def floating(B, c, reset_years):
    m = max(int(round(reset_years * 12)), 1)
    return np.array([B]), np.array([B * c * reset_years]), np.array([m])


class Position:
    def __init__(self, name, side, B, target, fn, group=None, kind=""):
        self.name, self.side, self.B, self.target, self.fn, self.group, self.kind = name, side, B, target, fn, group, kind
        self.s = 0.0


def pv(cf, r0, delta_bp, s):
    r = np.maximum(r0 + delta_bp / 1e4, 0.0) + s
    return float((cf * np.exp(-r * MID)).sum())


def solve_spreads(pos, r0):
    zero = np.zeros(19)
    base = {p.name: p.fn(1.0) for p in pos}
    groups = {}
    for p in pos:
        groups.setdefault(p.group or p.name, []).append(p)
    for g, ps in groups.items():
        tgt = sum(p.target for p in ps)
        f = lambda s: sum(pv(base[p.name], r0, zero, s) for p in ps) - tgt
        s = brentq(f, -0.6, 0.6)
        for p in ps:
            p.s = s


def run(P, bs4, q, frame, r0, tier1, detail=False):
    L = bs4["loans"]
    lf = nii.loan_split(bs4)
    float_loans = L * lf
    mort = bs4["mort_first"] + bs4["mort_junior"]
    fixed_other = max(L * (1 - lf) - mort, 0.0)
    rate = lambda cat: float(frame.loc[frame.category == cat, "annualized_rate_pct"].iloc[0]) / 100
    ib_rate, sec_rate, loan_rate = rate("Interest-bearing deposits"), rate("Debt securities (AFS + HTM)"), rate("Loans and leases")
    rr = P["retail_share"]
    nibD, ibD = bs4["dep_nib_dom"], bs4["dep_ib_dom"]
    foreign = bs4["dep_nib_for"] + bs4["dep_ib_for"]
    cats = {   # (balance, coupon, category)
        "Retail transactional": (nibD * rr, 0.0, 0), "Retail non-transactional": (ibD * rr, ib_rate, 1),
        "Wholesale (NIB)": (nibD * (1 - rr), 0.0, 2), "Wholesale (interest-bearing + foreign)": (ibD * (1 - rr) + foreign, ib_rate, 2)}
    caps_core, caps_mat = [0.90, 0.70, 0.50], [5.0, 4.5, 4.0]
    sc = P["nmd_cap_scale"]
    pos = []
    # ---------------- assets
    cash = bs4["bal_dom"] + bs4["bal_for"]
    pos.append(Position("Cash and interest-bearing balances", "A", cash, cash, lambda g: cf_vector(*floating(cash, 0.04, 1 / 365)), kind="floating"))
    pos.append(Position("Reverse repo", "A", bs4["reverse_repo"], bs4["reverse_repo"], lambda g: cf_vector(*floating(bs4["reverse_repo"], 0.045, 1 / 12)), kind="floating"))
    pos.append(Position("Loans: floating", "A", float_loans, float_loans, lambda g: cf_vector(*floating(float_loans, loan_rate, P["float_reset"])), kind="floating"))
    fx_tot = max(mort + fixed_other, 1e-9)
    mort_t = mort - LOAN_FV_DISCOUNT * mort / fx_tot
    oth_t = fixed_other - LOAN_FV_DISCOUNT * fixed_other / fx_tot
    pos.append(Position("Loans: residential mortgages (fixed)", "A", mort, mort_t,
                        lambda g: cf_vector(*level_pay(mort, P["mort_coupon"], P["mort_term"] * 12, min(1, g * P["cpr_mort"]))), kind="fixed"))
    pos.append(Position("Loans: other fixed", "A", fixed_other, oth_t,
                        lambda g: cf_vector(*level_pay(fixed_other, P["other_coupon"], 60, min(1, g * P["cpr_other"]))), kind="fixed"))
    s1, s2, s3 = bs4["sec_le1y"], bs4["sec_1_5y"], bs4["sec_gt5y"]
    fair = bs4["afs_fv"] + bs4["htm_fv"]
    scale = fair / (s1 + s2 + s3)
    pos.append(Position("Securities <=1y", "A", s1, s1 * scale, lambda g: cf_vector(*uniform(s1, sec_rate, 0, 1)), "sec", "fixed"))
    pos.append(Position("Securities 1-5y", "A", s2, s2 * scale, lambda g: cf_vector(*uniform(s2, sec_rate, 1, 5)), "sec", "fixed"))
    pos.append(Position("Securities >5y (MBS-like)", "A", s3, s3 * scale,
                        lambda g: cf_vector(*level_pay(s3, sec_rate, 25 * 12, min(1, g * P["cpr_mbs"]))), "sec", "fixed"))
    # ---------------- liabilities
    for k, (B, c, cat) in cats.items():
        core = B * min(caps_core[cat] * sc, 1.0)
        avg = caps_mat[cat] * sc
        pos.append(Position(f"NMD core: {k}", "L", core, core, (lambda core=core, c=c, avg=avg: lambda g: cf_vector(*uniform(core, c, 0, 2 * avg)))(), kind="nmd"))
        non = B - core
        pos.append(Position(f"NMD non-core (overnight): {k}", "L", non, non, (lambda non=non, c=c: lambda g: cf_vector(*floating(non, c, 1 / 365)))(), kind="floating"))
    pos.append(Position("Repo", "L", bs4["repo"], bs4["repo"], lambda g: cf_vector(*floating(bs4["repo"], 0.045, 1 / 12)), kind="floating"))
    ob1 = bs4["ob_le1y"]
    pos.append(Position("Other borrowed money <=1y", "L", ob1, ob1, lambda g: cf_vector(*uniform(ob1, 0.04, 0, 1)), kind="fixed"))
    ltd = bs4["ob_gt1y"]
    lf_ = P["ltd_float"]
    pos.append(Position("Long-term debt: floating / swapped", "L", ltd * lf_, ltd * lf_, lambda g: cf_vector(*floating(ltd * lf_, 0.045, 0.25)), kind="floating"))
    tot_l = sum(w for _, _, w in LTD_LADDER)
    for lo, hi, w in LTD_LADDER:
        bal_ = ltd * (1 - lf_) * w / tot_l
        pos.append(Position(f"Long-term debt: fixed, maturing {lo}-{hi}y", "L", bal_, bal_,
                            (lambda bal_=bal_, lo=lo, hi=hi: lambda g: cf_vector(*uniform(bal_, 0.040, lo, hi)))(), kind="fixed"))
    sd = bs4["sub_debt"]
    pos.append(Position("Subordinated debt", "L", sd, sd, lambda g: cf_vector(*uniform(sd, 0.048, 5, 15)), kind="fixed"))
    solve_spreads(pos, r0)
    sh = shocks()
    res, det = [], []
    for i, name in enumerate(SCEN):
        d_total, d_fixed_assets, d_sec = 0.0, 0.0, 0.0
        for p in pos:
            cf0, cfi = p.fn(1.0), p.fn(GAMMA[i])
            dv = pv(cfi, r0, sh[i], p.s) - pv(cf0, r0, np.zeros(19), p.s)
            sign = 1 if p.side == "A" else -1
            d_total += sign * dv
            if p.side == "A" and p.kind == "fixed":
                d_fixed_assets += dv
            if p.name.startswith("Securities"):
                d_sec += dv
            det.append(dict(scenario=name, position=p.name, side=p.side, balance_b=p.B, delta_value_b=sign * dv))
        d_total -= P["hedge_sec"] * d_sec                          # pay-fixed swaps on hedged AFS securities offset their value change
        d_total -= P["hedge_offset"] * d_fixed_assets            # pay-fixed swaps offset a share of the fixed-asset value change
        res.append(d_total)
    return res, pd.DataFrame(det), pos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--y9c-q4", default="data/y9c/FRY9C_1073757_20251231.csv")
    ap.add_argument("--y9c-q3", default="data/y9c/FRY9C_1073757_20250930.csv")
    ap.add_argument("--y9c-q4-prev", default="data/y9c/FRY9C_1073757_20241231.csv")
    ap.add_argument("--y9c-q3-prev", default="data/y9c/FRY9C_1073757_20240930.csv")
    ap.add_argument("--panel", default="data/panel/calibration_panel_full.csv")
    ap.add_argument("--outdir", default="data/rate_risk")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    bs4, bs3, q4 = nii.load(a.y9c_q4, a.y9c_q3)
    d4 = nii.read(a.y9c_q4)
    bs4["afs_fv"], bs4["htm_fv"] = float(d4["BHCK1773"]), float(d4["BHCK1771"])
    bs4["afs_am"], bs4["htm_am"] = float(d4["BHCK1772"]), float(d4["BHCK1754"])
    frame = nii.build_frame(bs4, bs3, q4)
    r0, crow = base_curve(a.panel)
    tier1 = bs4["tier1"]

    BASE = dict(retail_share=0.60, nmd_cap_scale=1.0, cpr_mort=0.06, cpr_other=0.10, cpr_mbs=0.07,
                mort_coupon=0.040, mort_term=25, other_coupon=0.065, float_reset=1 / 12, ltd_float=LTD_FLOAT_HEDGED,
                hedge_offset=0.0, hedge_sec=AFS_HEDGED_AMT / (bs4["afs_am"] + bs4["htm_am"]))
    GROSS = dict(BASE, ltd_float=LTD_FLOAT, hedge_sec=0.0)
    res_g, _, _ = run(GROSS, bs4, q4, frame, r0, tier1)
    res, det, pos = run(BASE, bs4, q4, frame, r0, tier1)
    R = pd.DataFrame(dict(scenario=SCEN, delta_eve_after_hedges_b=res, delta_eve_gross_b=res_g))
    R["delta_eve_b"] = R.delta_eve_after_hedges_b
    R["after_hedges_pct_tier1"] = R.delta_eve_b / tier1 * 100
    R["gross_pct_tier1"] = R.delta_eve_gross_b / tier1 * 100
    worst = R.loc[R.delta_eve_b.idxmin()]
    loss = max(0.0, -R.delta_eve_b.min())
    worst_g = R.loc[R.delta_eve_gross_b.idxmin()]
    loss_g = max(0.0, -R.delta_eve_gross_b.min())
    R = R.drop(columns="delta_eve_b")
    R.round(3).to_csv(os.path.join(a.outdir, "basel_eve_results.csv"), index=False)
    det.round(3).to_csv(os.path.join(a.outdir, "basel_eve_by_position.csv"), index=False)

    def worst_of(P):
        r, _, _ = run(P, bs4, q4, frame, r0, tier1)
        return min(r), SCEN[int(np.argmin(r))], r
    sens = []
    for lab, mod in [("base (after documented fair-value hedges)", {}), ("NMD core share and maturity at 75% of Basel caps", {"nmd_cap_scale": 0.75}),
                     ("NMD core share and maturity at 50% of caps", {"nmd_cap_scale": 0.50}),
                     ("retail share of domestic deposits 45%", {"retail_share": 0.45}),
                     ("retail share of domestic deposits 75%", {"retail_share": 0.75}),
                     ("mortgage CPR 3%", {"cpr_mort": 0.03}), ("mortgage CPR 10%", {"cpr_mort": 0.10}),
                     ("no documented hedges (gross cash instruments)", {"ltd_float": LTD_FLOAT, "hedge_sec": 0.0}),
                     ("hedged share of AFS 30% (vs 61%)", {"hedge_sec": 0.30 * bs4["afs_am"] / (bs4["afs_am"] + bs4["htm_am"])}),
                     ("hedged share of AFS 90%", {"hedge_sec": 0.90 * bs4["afs_am"] / (bs4["afs_am"] + bs4["htm_am"])}),
                     ("long-term debt effective floating 65% (vs 78%)", {"ltd_float": 0.65}),
                     ("long-term debt effective floating 95%", {"ltd_float": 0.95})]:
        P = dict(BASE)
        P.update(mod)
        w, ws, r = worst_of(P)
        sens.append(dict(case=lab, worst_delta_eve_b=w, worst_scenario=ws, worst_pct_tier1=w / tier1 * 100,
                         parallel_up_b=r[0], parallel_down_b=r[1]))
    S = pd.DataFrame(sens)
    S.round(2).to_csv(os.path.join(a.outdir, "basel_eve_sensitivity.csv"), index=False)

    sec_s = [p.s for p in pos if p.group == "sec"][0]
    A = [("USD shocks: parallel / short / long (bp)", f"{PAR:.0f} / {SHORT:.0f} / {LONG:.0f}", "BASEL", "BCBS d368 Annex 2 Table 1"),
         ("short-shock decay, steepener/flattener weights", "exp(-t/4); -0.65,+0.9; +0.8,-0.6", "BASEL", "Annex 2"),
         ("time buckets", "19 buckets, midpoints 0.0028y to 25y", "BASEL", "Table 1"),
         ("NMD core caps / maturity caps", "90/70/50%; 5/4.5/4y", "BASEL", "Table 2"),
         ("prepayment multipliers", "0.8 rates up, 1.2 rates down", "BASEL", "Table 3"),
         ("outlier test", "15% of Tier 1", "BASEL", "paragraph 94-ish, Pillar 2 threshold"),
         ("balances, deposit split, securities buckets, securities fair value, Tier 1", "see data", "DATA", "Y-9C 2025Q4"),
         ("securities spread to base curve (solved so PV = fair value)", f"{sec_s*1e4:.0f} bp", "DATA-DERIVED", "HTM + AFS fair value"),
         ("base curve", f"3m {crow['macro_tbill_3m']:.2f}%, 5y {crow['macro_ust_5y']:.2f}%, 10y {crow['macro_ust_10y']:.2f}%", "ASSUMED (Fed series, 2025Q4)", "linear between nodes"),
         ("retail share of domestic deposits", BASE["retail_share"], "ASSUMED", "not in Y-9C; BAC segment mix is the guide"),
         ("NMD core share and maturity vs caps", BASE["nmd_cap_scale"], "ASSUMED (=caps)", "sensitivity at 75% and 50%"),
         ("mortgage CPR / coupon / remaining term", f"{BASE['cpr_mort']:.0%} / {BASE['mort_coupon']:.1%} / {BASE['mort_term']}y", "ASSUMED", ""),
         ("other fixed loans: coupon / life / CPR", f"{BASE['other_coupon']:.1%} / 5y / {BASE['cpr_other']:.0%}", "ASSUMED", ""),
         ("MBS-like >5y securities: remaining term / CPR", f"25y / {BASE['cpr_mbs']:.0%}", "ASSUMED", ""),
         ("floating-rate reset", "1 month", "ASSUMED", ""),
         ("floating share of long-term debt: contractual / after hedges", f"{LTD_FLOAT:.0%} / {LTD_FLOAT_HEDGED:.0%}", "DATA (10-K Note 11 + FY2025 fair-value-hedge table)", "hedged LTD $175.7B plus contractual floating $31.4B, of $264.0B excluding structured liabilities"),
         ("hedged share of AFS securities", f"${AFS_HEDGED_AMT:.1f}B of AFS amortised cost (about {BASE['hedge_sec']:.0%} of all securities)", "DATA (FY2025 10-K)", "HTM treated as unhedged; hedged AFS assumed proportional across maturity buckets"),
         ("long-term debt maturity ladder", "2026-2030 and thereafter, from the note", "DATA", "coupon 4.0% assumed"),
         ("loans fair value vs carrying", f"-{LOAN_FV_DISCOUNT:.1f}B, all assigned to fixed loans", "DATA (10-K fair value footnote)", "deposits (+0.1%) and LTD (+2.1%) left at book"),
         ("post-shock rate floor", "0%", "ASSUMED", "Basel allows floors not above zero"),
         ("derivatives / hedges", "none in headline", "NOT VISIBLE", "see hedge sensitivity")]
    pd.DataFrame(A, columns=["parameter", "value", "source", "note"]).to_csv(os.path.join(a.outdir, "basel_eve_assumptions.csv"), index=False)

    byp = det.pivot_table(index=["side", "position"], columns="scenario", values="delta_value_b", sort=False)[SCEN]
    L = ["BASEL STANDARDISED EVE (BCBS d368), BAC, 2025Q4, banking book, USD. After documented fair-value hedges (FY2025 10-K) and gross", "",
         f"Base curve: 3m {crow['macro_tbill_3m']:.2f}%, 5y {crow['macro_ust_5y']:.2f}%, 10y {crow['macro_ust_10y']:.2f}%. Tier 1 capital ${tier1:.1f}B.", "",
         "CHANGE IN EVE BY SCENARIO ($B; negative = loss):", R.round(2).to_string(index=False), "",
         f"Standardised EVE risk measure AFTER documented hedges (worst: {worst.scenario}): ${loss:.1f}B = {loss/tier1*100:.1f}% of Tier 1. "
         f"Outlier threshold 15% of Tier 1 = ${0.15*tier1:.1f}B -> {'ABOVE' if loss > 0.15*tier1 else 'below'} the threshold.",
         f"GROSS (no swaps; worst: {worst_g.scenario}): ${loss_g:.1f}B = {loss_g/tier1*100:.1f}% of Tier 1.", "",
         "CONTRIBUTION BY POSITION ($B change in value, signed for EVE impact):", byp.round(1).to_string(), "",
         "SENSITIVITY OF THE WORST-SCENARIO RESULT TO THE ASSUMED INPUTS:", S.round(1).to_string(index=False), "",
         "Securities are valued at Y-9C fair value (carries the $80B HTM unrealized loss); fixed loans at 10-K fair value (about $12.7B below carrying); deposits and long-term debt at book.",
         "Hedges: long-term debt effective floating share and the hedged share of AFS come from the FY2025 10-K fair-value-hedge table. Cash flow hedges and ALM swaps outside fair value hedges are not included. All ASSUMED inputs are in basel_eve_assumptions.csv."]
    txt = "\n".join(L)
    open(os.path.join(a.outdir, "basel_eve_report.txt"), "w").write(txt + "\n")
    print(txt)


if __name__ == "__main__":
    main()
