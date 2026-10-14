"""
build_cet1_waterfall.py
------------------------
Allowance (CECL-style lifetime loss) and CET1 waterfall for Bank of America, 2026Q1-2029Q1, under the
Fed Baseline and Severely Adverse paths.

    py build_cet1_waterfall.py

Inputs
  data/calibration/bac_loss_paths.csv   BAC net charge-off rates (calibrate_bac_losses.py)
  data/models/ppnr_projections.csv      PPNR by scenario (estimate_ppnr_model.py)
  data/y9c/FRY9C_1073757_20251231.csv   starting CET1, RWA, loans, allowance, dividends, tax rate
Outputs (data/waterfall): cet1_waterfall.csv, cet1_summary.csv, cet1_report.txt

Waterfall each quarter
  PPNR - provision - [securities credit losses, trading/counterparty losses, other] = pre-tax income
  pre-tax income - taxes = net income;  CET1_t = CET1_{t-1} + net income - common and preferred dividends
  CET1 ratio = CET1 / RWA          (RWA held flat at the 2025Q4 level)

CECL-style allowance: allowance_t = loans x (sum of expected annualized net charge-off rates over the next
H quarters) / 4, using the scenario's own path (reasonable-and-supportable period) and reverting over 4
quarters to the baseline's final-quarter rate afterwards. H is the one calibrated number: it is solved so
that the BASELINE path reproduces BAC's actual 2025Q4 allowance. Provision = net charge-offs + change in allowance.

Inputs taken from the Fed's published BAC result and NOT modelled (they come from the Fed's global market
shock, a specified shock, not a forecast): securities credit losses 0.7, trading and counterparty losses 6.6,
other losses 1.9 ($B, 9 quarters; spread evenly over the first 9 quarters; 100% in severe, 50% in the constructed Adverse,
zero in baseline). Source below.
Not modelled: other comprehensive income (Fed: +4.7B), balance-sheet growth, deferred tax limits, buybacks (zero).
"""
import argparse
import os

import numpy as np
import pandas as pd

FED_SEVERE_ITEMS_9Q = {"securities_credit_losses": 0.7, "trading_counterparty_losses": 6.6, "other_losses": 1.9}
ADV_ITEM_SCALE = 0.5      # constructed Adverse takes half of the Fed-set severe items (same weight as the scenario construction)
FED_SRC = "Fed 2026 DFAST results, Bank of America one-page summary, severely adverse, 2026Q1-2028Q1"
FED_BENCH = {"cet1_start_pct": 11.4, "cet1_min_pct": 9.9, "cet1_end_2028Q1_pct": 10.8, "ppnr_9q": 66.2,
             "provisions_9q": 69.5, "pretax_9q": -12.5}
REQUIREMENT_PCT = 10.0       # 4.5 minimum + 3.0 GSIB surcharge + 2.5 stress capital buffer (YE2025)


def y9c_start(path):
    d = pd.read_csv(path, dtype=str).drop_duplicates("ItemName").set_index("ItemName")
    raw = pd.to_numeric(d["Value"], errors="coerce")
    v = raw / 1e6                                                        # $ thousands -> $ billions
    ratio = float(raw["BHCAP793"])                                       # already a percent
    cet1 = ratio / 100 * float(v["BHCAA223"])                            # reported ratio x RWA
    return dict(rwa=float(v["BHCAA223"]), cet1=cet1, cet1_ratio=ratio,
                loans=float(v["BHCKB528"]), allowance=float(v["BHCK3123"]),
                div_common_q=float(v["BHCK4460"]) / 4, div_pref_q=float(v["BHCK4598"]) / 4,
                tax_rate=float(v["BHCK4302"]) / float(v["BHCK4301"]))


def lifetime_loss(rates, start, H):
    """Sum of annualized rates/4 over quarters start+1 .. start+H (fractional H allowed), as a fraction of loans."""
    tot, full = 0.0, int(np.floor(H))
    for h in range(1, full + 1):
        tot += rates[start + h] / 4
    frac = H - full
    if frac > 0:
        tot += frac * rates[start + full + 1] / 4
    return tot


def extend(rates13, target, n_extra=40, revert=4):
    """Scenario path, then linear reversion to target over `revert` quarters, then flat."""
    r = list(rates13)
    last = r[-1]
    for j in range(1, n_extra + 1):
        w = min(j / revert, 1.0)
        r.append(last + (target - last) * w)
    return np.array([np.nan] + r)            # index 0 = current quarter (unused); 1.. = projections


def solve_H(base_ext, loans, allowance):
    target = allowance / loans
    lo, hi = 0.5, 30.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if lifetime_loss(base_ext, 0, mid) < target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def sv_nco(g):
    g = g[(g.scenario == "Severely Adverse") & (g.h <= 9)]
    return g.nco.sum()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paths", default="data/calibration/bac_loss_paths.csv")
    ap.add_argument("--ppnr", default="data/models/ppnr_projections.csv")
    ap.add_argument("--y9c", default="data/y9c/FRY9C_1073757_20251231.csv")
    ap.add_argument("--fed9q", default="data/raw/2026_Detailed_Nine_Quarter_Paths.csv")
    ap.add_argument("--outdir", default="data/waterfall")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)

    s0 = y9c_start(a.y9c)
    LP = pd.read_csv(a.paths)
    PP = pd.read_csv(a.ppnr)
    names = {"Baseline": ("Supervisory Baseline", "Supervisory Baseline"),
             "Adverse (constructed)": ("Constructed Adverse", "Constructed Adverse (with overlays)"),
             "Severely Adverse": ("Supervisory Severely Adverse", "Supervisory Severely Adverse (with overlays)")}
    rate = {k: LP[LP.scenario == v[0]].sort_values("quarter").bac_nco_rate_pct.values / 100 for k, v in names.items()}
    ext = {k: extend(r, rate["Baseline"][-1]) for k, r in rate.items()}
    H = solve_H(ext["Baseline"], s0["loans"], s0["allowance"])

    # Timing sensitivity: re-time the severe loss path to the SHAPE of the Fed's aggregate (all-bank) loan-loss path, keeping BAC's own
    # 9Q total. Quarters 10-13 (needed for the four-quarter-ahead allowance) continue at the Fed's last quarterly decline, then revert.
    ext_fed = None
    if os.path.exists(a.fed9q):
        import csv
        raw9 = list(csv.reader(open(a.fed9q, encoding="utf-8-sig")))
        fed_tot = next([float(x) for x in r[1:10]] for r in raw9 if r and r[0] == "Loan losses, total")
        r13 = rate["Severely Adverse"].copy()
        tot9 = r13[:9].sum()
        r13[:9] = tot9 * np.array(fed_tot) / sum(fed_tot)
        decay = fed_tot[8] / fed_tot[7]
        for j in range(9, len(r13)):
            r13[j] = r13[j - 1] * decay
        rate_fed = dict(rate); rate_fed["Severely Adverse"] = r13
        ext_fed = {k: extend(r, rate["Baseline"][-1]) for k, r in rate_fed.items()}

    def simulate(Hq, anchored=False, cov_cap=None, oci_9q=0.0, loss_scale=1.0, div_9q=None, fed_rule=True, fed_div=True, timing_fed=False):
        """Waterfall for all scenarios. anchored=False: allowance = loans x lifetime loss (base method).
        anchored=True: allowance = opening allowance + loans x (lifetime loss_t - lifetime loss today on the Baseline path),
        so the opening allowance is held at its actual value whatever the horizon (used for the horizon sensitivity).
        BASE CASE FOLLOWS THE FED'S PUBLISHED RULES:
          fed_rule: allowance at the end of each quarter = projected losses of the next four quarters (the Fed projects 13 quarters of
                    losses); the difference between that supervisory allowance at the start and the firm's actual allowance is
                    smoothed linearly into provisions over the nine quarters (Fed methodology, 'Overview of Modeling Framework').
          fed_div:  capital-action assumptions of 12 CFR 252.44(c): preferred dividends paid every quarter, no repurchases or issuance,
                    no common dividends after the first quarter (the first-quarter common dividend is the planned one). The Fed's
                    published CET1 path implies about $5.5B of dividends and other effects over 9Q; preferred (9 x 1.4B/4) plus one
                    quarter of common (2.0B) is $5.3B.
        Options: cov_cap = ceiling on allowance/loans; oci_9q = total OCI spread evenly over 9Q; loss_scale = multiplier on loan
        losses; div_9q = total 9Q dividends. fed_rule=False, fed_div=False reproduce the earlier calibrated-horizon / continuing-dividend case."""
        lt0 = lifetime_loss(ext["Baseline"], 0, Hq)
        rows = []
        for key, (lname, pname) in names.items():
            use_fed = timing_fed and ext_fed is not None and key == "Severely Adverse"
            rate_u = rate_fed if use_fed else rate
            ext_u = ext_fed if use_fed else ext
            q = LP[LP.scenario == lname].sort_values("quarter").quarter.values
            ppnr = PP[PP.scenario == pname].sort_values("quarter").set_index("quarter").ppnr
            allow_prev, cet1 = s0["allowance"], s0["cet1"]
            for i, qq in enumerate(q, start=1):
                nco = rate_u[key][i - 1] / 4 * s0["loans"] * loss_scale
                lt = lifetime_loss(ext_u[key], i, Hq)
                allow = s0["allowance"] + s0["loans"] * (lt - lt0) if anchored else s0["loans"] * lt
                if fed_rule:      # Fed method: allowance = next 4 quarters of projected losses; start-of-horizon difference smoothed linearly over 9Q
                    a0_sup = s0["loans"] * lifetime_loss(ext_u[key], 0, 4.0)
                    allow = s0["loans"] * lifetime_loss(ext_u[key], i, 4.0) - (a0_sup - s0["allowance"]) * (1 - min(i, 9) / 9)
                if cov_cap is not None:
                    allow = min(allow, cov_cap * s0["loans"])
                prov = nco + (allow - allow_prev)
                sc_w = {"Severely Adverse": 1.0, "Adverse (constructed)": ADV_ITEM_SCALE}.get(key, 0.0) if i <= 9 else 0.0
                sec = sc_w * FED_SEVERE_ITEMS_9Q["securities_credit_losses"] / 9
                trd = sc_w * FED_SEVERE_ITEMS_9Q["trading_counterparty_losses"] / 9
                oth = sc_w * FED_SEVERE_ITEMS_9Q["other_losses"] / 9
                pre = float(ppnr.loc[qq]) - prov - sec - trd - oth
                tax = s0["tax_rate"] * pre
                ni = pre - tax
                if div_9q is not None:
                    div_q = div_9q / 9
                elif fed_div:
                    div_q = s0["div_pref_q"] + (s0["div_common_q"] if i == 1 else 0.0)
                else:
                    div_q = s0["div_common_q"] + s0["div_pref_q"]
                cet1 = cet1 + ni - div_q + (oci_9q / 9 if i <= 9 else 0.0)
                rows.append(dict(scenario=key, quarter=qq, h=i, ppnr=ppnr.loc[qq], nco=nco, allowance=allow,
                                 provision=prov, other_stress_items=sec + trd + oth, pretax=pre, tax=tax,
                                 net_income=ni, dividends=div_q, cet1=cet1,
                                 rwa=s0["rwa"], cet1_ratio_pct=cet1 / s0["rwa"] * 100))
                allow_prev = allow
        return pd.DataFrame(rows)

    W = simulate(H)
    W.round(3).to_csv(os.path.join(a.outdir, "cet1_waterfall.csv"), index=False)

    summ = []
    for key, g in W.groupby("scenario"):
        g9 = g[g.h <= 9]
        summ.append(dict(scenario=key, cet1_start_pct=s0["cet1_ratio"], cet1_min_pct_9q=g9.cet1_ratio_pct.min(),
                         min_quarter=g9.loc[g9.cet1_ratio_pct.idxmin(), "quarter"],
                         cet1_end_9q_pct=g9.cet1_ratio_pct.iloc[-1], ppnr_9q=g9.ppnr.sum(), nco_9q=g9.nco.sum(),
                         provisions_9q=g9.provision.sum(), allowance_end_9q=g9.allowance.iloc[-1],
                         pretax_9q=g9.pretax.sum(), net_income_9q=g9.net_income.sum()))
    Sdf = pd.DataFrame(summ)
    Sdf.round(2).to_csv(os.path.join(a.outdir, "cet1_summary.csv"), index=False)

    # ---- allowance / capital-action method sensitivity (severely adverse)
    FED_LOSS_9Q, FED_PROV_9Q = 64.9, FED_BENCH["provisions_9q"]       # Fed 2026 DFAST, BAC severely adverse
    fed_end_allow = s0["allowance"] + (FED_PROV_9Q - FED_LOSS_9Q)      # implied: provisions - loan losses = allowance build
    srows = []
    for lab, kw in [("Fed rules: 4-quarter allowance smoothed over 9Q, no common dividends after Q1 (BASE)", {}),
                    ("Fed allowance rule, common dividends continue", dict(fed_div=False)),
                    (f"Calibrated {H:.1f}-quarter allowance, Fed dividends", dict(fed_rule=False)),
                    (f"Calibrated {H:.1f}-quarter allowance, common dividends continue (earlier base)", dict(fed_rule=False, fed_div=False))] \
                   + ([("Fed rules + loss timing taken from the Fed's aggregate path (BAC 9Q total unchanged)", dict(timing_fed=True)),
                       ("  ... and Fed OCI +4.7B", dict(timing_fed=True, oci_9q=4.7))] if ext_fed is not None else []):
        g = simulate(H, **kw)
        g = g[(g.scenario == "Severely Adverse") & (g.h <= 9)]
        srows.append(dict(method=lab, nco_9q=g.nco.sum(), provisions_9q=g.provision.sum(), allowance_end_9q=g.allowance.iloc[-1],
                          dividends_9q=g.dividends.sum(), pretax_9q=g.pretax.sum(), cet1_min_pct=g.cet1_ratio_pct.min(), cet1_end_pct=g.cet1_ratio_pct.iloc[-1]))
    srows.append(dict(method="Fed published / implied", nco_9q=FED_LOSS_9Q, provisions_9q=FED_PROV_9Q, allowance_end_9q=fed_end_allow,
                      dividends_9q=5.5, pretax_9q=FED_BENCH["pretax_9q"], cet1_min_pct=FED_BENCH["cet1_min_pct"], cet1_end_pct=FED_BENCH["cet1_end_2028Q1_pct"]))
    SENS = pd.DataFrame(srows)
    SENS.round(2).to_csv(os.path.join(a.outdir, "cet1_allowance_sensitivity.csv"), index=False)

    # ---- remaining differences to the Fed after applying its allowance and capital-action rules: cumulative steps
    FED_OCI = 4.7
    ls = FED_LOSS_9Q / sv_nco(simulate(H))
    steps = [("Model with the Fed's allowance and capital-action rules", {}),
             (f"+ Fed other comprehensive income (+{FED_OCI}B, not modelled)", dict(oci_9q=FED_OCI)),
             (f"+ loan losses set to the Fed's ({FED_LOSS_9Q}B)", dict(oci_9q=FED_OCI, loss_scale=ls))]
    brow = []
    for lab, kw in steps:
        g = simulate(H, **kw)
        g = g[(g.scenario == "Severely Adverse") & (g.h <= 9)]
        brow.append(dict(step=lab, provisions_9q=g.provision.sum(), pretax_9q=g.pretax.sum(),
                         cet1_min_pct=g.cet1_ratio_pct.min(), cet1_end_pct=g.cet1_ratio_pct.iloc[-1]))
    brow.append(dict(step="Fed published", provisions_9q=FED_PROV_9Q, pretax_9q=FED_BENCH["pretax_9q"],
                     cet1_min_pct=FED_BENCH["cet1_min_pct"], cet1_end_pct=FED_BENCH["cet1_end_2028Q1_pct"]))
    BR = pd.DataFrame(brow)
    BR.round(2).to_csv(os.path.join(a.outdir, "cet1_bridge_to_fed.csv"), index=False)

    sv = Sdf[Sdf.scenario == "Severely Adverse"].iloc[0]
    L = [f"Start (Y-9C 2025Q4): CET1 {s0['cet1']:.1f}B / RWA {s0['rwa']:,.1f}B = {s0['cet1_ratio']:.2f}%; allowance {s0['allowance']:.1f}B on loans {s0['loans']:,.0f}B; "
         f"tax rate {s0['tax_rate']*100:.1f}% (BAC 2025 effective); common dividends {s0['div_common_q']*4:.1f}B/yr",
         f"Calibrated CECL horizon H (used only in the sensitivity rows) solved so baseline reproduces actual allowance: {H:.1f} quarters (about {H/4:.1f} years).", "",
         Sdf.round(2).to_string(index=False), "",
         f"Severely adverse vs the {REQUIREMENT_PCT:.1f}% requirement (4.5 + 3.0 GSIB + 2.5 SCB): minimum {sv.cet1_min_pct_9q:.2f}% in {sv.min_quarter} "
         f"-> {'ABOVE' if sv.cet1_min_pct_9q >= REQUIREMENT_PCT else 'BELOW'} by {abs(sv.cet1_min_pct_9q - REQUIREMENT_PCT):.2f}pp",
         "", "Fed published for BAC (severely adverse, benchmark only): " + ", ".join(f"{k} {v}" for k, v in FED_BENCH.items()),
         f"Fed-set stress items included (severe only, not modelled): {FED_SEVERE_ITEMS_9Q}; source: {FED_SRC}.",
         "", "Provisions gap decomposition (severe, 9Q): loan losses model %.1fB vs Fed %.1fB; allowance build model %.1fB vs Fed %.1fB (Fed provisions - loan losses)."
         % (sv.nco_9q, FED_LOSS_9Q, sv.allowance_end_9q - s0["allowance"], FED_PROV_9Q - FED_LOSS_9Q),
         "Allowance / capital-action method (severe). Base follows the Fed: allowance = next four quarters of losses, start difference smoothed over 9Q, no common dividends after Q1. The Fed's end allowance is implied from its published provisions less loan losses:",
         SENS.round(2).to_string(index=False), "",
         "Remaining steps from the model's severe CET1 to the Fed's (cumulative):",
         BR.round(2).to_string(index=False), "",
         "Not modelled in the base case: OCI (Fed +4.7B, about +0.26pp of CET1), balance-sheet growth, buybacks (zero), tax-asset limits."]
    txt = "\n".join(L)
    open(os.path.join(a.outdir, "cet1_report.txt"), "w").write(txt + "\n")
    print(txt)


if __name__ == "__main__":
    main()
