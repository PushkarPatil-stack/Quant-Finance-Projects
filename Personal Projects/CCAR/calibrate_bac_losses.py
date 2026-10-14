"""
calibrate_bac_losses.py
------------------------
Turns the industry (FRED top-100) segment loss curves into a Bank of America loss path, using
BAC's OWN history, and then compares the result with the Fed's published BAC loss rates.

    py calibrate_bac_losses.py

Inputs (defaults assume your CCAR folder layout)
    data/panel/calibration_panel_full.csv     FRED segment charge-off rates, BAC XBRL NCO
    data/models/loss_projections.csv          from estimate_loss_models.py
    data/xbrl/bac_companyfacts.json           SEC download (for BAC loan balances)
    data/y9c/FRY9C_1073757_20251231.csv       latest Y-9C, used only for BAC's current loan mix

Method
  1. BAC loan mix (latest quarter): cards, C&I, CRE, residential mortgage + HELOC, other consumer,
     as a share of total loans. Loans outside those five (loans to non-bank financials, foreign
     governments, other) get no industry curve; the scale factor absorbs them.
  2. Industry-implied BAC charge-off rate_t = sum(mix_i x industry_NCO_rate_i,t).
  3. BAC actual NCO rate_t = net charge-offs x 4 / average loans (from XBRL).
  4. Scale factor k = least-squares ratio actual/implied, fitted on 2014Q1-2019Q4 (2011-13 legacy run-off excluded, reported separately) and checked on
     2020Q1-2025Q4 (out of sample). Also reported over the full window.
  5. BAC projected loss = k x sum(mix_i x projected industry NCO_i) under each Fed scenario, on
     balances held flat at the latest loan level (simplification; DFAST uses its own balance path).
  6. The Fed's published BAC loss rates (DFAST 2026 Table 10) are shown NEXT TO the result as a
     benchmark. Nothing is fitted to them.

Outputs (data/calibration): calibration_overlays.csv, bac_loss_paths.csv,
    benchmark_vs_fed.csv, calibration_report.txt
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

SEGMENTS = ["cards", "business", "cre", "mortgage", "other_consumer"]

# Y-9C items (all $ thousands). Residential mortgage + HELOC + multifamily is the residual of
# loans secured by real estate after construction and nonfarm-nonresidential CRE; the extract does
# not carry the residential/multifamily split, so multifamily sits in the mortgage bucket.
MIX_ITEMS = {
    "cards": ["BHCKB538"],
    "business": ["BHCK1763", "BHCK1764"],
    "cre": ["BHCKF158", "BHCKF159", "BHCKF160", "BHCKF161"],
    "other_consumer": ["BHCKK137", "BHCKK207", "BHCKB539"],
}
RE_TOTAL = "BHCK1410"
LOANS_TOTAL = "BHCK2122"

# Fed DFAST 2026 Table 10 (severely adverse, 9-quarter cumulative loss, % of average balances)
FED_BAC = {"total": 5.2, "mortgage_first_lien": 1.3, "heloc_junior": 2.4, "business": 6.5,
           "cre": 9.3, "cards": 16.2, "other_consumer": 2.0}
FED_ALL = {"total": 6.9, "mortgage_first_lien": 1.5, "heloc_junior": 3.2, "business": 9.0,
           "cre": 8.8, "cards": 17.1, "other_consumer": 7.3}
FED_SRC = "Fed 2026 DFAST results, Table 10, severely adverse, 2026Q1-2028Q1"


def y9c_mix(path):
    d = pd.read_csv(path, dtype=str).drop_duplicates("ItemName").set_index("ItemName")
    v = pd.to_numeric(d["Value"], errors="coerce") / 1e6        # $ billions
    bal = {k: float(sum(v[c] for c in codes)) for k, codes in MIX_ITEMS.items()}
    bal["mortgage"] = float(v[RE_TOTAL]) - bal["cre"]
    total = float(v[LOANS_TOTAL])
    return bal, total


def bac_loans(cf_path):
    """Quarter-end BAC loans ($B), stitched across two XBRL tags, gaps up to 4q interpolated."""
    cf = json.load(open(cf_path))["facts"]["us-gaap"]
    parts = []
    for tag in ["FinancingReceivableExcludingAccruedInterestBeforeAllowanceForCreditLoss",
                "LoansAndLeasesReceivableNetOfDeferredIncome"]:     # newest tag first
        u = [x for x in cf[tag]["units"]["USD"] if "start" not in x]
        s = (pd.DataFrame(u).sort_values("filed").drop_duplicates("end", keep="last")
             .assign(end=lambda d: pd.to_datetime(d["end"])).set_index("end")["val"] / 1e9)
        parts.append(s)
    s = parts[0].combine_first(parts[1]).sort_index()
    s.index = s.index.to_period("Q")
    s = s[~s.index.duplicated(keep="last")]
    full = pd.period_range(s.index.min(), s.index.max(), freq="Q")
    s = s.reindex(full).interpolate(limit=4)
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", default="data/panel/calibration_panel_full.csv")
    ap.add_argument("--proj", default="data/models/loss_projections.csv")
    ap.add_argument("--companyfacts", default="data/xbrl/bac_companyfacts.json")
    ap.add_argument("--y9c", default="data/y9c/FRY9C_1073757_20251231.csv")
    ap.add_argument("--outdir", default="data/calibration")
    ap.add_argument("--fit-start", default="2014Q1",
                    help="2011-2013 excluded: BAC's legacy Countrywide-era book ran off then (loss ratio 1.4-1.8x industry)")
    ap.add_argument("--fit-end", default="2019Q4")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)

    P = pd.read_csv(a.panel)
    P.index = pd.PeriodIndex(P.pop("quarter"), freq="Q")
    bal, total = y9c_mix(a.y9c)
    w = {k: bal[k] / total for k in SEGMENTS}
    covered = sum(w.values())

    # ---- steps 2-3: implied vs actual BAC NCO rate
    implied = sum(w[k] * P[f"fred_co_{k}"] for k in SEGMENTS)            # % annualized
    loans = bac_loans(a.companyfacts)
    avg_loans = (loans + loans.shift(1)) / 2
    actual = (P["bac_nco_total"] * 4 / avg_loans.reindex(P.index)) * 100
    D = pd.concat([implied.rename("implied"), actual.rename("actual")], axis=1).loc["2011Q1":"2025Q4"].dropna()

    def kfit(df):
        return float((df.actual * df.implied).sum() / (df.implied ** 2).sum())

    def stats(df, k):
        e = df.actual - k * df.implied
        r2 = 1 - (e ** 2).sum() / ((df.actual - df.actual.mean()) ** 2).sum()
        return r2, float(np.sqrt((e ** 2).mean())), float(e.mean())

    legacy = D.loc[:pd.Period(a.fit_start, "Q") - 1]
    D = D.loc[a.fit_start:]
    tr, te = D.loc[:a.fit_end], D.loc[pd.Period(a.fit_end, "Q") + 1:]
    k_tr, k_all = kfit(tr), kfit(D)
    r2_tr, rmse_tr, _ = stats(tr, k_tr)
    r2_te, rmse_te, bias_te = stats(te, k_tr)
    k = k_tr

    # ---- sensitivity of k to the mix assumption
    sens = []
    for label, adj in [("base", {}), ("cards +20%", {"cards": 1.2}), ("cards -20%", {"cards": 0.8}),
                       ("business +20%", {"business": 1.2}), ("CRE +50%", {"cre": 1.5}),
                       ("mortgage -20%", {"mortgage": 0.8})]:
        ww = {s: w[s] * adj.get(s, 1.0) for s in SEGMENTS}
        imp = sum(ww[s] * P[f"fred_co_{s}"] for s in SEGMENTS)
        dd = pd.concat([imp.rename("implied"), actual.rename("actual")], axis=1).loc[a.fit_start:a.fit_end].dropna()
        sens.append((label, kfit(dd)))

    # ---- step 5: project
    Pr = pd.read_csv(a.proj)
    L0 = float(loans.dropna().iloc[-1])
    paths = []
    for scen, g in Pr.groupby("scenario"):
        wide = g.pivot(index="quarter", columns="segment", values="nco_rate_ann")
        ind = sum(w[s] * wide[s] for s in SEGMENTS)                       # decimal, annualized
        bac = k * ind
        out = pd.DataFrame({"scenario": scen, "quarter": wide.index,
                            "industry_implied_nco_rate_pct": (ind * 100).values,
                            "bac_nco_rate_pct": (bac * 100).values,
                            "bac_nco_usd_b": (bac * L0 / 4).values})
        for s in SEGMENTS:
            out[f"industry_{s}_pct"] = (wide[s] * 100).values
        paths.append(out)
    paths = pd.concat(paths)
    paths.round(4).to_csv(os.path.join(a.outdir, "bac_loss_paths.csv"), index=False)

    sa = paths[paths.scenario.str.contains("Severely")].head(9)
    cum_total = float(sa.bac_nco_usd_b.sum() / L0 * 100)

    # ---- step 6: benchmark vs Fed (cumulative 9Q % of balances)
    seg9 = {s: float(Pr[(Pr.scenario.str.contains("Severely")) & (Pr.segment == s)]
                     .sort_values("quarter").nco_rate_ann.head(9).sum() / 4 * 100) for s in SEGMENTS}
    bench = [("total loans", cum_total, FED_BAC["total"], FED_ALL["total"]),
             ("credit cards", seg9["cards"], FED_BAC["cards"], FED_ALL["cards"]),
             ("commercial & industrial", seg9["business"], FED_BAC["business"], FED_ALL["business"]),
             ("commercial real estate", seg9["cre"], FED_BAC["cre"], FED_ALL["cre"]),
             ("residential mortgage (model) vs first-lien (Fed)", seg9["mortgage"],
              FED_BAC["mortgage_first_lien"], FED_ALL["mortgage_first_lien"]),
             ("other consumer", seg9["other_consumer"], FED_BAC["other_consumer"], FED_ALL["other_consumer"])]
    B = pd.DataFrame(bench, columns=["portfolio", "model_9q_cum_loss_pct", "fed_bac_pct", "fed_32bank_pct"])
    B["model_total_is"] = ["BAC-scaled (k x industry mix)"] + ["industry curve, not scaled"] * 5
    B["model_minus_fed_bac_pp"] = B.model_9q_cum_loss_pct - B.fed_bac_pct
    B["source"] = FED_SRC
    B.round(2).to_csv(os.path.join(a.outdir, "benchmark_vs_fed.csv"), index=False)

    ov = pd.DataFrame([{"item": "scale_factor_k_fit_train", "value": k_tr},
                       {"item": "scale_factor_k_fit_all", "value": k_all},
                       {"item": "fit_R2_in_sample", "value": r2_tr},
                       {"item": "oos_R2_2020_2025", "value": r2_te},
                       {"item": "oos_rmse_pp", "value": rmse_te},
                       {"item": "oos_bias_pp", "value": bias_te},
                       {"item": "latest_total_loans_usd_b", "value": L0},
                       {"item": "mix_covered_share", "value": covered}] +
                      [{"item": f"mix_weight_{s}", "value": w[s]} for s in SEGMENTS])
    ov["source"] = "estimated by calibrate_bac_losses.py from BAC XBRL NCO, XBRL loans, Y-9C mix, FRED"
    ov.round(4).to_csv(os.path.join(a.outdir, "calibration_overlays.csv"), index=False)

    L = [f"BAC loan mix (share of total loans {total:,.0f}B, latest Y-9C): " +
         ", ".join(f"{s} {w[s]*100:.1f}%" for s in SEGMENTS) + f"; covered {covered*100:.0f}%, "
         f"uncovered {100*(1-covered):.0f}% gets no industry curve (k absorbs it).",
         f"Fit sample {D.index.min()} to {D.index.max()} ({len(D)}q). Mean implied {D.implied.mean():.2f}% vs actual {D.actual.mean():.2f}%.",
         f"Legacy period {legacy.index.min()} to {legacy.index.max()} excluded from the fit: actual/implied = "
         f"{legacy.actual.mean()/legacy.implied.mean():.2f}x (Countrywide-era book run-off).",
         f"Scale factor k: fit {a.fit_start}-{a.fit_end} = {k_tr:.3f}; {a.fit_start}-2025 = {k_all:.3f}",
         f"  in-sample R2 {r2_tr:.2f}, RMSE {rmse_tr:.2f}pp | out-of-sample 2020-2025: R2 {r2_te:.2f}, RMSE {rmse_te:.2f}pp, bias {bias_te:+.2f}pp",
         "k under alternative mix assumptions (fit window): " + "; ".join(f"{n} {v:.3f}" for n, v in sens), "",
         f"BAC severely adverse, 9Q cumulative net charge-offs: {cum_total:.1f}% of loans "
         f"(Fed published for BAC: {FED_BAC['total']}%)", "", B.round(2).drop(columns=["source"]).to_string(index=False), "",
         "Balances held flat at latest loans; the industry mortgage curve covers first-lien + HELOC +",
         "multifamily (the Y-9C extract does not split them); the Fed rows are a benchmark only."]
    txt = "\n".join(L)
    open(os.path.join(a.outdir, "calibration_report.txt"), "w").write(txt + "\n")
    print(txt)


if __name__ == "__main__":
    main()
