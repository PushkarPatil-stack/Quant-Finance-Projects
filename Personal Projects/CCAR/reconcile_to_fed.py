"""
reconcile_to_fed.py
-------------------
Measures (does not tune away) the gaps between this model's severely adverse result and the Fed's published result for BAC.

  A. PPNR components against BAC's own trailing run-rate (2025Q1-2025Q4 actuals x 9/4), model and Fed side by side.
     Shows whether each component's gap is a model move or a Fed move, and how much of the component gaps cancel in revenue.
  B. Segment mix: what total nine-quarter loss would be if the CRE and mortgage curves were set to the Fed's values
     (same BAC loan mix and scale factor k as calibrate_bac_losses.py). Shows how much the two big segment misses matter in total.

    py reconcile_to_fed.py        (needs data/panel/calibration_panel_full.csv, data/models/ppnr_projections.csv,
                                   data/calibration/benchmark_vs_fed.csv; run after steps 5 and 7)
Output: data/waterfall/reconcile_to_fed.csv, reconcile_segment_mix.csv, reconcile_to_fed.txt
"""
import os
import pandas as pd

D = "data/"
FED = dict(nii=151.1, nonint_income=89.1, nonint_expense=174.0, ppnr=66.2)       # Fed 2026 DFAST, BAC severe, 2026Q1-2028Q1, $B
MIX = dict(cards=0.085, business=0.290, cre=0.056, mortgage=0.223, other_consumer=0.072)   # BAC loan mix, Y-9C 2025Q4
K = 0.955                                                                                    # BAC scale factor, 2014-2019 fit

P = pd.read_csv(D + "panel/calibration_panel_full.csv").set_index("quarter")
t4 = P.loc["2025Q1":"2025Q4", ["bac_nii", "bac_nonint_income", "bac_nonint_expense", "bac_ppnr"]].sum() * 9 / 4
M = pd.read_csv(D + "models/ppnr_projections.csv")
sev = M[M.scenario == "Supervisory Severely Adverse (with overlays)"].head(9)
mod = dict(nii=sev.nii.sum(), nonint_income=sev.nonint_income.sum(), nonint_expense=sev.nonint_expense.sum(), ppnr=sev.ppnr.sum())
run = dict(nii=t4.bac_nii, nonint_income=t4.bac_nonint_income, nonint_expense=t4.bac_nonint_expense, ppnr=t4.bac_ppnr)
rows = []
for k, lab in [("nii", "Net interest income"), ("nonint_income", "Noninterest income"), ("nonint_expense", "Noninterest expense (model incl. operational-risk loss)")]:
    rows.append(dict(item=lab, run_rate_9q=run[k], model=mod[k], fed=FED[k], model_vs_run_pct=100 * (mod[k] / run[k] - 1),
                     fed_vs_run_pct=100 * (FED[k] / run[k] - 1), model_minus_fed=mod[k] - FED[k]))
rv_run, rv_mod, rv_fed = run["nii"] + run["nonint_income"], mod["nii"] + mod["nonint_income"], FED["nii"] + FED["nonint_income"]
rows.insert(2, dict(item="Net revenue (NII + noninterest income)", run_rate_9q=rv_run, model=rv_mod, fed=rv_fed,
                    model_vs_run_pct=100 * (rv_mod / rv_run - 1), fed_vs_run_pct=100 * (rv_fed / rv_run - 1), model_minus_fed=rv_mod - rv_fed))
rows.append(dict(item="PPNR", run_rate_9q=run["ppnr"], model=mod["ppnr"], fed=FED["ppnr"], model_vs_run_pct=100 * (mod["ppnr"] / run["ppnr"] - 1),
                 fed_vs_run_pct=100 * (FED["ppnr"] / run["ppnr"] - 1), model_minus_fed=mod["ppnr"] - FED["ppnr"]))
A = pd.DataFrame(rows)
os.makedirs(D + "waterfall", exist_ok=True)
A.round(2).to_csv(D + "waterfall/reconcile_to_fed.csv", index=False)

B = pd.read_csv(D + "calibration/benchmark_vs_fed.csv").set_index("portfolio")
col = "model_9q_cum_loss_pct"
seg = {"cards": "credit cards", "business": "commercial & industrial", "cre": "commercial real estate",
       "mortgage": "residential mortgage (model) vs first-lien (Fed)", "other_consumer": "other consumer"}
loans = 1249.0
def total(cre_fed, mort_fed):
    v = 0.0
    for s, name in seg.items():
        x = B.loc[name, col]
        if s == "cre" and cre_fed: x = B.loc[name, "fed_bac_pct"]
        if s == "mortgage" and mort_fed: x = B.loc[name, "fed_bac_pct"]
        v += MIX[s] * x
    return K * v
variants = [("Model as built", False, False), ("CRE set to the Fed's", True, False), ("Mortgage set to the Fed's", False, True), ("CRE and mortgage set to the Fed's", True, True)]
C = pd.DataFrame([dict(variant=n, total_9q_loss_pct=total(c, m), total_9q_loss_b=total(c, m) / 100 * loans) for n, c, m in variants])
C["change_vs_model_b"] = C.total_9q_loss_b - C.total_9q_loss_b.iloc[0]
C["fed_total_pct"] = B.loc["total loans", "fed_bac_pct"]
C.round(3).to_csv(D + "waterfall/reconcile_segment_mix.csv", index=False)
txt = ("A. PPNR components, 9 quarters, $B (run-rate = 2025Q1-Q4 actual x 9/4)\n" + A.round(1).to_string(index=False) +
       "\n\nB. Total nine-quarter loss with the CRE and mortgage curves set to the Fed's (same mix and k)\n" + C.round(2).to_string(index=False) + "\n")
open(D + "waterfall/reconcile_to_fed.txt", "w").write(txt)
print(txt)
