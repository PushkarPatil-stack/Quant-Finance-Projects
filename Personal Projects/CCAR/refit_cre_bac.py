"""
refit_cre_bac.py
----------------
Refits the CRE loss equation on Bank of America's OWN charge-off history instead of the industry (FRED top-100) series.

WHY: diagnose_cre.py showed the industry CRE curve under-predicts even the 2008-10 episode (2.0% vs 4.4% realised), and the
Y-9C history shows BAC's own CRE experience (3.9% worst nine quarters at today's mix) is far below the Fed's 9.3%.

HOW
  1. Sub-portfolio net charge-off rates (construction, multifamily, non-farm non-residential) from Y-9C HI-B / HC-C
     (build_loan_type_history.py), combined with TODAY'S loan-mix weights, so the 2007-10 construction-heavy book does not
     distort the series (a fixed-weight, mix-standardised CRE rate).
  2. Same ARDL(1,1) equation and drivers as estimate_loss_models.py (log rate; unemployment and 4-quarter CRE price change),
     plus alternative specifications (price drawdown) as a range. Fitted 2007Q1 to the last quarter available.
     Quarters with rate <= 3bp are excluded from the fit (log), as for the industry model.
  3. Projections under the Fed Baseline, constructed Adverse and Severely Adverse paths, drivers capped to the fit range.
  4. Effect on BAC's total nine-quarter loss when this curve replaces the industry CRE curve in the segment mix.

    py refit_cre_bac.py      (needs data/y9c/bac_y9c_history.csv, data/panel/*, data/calibration/benchmark_vs_fed.csv;
                              estimate_loss_models.py and build_loan_type_history.py in the same folder)
Outputs: data/models/cre_bac_series.csv, cre_bac_refit.csv, cre_bac_projections.csv, cre_bac_refit.txt
CAUTION: about 34 usable observations from one stress episode; coefficients are individually weak (see t-stats in the output).
"""
import os
import numpy as np
import pandas as pd
import build_loan_type_history as b
import estimate_loss_models as m

D = "data/"
FED_CRE = 9.3
TAX, RWA = 0.191, 1772.9                              # tax rate and RWA ($B) as in build_cet1_waterfall.py
MIX_CRE, K, LOANS = 0.056, 0.955, 1249.0          # BAC CRE share of loans, scale factor k, loans ($B) - as in calibrate_bac_losses.py

H = b.load(D + "y9c")
subs = [k for k in b.SUB if k.startswith("CRE")]
rates, bals = {}, {}
for k in subs:
    rates[k], bals[k] = b.rate_series(H, *b.SUB[k])
w = {k: bals[k].dropna().iloc[-1] for k in subs}
tot = sum(w.values())
w = {k: v / tot for k, v in w.items()}
std = (sum(w[k] * rates[k] for k in subs) / 100).dropna()
std = std.loc["2007Q1":]
os.makedirs(D + "models", exist_ok=True)
std.rename("cre_std_nco_rate").to_csv(D + "models/cre_bac_series.csv", index_label="quarter")
last_q = pd.Period(std.index[-1], "Q")

P = pd.read_csv(D + "panel/calibration_panel_full.csv")
P.index = pd.PeriodIndex(P.pop("quarter"), freq="Q")
S = pd.read_csv(D + "panel/scenarios_long.csv")
mc = [c for c in P.columns if c.startswith("macro_")]


def feats(M):
    F = m.features(M)
    cr = np.log(M["macro_cre_index"])
    F["cre_dd"] = cr - cr.rolling(12, min_periods=4).max()
    return F


Fh = feats(P[mc])
y = np.log(std.set_axis(pd.PeriodIndex(std.index, freq="Q")).where(lambda s: s > m.FLOOR)).reindex(P.index)
actual = pd.Series(std.values, index=pd.PeriodIndex(std.index, freq="Q"))

# starting rate for projections: BAC's latest full-year CRE rate (latest Q4 year-to-date over average balance), else the last quarter
start = float(actual.iloc[-1])
for yr in sorted({q[:4] for q in H.index}, reverse=True):
    if f"{yr}Q4" in H.index and f"{int(yr)-1}Q4" in H.index:
        parts = []
        for k in subs:
            co, rec, bal = b.SUB[k]
            ytd = (b.num(H, co) - b.num(H, rec)).loc[f"{yr}Q4"] / 1e6
            bl = (b.num(H, bal) / 1e6)
            parts.append(w[k] * ytd / ((bl.loc[f"{yr}Q4"] + bl.loc[f"{int(yr)-1}Q4"]) / 2))
        start = max(sum(parts), 0.0005)
        start_note = f"{yr} full-year rate"
        break
else:
    start_note = f"last fitted quarter ({last_q})"

SPECS = [("ur + cre4 (same as industry model)", ["ur", "cre4"]), ("drawdown only", ["cre_dd"]),
         ("ur + drawdown", ["ur", "cre_dd"]), ("cre4 + drawdown", ["cre4", "cre_dd"])]
bt_q = list(pd.period_range("2008Q1", "2010Q1", freq="Q"))
act_bt = actual.loc["2008Q1":"2010Q1"].sum() / 4 * 100
rows, projs, coefs = [], [], []
for si, (name, cols) in enumerate(SPECS):
    mod = m.fit_eq(y, Fh, cols, P.index, "2007Q1", str(last_q))
    bt = np.exp(m.simulate(mod, float(np.log(max(actual.loc["2007Q4"], m.FLOOR))), Fh, bt_q)).sum() / 4 * 100
    sev_cum = None
    for sname, g in S.groupby("scenario"):
        g = g.set_index(pd.PeriodIndex(g["quarter"], freq="Q"))
        Fa = feats(pd.concat([P[mc].loc[:"2025Q4"], g[mc]]))
        Hh = Fh.loc["2007Q1":str(last_q), cols]
        for c in cols:
            Fa[c] = Fa[c].clip(Hh[c].min(), Hh[c].max())
        path = np.exp(m.simulate(mod, float(np.log(start)), Fa, list(g.index)))
        if si == 0:
            projs.append(pd.DataFrame({"scenario": sname, "segment": "cre", "quarter": [str(q) for q in g.index], "nco_rate_ann": path}))
        if sname == "Supervisory Severely Adverse":
            sev_cum, sev_peak = path[:9].sum() / 4 * 100, path.max() * 100
    rows.append(dict(spec=name, n=mod["n"], R2=round(mod["r2"], 3), rho=round(mod["b"][1], 2), backtest_pred_9q=round(bt, 2),
                     backtest_actual_9q=round(act_bt, 2), severe_9q_cum_pct=round(sev_cum, 2), severe_peak_pct=round(sev_peak, 2)))
    for n_, b_, s_ in zip(mod["names"], mod["b"], mod["se"]):
        coefs.append(dict(spec=name, term=n_, coef=round(b_, 3), t=round(b_ / s_, 2) if s_ == s_ else np.nan))
R = pd.DataFrame(rows)
R.to_csv(D + "models/cre_bac_refit.csv", index=False)
pd.DataFrame(coefs).to_csv(D + "models/cre_bac_coefficients.csv", index=False)
pd.concat(projs).round(5).to_csv(D + "models/cre_bac_projections.csv", index=False)

# effect on total nine-quarter loss when the BAC-own CRE curve replaces the industry one
B = pd.read_csv(D + "calibration/benchmark_vs_fed.csv").set_index("portfolio")
ind = float(B.loc["commercial real estate", "model_9q_cum_loss_pct"])
tot_old = float(B.loc["total loans", "model_9q_cum_loss_pct"])
new_cre = float(R.severe_9q_cum_pct.iloc[0])
tot_new = tot_old + K * MIX_CRE * (new_cre - ind)
d_b = (tot_new - tot_old) / 100 * LOANS
txt = (f"CRE refit on BAC's own mix-standardised series ({len(std)} quarters, 2007Q1-{last_q}; today's weights "
       + ", ".join(f"{k.split(': ')[1]} {v:.0%}" for k, v in w.items()) + ")\n"
       f"Projection start rate {start*100:.2f}% ({start_note}).\n\n" + R.to_string(index=False) +
       f"\n\nFirst specification (same as the industry model): severe nine-quarter CRE loss {new_cre:.2f}% vs industry model {ind:.2f}% and Fed {FED_CRE}%.\n"
       f"Total nine-quarter loss with this CRE curve: {tot_new:.2f}% vs {tot_old:.2f}% (change {d_b:+.1f}B; about {-d_b * (1 - TAX) / RWA * 100:+.2f}pp of CET1 before any reserve effect).\n"
       "Coefficient t-statistics are in cre_bac_coefficients.csv; most are below 2, so treat the specifications as a range.\n")
open(D + "models/cre_bac_refit.txt", "w").write(txt)
print(txt)
