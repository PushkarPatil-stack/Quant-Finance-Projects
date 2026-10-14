"""
estimate_loss_models.py  (v2)
------------------------------
Loss models by segment on FRED top-100-bank data, window 2000Q1-2025Q4, projected under
the Fed Baseline and Severely Adverse paths.

    py estimate_loss_models.py            (needs data/panel/calibration_panel_full.csv, scenarios_long.csv)

WHY v2: v1 used a two-step Engle-Granger ECM on delinquency vs unemployment LEVELS. The data rejected
it (delinquency is near unit-root; residual AR(1) 0.85-0.97; wrong-signed error-correction terms).
v2 uses the unrestricted (one-step) error-correction form, which is algebraically an ARDL(1,1):
    y_t = a + rho*y_{t-1} + b0'x_t + b1'x_{t-1} + covid_t + e_t
    <=>  dy_t = a + (rho-1)*y_{t-1} + b0'dx_t + (b0+b1)'x_{t-1} + ...      (speed of adjustment = 1-rho)
Drivers are the unemployment GAP (unemployment minus its trailing 12-quarter minimum - captures the
turn in the cycle, not the level), 4-quarter change in unemployment, BBB spread, 4-quarter house
price / CRE price growth.

Two equations per segment:
  LOSS : y = log(net charge-off rate)         <- the loss engine (what flows to provisions)
  PD   : y = logit(delinquency rate)          <- PD proxy (not available for other_consumer)
  LGD (implied) = NCO_t / DQ_{t-1}            <- reported as a decomposition, not separately forecast
Mortgage/CRE quarters with NCO <= 3bp (net recoveries) are excluded from the LOSS fit and the
projection is floored at 3bp.

Outputs in data/models: loss_model_coefficients.csv, loss_model_fit.csv,
loss_projections.csv, loss_model_report.txt
"""
import argparse
import os

import numpy as np
import pandas as pd

FLOOR = 0.0003
DRIVERS = {"cards": ["urmin", "dur4"], "business": ["urmin", "spread"],
           "other_consumer": ["urmin", "dur4"], "cre": ["ur", "cre4"], "mortgage": ["ur", "hpi4"]}
COVID = ("2020Q2", "2021Q4")


def features(m):
    ur = m["macro_unemployment"]
    return pd.DataFrame({
        "ur": ur, "urmin": ur - ur.rolling(12, min_periods=12).min(), "dur4": ur.diff(4),
        "spread": m["macro_bbb_spread"], "hpi4": np.log(m["macro_hpi"]).diff(4),
        "cre4": np.log(m["macro_cre_index"]).diff(4)})


def covid_dummy(idx):
    c = pd.Series(0.0, index=idx)
    c.loc[COVID[0]:COVID[1]] = 1.0
    return c


def logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def design(y, F, cols, idx):
    Z = pd.concat([y.rename("y"), y.shift(1).rename("ylag"), F[cols], F[cols].shift(1).add_suffix("_l1"),
                   covid_dummy(idx).rename("covid")], axis=1)
    return Z


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
    r2 = 1 - e @ e / ((y - y.mean()) ** 2).sum()
    return b, se, e, r2


def fit_eq(y, F, cols, idx, lo, hi):
    Z = design(y, F, cols, idx).loc[lo:hi].dropna()
    dropped = Z["covid"].sum() == 0          # no COVID-period observations left -> dummy not identifiable
    if dropped:
        Z = Z.drop(columns="covid")
    names = ["const"] + list(Z.columns[1:])
    A = np.column_stack([np.ones(len(Z)), Z.drop(columns="y").values])
    b, se, e, r2 = nw_ols(A, Z["y"].values)
    if dropped:
        b, se, names = np.append(b, 0.0), np.append(se, np.nan), names + ["covid(not identified)"]
    ar1 = np.corrcoef(e[1:], e[:-1])[0, 1]
    # fitted-vs-actual (in-sample, one-step ahead)
    return dict(names=names, b=b, se=se, r2=r2, ar1=ar1, n=len(Z), sigma=e.std(ddof=len(b)),
                cols=cols)


def step(model, ylag, F_t, F_l1):
    v = [1.0, ylag] + list(F_t[model["cols"]]) + list(F_l1[model["cols"]]) + [0.0]
    return float(np.dot(model["b"], v))


def simulate(model, y0, Fall, quarters):
    out, y = [], y0
    for q in quarters:
        y = step(model, y, Fall.loc[q], Fall.loc[q - 1])
        out.append(y)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", default="data/panel/calibration_panel_full.csv")
    ap.add_argument("--scen", default="data/panel/scenarios_long.csv")
    ap.add_argument("--outdir", default="data/models")
    ap.add_argument("--start", default="2000Q1")
    ap.add_argument("--end", default="2025Q4")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    P = pd.read_csv(a.panel)
    P.index = pd.PeriodIndex(P.pop("quarter"), freq="Q")
    S = pd.read_csv(a.scen)
    macro_cols = [c for c in P.columns if c.startswith("macro_")]
    last = pd.Period(a.end, "Q")
    coefs, fits, projs, lgdrows = [], [], [], []

    for seg, cols in DRIVERS.items():
        if f"fred_co_{seg}" not in P.columns:
            continue
        co = P[f"fred_co_{seg}"] / 100
        eqs = {"LOSS": (np.log(co.where(co > FLOOR)), "co")}
        if f"fred_dq_{seg}" in P.columns:
            eqs["PD"] = (logit(P[f"fred_dq_{seg}"] / 100), "dq")
        F_hist = features(P[macro_cols])
        models = {}
        for eq, (y, _) in eqs.items():
            m = fit_eq(y, F_hist, cols, P.index, a.start, a.end)
            models[eq] = m
            for n_, b_, s_ in zip(m["names"], m["b"], m["se"]):
                coefs.append((seg, eq, n_, b_, s_, b_ / s_ if s_ == s_ else np.nan))
            rho = m["b"][1]
            fits.append((seg, eq, m["n"], m["r2"], m["ar1"], rho, 1 - rho,
                         np.log(0.5) / np.log(rho) if 0 < rho < 1 else np.nan, m["sigma"]))
        for sname, g in S.groupby("scenario"):
            g = g.set_index(pd.PeriodIndex(g["quarter"], freq="Q"))
            M = pd.concat([P[macro_cols].loc[:last], g[macro_cols]])
            F_all = features(M)
            qs = list(g.index)
            res = {"quarter": [str(q) for q in qs]}
            y0 = {"LOSS": float(np.log(max(co.loc[last], FLOOR))),
                  "PD": float(logit(P.loc[last, f"fred_dq_{seg}"] / 100)) if "PD" in models else None}
            for eq in models:
                path = simulate(models[eq], y0[eq], F_all, qs)
                res[eq] = np.exp(path) if eq == "LOSS" else 1 / (1 + np.exp(-np.array(path)))
            df = pd.DataFrame(res)
            df.insert(0, "scenario", sname)
            df.insert(1, "segment", seg)
            df = df.rename(columns={"LOSS": "nco_rate_ann", "PD": "dq_rate"})
            if "dq_rate" in df:
                df["lgd_implied"] = df["nco_rate_ann"] / df["dq_rate"].shift(1).fillna(P.loc[last, f"fred_dq_{seg}"] / 100)
            projs.append(df)

    coef = pd.DataFrame(coefs, columns=["segment", "equation", "term", "coef", "se_NW", "t"])
    fit = pd.DataFrame(fits, columns=["segment", "equation", "n", "R2", "resid_AR1", "rho", "speed_1-rho",
                                      "half_life_q", "resid_sd"])
    proj = pd.concat(projs)
    coef.round(4).to_csv(os.path.join(a.outdir, "loss_model_coefficients.csv"), index=False)
    fit.round(3).to_csv(os.path.join(a.outdir, "loss_model_fit.csv"), index=False)
    proj.round(5).to_csv(os.path.join(a.outdir, "loss_projections.csv"), index=False)

    L = [f"Window {a.start}-{a.end}; one-step ECM/ARDL(1,1) + COVID dummy 2020Q2-2021Q4", "",
         "FIT (resid_AR1 near 0 = residuals are not autocorrelated; rho = persistence; "
         "half_life = quarters for a shock to decay by half)", fit.round(3).to_string(index=False), "",
         "PROJECTED ANNUALIZED NCO RATE (%), 2026Q1-2029Q1 (13 quarters):"]
    for seg, g in proj.groupby("segment"):
        parts = []
        for s, h in g.groupby("scenario"):
            parts.append(f"{s.replace('Supervisory ', '')}: peak {h.nco_rate_ann.max()*100:.2f} "
                         f"(Q{int(h.nco_rate_ann.values.argmax())+1}), avg9 {h.nco_rate_ann.head(9).mean()*100:.2f}")
        L.append(f"  {seg:15s} " + " | ".join(parts))
    txt = "\n".join(L)
    open(os.path.join(a.outdir, "loss_model_report.txt"), "w").write(txt + "\n")
    print(txt)


if __name__ == "__main__":
    main()
