"""
Bank of America stress-testing dashboard (real-data pipeline).

    py -m streamlit run dashboard_app.py -- --data "C:\\path\\to\\CCAR\\data"
    (default data folder: ./data, then ../data)

Reads only the CSV/TXT files written by the pipeline scripts; it never recomputes a model.
"""
import argparse
import os
import sys

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

REQ = 10.0                        # 4.5 minimum + 3.0 G-SIB surcharge + 2.5 stress capital buffer (YE2025)
FED = {"min": 9.9, "end": 10.8, "start": 11.4, "ppnr": 66.2, "prov": 69.5, "pretax": -12.5}
SEV, BASE, ADV = "Severely Adverse", "Baseline", "Adverse (constructed)"
PATH_NAME = {BASE: "Supervisory Baseline", ADV: "Constructed Adverse", SEV: "Supervisory Severely Adverse"}
COL = {BASE: "#3b82f6", ADV: "#f59e0b", SEV: "#dc2626", "model": "#9ca3af", "fed": "#16a34a"}


def data_dir():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None)
    a, _ = ap.parse_known_args(sys.argv[1:])
    here = os.path.dirname(os.path.abspath(__file__))
    for c in [a.data, "data", os.path.join("..", "data"), os.path.join(here, "data"), os.path.join(here, "..", "data"),
                 os.path.join(here, ".."), os.path.join(here, "..", "..")]:
        if c and os.path.isdir(os.path.join(c, "waterfall")):
            return c
    return "data"


D = data_dir()


@st.cache_data
def load(rel, **kw):
    p = os.path.join(D, rel)
    return pd.read_csv(p, **kw) if os.path.exists(p) else None


@st.cache_data
def text(rel):
    p = os.path.join(D, rel)
    return open(p, encoding="utf-8").read() if os.path.exists(p) else None


def need(df, name):
    if df is None:
        st.warning(f"`{name}` not found in `{D}`. Run the pipeline script that writes it (see Governance tab).")
        return False
    return True


def line_fig(title, ytitle, series, hline=None, hline_label=None, height=380):
    f = go.Figure()
    for name, x, y, color, dash in series:
        f.add_trace(go.Scatter(x=x, y=y, name=name, mode="lines", line=dict(color=color, width=2.4, dash=dash)))
    if hline is not None:
        f.add_hline(y=hline, line=dict(color="#6b7280", dash="dot"), annotation_text=hline_label,
                    annotation_position="bottom right")
    f.update_layout(title=title, yaxis_title=ytitle, height=height, margin=dict(l=10, r=10, t=50, b=10),
                    legend=dict(orientation="h", y=-0.2), hovermode="x unified")
    return f


st.set_page_config(page_title="BAC Stress Test", layout="wide")
st.title("Bank of America: capital stress test")
st.caption("Fed 2026 supervisory scenarios, 2026Q1-2029Q1. Industry loss curves scaled to BAC history, "
           "PPNR from BAC income with stress overlays, CET1 waterfall. Fed's published BAC result is shown as a benchmark only.")

W = load("waterfall/cet1_waterfall.csv")
SM = load("waterfall/cet1_summary.csv")
tabs = st.tabs(["Summary", "Scenarios", "Credit losses", "PPNR", "Capital waterfall", "Rate risk (NII / EVE)", "Governance"])

# ------------------------------------------------------------------ Summary
with tabs[0]:
    if need(W, "waterfall/cet1_waterfall.csv") and need(SM, "waterfall/cet1_summary.csv"):
        sv = SM[SM.scenario == SEV].iloc[0]
        bs = SM[SM.scenario == BASE].iloc[0]
        c = st.columns(4)
        c[0].metric("CET1 ratio, start (2025Q4)", f"{sv.cet1_start_pct:.2f}%")
        c[1].metric("Severe minimum", f"{sv.cet1_min_pct_9q:.2f}%", f"{sv.cet1_min_pct_9q - REQ:+.2f} pp vs {REQ:.1f}% requirement",
                    delta_color="normal")
        c[2].metric("Severe at 2028Q1", f"{sv.cet1_end_9q_pct:.2f}%")
        c[3].metric("Baseline at 2028Q1", f"{bs.cet1_end_9q_pct:.2f}%")
        fig = go.Figure()
        for s in (BASE, ADV, SEV):
            g = W[W.scenario == s]
            fig.add_trace(go.Scatter(x=g.quarter, y=g.cet1_ratio_pct, name=s, mode="lines+markers",
                                     line=dict(color=COL[s], width=2.6)))
        fig.add_hline(y=REQ, line=dict(color="#6b7280", dash="dot"), annotation_text="Requirement 10.0%",
                      annotation_position="top left")
        fig.add_hline(y=FED["min"], line=dict(color=COL["fed"], dash="dash"), annotation_text="Fed minimum for BAC 9.9%",
                      annotation_position="bottom right")
        fig.update_layout(title="CET1 ratio path", yaxis_title="% of RWA", height=420, margin=dict(l=10, r=10, t=50, b=10),
                          legend=dict(orientation="h", y=-0.15), hovermode="x unified")
        st.plotly_chart(fig, width="stretch")
        cmp = pd.DataFrame({
            "Severely adverse, 9 quarters ($B)": ["PPNR", "Provisions", "Pre-tax income", "CET1 minimum (%)", "CET1 at 2028Q1 (%)"],
            "This model": [sv.ppnr_9q, sv.provisions_9q, sv.pretax_9q, sv.cet1_min_pct_9q, sv.cet1_end_9q_pct],
            "Fed, BAC (benchmark)": [FED["ppnr"], FED["prov"], FED["pretax"], FED["min"], FED["end"]]})
        cmp["Difference"] = cmp["This model"] - cmp["Fed, BAC (benchmark)"]
        st.dataframe(cmp.round(2), hide_index=True, width="stretch")
        st.caption(f"Nothing is fitted to the Fed figures. Modelled PPNR is {sv.ppnr_9q - FED['ppnr']:+.1f}B and provisions "
                   f"{sv.provisions_9q - FED['prov']:+.1f}B versus the Fed's, so pre-tax income is {sv.pretax_9q - FED['pretax']:+.1f}B apart. "
                   "The provision gap comes from segment misses (CRE, mortgage); OCI and capital actions are not modelled (see Governance).")

# ------------------------------------------------------------------ Scenarios
with tabs[1]:
    sc, panel = load("panel/scenarios_long.csv"), load("panel/calibration_panel_full.csv")
    if need(sc, "panel/scenarios_long.csv"):
        cols = [c for c in sc.columns if c.startswith("macro_")]
        nice = {c: c.replace("macro_", "").replace("_", " ") for c in cols}
        pick = st.selectbox("Macro variable", cols, index=cols.index("macro_unemployment") if "macro_unemployment" in cols else 0,
                            format_func=lambda c: nice[c])
        ser = []
        if panel is not None and pick in panel.columns:
            h = panel[["quarter", pick]].dropna().tail(32)
            ser.append(("History", h.quarter, h[pick], "#6b7280", "solid"))
        for n, g in sc.groupby("scenario"):
            ser.append((n.replace("Supervisory ", ""), g.quarter, g[pick], COL[SEV] if "Severe" in n else (COL[ADV] if "Constructed" in n else COL[BASE]), "solid"))
        st.plotly_chart(line_fig(nice[pick].title(), "", ser), width="stretch")
        st.caption("Source: Fed 2026 supervisory scenario files. History: Fed historic file. The Fed publishes no Adverse scenario; 'Constructed Adverse' = Baseline + 0.5 x (Severely Adverse - Baseline), an assumption.")

# ------------------------------------------------------------------ Credit losses
with tabs[2]:
    P, B, O = load("calibration/bac_loss_paths.csv"), load("calibration/benchmark_vs_fed.csv"), load("calibration/calibration_overlays.csv")
    if need(P, "calibration/bac_loss_paths.csv"):
        seg = {"cards": "Credit cards", "business": "Business (C&I)", "cre": "CRE", "mortgage": "Residential mortgage",
               "other_consumer": "Other consumer"}
        scn = st.radio("Scenario", [BASE, ADV, SEV], horizontal=True, key="loss_scn")
        g = P[P.scenario == PATH_NAME[scn]]
        ser = [(seg[k], g.quarter, g[f"industry_{k}_pct"], c, "solid") for k, c in
               zip(seg, ["#ef4444", "#3b82f6", "#f59e0b", "#10b981", "#8b5cf6"])]
        ser.append(("BAC total (scaled)", g.quarter, g.bac_nco_rate_pct, "#111827", "dash"))
        st.plotly_chart(line_fig("Annualized net charge-off rate: industry segment curves and BAC total", "%", ser), width="stretch")
        if need(B, "calibration/benchmark_vs_fed.csv"):
            st.subheader("Nine-quarter cumulative loss, severely adverse, versus the Fed's published BAC rates")
            bt = B.drop(columns=[c for c in ["source", "model_total_is"] if c in B]).rename(columns={
                "portfolio": "Portfolio", "model_9q_cum_loss_pct": "Model %", "fed_bac_pct": "Fed BAC %",
                "fed_32bank_pct": "Fed 32 banks %", "model_minus_fed_bac_pp": "Model minus Fed (pp)"})
            st.dataframe(bt.round(2), hide_index=True, width="stretch")
            st.caption("Only the total is scaled to BAC (one factor fitted on BAC's own 2014-2019 charge-offs). Segment rows are "
                       "unscaled industry curves; the CRE and mortgage misses are known limitations and are not tuned away.")
        if O is not None:
            o = O.set_index("item").value
            c = st.columns(4)
            c[0].metric("BAC scale factor k", f"{o['scale_factor_k_fit_train']:.2f}")
            c[1].metric("Out-of-sample R²", f"{o['oos_R2_2020_2025']:.2f}")
            c[2].metric("Out-of-sample RMSE", f"{o['oos_rmse_pp']:.2f} pp")
            c[3].metric("Loan mix covered", f"{o['mix_covered_share']*100:.0f}%")
        L = load("models/loss_model_fit.csv")
        if L is not None:
            st.subheader("Segment loss equations (ARDL/ECM form, 2000Q1-2025Q4)")
            st.dataframe(L.round(3), hide_index=True, width="stretch")
            st.caption("LOSS = log net charge-off rate (used). PD = logit delinquency rate: near unit-root, speed of adjustment "
                       "near zero, shown for transparency and not used for projection.")

# ------------------------------------------------------------------ PPNR
with tabs[3]:
    PP = load("models/ppnr_projections.csv")
    if need(PP, "models/ppnr_projections.csv"):
        ser = []
        for n, g in PP.groupby("scenario"):
            color = COL[BASE] if n.endswith("Baseline") else (COL[ADV] if "Constructed" in n else (COL[SEV] if "overlays" in n else COL["model"]))
            ser.append((n.replace("Supervisory ", ""), g.quarter, g.ppnr, color, "solid" if "model only" not in n else "dot"))
        st.plotly_chart(line_fig("Quarterly PPNR ($B)", "$B", ser), width="stretch")
        rows = []
        for n, g in PP.groupby("scenario"):
            g9 = g.head(9)
            rows.append({"Scenario": n.replace("Supervisory ", ""), "NII": g9.nii.sum(), "Noninterest income": g9.nonint_income.sum(),
                         "Noninterest expense": g9.nonint_expense.sum(), "PPNR (9Q)": g9.ppnr.sum()})
        rows.append({"Scenario": "Fed, BAC severe (benchmark)", "NII": 151.1, "Noninterest income": 89.1,
                     "Noninterest expense": 174.0, "PPNR (9Q)": 66.2})
        st.dataframe(pd.DataFrame(rows).round(1), hide_index=True, width="stretch")
        OV = load("models/ppnr_overlays.csv")
        if OV is not None:
            st.subheader("Stress overlays (stressed scenarios; the NII effect comes from the Rate risk tab)")
            st.dataframe(OV.round(4), hide_index=True, width="stretch")
            st.caption("The 2014-2025 sample has no deep recession, so the stress response is explicit parameters. Each overlay lists its source and observed range; the sales and trading haircut is set to zero.")
        rep = text("models/ppnr_report.txt")
        if rep:
            with st.expander("Sensitivity and fit report"):
                st.code(rep)

# ------------------------------------------------------------------ Capital waterfall
with tabs[4]:
    if need(W, "waterfall/cet1_waterfall.csv"):
        scn = st.radio("Scenario", [SEV, ADV, BASE], horizontal=True, key="wf_scn")
        g = W[(W.scenario == scn) & (W.h <= 9)]
        start = SM[SM.scenario == scn].cet1_start_pct.iloc[0] / 100 * g.rwa.iloc[0]
        steps = [("Start CET1", start), ("PPNR", g.ppnr.sum()), ("Provisions", -g.provision.sum()),
                 ("Fed-set stress items", -g.other_stress_items.sum()), ("Taxes", -g.tax.sum()), ("Dividends", -g.dividends.sum())]
        end = g.cet1.iloc[-1]
        fig = go.Figure(go.Waterfall(x=[s[0] for s in steps] + ["End CET1 (2028Q1)"], y=[s[1] for s in steps] + [end],
                                     measure=["absolute"] + ["relative"] * 5 + ["total"],
                                     increasing=dict(marker=dict(color="#16a34a")), decreasing=dict(marker=dict(color="#dc2626")),
                                     totals=dict(marker=dict(color="#1f2937")), connector=dict(line=dict(color="#9ca3af"))))
        fig.update_layout(title=f"CET1 capital bridge, nine quarters, {scn} ($B)", height=420, margin=dict(l=10, r=10, t=50, b=10))
        st.plotly_chart(fig, width="stretch")
        show = W[W.scenario == scn][["quarter", "ppnr", "nco", "allowance", "provision", "pretax", "net_income", "dividends", "cet1", "cet1_ratio_pct"]]
        st.dataframe(show.round(2), hide_index=True, width="stretch")
        st.caption("Allowance follows the Fed's published rule: the next four quarters of projected losses at each quarter-end, with the difference from the starting allowance "
                   "smoothed over nine quarters. Dividends follow the Fed's capital-action assumptions: preferred each quarter, common only in the first quarter. "
                   "RWA and balances are held flat. OCI is not modelled.")
        AS = load("waterfall/cet1_allowance_sensitivity.csv")
        if AS is not None:
            st.markdown("**Allowance and dividend method, severely adverse** (the Fed's end allowance is implied as 13.2 + provisions minus loan losses)")
            st.dataframe(AS.round(2), hide_index=True, width="stretch")
            st.caption("The base follows the Fed's rules. The last rows show the earlier calibrated 7.2-quarter horizon and continuing common dividends, which put the minimum at 8.84%. "
                       "Remaining difference to the Fed: the model's loan losses are $5.9B lower but its end allowance ($29.1B) is above the Fed's implied $17.8B.")
        BRG = load("waterfall/cet1_bridge_to_fed.csv")
        if BRG is not None:
            st.markdown("**Remaining steps from the model's severe CET1 to the Fed's** (cumulative)")
            st.dataframe(BRG.round(2), hide_index=True, width="stretch")
            st.caption("The Fed's OCI is published; the last step sets loan losses to the Fed's. It shows what explains the difference; it does not make the model match the Fed.")

# ------------------------------------------------------------------ Rate risk
with tabs[5]:
    NR, FT, OS = load("rate_risk/nii_eve_results.csv"), load("rate_risk/nii_fit_vs_disclosed.csv"), load("rate_risk/nii_out_of_sample_2025.csv")
    NE, EV, ES = load("rate_risk/nii_scenario_effect.csv"), load("rate_risk/basel_eve_results.csv"), load("rate_risk/basel_eve_sensitivity.csv")
    st.subheader("Net interest income sensitivity, 12 months, instantaneous parallel shift ($B)")
    if need(NR, "rate_risk/nii_eve_results.csv"):
        st.dataframe(NR.rename(columns={"shock": "Shock", "static_deposits_b": "Static deposits", "dynamic_deposits_b": "Dynamic deposits",
                                        "dynamic_pct_of_run_rate": "Dynamic, % of run-rate NII"}).round(2), hide_index=True, width="stretch")
    if FT is not None:
        FT = FT.assign(key=FT.year.astype(str) + " " + FT.basis)
        pick = st.selectbox("Fit to BAC's disclosed sensitivities", list(dict.fromkeys(FT.key)), index=len(set(FT.key)) - 1)
        g = FT[FT.key == pick]
        f = go.Figure()
        f.add_trace(go.Bar(x=g.shock, y=g.disclosed_b, name="Disclosed (10-K)", marker_color="#1f2937"))
        f.add_trace(go.Bar(x=g.shock, y=g.model_b, name="Model", marker_color="#3b82f6"))
        f.update_layout(barmode="group", height=360, yaxis_title="$B", margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h", y=-0.3))
        st.plotly_chart(f, width="stretch")
        rm = FT.groupby("basis").apply(lambda x: (x.diff_b ** 2).mean() ** 0.5)
        st.caption("Fit RMSE ($B): " + ", ".join(f"{k} {v:.2f}" for k, v in rm.items()) +
                   ". Static = no deposit balance change; dynamic = deposit migration allowed. Sources: BAC 10-K Tables 44 and 45.")
    if OS is not None:
        st.subheader("Out-of-sample test: parameters fitted without 2025, 2025 disclosed sensitivities predicted")
        st.dataframe(OS.round(2), hide_index=True, width="stretch")
    if NE is not None:
        ser = [(n.replace("Supervisory ", ""), g.quarter, g.nii_effect_vs_baseline_b, COL[SEV] if "Severe" in n else (COL[ADV] if "Constructed" in n else COL[BASE]), "solid")
               for n, g in NE.groupby("scenario")]
        st.plotly_chart(line_fig("Quarterly NII effect vs Baseline used in the PPNR stress ($B)", "$B", ser, height=320), width="stretch")
    st.subheader("Economic value of equity: Basel standardised framework")
    if need(EV, "rate_risk/basel_eve_results.csv"):
        f = go.Figure()
        f.add_trace(go.Bar(x=EV.scenario, y=EV.after_hedges_pct_tier1, name="After documented fair-value hedges", marker_color="#3b82f6"))
        f.add_trace(go.Bar(x=EV.scenario, y=EV.gross_pct_tier1, name="Gross (no hedges)", marker_color="#9ca3af"))
        f.add_hline(y=-15, line=dict(color="#dc2626", dash="dash"), annotation_text="15% of Tier 1 outlier line", annotation_position="bottom right")
        f.update_layout(barmode="group", height=380, yaxis_title="Change in EVE, % of Tier 1", margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h", y=-0.3))
        st.plotly_chart(f, width="stretch")
        st.dataframe(EV.round(2), hide_index=True, width="stretch")
        st.caption("BAC does not disclose EVE. This is built from FR Y-9C balances and BAC 10-K fair values and fair-value hedges, with Basel's "
                   "shocks, buckets, deposit caps and prepayment multipliers (BCBS d368). Deposit split, core share, prepayment and coupons are assumptions.")
    if ES is not None:
        st.subheader("How much the worst-case EVE depends on the assumed inputs")
        st.dataframe(ES.round(1), hide_index=True, width="stretch")
    for nm, rel in [("NII assumptions and sources", "rate_risk/nii_eve_assumptions.csv"), ("EVE assumptions and sources", "rate_risk/basel_eve_assumptions.csv")]:
        A_ = load(rel)
        if A_ is not None:
            with st.expander(nm):
                st.dataframe(A_, hide_index=True, width="stretch")

# ------------------------------------------------------------------ Governance
with tabs[6]:
    st.subheader("Data windows")
    st.markdown("""
| Block | Window | Why |
|---|---|---|
| Segment loss models | 2000Q1-2025Q4 (104 quarters) | Includes the GFC and COVID |
| BAC loss scale factor | 2014Q1-2019Q4 fit, 2020-2025 test | 2011-2013 excluded: Countrywide-era legacy book ran 1.7x industry |
| PPNR | 2014Q1-2025Q4 | 2008-2013 income dominated by one-offs (goodwill, litigation, rep and warranty) |
| Allowance and CET1 start | 2025Q4 Y-9C | Matches the Fed's starting point |
| NII model | 2025Q4 balance sheet; fitted to 10-K sensitivities for 2023, 2024, 2025 | 30 disclosed points, 2025 held out as a test |
| EVE | 2025Q4 balance sheet | Basel standardised framework, fair-value hedges from the FY2025 10-K |
""")
    st.subheader("Known limitations")
    st.markdown("""
- PD/LGD from FRED are proxies (delinquency rate; charge-off over lagged delinquency). A stable PD error-correction model was rejected: delinquency is near unit-root.
- Segment calibration to BAC is not possible without BAC's loan-type history; only the total is scaled. CRE (model 1.5% vs Fed 9.3%) and mortgage (4.1% vs 1.3%) miss and are shown, not tuned.
- PPNR stress response relies on overlays; each is sourced to BAC 10-K history with a sensitivity range (sales and trading haircut set to zero). The NII effect comes from the fitted rate-risk model.
- Fee-income baseline path looks high versus the current run-rate.
- Balances are held flat; OCI, buybacks and deferred-tax limits are not modelled; the Fed's dividend assumption is not published in the results PDF.
- The Fed publishes no Adverse scenario. "Constructed Adverse" is a halfway interpolation between Baseline and Severely Adverse (price indices in logs); it is an assumption, and because losses respond non-linearly it comes out mild (CET1 minimum 11.2%).
""")
    for name, rel in [("Window report", "panel/window_report.txt"), ("Calibration report", "calibration/calibration_report.txt"),
                      ("Waterfall report", "waterfall/cet1_report.txt"), ("NII model report", "rate_risk/nii_eve_report.txt"),
                      ("Basel EVE report", "rate_risk/basel_eve_report.txt")]:
        t = text(rel)
        if t:
            with st.expander(name):
                st.code(t)
