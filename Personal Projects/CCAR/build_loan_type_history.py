"""
build_loan_type_history.py
--------------------------
Quarterly net charge-off rates by loan type for Bank of America from the FR Y-9C (Schedules HI-B Part I and HC-C).

    py build_loan_type_history.py        (needs data/y9c/bac_y9c_history.csv from fetch_bhc_history.py; optionally the
                                          FRY9C_1073757_YYYYMMDD.csv files in data/y9c for 2021Q2 onward)

HOW
  * Charge-offs and recoveries on Schedule HI-B are YEAR-TO-DATE, so quarterly = YTD(t) - YTD(t-1) within a year (Q1 = YTD).
  * Net charge-off rate (annualised, %) = 4 x quarterly net charge-offs / average of opening and closing balance.
  * Y-9C amounts are in $ thousands.
GROUPS (mnemonics are the same in the bulk history and in the per-quarter files)
  CRE       construction (C891, C893) + multifamily (3588) + non-farm non-residential (C895, C897)  [recoveries: C892, C894, 3589, C896, C898]
            balances: F158 + F159 + BHDM1460 + F160 + F161                      (domestic real estate only; foreign RE is tiny)
  MORTGAGE  open-end (5411/5412) + closed-end first lien (C234/C217) + junior lien (C235/C218)
            balances: BHDM1797 + BHDM5367 + BHDM5368
  CARDS     B514 / B515 ; balance B538
  C&I       4645 + 4646 / 4617 + 4618 ; balance 1763 + 1764
KNOWN BREAKS: 2008-2013 includes the Countrywide and Merrill portfolios; loan-category definitions changed in 2008 and 2010-12.
Output: data/models/loan_type_history.csv  (quarter, <group>_nco_b, <group>_bal_b, <group>_nco_rate_pct)
"""
import glob
import os

import pandas as pd

GROUPS = {
    "cre": dict(co=["C891", "C893", "3588", "C895", "C897"], rec=["C892", "C894", "3589", "C896", "C898"],
                bal=["F158", "F159", "DM1460", "F160", "F161"]),
    "mortgage": dict(co=["5411", "C234", "C235"], rec=["5412", "C217", "C218"], bal=["DM1797", "DM5367", "DM5368"]),
    "cards": dict(co=["B514"], rec=["B515"], bal=["B538"]),
    "business": dict(co=["4645", "4646"], rec=["4617", "4618"], bal=["1763", "1764"]),
}


def col(code):
    return ("BH" + code) if code.startswith("DM") else ("BHCK" + code)


def load(dirpath):
    H = pd.read_csv(os.path.join(dirpath, "bac_y9c_history.csv"), dtype=str, low_memory=False).set_index("quarter")
    extra = []
    for f in sorted(glob.glob(os.path.join(dirpath, "FRY9C_1073757_*.csv"))):
        d = pd.read_csv(f, dtype=str)
        q = os.path.basename(f)[-12:-4]
        qq = f"{q[:4]}Q{int(q[4:6]) // 3}"
        if qq in H.index:
            continue
        extra.append(d.drop_duplicates("ItemName").set_index("ItemName")["Value"].rename(qq))
    if extra:
        H = pd.concat([H, pd.concat(extra, axis=1).T], sort=False)
    return H.sort_index()


def num(H, codes):
    cols = [col(c) for c in codes]
    miss = [c for c in cols if c not in H.columns]
    if miss:
        raise KeyError(f"missing in history: {miss}")
    return H[cols].apply(pd.to_numeric, errors="coerce").fillna(0).sum(axis=1)


def quarterly_from_ytd(ytd):
    """YTD -> quarterly. Q1 = YTD; Q2-Q4 = YTD minus the previous quarter's YTD, or NaN when that quarter's file is missing."""
    out = ytd.copy().astype(float)
    for i, q in enumerate(ytd.index):
        if q.endswith("Q1"):
            continue
        prev = f"{q[:4]}Q{int(q[-1]) - 1}"
        out.iloc[i] = ytd.iloc[i] - ytd.loc[prev] if prev in ytd.index else float("nan")
    return out


SUB = {   # sub-portfolios for the worst-window table: (charge-off codes, recovery codes, balance codes)
    "CRE: construction": (["C891", "C893"], ["C892", "C894"], ["F158", "F159"]),
    "CRE: multifamily": (["3588"], ["3589"], ["DM1460"]),
    "CRE: non-farm non-residential": (["C895", "C897"], ["C896", "C898"], ["F160", "F161"]),
    "Mortgage: closed-end first lien": (["C234"], ["C217"], ["DM5367"]),
    "Mortgage: closed-end junior lien": (["C235"], ["C218"], ["DM5368"]),
    "Mortgage: home equity (open-end)": (["5411"], ["5412"], ["DM1797"]),
    "Credit cards": (["B514"], ["B515"], ["B538"]),
    "Commercial and industrial": (["4645", "4646"], ["4617", "4618"], ["1763", "1764"]),
}


def rate_series(H, co, rec, bal):
    net = quarterly_from_ytd(num(H, co) - num(H, rec)) / 1e6
    bl = num(H, bal) / 1e6
    return 4 * net / (bl + bl.shift(1)).div(2).fillna(bl) * 100, bl


def worst_windows(H, lo="2007Q1", hi="2021Q1", n=9):
    """Worst n-quarter cumulative net charge-off rate (sum of annualised rates / 4) for each sub-portfolio, and the latest balance."""
    rows = []
    for name, (co, rec, bal) in SUB.items():
        r, bl = rate_series(H, co, rec, bal)
        r = r.loc[lo:hi]
        w = r.rolling(n).sum() / 4
        rows.append(dict(portfolio=name, worst_9q_cum_pct=w.max(), window_ends=w.idxmax(), peak_quarter_rate_pct=r.max(),
                         peak_quarter=r.idxmax(), balance_latest_b=bl.dropna().iloc[-1]))
    return pd.DataFrame(rows)


def allowance_history(H):
    net = quarterly_from_ytd(num(H, ["4635"]) - num(H, ["4605"])) / 1e6
    A, L = num(H, ["3123"]) / 1e6, num(H, ["B528"]) / 1e6
    df = pd.DataFrame({"allowance_b": A, "loans_b": L, "net_chargeoffs_q_b": net})
    df["coverage_pct"] = df.allowance_b / df.loans_b * 100
    df["next8q_net_chargeoffs_b"] = df.net_chargeoffs_q_b[::-1].rolling(8).sum()[::-1].shift(-1)
    return df


def main():
    H = load("data/y9c")
    res = pd.DataFrame(index=H.index)
    for g, c in GROUPS.items():
        net_ytd = num(H, c["co"]) - num(H, c["rec"])
        net_q = quarterly_from_ytd(net_ytd) / 1e6                       # $B
        bal = num(H, c["bal"]) / 1e6                                    # $B
        avg = (bal + bal.shift(1)).div(2).fillna(bal)
        res[f"{g}_nco_b"] = net_q
        res[f"{g}_bal_b"] = bal
        res[f"{g}_nco_rate_pct"] = 4 * net_q / avg * 100
    os.makedirs("data/models", exist_ok=True)
    res.round(4).to_csv("data/models/loan_type_history.csv", index_label="quarter")
    print(res[[c for c in res.columns if c.endswith("rate_pct")]].round(2).to_string())
    W = worst_windows(H)
    W.round(2).to_csv("data/models/loan_type_worst_windows.csv", index=False)
    cre = W[W.portfolio.str.startswith("CRE")]
    mix = (cre.worst_9q_cum_pct * cre.balance_latest_b).sum() / cre.balance_latest_b.sum()
    print("\nWorst 9-quarter cumulative net charge-off rate, 2007Q1-2021Q1 (%):\n" + W.round(2).to_string(index=False))
    print(f"\nBAC's own worst CRE experience weighted to the latest CRE mix: {mix:.2f}% (Fed severe: 9.3%)")
    A = allowance_history(H)
    A.loc["2007Q1":"2021Q1"].round(2).to_csv("data/models/allowance_history.csv", index_label="quarter")
    print("\nAllowance and coverage (selected):\n" + A.loc[[q for q in ["2007Q4", "2008Q4", "2009Q4", "2010Q4", "2011Q4", "2013Q4", "2019Q4", "2020Q2", "2020Q4"] if q in A.index]].round(2).to_string())
    return res


if __name__ == "__main__":
    main()
