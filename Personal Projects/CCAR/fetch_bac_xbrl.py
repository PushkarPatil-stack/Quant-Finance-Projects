"""
fetch_bac_xbrl.py  (v2)
------------------------
Pulls Bank of America's XBRL "company facts" from the SEC, audits the tags,
and builds a quarterly table by stitching tags across BAC's tagging changes.

    python fetch_bac_xbrl.py --email you@example.com          # download + build
    python fetch_bac_xbrl.py --local data/xbrl/bac_companyfacts.json   # rebuild only

Outputs in --outdir (default data/xbrl):
    bac_companyfacts.json   raw download (only when fetched)
    tag_audit.csv           every tag matching the keyword scan, with coverage
    bac_quarterly_xbrl.csv  quarterly table ($ billions; ratios in percent)
    resolution.txt          for each concept: tags used, coverage, overlap checks

Notes
 * The SEC API returns consolidated (undimensioned) values only. CET1 capital
   and RWA are not tagged that way for BAC, so they stay NOT TAGGED here and
   come from the Y-9C / filings instead.
 * Each concept is a chain of tags, newest first. A later tag only fills
   quarters the earlier ones lack. Where tags overlap, the script reports the
   largest difference, so a definition change shows up instead of hiding.
 * Quarterly flows: 3-month facts are used directly. Missing quarters are
   derived from year-to-date facts (6M-3M, 9M-6M, 12M-9M).
 * The same quarter is sometimes reported with start dates a day apart; the
   latest filing wins.
"""

import argparse
import json
import os
import re
import sys
import urllib.request

import pandas as pd

CIK = "0000070858"
URL = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{CIK}.json"
P = "FinancingReceivableExcludingAccruedInterest"

FLOW_CHAINS = {
    "nii": ["InterestIncomeExpenseNet"],
    "nonint_income": ["NoninterestIncome"],
    "nonint_expense": ["NoninterestExpense"],
    "net_income": ["NetIncomeLoss"],
    # loans-only provision from 2019; before 2019 the old tag also includes unfunded commitments
    "provision_loans": [P + "CreditLossExpenseReversal", "ProvisionForLoanLossesExpensed",
                        "ProvisionForLoanLeaseAndOtherLosses"],
    "provision_unfunded": ["OffBalanceSheetCreditLossLiabilityCreditLossExpenseReversal"],
    "chargeoffs": [P + "AllowanceForCreditLossWriteoff",
                   "FinancingReceivableAllowanceForCreditLossesWriteOffs",
                   "AllowanceForLoanAndLeaseLossesWriteOffs"],
    "recoveries": [P + "AllowanceForCreditLossRecovery",
                   "FinancingReceivableAllowanceForCreditLossesRecovery",
                   "AllowanceForLoanAndLeaseLossRecoveryOfBadDebts",
                   "AllowanceForLoanAndLeaseLossesRecoveriesOfBadDebts"],
    "nco_reported": [P + "AllowanceForCreditLossWriteoffAfterRecovery",
                     "FinancingReceivableAllowanceForCreditLossWriteoffAfterRecovery",
                     "AllowanceForLoanAndLeaseLossesWriteoffsNet"],
}
STOCK_CHAINS = {
    "allowance": ["FinancingReceivableAllowanceForCreditLossExcludingAccruedInterest",
                  "FinancingReceivableAllowanceForCreditLosses",
                  "LoansAndLeasesReceivableAllowance"],
    "total_assets": ["Assets"],
    "cet1_capital": ["CommonEquityTier1Capital", "CommonEquityTierOneCapital"],
    "rwa": ["RiskWeightedAssets"],
    "cet1_ratio": ["CommonEquityTier1CapitalRatio", "CommonEquityTierOneCapitalRatio"],
    "tier1_capital": ["Tier1Capital"],
}
# Known data fixes (found by checking the stitched series against the Y-9C and overlaps)
# - old recoveries tag is reported as a negative number from 2011 on -> use absolute value
SIGN_FIX_ABS = {"AllowanceForLoanAndLeaseLossesRecoveriesOfBadDebts"}
# - these observations are one-time opening balances dated 12/31 (CECL adoption, and the
#   2010 consolidation of off-balance-sheet vehicles), not year-end balances -> drop them
DROP_OBS = {
    "FinancingReceivableAllowanceForCreditLossExcludingAccruedInterest": ["2019-12-31"],
    "LoansAndLeasesReceivableAllowance": ["2009-12-31"],
}
# - charge-off / recovery values before this date came out inconsistent (2009-10 YTD scoping)
RELIABLE_FROM = {"chargeoffs": "2011-03-31", "recoveries": "2011-03-31",
                 "nco_reported": "2011-03-31"}

SCAN = re.compile(r"tier1|tierone|riskweighted|chargeoff|writeoff|recover|allowance|"
                  r"provisionfor|creditloss|expensereversal|interestincomeexpensenet|"
                  r"noninterest|netincomeloss", re.I)


def load_facts(args):
    os.makedirs(args.outdir, exist_ok=True)
    if args.local:
        with open(args.local) as f:
            return json.load(f)
    if not args.email:
        sys.exit("Provide --email (SEC requires it) or --local <json>.")
    req = urllib.request.Request(URL, headers={"User-Agent": f"research script {args.email}"})
    with urllib.request.urlopen(req, timeout=60) as r:
        raw = r.read()
    path = os.path.join(args.outdir, "bac_companyfacts.json")
    with open(path, "wb") as f:
        f.write(raw)
    print(f"Downloaded {len(raw) / 1e6:.1f} MB -> {path}")
    return json.loads(raw)


def to_frame(rows):
    df = pd.DataFrame(rows)
    for c in ("start", "end", "filed"):
        if c in df.columns:
            df[c] = pd.to_datetime(df[c])
    if "start" not in df.columns:
        df["start"] = pd.NaT
    df["days"] = (df["end"] - df["start"]).dt.days
    return df


def period_kind(days):
    if pd.isna(days):
        return "instant"
    for lo, hi, k in ((80, 100, "3M"), (170, 190, "6M"), (260, 280, "9M"), (355, 375, "12M")):
        if lo <= days <= hi:
            return k
    return "other"


def audit(cf):
    rows = []
    for taxo, tags in cf["facts"].items():
        for tag, obj in tags.items():
            if not (SCAN.search(tag) or SCAN.search(obj.get("label") or "")):
                continue
            for unit, facts in obj["units"].items():
                df = to_frame(facts)
                kinds = sorted({period_kind(d) for d in df["days"]})
                rows.append({"taxonomy": taxo, "tag": tag, "label": obj.get("label"),
                             "unit": unit, "n_facts": len(df),
                             "first_end": df["end"].min().date(), "last_end": df["end"].max().date(),
                             "period_kinds": "/".join(kinds)})
    return pd.DataFrame(rows).sort_values(["taxonomy", "tag"])


def get_rows(cf, tag):
    for taxo in ("us-gaap", *[t for t in cf["facts"] if t != "us-gaap"]):
        obj = cf["facts"].get(taxo, {}).get(tag)
        if obj:
            unit = "USD" if "USD" in obj["units"] else next(iter(obj["units"]))
            return taxo, unit, obj["units"][unit]
    return None


def dedupe_latest(df, keys):
    """Latest filing wins; also count periods whose value changed across filings."""
    changed = int((df.groupby(keys)["val"].nunique() > 1).sum())
    return df.sort_values("filed").drop_duplicates(keys, keep="last"), changed


def quarterly_flow(rows):
    df, changed = dedupe_latest(to_frame(rows), ["start", "end"])
    df = df.assign(kind=df["days"].map(period_kind))
    q = (df[df.kind == "3M"].sort_values("filed").drop_duplicates("end", keep="last")
         .set_index("end")["val"].copy())
    prev_kind = {"6M": "3M", "9M": "6M", "12M": "9M"}
    for k in ("6M", "9M", "12M"):
        prev = df[df.kind == prev_kind[k]]
        for _, r in df[df.kind == k].iterrows():
            if r["end"] in q.index:
                continue
            m = prev[(prev["start"] - r["start"]).abs().dt.days <= 3]
            if not m.empty:
                q[r["end"]] = r["val"] - m.sort_values("filed").iloc[-1]["val"]
    return q.sort_index(), changed


def quarter_end_stock(rows):
    df = to_frame(rows)
    df = df[df["start"].isna()]
    df, changed = dedupe_latest(df, ["end"])
    return df.set_index("end")["val"].sort_index(), changed


def build_chain(cf, concept, tags, is_flow):
    acc, notes = None, []
    for tag in tags:
        hit = get_rows(cf, tag)
        if not hit:
            continue
        s, changed = quarterly_flow(hit[2]) if is_flow else quarter_end_stock(hit[2])
        s = s[~s.index.duplicated(keep="last")]
        if tag in DROP_OBS:
            s = s.drop(pd.to_datetime(DROP_OBS[tag]), errors="ignore")
        if not len(s):
            continue
        if tag in SIGN_FIX_ABS:
            s = s.abs()
        if hit[1] == "USD":
            s = s / 1e9
        elif s.abs().max() < 1:
            s = s * 100  # ratio -> percent
        if acc is None:
            overlap = "primary"
            acc = s
        else:
            ov = acc.index.intersection(s.index)
            overlap = (f"overlap {len(ov)}q, max abs diff {(acc[ov] - s[ov]).abs().max():.3f}"
                       if len(ov) else "no overlap")
            acc = acc.combine_first(s)
        notes.append(f"    {tag}: {len(s)}q, {s.index.min().date()} to {s.index.max().date()}; "
                     f"{overlap}; periods restated across filings: {changed}")
    if acc is not None and concept in RELIABLE_FROM:
        acc = acc[acc.index >= pd.Timestamp(RELIABLE_FROM[concept])]
        notes.append(f"    (values before {RELIABLE_FROM[concept]} dropped as unreliable)")
    return acc, notes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--email")
    ap.add_argument("--local")
    ap.add_argument("--outdir", default="data/xbrl")
    args = ap.parse_args()

    cf = load_facts(args)
    print(f"Entity in file: {cf.get('entityName')} (CIK {cf.get('cik')})")
    aud = audit(cf)
    aud.to_csv(os.path.join(args.outdir, "tag_audit.csv"), index=False)
    print(f"Audit: {len(aud)} tag/unit rows matched -> tag_audit.csv")

    series, lines = {}, []
    for concept, tags in {**FLOW_CHAINS, **STOCK_CHAINS}.items():
        acc, notes = build_chain(cf, concept, tags, concept in FLOW_CHAINS)
        if acc is None:
            lines.append(f"{concept}: NOT TAGGED (tried {', '.join(tags)}) -> use Y-9C / filings")
            continue
        series[concept] = acc
        lines.append(f"{concept}: {len(acc)}q, {acc.index.min().date()} to {acc.index.max().date()}")
        lines += notes

    out = pd.DataFrame(series).sort_index()
    out = out[out.index.month.isin([3, 6, 9, 12])]
    if {"nii", "nonint_income", "nonint_expense"} <= set(out.columns):
        out["ppnr"] = out["nii"] + out["nonint_income"] - out["nonint_expense"]
    if {"chargeoffs", "recoveries"} <= set(out.columns):
        out["nco_total"] = out["chargeoffs"] - out["recoveries"]
        if "nco_reported" in out.columns:
            d = (out["nco_total"] - out["nco_reported"]).abs().max()
            lines.append(f"CHECK nco_total (chargeoffs-recoveries) vs nco_reported: max abs diff {d:.3f}")
    if {"provision_loans", "provision_unfunded"} <= set(out.columns):
        out["provision_total"] = out["provision_loans"] + out["provision_unfunded"].fillna(0)
    if "total_assets" in out.columns and out["total_assets"].dropna().iloc[-1] < 1000:
        lines.append("WARNING: latest total assets under $1T; this may not be BAC consolidated data")

    out.index.name = "quarter_end"
    out.round(4).to_csv(os.path.join(args.outdir, "bac_quarterly_xbrl.csv"))
    with open(os.path.join(args.outdir, "resolution.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nWrote bac_quarterly_xbrl.csv ({len(out)} rows).")


if __name__ == "__main__":
    main()
