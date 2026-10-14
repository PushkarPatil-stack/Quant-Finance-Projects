"""
fetch_fred_losses.py
---------------------
Pulls the loss-side series from FRED (charge-off rates, delinquency rates,
optional loan balances), converts everything to quarterly, and writes an
audit of what was found and over which dates.

Run on your laptop (FRED is not reachable from Claude's sandbox):

    py fetch_fred_losses.py --api-key YOUR_FRED_KEY
    py fetch_fred_losses.py                      # no key: works, but no series titles

A free key (fred.stlouisfed.org > My Account > API Keys) is recommended: it
returns each series' title, units and seasonal adjustment, so you can confirm
that an ID really is "top 100 banks" and not "all commercial banks".

IMPORTANT: the series IDs below are written from memory and have NOT been
verified. Each concept lists candidates (top-100 banks first, all-commercial-
banks as fallback); the first one that returns data wins, and every attempt is
recorded in fred_audit.csv. If a concept ends up NOT FOUND, or resolves to a
fallback you don't want, supply the right ID yourself:

    py fetch_fred_losses.py --extra co_cre=SERIESID --extra dq_mortgage=SERIESID

Outputs (in --outdir, default data/fred):
    raw/<ID>.csv + <ID>.meta.json   cache of every series fetched (re-runs reuse it;
                                    use --refresh to download again)
    fred_audit.csv                  every candidate tried: status, scope, title, dates
    fred_losses_quarterly.csv       one column per concept, quarterly (fred_<concept>)
"""

import argparse
import io
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

import pandas as pd

# concept -> (kind, [candidate FRED IDs: preferred first, fallback after])
SERIES = {
    "co_cards": ("loss", ["CORCCT100S", "CORCCACBS"]),
    "co_business": ("loss", ["CORBLT100S", "CORBLACBS"]),
    "co_cre": ("loss", ["CORCREXFT100S", "CORCREXFACBS"]),
    "co_mortgage": ("loss", ["CORSFRMT100S", "CORSFRMACBS"]),
    "co_other_consumer": ("loss", ["COROCLT100S", "COROCLACBS"]),
    "dq_cards": ("loss", ["DRCCLT100S", "DRCCLACBS"]),
    "dq_business": ("loss", ["DRBLT100S", "DRBLACBS"]),
    "dq_cre": ("loss", ["DRCRELEXFT100S", "DRCRELEXFACBS"]),
    "dq_mortgage": ("loss", ["DRSFRMT100S", "DRSFRMACBS"]),
    # optional balances (all commercial banks), used only for weighting
    "bal_ci": ("balance", ["BUSLOANS"]),
    "bal_real_estate": ("balance", ["REALLN"]),
    "bal_cards": ("balance", ["CCLACBW027SBOG"]),
}
MIN_QUARTERS = 20          # a series shorter than this is treated as "too short"
# observations needed in a quarter before a daily/weekly/monthly mean is trusted
MIN_OBS_PER_QUARTER = {"D": 55, "W": 12, "M": 3, "Q": 1}
UA = "calibration-data-pull research script"


def http_get(url, timeout=30):
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode("utf-8", errors="replace")


def parse_fredgraph(text):
    """fredgraph.csv: first column is the date, second the value ('.' or blank = missing)."""
    df = pd.read_csv(io.StringIO(text))
    if df.shape[1] < 2:
        raise ValueError("unexpected fredgraph format")
    s = pd.Series(pd.to_numeric(df.iloc[:, 1], errors="coerce").values,
                  index=pd.to_datetime(df.iloc[:, 0], errors="coerce"))
    return s[s.index.notna()].dropna()


def parse_api(meta_json, obs_json):
    meta = json.loads(meta_json)["seriess"][0]
    obs = json.loads(obs_json)["observations"]
    vals = pd.to_numeric(pd.Series([o["value"] for o in obs], dtype="object"), errors="coerce")
    s = pd.Series(vals.values, index=pd.to_datetime([o["date"] for o in obs]))
    info = {"title": meta.get("title"), "freq": meta.get("frequency_short"),
            "units": meta.get("units_short"), "sa": meta.get("seasonal_adjustment_short")}
    return s.dropna(), info


def fetch_remote(sid, api_key):
    """Download one series from FRED. Returns (series, info). Raises on any failure."""
    if api_key:
        base = "https://api.stlouisfed.org/fred/series"
        q = urllib.parse.urlencode({"series_id": sid, "api_key": api_key, "file_type": "json"})
        meta_json = http_get(f"{base}?{q}")
        obs_json = http_get(f"{base}/observations?{q}")
        return parse_api(meta_json, obs_json)
    return parse_fredgraph(http_get(
        f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}")), {}


def get_series(sid, api_key, cache_dir, refresh):
    """Return (series, info). Uses the cache unless refresh is set."""
    csv_path = os.path.join(cache_dir, f"{sid}.csv")
    meta_path = os.path.join(cache_dir, f"{sid}.meta.json")
    if os.path.exists(csv_path) and not refresh:
        df = pd.read_csv(csv_path, parse_dates=["date"])
        s = pd.Series(df["value"].values, index=df["date"]).dropna()
        info = json.load(open(meta_path)) if os.path.exists(meta_path) else {}
        return s, info
    s, info = fetch_remote(sid, api_key)
    if s.empty:
        raise ValueError("no observations")
    os.makedirs(cache_dir, exist_ok=True)
    pd.DataFrame({"date": s.index, "value": s.values}).to_csv(csv_path, index=False)
    json.dump(info, open(meta_path, "w"))
    return s, info


def explain_failure(e, api_key):
    """Plain-English reason for a failed download."""
    if isinstance(e, urllib.error.HTTPError):
        try:
            body = e.read().decode("utf-8", errors="replace")[:200].replace("\n", " ")
        except Exception:
            body = ""
        if e.code in (400, 401, 403) and api_key:
            return (f"FRED rejected the request (HTTP {e.code}). Most likely the API key is wrong: use the "
                    f"real 32-character key from your FRED account, with no quotes, not the word "
                    f"YOUR_KEY. Server said: {body}")
        if e.code == 429:
            return "FRED is rate-limiting you (HTTP 429). Wait a few minutes and run again."
        return f"HTTP {e.code} from FRED. Server said: {body}"
    reason = getattr(e, "reason", e)
    text = str(reason)
    if "CERTIFICATE" in text.upper() or "SSL" in text.upper():
        return (f"Secure connection failed ({text}). This usually means a company network, VPN or "
                f"antivirus is inspecting HTTPS traffic. Try another network or turn the VPN off.")
    return (f"Could not reach FRED ({text}). Check your internet connection, VPN or firewall; "
            f"also try opening https://fred.stlouisfed.org in a browser.")


def preflight(api_key):
    """Download one well-known series to confirm the connection and key work."""
    try:
        s, _ = fetch_remote("UNRATE", api_key)
        if s.empty:
            raise ValueError("empty response")
    except Exception as e:
        sys.exit("\nCannot download from FRED, so nothing was attempted.\n  " +
                 explain_failure(e, api_key) +
                 "\n(Use --no-preflight to skip this check when working from the cache.)")
    print(f"Connection check OK (UNRATE: {len(s)} observations, last {s.index.max().date()}).")


def infer_freq(s):
    d = s.index.to_series().diff().dt.days.median()
    return "D" if d <= 4 else "W" if d <= 10 else "M" if d <= 45 else "Q"


def to_quarterly(s, freq):
    freq = (freq or infer_freq(s))[:1].upper()
    freq = freq if freq in MIN_OBS_PER_QUARTER else infer_freq(s)
    p = s.index.to_period("Q")
    if freq == "Q":
        q = pd.Series(s.values, index=p)
        return q[~q.index.duplicated(keep="last")], freq
    g = s.groupby(p)
    q = g.mean()[g.count() >= MIN_OBS_PER_QUARTER[freq]]   # drops incomplete quarters
    return q, freq


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api-key", default=os.environ.get("FRED_API_KEY"))
    ap.add_argument("--outdir", default="data/fred")
    ap.add_argument("--refresh", action="store_true", help="ignore the cache and download again")
    ap.add_argument("--no-preflight", action="store_true", help="skip the connection check")
    ap.add_argument("--extra", action="append", default=[],
                    help="CONCEPT=SERIESID, tried before the built-in candidates (repeatable)")
    args = ap.parse_args()

    series = {k: (v[0], list(v[1])) for k, v in SERIES.items()}
    for item in args.extra:
        concept, sid = item.split("=", 1)
        kind, cands = series.get(concept, ("loss", []))
        series[concept] = (kind, [sid.strip()] + cands)

    cache_dir = os.path.join(args.outdir, "raw")
    os.makedirs(cache_dir, exist_ok=True)
    audit, out = [], {}
    all_ids = [sid for _, (_, c) in series.items() for sid in c]
    if not args.no_preflight and (args.refresh or any(
            not os.path.exists(os.path.join(cache_dir, f"{i}.csv")) for i in all_ids)):
        preflight(args.api_key)

    for concept, (kind, cands) in series.items():
        resolved = False
        for sid in cands:
            row = {"concept": concept, "series_id": sid, "kind": kind,
                   "scope": ("top100" if "T100" in sid else
                             "all_banks" if ("ACB" in sid or kind == "balance") else "other"),
                   "status": "", "used": False, "title": "", "freq": "", "units": "", "sa": "",
                   "first_obs": "", "last_obs": "", "n_obs": "", "first_quarter": "",
                   "last_quarter": "", "n_quarters": ""}
            if resolved:
                row["status"] = "not tried (earlier candidate resolved)"
                audit.append(row)
                continue
            cached = os.path.exists(os.path.join(cache_dir, f"{sid}.csv")) and not args.refresh
            try:
                s, info = get_series(sid, args.api_key, cache_dir, args.refresh)
                q, freq = to_quarterly(s, info.get("freq"))
            except (urllib.error.URLError, ValueError, KeyError, IndexError, OSError) as e:
                why = (f"HTTP {e.code} {e.reason}" if isinstance(e, urllib.error.HTTPError)
                       else f"{type(e).__name__}: {getattr(e, 'reason', e)}")
                row["status"] = f"not available ({why})"
                audit.append(row)
                print(f"  {concept:18s} {sid:16s} not available ({why})")
                if not cached:
                    time.sleep(0.3)
                continue
            row.update({"title": info.get("title", ""), "freq": freq, "units": info.get("units", ""),
                        "sa": info.get("sa", ""), "first_obs": s.index.min().date(),
                        "last_obs": s.index.max().date(), "n_obs": len(s),
                        "first_quarter": str(q.index.min()), "last_quarter": str(q.index.max()),
                        "n_quarters": len(q)})
            if len(q) < MIN_QUARTERS:
                row["status"] = f"too short ({len(q)} quarters)"
                audit.append(row)
                continue
            row["status"], row["used"] = "ok", True
            audit.append(row)
            out[f"fred_{concept}"] = q
            resolved = True
            print(f"  {concept:18s} {sid:16s} {q.index.min()} to {q.index.max()} ({len(q)}q)")
        if not resolved:
            print(f"  {concept:18s} NOT FOUND from {', '.join(cands)}")

    aud = pd.DataFrame(audit)
    aud.to_csv(os.path.join(args.outdir, "fred_audit.csv"), index=False)
    if not out:
        sys.exit("No series could be loaded. Check your internet connection or pass --extra IDs.")
    panel = pd.DataFrame(out).sort_index()
    panel.index = panel.index.astype(str)
    panel.index.name = "quarter"
    panel.round(6).to_csv(os.path.join(args.outdir, "fred_losses_quarterly.csv"))

    used = aud[aud["used"]]
    print(f"\nResolved {len(used)} of {len(series)} concepts. Wrote fred_audit.csv and "
          f"fred_losses_quarterly.csv ({len(panel)} quarters).")
    fallbacks = used[(used["kind"] == "loss") & (used["scope"] != "top100")]
    if len(fallbacks):
        print("Loss series using all-banks (not top-100) fallbacks: " + ", ".join(fallbacks["concept"]))
    if not args.api_key:
        print("No API key given, so titles are blank. Confirm series meaning on fred.stlouisfed.org "
              "or re-run with --api-key.")


if __name__ == "__main__":
    main()
