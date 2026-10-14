"""
fetch_bhc_history.py
--------------------
Builds a quarterly FR Y-9C history for Bank of America (RSSD 1073757) from the Chicago Fed's public holding-company files
(one file per quarter, all holding companies, 1986Q3-2021Q1):
    https://www.chicagofed.org/~/media/others/banking/financial-institution-reports/bhc-data/bhcfYYMM.csv
Each file is downloaded, Bank of America's row is kept (every column, so no item codes are guessed), and the big file is deleted.

    py fetch_bhc_history.py                      downloads 2007Q1-2021Q1 and writes data/y9c/bac_y9c_history.csv
    py fetch_bhc_history.py --start 2005Q1       change the window
    py fetch_bhc_history.py --local-dir C:\\files   use bhcfYYMM.csv files you already downloaded (no internet needed)

Quarters from 2021Q2 on are not on that page; use the FFIEC National Information Center "Financial Data Download" for BAC
and keep them in data/y9c as FRY9C_1073757_YYYYMMDD.csv (same format as your 2023-2025 files). build_loan_type_history.py
combines both.
Output: data/y9c/bac_y9c_history.csv  (one row per quarter, columns = Y-9C mnemonics, e.g. BHCKB514)
"""
import argparse
import io
import os
import sys
import urllib.request

import pandas as pd

URL = "https://www.chicagofed.org/~/media/others/banking/financial-institution-reports/bhc-data/bhcf{yy}{mm}.csv"
RSSD = "1073757"


def quarters(start, end):
    y, q = int(start[:4]), int(start[-1])
    ye, qe = int(end[:4]), int(end[-1])
    while (y, q) <= (ye, qe):
        yield y, q
        y, q = (y, q + 1) if q < 4 else (y + 1, 1)


def bac_row(raw_bytes):
    """Return Bank of America's row from one bhcf file as a one-row DataFrame (all columns)."""
    text = raw_bytes.decode("latin-1")
    first = text.split("\n", 1)[0]
    sep = "^" if first.count("^") > first.count(",") else ("\t" if first.count("\t") > first.count(",") else ",")
    df = pd.read_csv(io.StringIO(text), sep=sep, dtype=str, low_memory=False)
    df.columns = [c.strip().strip('"').upper() for c in df.columns]
    key = next((c for c in ("RSSD9001", "RSSD_ID", "RSSD") if c in df.columns), None)
    if key is None:
        raise ValueError("no RSSD9001 column; columns start with " + ", ".join(df.columns[:8]))
    return df[df[key].str.strip() == RSSD]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2007Q1")
    ap.add_argument("--end", default="2021Q1")
    ap.add_argument("--local-dir", default=None)
    ap.add_argument("--out", default="data/y9c/bac_y9c_history.csv")
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    rows, missing = [], []
    for y, q in quarters(a.start, a.end):
        yy, mm = f"{y % 100:02d}", f"{q * 3:02d}"
        name = f"bhcf{yy}{mm}.csv"
        try:
            if a.local_dir:
                with open(os.path.join(a.local_dir, name), "rb") as f:
                    raw = f.read()
            else:
                req = urllib.request.Request(URL.format(yy=yy, mm=mm), headers={"User-Agent": "Mozilla/5.0 (research script)"})
                raw = urllib.request.urlopen(req, timeout=120).read()
            r = bac_row(raw)
            if r.empty:
                raise ValueError("Bank of America row not found")
            r.insert(0, "quarter", f"{y}Q{q}")
            rows.append(r.iloc[[0]])
            print(f"{y}Q{q}: ok ({r.shape[1]} columns)")
        except Exception as e:                      # keep going; report at the end
            missing.append((f"{y}Q{q}", str(e)[:80]))
            print(f"{y}Q{q}: FAILED {str(e)[:80]}")
    if not rows:
        sys.exit("nothing downloaded")
    H = pd.concat(rows, ignore_index=True)           # different years have different columns -> union, blanks where absent
    H.to_csv(a.out, index=False)
    print(f"\nwrote {a.out}: {len(H)} quarters, {H.shape[1]} columns")
    if missing:
        print("missing:", missing)


if __name__ == "__main__":
    main()
