"""CRE current-book cross-check.
Inputs: (1) BAC 2025 10-K CRE tables, transcribed below ($ millions, 31 Dec 2025 unless stated);
        (2) Fed 2026 'Detailed Nine Quarter Paths' CSV (AGGREGATE of all banks, not BAC) in data/raw/.
Outputs: data/models/cre_current_book.csv and cre_current_book.txt
Run: py cre_current_book_check.py
"""
import csv, os, io
import pandas as pd

FED9Q = "data/raw/2026_Detailed_Nine_Quarter_Paths.csv"
OUT = "data/models"; os.makedirs(OUT, exist_ok=True)

# ---- BAC 10-K 2025 (screenshots: CRE by property type; Commercial credit quality by vintage; Table 30; MBFD) ----
prop = {"Industrial/Warehouse": (13031, 13166), "Office": (12447, 15061), "Multi-family rental": (10986, 11022),
        "Shopping centers/Retail": (6947, 5603), "Hotel/Motels": (4629, 4680), "Multi-use": (2509, 2162),
        "Other non-residential": (17295, 13179), "Residential": (904, 857)}
TOTAL25, TOTAL24 = 68748, 65730
vint = {"2025": (11698, 5), "2024": (5856, 249), "2023": (4784, 366), "2022": (10430, 2294),
        "2021": (8161, 1986), "Prior": (16670, 2874), "Revolving": (11149, 539)}   # (total, reservable criticized)
PASS, CRIT, GROSS_CO = 60435, 8313, 520
NPL25, NPL24, ACC90_25 = 1596, 2068, 10
MBFD25, MBFD24, MBFD90_25, MBFD90_24 = 1978, 2393, 558, 203
FED_CRE_LOSS_B, FED_CRE_RATE = 7.1, 0.093      # Fed Table 10, BAC severe
MODEL_CRE_RATE = 0.0378                         # refit_cre_bac.py severe, matching spec (range 2.3-3.8%)

rows = []
def add(section, item, value, note=""): rows.append(dict(section=section, item=item, value=value, note=note))

assert sum(v[0] for v in prop.values()) == TOTAL25 and sum(v[1] for v in prop.values()) == TOTAL24
for k, (a, b) in prop.items():
    add("property_mix", k, round(a/TOTAL25*100, 1), f"% of CRE; 2024 {b/TOTAL24*100:.1f}%; balance {a} vs {b} $M ({(a/b-1)*100:+.1f}%)")
add("book", "reservable criticized % of CRE", round(CRIT/TOTAL25*100, 1), f"{CRIT} of {TOTAL25}")
add("book", "nonperforming % of CRE 2025", round(NPL25/TOTAL25*100, 2), f"2024 {NPL24/TOTAL24*100:.2f}%")
add("book", "accruing 90+ past due % 2025", round(ACC90_25/TOTAL25*100, 3), "")
add("book", "gross charge-off rate 2025 %", round(GROSS_CO/TOTAL25*100, 2), f"{GROSS_CO} $M")
add("book", "modified loans (borrowers in difficulty) % 2025", round(MBFD25/TOTAL25*100, 2), f"2024 {MBFD24/TOTAL24*100:.2f}% (10-K reports 2.88 / 3.64)")
add("book", "modified loans 90+ past due $M", MBFD90_25, f"2024 {MBFD90_24}; share of modified {MBFD90_25/MBFD25*100:.0f}% vs {MBFD90_24/MBFD24*100:.0f}%")
for k, (t, c) in vint.items():
    add("criticized_by_vintage", k, round(c/t*100, 1), f"% criticized; {c} of {t} $M")

# ---- Fed aggregate 9Q path: SHAPE only ----
raw = list(csv.reader(open(FED9Q, encoding="utf-8-sig")))
hdr = next(i for i, r in enumerate(raw) if r and r[0] == "Item")
fed = {r[0]: [float(x) for x in r[1:10]] for r in raw[hdr+1:] if len(r) >= 10 and r[0] and r[1] not in ("",)}
def shape(name, key):
    s = fed[key]; pk = max(range(9), key=lambda i: s[i])
    add("fed_aggregate_shape", name, f"peak PQ{pk+1}; PQ9/peak = {s[8]/s[pk]:.2f}", f"9Q total {sum(s):.1f}; path {s}")
shape("loan losses total", "Loan losses, total")
shape("loan losses CRE", "Loan losses, commercial real estate, domestic")
shape("loan losses C&I", "Loan losses, commercial and industrial")
shape("loan losses credit cards", "Loan losses, credit cards")
L, P = fed["Loan losses, total"], fed["Provisions for loan and lease losses"]
first_rel = next(i for i in range(9) if P[i] < L[i]); 
add("fed_aggregate_shape", "provisions minus loan losses by quarter", "; ".join(f"{p-l:+.1f}" for p, l in zip(P, L)),
    f"9Q build {sum(P)-sum(L):+.1f} on losses {sum(L):.1f} = {(sum(P)-sum(L))/sum(L)*100:.1f}% (BAC Fed: 4.6/64.9 = 7.1%); provisions fall below losses from PQ{first_rel+1}")
add("fed_aggregate_shape", "CRE share of aggregate loan losses %", round(sum(fed['Loan losses, commercial real estate, domestic'])/sum(L)*100, 1), "")

# ---- Illustrative: what the Fed's CRE loss means for BAC's current book ----
fed_total = FED_CRE_LOSS_B * 1000
model_total = MODEL_CRE_RATE * TOTAL25
add("bridge", "Fed CRE 9Q loss $M / balance", round(fed_total/TOTAL25*100, 1), f"{fed_total:.0f} $M on {TOTAL25}; Fed's 9.3% implies avg balance {fed_total/FED_CRE_RATE:,.0f} $M (definition/balance differs from 10-K)")
add("bridge", f"model CRE loss $M at {MODEL_CRE_RATE*100:.2f}%", round(model_total), "")
for lgd in (0.25, 0.35, 0.45):
    c = CRIT*lgd
    add("bridge", f"criticized loans lost at LGD {int(lgd*100)}% $M", round(c),
        f"= {c/model_total*100:.0f}% of model loss, {c/fed_total*100:.0f}% of Fed loss; "
        f"remaining to Fed {fed_total-c:.0f} $M = {(fed_total-c)/PASS*100:.1f}% of pass-rated; remaining to model {model_total-c:.0f} $M = {(model_total-c)/PASS*100:.1f}%")
add("bridge", "office at 20/30/40% loss $M", "; ".join(str(round(prop['Office'][0]*x)) for x in (.2, .3, .4)), "illustrative")

df = pd.DataFrame(rows); df.to_csv(f"{OUT}/cre_current_book.csv", index=False)
with open(f"{OUT}/cre_current_book.txt", "w") as f:
    for r in rows: f.write(f"[{r['section']}] {r['item']}: {r['value']}  {r['note']}\n")
print(open(f"{OUT}/cre_current_book.txt").read())
