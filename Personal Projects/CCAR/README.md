# Bank of America capital stress test (real-data pipeline)

Projects BAC's credit losses, pre-provision net revenue (PPNR), allowance and CET1 ratio under the Fed's 2026
supervisory Baseline and Severely Adverse scenarios, using public data only. The Fed's published BAC result is
used as an outside benchmark and is never fitted to.

## Result (severely adverse, 2026Q1-2028Q1)

| | This model | Fed, BAC |
|---|---|---|
| CET1 ratio, start | 11.36% | 11.4% |
| CET1 ratio, minimum | 10.26% | 9.9% |
| CET1 ratio, 2028Q1 | 10.30% | 10.8% |
| PPNR | $67.6B | $66.2B |
| Provisions | $75.3B | $69.5B |
| Pre-tax income | -$16.8B | -$12.5B |

The allowance and capital-action steps follow the Fed's published rules (Fed supervisory stress test methodology, "Overview of
Modeling Framework"; 12 CFR 252.44(c)): the allowance at the end of each quarter equals the projected losses of the next four
quarters (the Fed projects 13 quarters of losses), the difference between that and the firm's starting allowance is smoothed
linearly into provisions over the nine quarters, preferred dividends are paid, there are no repurchases or issuance, and no common
dividends after the first quarter. An earlier version used a calibrated 7.2-quarter horizon and continuing common dividends
(CET1 minimum 8.83%, end 8.89%); it is kept as a sensitivity row in `data/waterfall/cet1_allowance_sensitivity.csv`.
Neither the Fed's 9.9% minimum nor this model's is the pass/fail test: the 10.0% line (4.5 + 3.0 G-SIB + 2.5 SCB) is a reference,
and the Fed's own minimum is 0.1 below it.

Where the remaining gaps come from: PPNR is $1.4B above the Fed's. Provisions are $5.8B above: loan losses are $5.6B below the Fed's
($59.3B vs $64.9B) and the allowance build is $11.4B above ($16.0B vs $4.6B), because the model's losses stay high through the
13th quarter (unemployment is still 9.4% in 2028Q1) so the four-quarter forward allowance at the end is $29.2B against the Fed's implied
$17.8B. OCI of +$4.7B is not modelled (+0.26pp of CET1). `data/waterfall/cet1_bridge_to_fed.csv` shows the remaining steps.
The Fed holds balance sheets constant and RWA flat ($1,772.9B to $1,771.1B), as this model does.

Loss-timing sensitivity: `build_cet1_waterfall.py` also reads the Fed's aggregate nine-quarter file (`data/raw/2026_Detailed_Nine_Quarter_Paths.csv`) and re-times the severe loss path to the Fed's shape (peak in quarter 5, about 75% of peak by quarter 9), keeping BAC's own 9Q total; quarters 10-13 continue at the Fed's last decline. End allowance falls from $29.2B to $16.9B (Fed implied $17.8B), provisions to $63.0B (Fed $69.5B), severe CET1 minimum/end to 10.74%/10.86% (Fed 9.9%/10.8%). It is a sensitivity, not the base, because the shape is the all-bank aggregate and not BAC's. It shows the reserve gap is mostly loss timing.

Correction note: an earlier version of the PPNR step applied the NII rate sensitivity with the wrong sign (lower rates raised
NII). That inflated severe PPNR to $82B and made pre-tax income appear to match the Fed's; the offset was an error, not a
result. The NII stress now comes from the fitted NII model (below).

## Rate risk (NII and EVE)

| | Result |
|---|---|
| NII, +100bp / -100bp (12m, dynamic deposits) | +$0.9B / -$2.1B (BAC discloses +$0.7B / -$2.0B) |
| NII fit | 30 disclosed points (10-K Tables 44/45, 2023-2025): static RMSE $0.28B, dynamic $0.22B |
| NII out-of-sample | fitted without 2025, predicts 2025 at RMSE $0.41B (largest miss: +200bp) |
| EVE, Basel standardised, worst scenario | -$32.0B = 14.1% of Tier 1 (parallel down); gross of hedges -$35.8B = 15.7% |

EVE is an estimate from public data and Basel's standardised parameters (BCBS d368); BAC does not disclose EVE. Its plausible
range across the unobservable inputs (deposit retail share and stability, prepayment, hedged share of AFS) is about 9% to 23% of
Tier 1, straddling the 15% supervisory outlier line. See `data/rate_risk/basel_eve_sensitivity.csv`.

## Run order (from the project folder, Windows: `py`)

| Step | Script | Reads | Writes |
|---|---|---|---|
| 1 | `fetch_bac_xbrl.py --email you@example.com` | SEC company facts | `data/xbrl/bac_quarterly_xbrl.csv` |
| 2 | `fetch_fred_losses.py --api-key <key>` | FRED | `data/fred/fred_losses_quarterly.csv` |
| 3 | `build_calibration_panel.py` | Fed CSVs in `data/raw/fed`, steps 1-2 | `data/panel/*` |
| 4 | `estimate_loss_models.py` | panel | `data/models/loss_*` |
| 0 | `fetch_bhc_history.py` then `build_loan_type_history.py` | Chicago Fed holding-company files 2007Q1-2021Q1, per-quarter FRY9C files after | `data/y9c/bac_y9c_history.csv`, `data/models/loan_type_*`, `allowance_history.csv` |
| 0b | `refit_cre_bac.py` | `bac_y9c_history.csv`, panel, `benchmark_vs_fed.csv` | `data/models/cre_bac_*` |
| 4b | `diagnose_cre.py` | panel, `estimate_loss_models.py` | `data/models/cre_diagnostic.*` |
| 5 | `calibrate_bac_losses.py` | panel, loss paths, SEC JSON, Y-9C 2025Q4 | `data/calibration/*` |
| 6 | `build_nii_eve_model.py` (about 75 seconds) | panel scenarios, Y-9C 2023-2025 Q3 and Q4 | `data/rate_risk/nii_*` |
| 7 | `estimate_ppnr_model.py` | panel, `nii_scenario_effect.csv` | `data/models/ppnr_*` |
| 8 | `build_cet1_waterfall.py` | steps 5 and 7, Y-9C 2025Q4 | `data/waterfall/*` (includes the allowance sensitivity and the bridge to the Fed) |
| 9 | `build_basel_eve.py` | Y-9C 2025Q4 and Q3, panel | `data/rate_risk/basel_eve_*` |
| 9b | `reconcile_to_fed.py` | steps 5, 7 and 9 | `data/waterfall/reconcile_*` |
| 9c | `cre_current_book_check.py` | `data/raw/2026_Detailed_Nine_Quarter_Paths.csv` (Fed, aggregate of all banks), BAC 10-K 2025 CRE tables (transcribed in the script) | `data/models/cre_current_book.*` |
| 10 | `py -m streamlit run dashboard/dashboard_app.py -- --data data` | all of the above | dashboard |

Manual inputs: three Fed scenario CSVs (`data/raw/fed`); FR Y-9C files for BAC in `data/y9c`
(`FRY9C_1073757_<yyyymmdd>.csv` for 20230930, 20231231, 20240930, 20241231, 20250930, 20251231);
`bac_companyfacts.json` (`data/xbrl`, written by step 1). The rate-risk constants taken from BAC's 10-K (Table 44/45 NII
sensitivities, long-term debt note, fair-value footnote, fair-value-hedge table) are typed into the two rate-risk scripts with
the table named beside each.

## Method

- **Credit losses.** Five segments (cards, C&I, CRE, mortgage, other consumer) from FRED top-100-bank charge-off rates,
  2000Q1-2025Q4. One-step error-correction (ARDL) form on log net charge-off rate; drivers are the unemployment gap
  (vs trailing 12-quarter minimum), 4-quarter unemployment change, BBB spread, house/CRE price growth, COVID dummy.
  A delinquency (PD-proxy) error-correction model was tested and rejected: delinquency is near unit-root.
- **BAC scaling.** Segment curves weighted by BAC's current loan mix; one scale factor fitted to BAC's own net
  charge-offs 2014Q1-2019Q4 (k = 0.96), tested out of sample on 2020-2025 (R2 0.72, RMSE 0.07 pp). 2011-2013 is
  excluded: BAC's legacy Countrywide-era book ran at 1.7x the industry.
- **PPNR.** NIM, fee and expense equations on BAC income 2014Q1-2025Q4, plus stress overlays for the stressed cases:
  the NII effect from the rate-risk model (quarterly gap in the 3-month rate vs Baseline, dynamic deposits), equity-linked fee
  elasticity (0.6, from 2007-2009 GWIM fees), IB haircut (0.40, range 0.24-0.54), sales and trading haircut (0, see limitations), and operational-risk loss from the Fed's aggregate (0.88% of assets).
- **NII model.** Static-balance, 12-month, instantaneous shifts on the 2025Q4 balance sheet by repricing class. Deposit beta,
  low-rate deposit tier, hedge-adjusted long-end repricing and floating long debt are fitted by least squares to BAC's disclosed
  sensitivities (parallel, flattener, steepener; static and dynamic deposits; 2023-2025). A separate deposit-migration term
  (non-interest-bearing balances leaving as rates rise) accounts for the dynamic-minus-static gap.
- **EVE.** Basel standardised framework: six scenarios, USD shocks 200/300/150bp, 19 time buckets, core-deposit caps, prepayment
  multipliers. Securities at fair value, fixed loans at the 10-K fair value, long-term debt and AFS after the fair-value hedges
  in the 10-K. Deposit split, core share, prepayment speeds and coupons are assumptions (listed in `basel_eve_assumptions.csv`).
- **Allowance and capital.** The Fed's published rules: allowance = next four quarters of projected losses, starting difference
  smoothed over nine quarters into provisions; provision = charge-offs + allowance change; CET1 = prior + net income - dividends
  (preferred every quarter, common only in the first quarter); RWA and balances flat. The earlier calibrated 7.2-quarter lifetime
  horizon is a sensitivity.

## Data windows

| Block | Window | Reason |
|---|---|---|
| Segment loss models | 2000Q1-2025Q4 | Includes GFC and COVID |
| BAC scale factor | 2014Q1-2019Q4 fit | Legacy run-off excluded |
| PPNR | 2014Q1-2025Q4 | 2008-2013 dominated by one-offs |
| Capital starting point | 2025Q4 | Matches the Fed |

## Known limitations

- PD/LGD are industry proxies, not BAC loan-level estimates.
- Only the total loss rate is scaled to BAC. Segment misses vs the Fed (CRE 1.5% vs 9.3%, mortgage 4.1% vs 1.3%)
  are shown, not tuned away. `diagnose_cre.py` tests the CRE gap: a back-test on 2008-10 predicts 2.0% against 4.4% realised, so about half the miss is the model's dynamics; correcting for that gives 3.2-4.5%, still below the Fed's 9.3% (which is about 1.6x the worst realised industry nine-quarter loss, 5.8%).
- Remaining differences to the Fed after applying its allowance and capital rules: the model's end allowance is $29.2B against the Fed's implied $17.8B (the model's losses stay high through quarter 13; re-timing them to the Fed's aggregate shape gives $16.9B), loan losses are $5.6B below the Fed's, and OCI (+$4.7B) is not modelled. `cet1_bridge_to_fed.csv` shows the steps.
- PPNR fee overlays are sourced to BAC 10-K history but rest on few stress episodes: the investment banking haircut (24-54%) is supported by 2022 alone (2008 +7%, 2020 +27%); the sales and trading haircut is set to zero (client revenue did not fall in stress; legacy mark-to-market losses sit in the Fed's separate trading and counterparty item). Severe 9Q PPNR sensitivity: $62-73B across NII and fee stress scaling; $63.7B with a 15% sales and trading haircut.
- BAC's own 2007-2021 loan-type history (`build_loan_type_history.py`): weighted to today's CRE mix its worst nine-quarter loss was 3.9% (Fed 9.3%); mortgage 3.8% overall, 3.0% first lien (model 4.0%, Fed first-lien 1.3%); allowance peaked at 4.37% of loans in 2010Q4 (pre-CECL). `refit_cre_bac.py` refits the CRE equation on that history (mix-standardised, 2007Q1-2025Q4, 60 quarters, 34 usable in the regression, start rate 0.65%): persistence 0.87 to 0.52, severe nine-quarter loss 3.8% (2.3-3.8% across specifications; industry curve 1.5%; Fed 9.3%), total loss +$1.5B, CET1 about -0.07pp before reserve effects. It is reported alongside the headline results, not wired into them; coefficients are mostly insignificant. The model's CRE shortfall is therefore partly model dynamics (2.0% predicted vs 4.4-4.6% realised in 2008-10) and partly supervisory severity; the mortgage figure is crisis-anchored.
- PPNR components differ from the Fed's even though the total is close. `reconcile_to_fed.py` (run after steps 5, 7 and 9) sets them against BAC's 2025 run-rate: the model lowers NII 5% and fees 6%, the Fed raises NII 12% and cuts fees 28%. Net revenue agrees within $3.5B (1.5%) and PPNR within $1.4B; the $14.8B pre-tax gap is $16.3B of extra provisions less $1.4B of higher PPNR. The NII/fee split is not validated. Setting CRE and mortgage to the Fed's values moves total losses +$5.2B and -$7.2B, net -$2.0B.
- Balances held flat; no OCI, buybacks, or deferred-tax limits.
- The Fed publishes no Adverse scenario. A constructed one (halfway between Baseline and Severely Adverse; price indices interpolated in logs) is included and labelled as an assumption; it is mild (CET1 minimum 11.2%) because losses respond non-linearly to the macro path.
- NII model: the +200bp dynamic-deposit result is overstated ($1.3B vs $0.8B disclosed); a static-balance 12-month shift is applied each quarter as the rate gap vs Baseline, and shocks beyond -200bp are extrapolated.
- EVE: not disclosed by BAC. Cash flow hedges and swaps outside fair-value hedges are not included; deposit assumptions drive the result.
- Not a regulatory submission. The companion model review was done by the developer (a documented self-challenge, not an independent validation); no independent validation has been performed.

## Sources

Fed 2026 supervisory scenarios and DFAST results (June 2026); FFIEC FR Y-9C (BAC, RSSD 1073757); SEC XBRL
company facts (BAC, CIK 70858); FRED (Federal Reserve charge-off and delinquency rates, top-100 banks);
BAC FY2025 and FY2024 10-K (NII sensitivity Tables 44/45, long-term debt note, fair-value footnote, fair-value-hedge table) and 4Q25 earnings presentation (fee mix); Basel Committee BCBS d368 (April 2016).
