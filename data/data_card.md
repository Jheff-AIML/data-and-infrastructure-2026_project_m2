# Data Card — Healthcare Fraud Detection Dataset (Milestone 2)

> Data card for the **Milestone 2 pipeline** (separate repo:
> `data-and-infrastructure-2026_project_m2`). Documenting the dataset as it
> enters *this* pipeline — the source facts a tutor needs to reproduce §2
> (provenance/governance) and §3 (data description) of the report. Every number
> below is verified by the reproducible evidence scripts in `eda/`.

---

## Source & licence

- **Dataset:** Healthcare Fraud Detection Dataset
- **Author / maintainer:** Nudrat Abbas (Kaggle)
- **Source:** https://www.kaggle.com/datasets/nudratabbas/healthcare-fraud-detection-dataset
- **Licence:** **CC0 — public domain** (simulated data, no PII; redistribution permitted)
- **Nature:** fully **synthetic** medical-billing records engineered (log-normal
  distributions) to mimic real-world claim behaviour and billing anomalies.

> **Limitation to carry into the report (§2):** because the data is synthetic,
  a model trained on it may not generalise to real billing fraud. This is an
  honest, discussable limitation — we own it rather than paper over it.

---

## At a glance

| Property | Value |
|---|---|
| Rows × columns | **10,000 × 20** |
| Target | `Is_Fraud` (0 = clean, 1 = fraudulent) |
| Fraud rate (baseline) | **829 / 10,000 = 8.29%** |
| Date span (`Claim_Submission_Date`) | 2021-01-06 → 2025-01-26 (49 months) |
| Duplicate rows / duplicate `Claim_ID` | **0 / 0** (10,000 unique) |
| Missingness (before imputation) | **350 rows × 3 fields** (`Insurance_Type`, `Provider_Specialty`, `Prior_Visits_12m`) |
| Row granularity | one claim per row |

---

## Schema (20 columns, grouped by role in the M2 pipeline)

| Role in M2 | Columns |
|---|---|
| **Identifiers** (excluded from features) | `Provider_ID`, `Claim_ID` |
| **Patient** (protected; excluded at gold) | `Patient_Age`, `Patient_Gender` |
| **Clinical coding** (categorical features) | `Diagnosis_Code`, `Procedure_Code`, `Provider_Specialty` |
| **Payer / visit** (categorical features) | `Insurance_Type`, `Visit_Type`, `Patient_State` |
| **Financial** (numeric, log1p'd at gold) | `Claim_Amount`, `Approved_Amount` |
| **Temporal** (raw date dropped; month + weekend kept) | `Claim_Submission_Date`, `Days_Between_Service_and_Claim` |
| **Volume / stay / history** (numeric features) | `Number_of_Claims_Per_Provider_Monthly`, `Length_of_Stay`, `Prior_Visits_12m`, `Chronic_Condition_Flag` |
| **Settlement outcome** (excluded; target-leakage hazard) | `Claim_Status` |
| **Target** | `Is_Fraud` |

---

## What M2 established with the evidence (`eda/`)

- **Missingness is non-informative.** The three null fields each have 350 nulls,
  but missingness does not shift class balance (missing-insurance fraud 7.1% vs
  baseline 8.3%; missing-specialty 8.6%; missing-prior-visits 8.3%). Co-missing
  masks overlap only lightly; joint cells are tiny (n ≤ 13). → **simple
  imputation stands** (`Prior_Visits_12m`→0, categoricals→`"Unknown"`). *(EV1/EV7 → D1/D2)*
- **`Claim_Status` is a target-leakage hazard.** Fraud by status: Approved 2.2%
  vs Pending 22.7% / Rejected 23.1%. It is a downstream adjudication outcome,
  not available at prediction time → **excluded with documentation**. *(EV3 → D10)*
- **`Patient_Age` / `Patient_Gender` carry no signal and are protected
  characteristics → excluded at gold.** Gender 50/50 at ~8.3% fraud each
  (Cramer's V 0.0047); age point-biserial r −0.017. Exclusion costs ~no
  capability; fraud flags route to human investigation, never automated denial. *(EV5 → D12)*
- **Velocity of fraud is fast.** 100% of fraudulent claims filed ≤ 6 days after
  service (legit mean 15.45d / fraud mean 2.97d). → velocity feature uses
  `≤ 6 days`, not M1's `< 5`. *(EV2 → D14)*
- **Amounts are strongly right-skewed** (Claim 2.4, Approved 2.1) → `log1p`
  before scaling collapses skew (2.411→0.102). *(EV4/EV6 → D13)*
- **No categorical field is genuinely high-cardinality** (max = 10 unique).
  M1's `high_card` label was a naming artifact; `TargetEncoder` is retained for
  signal capture + smoothing, not dimension control. *(EV9 → D4 reframe)*
- **Splits are class-balanced.** 80/10/10 stratified → 8000/1000/1000, fraud
  0.0829/0.0830/0.0830; all 8 CV folds stay in the ~8.3% band. *(EV8/EV9 → D6/D8)*
- **No systematic calendar seasonality.** Monthly fraud swings 0.046–0.121
  (peak 2024-09 at 1.44× median) but with no stable seasonal pattern. *(EV9 → D3)*

---

## Provenance chain (bronze → silver → gold)

- **Bronze** = this raw CSV, versioned and kept untouched (the "keep the raw
  layer" rule).
- **Silver** = cleaned, typed, validated, imputed; written at close-of-silver.
- **Gold** = model-ready feature matrix: 19 pre-encoding features + `Is_Fraud` +
  `split`; encoders/scaler stay *fit-per-fold in the training pipeline* so the
  stored object never leaks. (D16/D17/D18)

Each tier carries a `manifest.json` (sha256 + shape + source) chaining it to the
previous one.