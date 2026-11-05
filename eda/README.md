# EDA Evidence Scripts

These scripts reproduce **every EDA number and figure** cited in the Milestone 2
report. Each `evidence_passN.py` is a self-contained, reproducible proof that
turns one or more preprocessing/splitting decisions into a real number or plot.

Running any script regenerates its figures under `eda/plots/` and prints the
exact figures quoted in the report.

## Requirements

Everything needed is already in the repo root `requirements.txt`
(`numpy`, `pandas`, `scikit_learn`, `matplotlib`, `seaborn`, plus `pyarrow` for
the gold-tier pass).

```bash
pip install -r requirements.txt
```

## Input data

Each script reads the raw dataset
`healthcare_fraud_detection.csv` (10,000 × 20, Kaggle "Healthcare Fraud
Detection", Nudrat Abbas). A copy is bundled in the repo at
`data/healthcare_fraud_detection.csv`, and every script **defaults to it** — so a
plain run from the repo works with no extra arguments, no machine-specific
paths, and no Kaggle auth wall:

```bash
python eda/evidence_pass1.py
```

To use a different copy of the dataset (e.g. a fresh download via the same
`kagglehub.dataset_download(...)` call as the milestone notebook, or your own
local file), pass it explicitly and the script uses that file instead:

```bash
python eda/evidence_pass1.py --csv path/to/healthcare_fraud_detection.csv
```

If the default bundled file is missing (e.g. the repo was cloned without
`data/`), the script stops with a clear message telling you what to do, rather
than a raw traceback.

## What each script proves (→ decision / report section)

| Script | Decision(s) earned | Key output |
|---|---|---|
| `evidence_pass1.py` | D1, D2, D7, D10, D11, D13, D14 (missingness, leakage audit, Timeline-Trap) | EV1–EV4: missingness pattern, velocity story, `Claim_Status` leakage rates, amount skew |
| `evidence_pass2.py` | D12 (exclude age/gender, fairness) | EV5: gender 50/50, Cramer's V 0.0047, age r −0.017 → no signal |
| `evidence_pass3.py` | D13 (log1p transform) | EV6: Claim_Amount skew 2.411→0.102, Approved 2.090→0.055 |
| `evidence_pass4.py` | D1, D2 (imputation replication) | EV7: 0 nulls, rows preserved, target untouched, fraud rate 0.0829 |
| `evidence_pass5_gold.py` | D16, D18 (gold-tier contract) | EV8: (10000,21) 19 features+target+split; 8000/1000/1000; 0 nulls |
| `evidence_pass6_queue.py` | D4, D6, D8, D3 | EV9: cardinality reframe, fold balance, 49-month seasonality |

## Figures produced

- `plots/ev5_age_fraud_rate.png`, `plots/ev6_amount_transform.png`
- `plots/ev9_d4_fraud_by_category.png`, `plots/ev9_d6_fold_balance.png`,
  `plots/ev9_d3_seasonality.png`

## One honest finding to reproduce

`evidence_pass6_queue.py` documents that **no categorical field is genuinely
high-cardinality** (max cardinality = 10). The Milestone 1 notebook labelled
four fields `high_card_features` → `TargetEncoder`; that label is a naming
artifact, not an empirical property — `TargetEncoder` is retained for per-category
fraud-signal capture + smoothing, not for dimension control. The script prints
the exact cardinality table backing this.