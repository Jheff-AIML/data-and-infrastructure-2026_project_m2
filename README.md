# Data & Infrastructure for AI — Milestone 2

A documented, reproducible **bronze → silver → gold** medallion pipeline for a healthcare-fraud detection dataset (SETU *Data Handling & Infrastructure for AI*, Milestone 2).

This milestone is **not** a model-building exercise. Its deliverable is a trustworthy, versioned data pipeline and a **model-ready feature matrix** (`gold`) whose every design decision is earned by EDA evidence (see `eda/` and the decision log), plus the stratified CV-fold infrastructure that Milestone 3's model will consume.

## Scope in one line
> "No model is trained or cross-validated in this milestone; the pipeline produces the gold feature matrix and the leak-free CV-fold structure for Milestone 3."

## Repo layout

```
.
├── README.md
├── requirements.txt            # pinned dependency set (reproducibility anchor)
├── data/
│   ├── healthcare_fraud_detection.csv   # bundled raw copy (byte-identical to the bronze object)
│   └── data_card.md                     # source + licence + schema-by-M2-role
├── eda/
│   ├── evidence_pass1.py … evidence_pass6.py   # reproducible EDA evidence (EV1–EV9)
│   ├── plots/                     # the 6 report figures (Figures 1–6)
│   └── gold_out/                  # local proof of the gold layer
└── notebooks/
    └── Milestone_Two.ipynb        # the Colab pipeline (acquisition → bronze/silver/gold)
```

## How to run

```bash
pip install -r requirements.txt
python eda/evidence_pass1.py   # (repeat for pass2 … pass6)
```

- The EDA evidence scripts default to the bundled CSV — **no Kaggle login needed**; see **[`eda/README.md`](eda/README.md)** for what each script proves (decision → evidence mapping) and the full `--csv <path>` interface.
- The full acquisition→gold flow (with **Kaggle/GCS auth via the Colab secrets manager — no credentials in this repo**) is in `notebooks/Milestone_Two.ipynb`.

## Reproducibility

- All evidence scripts run against the **pinned** `requirements.txt`.
- The `gold` artifact is **byte-identical on cold re-run** (sha256 `4541d519…`), verified under two environments.
- Reproducibility claims are **scoped strictly to the pinned dependency set** — we do *not* assert byte-identical output under unpinned or untested versions.

## Report

The full Milestone-2 report (§5 architecture, §6 reproducibility, §7 repo guide) is delivered as the PDF submitted with this milestone; it references this repo and its evidence throughout.

## Licence

GPL v2 — see `LICENSE`.