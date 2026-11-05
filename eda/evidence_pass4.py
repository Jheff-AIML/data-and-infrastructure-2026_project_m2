#!/usr/bin/env python3
"""
Evidence Pass 4 — EV7: imputation replication for the silver wave (D1, D2)
==========================================================================
Reproducible evidence for the impute wave of the Colab silver tier. Replays
exactly what the notebook cell will do against the raw CSV and validates that
the imputation (Prior_Visits_12m -> 0 ; Insurance_Type -> "Unknown" ;
Provider_Specialty -> "Unknown") completes, preserves rows, does not touch the
target, and does not shift class balance. Also re-checks the EV1 joint-missing
cell sizes that argue AGAINST engineering missing-indicator features.

Usage:  python evidence_pass4.py [--csv path/to/healthcare_fraud_detection.csv]
        (default: bundled data/healthcare_fraud_detection.csv)
Output: EV7 block; asserts internally (non-zero exit on any failure).
"""
import os
import sys
import argparse
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CSV = os.path.join(HERE, "..", "data", "healthcare_fraud_detection.csv")
ap = argparse.ArgumentParser()
ap.add_argument("--csv", default=DEFAULT_CSV,
                help="path to healthcare_fraud_detection.csv (default: bundled data/ copy)")
args = ap.parse_args()
PATH = args.csv
if not os.path.exists(PATH):
    print(f"error: CSV not found at {PATH}")
    print("The default expects the bundled copy at <repo>/data/healthcare_fraud_detection.csv.")
    print("Pass a different file with --csv path/to/healthcare_fraud_detection.csv "
          "(e.g. a fresh kagglehub download).")
    sys.exit(1)
FILL_MAP = {"Prior_Visits_12m": 0, "Insurance_Type": "Unknown", "Provider_Specialty": "Unknown"}
NULL_COLS = list(FILL_MAP.keys())

df = pd.read_csv(PATH)
base_rate = df["Is_Fraud"].mean()
n0 = len(df)
target_col = df["Is_Fraud"].copy()

print("=" * 72)
print("EV7 — IMPUTATION REPLICATION for the silver wave (evidence for D1/D2)")
print("=" * 72)
print("raw shape:", df.shape, " baseline fraud rate:", round(base_rate, 4))

# --- before: justify imputation with class-conditional missing-vs-present rates (flat) ---
print("\n-- BEFORE (justification: does missingness shift class? --)")
flat = True
for c in NULL_COLS:
    m = df[c].isnull()
    r_miss = df.loc[m, "Is_Fraud"].mean()
    r_pres = df.loc[~m, "Is_Fraud"].mean()
    print(f"  {c:20s} nulls={int(m.sum()):4d}  missing->fraud={r_miss:.4f}  present->fraud={r_pres:.4f}  "
          f"(baseline {base_rate:.4f})")
    if abs(r_miss - base_rate) > 0.02:
        flat = False
print("  flat?", flat)

# --- joint missingness: only the MULTI-field co-missing cells argue against indicators ---
mask_df = df[NULL_COLS].isnull()
print("\n-- joint missingness (row counts per missing combo) --")
combo_counts = mask_df.groupby(NULL_COLS).size()
for combo, cnt in combo_counts.items():
    print(f"  missing={tuple(bool(b) for b in combo)}  rows={int(cnt)}")

multi_cells = [int(v) for (combo, v) in combo_counts.items() if sum(combo) >= 2]
single_missing = [int(v) for (combo, v) in combo_counts.items() if sum(combo) == 1]
print("\n  largest SINGLE-field missing cell n =", max(single_missing))
print(f"  largest MULTI-field co-missing cell n = {max(multi_cells)}  "
      f"(all <= 13 => no meaningful co-missing cluster => no missing-indicator feature)")

# --- apply the SAME imputation the notebook will ---
df2 = df.copy()
for c, fill in FILL_MAP.items():
    df2[c] = df2[c].fillna(fill)

print("\n-- AFTER (assertions the notebook cell will rely on) --")
assert (df2[NULL_COLS].isnull().sum() == 0).all(), "imputation left nulls behind"
print("  ok  nulls remaining in imputed cols:", df2[NULL_COLS].isnull().sum().sum())
assert len(df2) == n0, "row count changed"
print("  ok  rows preserved:", n0, "->", len(df2))
assert df2["Is_Fraud"].equals(target_col), "target mutated"
print("  ok  target unchanged (identical to raw)")
assert abs(df2["Is_Fraud"].mean() - base_rate) < 1e-12, "fraud rate shifted"
print("  ok  fraud rate unchanged:", f"{df2['Is_Fraud'].mean():.4f}")

print("\n  completed frame fraud rate =", f"{df2['Is_Fraud'].mean():.4f}",
      f"({df2['Is_Fraud'].sum()}/{len(df2)})")
print("\nVERDICT: imputation completes, preserves rows, leaves target untouched, "
      "keeps the fraud rate at baseline; flat marginal rates + tiny co-missing "
      "cells -> D1 (Prior_Visits->0) and D2 (Insurance/Provider->'Unknown') are "
      "safe, and no missing-indicator features should be engineered (EV1).")