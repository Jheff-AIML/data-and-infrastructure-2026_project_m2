#!/usr/bin/env python3
"""
Evidence Pass 1 — Healthcare Fraud M2
=====================================
Reproducible EDA evidence for the M1-decision review (Session 2, 2026-11-01).
Every number quoted in `04-eda-decision-log.md` under EV1-EV4 can be
regenerated from this script against the raw CSV.

Usage:  python evidence_pass1.py [--csv path/to/healthcare_fraud_detection.csv]
        (default: bundled data/healthcare_fraud_detection.csv)
Output: prints EV1-EV4 blocks; if matplotlib is available and AGREED_SAVE=1,
        writes the headline Timeline-Trap plot to plots/.
"""
import os
import sys
import argparse
import pandas as pd
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))          # eda/
PLOT_DIR = os.path.join(HERE, "plots")                      # anchored, not cwd-relative
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
AGREED_SAVE = 1  # set 0 to skip plotting

df = pd.read_csv(PATH)

# ------------------------------------------------------------------
# EV1 — MISSINGNESS PATTERN (D1, D2)
# ------------------------------------------------------------------
null_cols = ["Insurance_Type", "Provider_Specialty", "Prior_Visits_12m"]
mask_df = df[null_cols].isnull()
print("=" * 70)
print("EV1 — MISSINGNESS PATTERN (evidence for D1/D2)")
print("=" * 70)
print("null count per column:", {c: int(df[c].isnull().sum()) for c in null_cols})
print("\nrows by number of null fields:", mask_df.sum(axis=1).value_counts().to_dict())
print("co-missing subsets (mask = Insurance_Type, Provider_Specialty, Prior_Visits_12m):")
for k, grp in mask_df.groupby(null_cols):
    print(f"  mask={tuple(bool(x) for x in k)}  rows={len(grp):5d}  fraud_rate={df.loc[grp.index,'Is_Fraud'].mean():.3f}")
print("\nclass-conditional missing rate (missing->fraud vs present->fraud):")
for c in null_cols:
    m = df[c].isnull()
    print(f"  {c:22s} missing n={m.sum():4d} fraud={df.loc[m,'Is_Fraud'].mean():.3f} "
          f"({df.loc[m,'Is_Fraud'].sum():.0f}) | present fraud={df.loc[~m,'Is_Fraud'].mean():.3f}")

# ------------------------------------------------------------------
# EV2 — TIMELINE TRAP (D7) — reproduces the README headline numbers
# ------------------------------------------------------------------
print("\n" + "=" * 70)
print("EV2 — TIMELINE TRAP (evidence for D7)")
print("=" * 70)
for lab, sub in [("legit (0)", df[df["Is_Fraud"] == 0]), ("fraud (1)", df[df["Is_Fraud"] == 1])]:
    s = sub["Days_Between_Service_and_Claim"]
    print(f"{lab:10s} n={len(sub):5d} mean={s.mean():.2f} median={s.median():.0f} "
          f"min={s.min()} max={s.max()} zeros={(s==0).sum()}")
    for t in (3, 5, 6, 10):
        print(f"    %<={t}d: {(s<=t).mean()*100:6.2f}")
print("README claim check: legit 15.45/15/2-29, fraud 2.97/3/0-6, zeros=115")
print("IMPORTANT: the velocity feature uses <5d as 'fast' but fraud<=5d is",
      f"{(df[df.Is_Fraud==1]['Days_Between_Service_and_Claim']<=5).mean()*100:.1f}% (not 100%)")

if AGREED_SAVE:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 5))
        for lab, sub, color in [("legit (0)", df[df["Is_Fraud"] == 0], "tab:blue"),
                                ("fraud (1)", df[df["Is_Fraud"] == 1], "tab:red")]:
            s = sub["Days_Between_Service_and_Claim"]
            ax.hist(s, bins=range(0, 31), density=True, alpha=0.55,
                    label=f"{lab} — mean {s.mean():.1f}d, median {s.median():.0f}d", color=color)
        ax.axvline(5, color="k", ls="--", lw=1.2, label="velocity 'fast' threshold (<5d)")
        ax.set(title="Timeline Trap: Days_Between_Service_and_Claim by class",
               xlabel="Days between service and claim submission", ylabel="density")
        ax.legend()
        os.makedirs(PLOT_DIR, exist_ok=True)
        plot_path = os.path.join(PLOT_DIR, "ev2_timeline_trap.png")
        fig.savefig(plot_path, dpi=150, bbox_inches="tight")
        print("\nsaved", plot_path)
    except Exception as e:
        print("plot save skipped:", e)

# ------------------------------------------------------------------
# EV3 — LEAKAGE / UNUSED-COLUMN AUDIT (new findings)
# ------------------------------------------------------------------
print("\n" + "=" * 70)
print("EV3 — LEAKAGE / UNUSED-COLUMN AUDIT (new findings)")
print("=" * 70)
print("Claim_Status value counts:", df["Claim_Status"].value_counts().to_dict())
print("\nClaim_Status x Is_Fraud (leak check, row-normalised):")
print(pd.crosstab(df["Claim_Status"], df["Is_Fraud"], normalize="index").round(3))
print("\nClaim_ID unique:", df["Claim_ID"].nunique(), "of", len(df))
print("full-row duplicates:", df.duplicated().sum(), "| dup Claim_IDs:", df["Claim_ID"].duplicated().sum())
print("Patient_Age:", df["Patient_Age"].min(), "-", df["Patient_Age"].max(),
      "| unique:", df["Patient_Age"].nunique())
print("Patient_Gender:", df["Patient_Gender"].value_counts().to_dict())
sub = pd.to_datetime(df["Claim_Submission_Date"])
print("Claim_Submission_Date range:", sub.min(), "->", sub.max())

# ------------------------------------------------------------------
# EV4 — AMOUNT SKEW (D5)
# ------------------------------------------------------------------
print("\n" + "=" * 70)
print("EV4 — AMOUNT SKEW (evidence for D5)")
print("=" * 70)
for c in ["Claim_Amount", "Approved_Amount"]:
    s = df[c]
    print(f"{c:16s} skew={s.skew():.1f} legit median={s[df['Is_Fraud']==0].median():.2f} "
          f"fraud median={s[df['Is_Fraud']==1].median():.2f} fraud max={s[df['Is_Fraud']==1].max():.2f}")
