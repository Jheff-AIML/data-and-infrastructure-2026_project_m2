#!/usr/bin/env python3
"""
Evidence Pass 3 — EV6: amount-transform skew before/after log1p (D13)
=====================================================================
Reproducible evidence for decision D13 (reconcile the README "log
transform" claim with the code that runs StandardScaler on raw amounts).

Answering: does np.log1p actually tame the right skew that EV4 measured
(Claim_Amount 2.4, Approved_Amount 2.1) to near-symmetric before scaling?
If yes, aligning code -> narrative (log1p then StandardScaler) is the
defensible move, and the report gains a before/after exhibit.

Usage:  python evidence_pass3.py [--csv path/to/healthcare_fraud_detection.csv]
        (default: bundled data/healthcare_fraud_detection.csv)
"""
import os
import sys
import argparse
import pandas as pd
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PLOT_DIR = os.path.join(HERE, "plots")
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
AGREED_SAVE = 1

df = pd.read_csv(PATH)

print("=" * 72)
print("EV6 — AMOUNT TRANSFORM: skew before/after log1p (evidence for D13)")
print("=" * 72)
for c in ["Claim_Amount", "Approved_Amount"]:
    raw = df[c]
    mn = raw.min()
    log = np.log1p(raw) if mn > -1 else None
    print(f"\n{c}")
    print(f"  raw            : min={mn:.2f} max={raw.max():.2f} skew={raw.skew():.3f}")
    if log is not None:
        print(f"  after log1p    : min={log.min():.2f} max={log.max():.2f} skew={log.skew():.3f}")
        red = abs(raw.skew()) - abs(log.skew())
        print(f"  skew reduction : {red:.3f}  ({red/abs(raw.skew())*100:.0f}% of raw |skew| removed)")

print("\nREADOUT: target for a defensible log1p is |skew| near 0 (symmetric); "
      "if it lands < ~0.5, the transform achieves its stated purpose and the "
      "code should implement log1p BEFORE StandardScaler (align code to README).")

if AGREED_SAVE:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 2, figsize=(11, 7))
        for i, c in enumerate(["Claim_Amount", "Approved_Amount"]):
            raw = df[c]
            log = np.log1p(raw)
            axes[i, 0].hist(raw, bins=60, color="tab:blue", alpha=0.8)
            axes[i, 0].set(title=f"{c} (raw) skew={raw.skew():.2f}", xlabel=c)
            axes[i, 1].hist(log, bins=60, color="tab:green", alpha=0.8)
            axes[i, 1].set(title=f"{c} after log1p skew={log.skew():.2f}", xlabel="log1p(" + c + ")")
        fig.suptitle("EV6: log1p tames right skew (evidence for D13)")
        fig.tight_layout()
        os.makedirs(PLOT_DIR, exist_ok=True)
        plot_path = os.path.join(PLOT_DIR, "ev6_amount_transform.png")
        fig.savefig(plot_path, dpi=150, bbox_inches="tight")
        print("\nsaved", plot_path)
    except Exception as e:
        print("plot save skipped:", e)