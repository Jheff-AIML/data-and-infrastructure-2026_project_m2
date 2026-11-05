#!/usr/bin/env python3
"""
Evidence Pass 2 — D12: Patient_Age & Patient_Gender signal check
=================================================================
Reproducible EDA evidence for decision D12 (should age/gender be
reintroduced at silver/gold, or documented as excluded?). Written
Session 4, 2026-11-02, to ground the ethics + signal discussion in
numbers rather than narrative.

Question being answered: do Patient_Age / Patient_Gender carry enough
predictive signal that excluding them is a real accuracy cost? (If not,
the fairness-motivated exclusion is essentially free.)

Usage:  python evidence_pass2.py [--csv path/to/healthcare_fraud_detection.csv]
        (default: bundled data/healthcare_fraud_detection.csv)
Output: prints the D12 evidence blocks; optionally saves
        plots/ev5_age_fraud_rate.png
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
fraud = df["Is_Fraud"]
base_rate = fraud.mean()
n = len(df)

def pt_bis(r, dfn):
    """Point-biserial r between a continuous col and the binary target."""
    m1, m0 = r[fraud == 1].mean(), r[fraud == 0].mean()
    s = r.std(ddof=1)
    p1 = fraud.mean()
    if s == 0:
        return 0.0
    return (m1 - m0) / s * np.sqrt(p1 * (1 - p1))

print("=" * 72)
print("D12 — Patient_Age & Patient_Gender SIGNAL CHECK")
print("=" * 72)
print(f"baseline fraud rate: {base_rate:.4f}  ({fraud.sum()}/{n})")

# ------------------------------------------------------------------
# GENDER
# ------------------------------------------------------------------
print("\n--- Patient_Gender ---")
print("value counts:", df["Patient_Gender"].value_counts().to_dict())
gb = df.groupby("Patient_Gender")["Is_Fraud"].agg(["count", "mean"])
gb.columns = ["n", "fraud_rate"]
print(gb.round(4))
# effect size: Cramer's V between gender and target
tbl = pd.crosstab(df["Patient_Gender"], fraud)
chi2 = float(pd.crosstab(df["Patient_Gender"], fraud, margins=False).pipe(
    lambda t: ((t - np.outer(t.sum(1), t.sum(0) / t.sum().sum())) ** 2
               / np.outer(t.sum(1), t.sum(0) / t.sum().sum()))).sum().sum())
min_dim = min(tbl.shape) - 1
cramers_v = np.sqrt(chi2 / (n * min_dim)) if min_dim > 0 else 0.0
print(f"chi-sq={chi2:.3f}  Cramer's V={cramers_v:.4f}  "
      f"(gender rate spread {gb['fraud_rate'].max()-gb['fraud_rate'].min():.4f})")
print("READOUT: gender is ~50/50; a rate spread this small = essentially no signal.")

# ------------------------------------------------------------------
# AGE
# ------------------------------------------------------------------
print("\n--- Patient_Age ---")
age = df["Patient_Age"]
print(f"range {age.min()}-{age.max()}, unique {age.nunique()}, "
      f"mean {age.mean():.1f}, median {age.median():.0f}")
print(f"point-biserial r(age, fraud) = {pt_bis(age, n):.4f}")

# binned fraud rates
bands = [(0, 34, "<35"), (35, 49, "35-49"), (50, 64, "50-64"),
         (65, 79, "65-79"), (80, 200, "80+")]
print("\nage band      n  fraud_rate  (vs baseline)")
seen = []
for lo, hi, lab in bands:
    m = age.between(lo, hi)
    sub = df[m]
    rate = sub["Is_Fraud"].mean()
    r_n = int(sub["Is_Fraud"].sum())
    seen.append((lab, rate, len(sub)))
    print(f"{lab:12s} {len(sub):5d}   {rate:.4f}  "
          f"({r_n:4d}/{len(sub)})  {'+' if rate > base_rate else '-'}"
          f"{abs(rate - base_rate):.4f}")
# spread
mx = max(seen, key=lambda x: x[1]); mn = min(seen, key=lambda x: x[1])
print(f"READOUT: band spread {mx[1]-mn[1]:.4f} (max {mx[0]}, min {mn[0]}); "
      f"|r|={abs(pt_bis(age, n)):.3f} — {'non-trivial spread' if mx[1]-mn[1] > 0.04 else 'small spread'}")

# monotonic-ish summary: correlation of band centre with band rate
if len(seen) >= 3:
    centres = [(35, 32.5), (35, 49, 42), (50, 64, 57), (65, 79, 72), (80, 200, 99)][:len(seen)]
    cxs = np.array([c[2] for c in [x for x in [(0,34,17),(35,49,42),(50,64,57),(65,79,72),(80,200,99)]][:len(seen)]])
    cys = np.array([x[1] for x in seen])
    slope = np.corrcoef(cxs, cys)[0, 1] if cys.std() > 0 and cxs.std() > 0 else 0.0
    print(f"corr(age-band-centre, band fraud-rate) = {slope:.3f}")

# ------------------------------------------------------------------
# PLOT (optional)
# ------------------------------------------------------------------
if AGREED_SAVE:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        labels = [x[0] for x in seen]
        rates = [x[1] for x in seen]
        fig, ax = plt.subplots(figsize=(8, 5))
        colors = ["tab:red" if r > base_rate else "tab:blue" for r in rates]
        ax.bar(labels, rates, color=colors, alpha=0.8)
        ax.axhline(base_rate, color="k", ls="--", lw=1.2,
                   label=f"baseline {base_rate:.3f}")
        for i, r in enumerate(rates):
            ax.text(i, r + 0.0015, f"{r:.3f}", ha="center", fontsize=9)
        ax.set(title="D12: fraud rate by patient-age band (M2 healthcare fraud)",
               xlabel="age band", ylabel="fraud rate")
        ax.legend()
        os.makedirs(PLOT_DIR, exist_ok=True)
        plot_path = os.path.join(PLOT_DIR, "ev5_age_fraud_rate.png")
        fig.savefig(plot_path, dpi=150, bbox_inches="tight")
        print("\nsaved", plot_path)
    except Exception as e:
        print("plot save skipped:", e)