#!/usr/bin/env python
"""evidence_pass6_queue.py — close out the three remaining evidence-queue items.

D4  : cardinality per categorical field + fraud-rate-by-category (top-N) plot
D6  : fold-wise / split-wise class balance table (stratified 80/10/10 + 8-fold on train)
D3  : month-level seasonality of fraud (Submission_Month) plot

Reproduces the split + fraud numbers from the raw CSV, matching EV8's gold-tier
split exactly (random_state=42, 80/10/10 stratify). Class-conditional rates are
computed ON the subset, never on the whole frame (the "rate without a
denominator" rule). Plots are anchored to `plots/` beside this script.

Usage:  python eda/evidence_pass6_queue.py [--csv PATH] [--out DIR]
"""
import argparse, os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold

DEFAULT_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "..", "data", "healthcare_fraud_detection.csv")
PLOT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "plots")

CATEGORICAL_FIELDS = [
    "Diagnosis_Code", "Procedure_Code", "Insurance_Type", "Visit_Type",
    "Provider_Specialty", "Patient_State",
]
# M1's own labels (notebook cell 5): these four go through TargetEncoder(smooth=10)
M1_TARGET_ENCODED = ["Diagnosis_Code", "Procedure_Code", "Provider_Specialty", "Patient_State"]
M1_ONE_HOT = ["Insurance_Type", "Visit_Type"]
TARGET = "Is_Fraud"
CARD_HI = 50  # a sane "high cardinality" bar for one-hot explosion concerns


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=DEFAULT_CSV,
                    help="path to healthcare_fraud_detection.csv (default: bundled data/ copy)")
    ap.add_argument("--out", default=PLOT_DIR)
    args = ap.parse_args()
    if not os.path.exists(args.csv):
        print(f"error: CSV not found at {args.csv}")
        print("The default expects the bundled copy at <repo>/data/healthcare_fraud_detection.csv.")
        print("Pass a different file with --csv path/to/healthcare_fraud_detection.csv "
              "(e.g. a fresh kagglehub download).")
        sys.exit(1)
    os.makedirs(args.out, exist_ok=True)

    df = pd.read_csv(args.csv)
    df[TARGET] = df[TARGET].astype(int)
    n_total = len(df)
    baseline = df[TARGET].mean()
    checks = []
    def check(cond, label):
        checks.append((bool(cond), label))

    print("=" * 60)
    print("D4 — CARDINALITY per categorical field  (report $3/$4)")
    print("=" * 60)
    print(f"{'field':<22}{'unique':>8}{'mode':>10}{'mode_share':>12}")
    for col in CATEGORICAL_FIELDS:
        vc = df[col].value_counts()
        unique = vc.size
        mode = vc.index[0]
        mode_share = vc.iloc[0] / n_total
        print(f"{col:<22}{unique:>8}{str(mode)[:9]:>10}{mode_share:>12.3f}")
        check(unique >= 1, f"D4 cardinality {col} computed ({unique})")

    # HONEST REFRAME: NO field is high-cardinality (all <= CARD_HI). M1's
    # "high_card_features" hardcode is a naming artifact, not an empirical fact.
    card = {col: df[col].nunique() for col in CATEGORICAL_FIELDS}
    truly_high = [c for c in CATEGORICAL_FIELDS if card[c] >= CARD_HI]
    m1_high = [c for c in CATEGORICAL_FIELDS if c in M1_TARGET_ENCODED]
    print(f"\n  FINDING: max cardinality across all 6 fields = {max(card.values())} (bar={CARD_HI}). "
          f"NO field is genuinely high-cardinality.")
    print(f"  M1 calls {len(m1_high)} fields 'high_card' -> TargetEncoder: {', '.join(m1_high)} "
          f"(actual unique: {', '.join(str(card[c]) for c in m1_high)})")
    explosion = sum(card[c] for c in CATEGORICAL_FIELDS)
    print(f"  TRUE one-hot cost if we encoded ALL six: ~{explosion} columns vs {len(CATEGORICAL_FIELDS)} "
          f"fields — trivial for 10k rows.")
    check(not truly_high, "D4 reframe: no field exceeds CARD_HI -> 'dimension-explosion' does not justify TargetEncoder")
    check(card["Diagnosis_Code"] <= CARD_HI and card["Procedure_Code"] <= CARD_HI,
          "D4 reframe: Diagnosis/Procedure are LOW-card, contradicting the M1 'high_card' hardcode")
    # legitimate remaining justification: per-category fraud signal + smoothing regularisation
    # (this is what the report should cite instead of explosion)

    # --- fraud-rate-by-category for the fields M1 target-encodes (signal capture) ---
    print("\n" + "=" * 60)
    print("D4 — FRAUD RATE BY CATEGORY (top-N by volume, M1's target-encoded fields)")
    print("=" * 60)
    n_top = 6
    top_plot_rows = []
    for col in M1_TARGET_ENCODED:  # Diagnosis, Procedure, Provider_Specialty, Patient_State
        vc = df[col].value_counts()
        top_cats = vc.index[:n_top].tolist()
        print(f"\n  [{col}] top-{n_top} categories by volume (unique={card[col]}):")
        print(f"    {'category':<16}{'count':>7}{'fraud_rate':>12}")
        for cat in top_cats:
            sub = df.loc[df[col] == cat]                              # denominator = the subset
            fr = sub[TARGET].mean()                                    # class-conditional on SUBSET
            print(f"    {str(cat)[:16]:<16}{len(sub):>7}{fr:>12.4f}")
        # capture for bar chart (skip NaN category neatly)
        top_plot_rows.append((col, top_cats,
                              [df.loc[df[col] == c, TARGET].mean() for c in top_cats]))

    # --- D4 plot: fraud rate by top-N category, with baseline line ---
    fig, axes = plt.subplots(1, len(top_plot_rows), figsize=(5.5 * len(top_plot_rows), 4), squeeze=False)
    for ax, (col, cats, rates) in zip(axes[0], top_plot_rows):
        colors = ["#c0392b" if r > baseline * 1.5 else "#7f8c8d" for r in rates]
        ax.bar([str(c)[:10] for c in cats], rates, color=colors)
        ax.axhline(baseline, color="#2c3e50", ls="--", lw=1, label=f"baseline {baseline:.3f}")
        ax.axhline(baseline * 2, color="#e74c3c", ls=":", lw=1, label="2x baseline")
        ax.set_title(f"{col}\nfraud rate by top-{len(cats)} category")
        ax.set_ylabel("fraud rate")
        ax.legend(fontsize=8)
        ax.tick_params(axis="x", rotation=30)
    fig.suptitle("D4 — fraud rate by category (high-cardinality fields)", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "ev9_d4_fraud_by_category.png"), dpi=110)
    plt.close(fig)
    print("\n  plot saved: ev9_d4_fraud_by_category.png")

    # =====================================================================
    print("=" * 60)
    print("D6 — SPLIT / FOLD-WISE CLASS BALANCE  (report $3 & splitting)")
    print("=" * 60)
    train, temp = train_test_split(df, test_size=0.20, random_state=42, stratify=df[TARGET])
    dev, test = train_test_split(temp, test_size=0.50, random_state=42, stratify=temp[TARGET])
    splits = {"train (80%)": train, "dev (10%)": dev, "test (10%)": test}
    print(f"\n  Split balance (matches EV8 exactly):")
    print(f"    {'split':<12}{'n':>7}{'fraud_rate':>12}")
    for name, s in splits.items():
        r = s[TARGET].mean()
        print(f"    {name:<12}{len(s):>7}{r:>12.4f}")
        check(0.075 < r < 0.10, f"D6 split {name} fraud {r:.4f} in ~8.3% band")

    # 8-fold stratified CV on train (D8) — show each fold stays ~8.3%
    skf = StratifiedKFold(n_splits=8, shuffle=True, random_state=42)
    fold_rates = []
    for i, (tr_idx, va_idx) in enumerate(skf.split(train, train[TARGET])):
        tr_r = train.iloc[tr_idx][TARGET].mean()
        va_r = train.iloc[va_idx][TARGET].mean()
        fold_rates.append((tr_r, va_r))
        print(f"    fold {i+1}: train-train rate {tr_r:.4f} | val rate {va_r:.4f}")
    check(all(0.075 < tr < 0.10 and 0.075 < va < 0.10 for tr, va in fold_rates),
          "D8 all 8 folds stay in ~8.3% band (leak-free CV denominator)")

    # --- D6 plot: fold + split balance ---
    fig, ax = plt.subplots(figsize=(8, 4.2))
    names = list(splits.keys()) + [f"fold{i+1}t" for i in range(8)] + [f"fold{i+1}v" for i in range(8)]
    rates = [s[TARGET].mean() for s in splits.values()] + [tr for tr, _ in fold_rates] + [va for _, va in fold_rates]
    xs = list(range(len(names)))
    colors = ["#3498db"] * 3 + ["#95a5a6"] * 8 + ["#bdc3c7"] * 8
    ax.bar(xs, rates, color=colors)
    ax.axhline(baseline, color="#2c3e50", ls="--", lw=1.2, label=f"baseline {baseline:.3f}")
    ax.axhspan(0.075, 0.10, color="#27ae60", alpha=0.08, label="~8.3% band")
    ax.set_xticks(xs)
    ax.set_xticklabels(names, rotation=60, fontsize=7)
    ax.set_title("D6 — class balance: splits, then 8 train-folds (train/val)", fontsize=11)
    ax.set_ylabel("fraud rate")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "ev9_d6_fold_balance.png"), dpi=110)
    plt.close(fig)
    print("  plot saved: ev9_d6_fold_balance.png")

    # =====================================================================
    print("=" * 60)
    print("D3 — MONTH-LEVEL SEASONALITY of fraud  (report $3 / $4)")
    print("=" * 60)
    sub = pd.to_datetime(df["Claim_Submission_Date"])
    df = df.assign(mo=sub.dt.month, yr=sub.dt.year)
    monthly = (df.groupby(["yr", "mo"], dropna=False)
                 .agg(n=("Claim_ID", "size"), n_fraud=("Is_Fraud", "sum"))
                 .reset_index())
    monthly["fraud_rate"] = monthly["n_fraud"] / monthly["n"]     # denominator = that month's claims
    monthly["ym"] = monthly["yr"].astype(str) + "-" + monthly["mo"].astype(str).str.zfill(2)
    print(f"\n  Date span: {sub.min().date()} -> {sub.max().date()}  ({monthly['ym'].nunique()} months)")
    print(f"  {'month':<8}{'n':>7}{'n_fraud':>9}{'fraud_rate':>12}")
    worst = monthly.nlargest(3, "fraud_rate")
    for _, r in monthly.iterrows():
        mark = "  <-- peak" if r["ym"] in set(worst["ym"]) else ""
        print(f"  {r['ym']:<8}{r['n']:>7}{r['n_fraud']:>9}{r['fraud_rate']:>12.4f}{mark}")
    # seasonality magnitude check: peak vs median month rate
    median_mo_rate = monthly["fraud_rate"].median()
    peak = worst.iloc[0]
    ratio = peak["fraud_rate"] / median_mo_rate if median_mo_rate > 0 else float("nan")
    print(f"\n  Median monthly rate {median_mo_rate:.4f}; peak month {peak['ym']} {peak['fraud_rate']:.4f} "
          f"(ratio {ratio:.2f}x)")
    check(monthly["n"].sum() == n_total, "D3 months sum to full dataset")
    check(len(monthly) >= 12, "D3 spans >=12 months (seasonality view valid)")

    # --- D3 plot: monthly fraud rate + volume ---
    fig, ax1 = plt.subplots(figsize=(11, 4))
    ax1.bar(monthly["ym"], monthly["n"], color="#d5dbdb", label="claims (all)")
    ax1.set_ylabel("claim count", color="#7f8c8d")
    ax1.tick_params(axis="y", labelcolor="#7f8c8d")
    ax2 = ax1.twinx()
    ax2.plot(monthly["ym"], monthly["fraud_rate"], color="#c0392b", marker="o", ms=4, label="fraud rate")
    ax2.axhline(baseline, color="#c0392b", ls="--", lw=1, alpha=0.6, label=f"baseline {baseline:.3f}")
    ax2.set_ylabel("fraud rate", color="#c0392b")
    ax2.tick_params(axis="y", labelcolor="#c0392b")
    ax1.set_xticks(range(len(monthly)))
    ax1.set_xticklabels(monthly["ym"], rotation=90, fontsize=7)
    ax1.set_title("D3 — monthly claim volume vs fraud rate (Submission_Month)", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "ev9_d3_seasonality.png"), dpi=110)
    plt.close(fig)
    print("  plot saved: ev9_d3_seasonality.png")

    # =====================================================================
    print("=" * 60)
    print("RESULT")
    print("=" * 60)
    for ok, label in checks:
        print(f"  {'PASS' if ok else 'FAIL'}  {label}")
    failed = [l for ok, l in checks if not ok]
    print(f"  checks passed: {sum(ok for ok,_ in checks)}/{len(checks)}")
    if failed:
        print("  CHECKS FAILED: " + "; ".join(failed))
        sys.exit(1)
    print("  EVIDENCE QUEUE CLOSED (D4 / D6 / D3)")


if __name__ == "__main__":
    main()