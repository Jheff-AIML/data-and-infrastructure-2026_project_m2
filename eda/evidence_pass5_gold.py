#!/usr/bin/env python
"""evidence_pass5_gold.py — reproduce the GOLD tier transformation and assert its contract.

GOLD reads SILVER from GCS. We do not have the GCS silver object locally, so we
reconstruct the *silver-equivalent* (cast + validate + impute per D1/D2/D17) from the
raw CSV first, then run the gold featurization (gold tier, design note 08).

This proves the gold outcome is correct regardless of GCS: feature set, exclusions,
split balance, velocity 'fast'=<=6 (D14), log1p amounts (D13), zero nulls, row counts.

Usage:  python eda/evidence_pass5_gold.py [--csv PATH] [--out DIR]
"""
import argparse, hashlib, json, os, sys, tempfile

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

DEFAULT_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "..", "data", "healthcare_fraud_detection.csv")
DEFAULT_OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gold_out")

# ---- M1 feature set moved into gold (design note 08), minus D10/D12/D3 exclusions ----
CATEGORICAL_FEATURES = [
    "Diagnosis_Code", "Procedure_Code", "Insurance_Type", "Visit_Type",
    "Provider_Specialty", "Patient_State",
]
NUMERIC_FEATURES = [
    "Claim_Amount", "Approved_Amount", "Number_of_Claims_Per_Provider_Monthly",
    "Length_of_Stay", "Chronic_Condition_Flag", "Prior_Visits_12m",
    "Hist_Pct_Fast_Claims", "Hist_Mean_Lag", "Hist_Lag_Std",
    "Current_vs_Hist_Mean_Diff", "Current_Speed_Z_Score",
    "Submission_Month", "Is_Submission_Weekend",
]
FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES  # 19 features
TARGET = "Is_Fraud"
SPLIT = "split"

# ---- (a) SILVER-equivalent: cast + validate + impute (D1/D2/D17) ----
def build_silver(df):
    df = df.copy()
    df["Is_Fraud"] = df["Is_Fraud"].astype(int)
    for c in ("Claim_Amount", "Approved_Amount", "Length_of_Stay"):
        df[c] = df[c].astype(float)
    for c in ("Insurance_Type", "Provider_Specialty"):
        df[c] = df[c].fillna("Unknown")
    df["Prior_Visits_12m"] = df["Prior_Visits_12m"].fillna(0.0)
    return df

# ---- (b) GOLD featurization (gold tier, design note 08) ----
def gold_tier(silver):
    gold = silver.copy()

    # (1) time features (D3); drop raw date
    sub = pd.to_datetime(gold["Claim_Submission_Date"])
    gold["Submission_Month"] = sub.dt.month.astype(int)
    gold["Is_Submission_Weekend"] = sub.dt.weekday.isin([5, 6]).astype(int)
    gold = gold.drop(columns=["Claim_Submission_Date"])

    # (2) stratified split + tag (D6/D15)
    train, temp = train_test_split(gold, test_size=0.20, random_state=42, stratify=gold["Is_Fraud"])
    dev, test = train_test_split(temp, test_size=0.50, random_state=42, stratify=temp["Is_Fraud"])
    train["split"], dev["split"], test["split"] = "train", "dev", "test"

    # (3) velocity profiling on TRAIN ONLY (D7/D14 leak-free; 'fast' = <=6)
    stats = (train.groupby("Provider_ID")
             .agg(Hist_Pct_Fast_Claims=("Days_Between_Service_and_Claim", lambda x: (x <= 6).mean()),
                  Hist_Mean_Lag=("Days_Between_Service_and_Claim", "mean"),
                  Hist_Lag_Std=("Days_Between_Service_and_Claim", "std"))
             .reset_index())
    stats["Hist_Lag_Std"] = stats["Hist_Lag_Std"].fillna(0.0)
    g_fast = (train["Days_Between_Service_and_Claim"] <= 6).mean()
    g_mean = train["Days_Between_Service_and_Claim"].mean()
    g_std = train["Days_Between_Service_and_Claim"].std()

    for name, df_ in (("train", train), ("dev", dev), ("test", test)):
        df_ = df_.merge(stats, on="Provider_ID", how="left")
        for col, fallback in (("Hist_Pct_Fast_Claims", g_fast), ("Hist_Mean_Lag", g_mean), ("Hist_Lag_Std", g_std)):
            df_[col] = df_[col].fillna(fallback)
        df_["Current_vs_Hist_Mean_Diff"] = df_["Days_Between_Service_and_Claim"] - df_["Hist_Mean_Lag"]
        df_["Current_Speed_Z_Score"] = df_["Current_vs_Hist_Mean_Diff"] / (df_["Hist_Lag_Std"] + 1e-5)
        df_[SPLIT] = name
        if name == "train":
            train = df_
        elif name == "dev":
            dev = df_
        else:
            test = df_

    full = pd.concat([train, dev, test])
    # (4) deterministic amount transform at gold (D13); scaling stays in pipeline
    for c in ("Claim_Amount", "Approved_Amount"):
        full[c] = np.log1p(full[c])

    # (5) assemble pre-encoding matrix + target + split (D16)
    return full[FEATURES + [TARGET, SPLIT]]


def sha256_of_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=DEFAULT_CSV,
                    help="path to healthcare_fraud_detection.csv (default: bundled data/ copy)")
    ap.add_argument("--out", default=DEFAULT_OUT)
    args = ap.parse_args()
    if not os.path.exists(args.csv):
        print(f"error: CSV not found at {args.csv}")
        print("The default expects the bundled copy at <repo>/data/healthcare_fraud_detection.csv.")
        print("Pass a different file with --csv path/to/healthcare_fraud_detection.csv "
              "(e.g. a fresh kagglehub download).")
        sys.exit(1)

    raw = pd.read_csv(args.csv)
    silver = build_silver(raw)
    assert silver.isnull().sum().sum() == 0, "silver must be null-free"
    assert silver.shape[0] == 10000, "row count"

    gold = gold_tier(silver)

    os.makedirs(args.out, exist_ok=True)
    csv_path = os.path.join(args.out, "healthcare_fraud_gold.csv")
    pqt_path = os.path.join(args.out, "healthcare_fraud_gold.parquet")
    gold.to_csv(csv_path, index=False)
    gold.to_parquet(pqt_path, index=False)

    # ---- ASSERTIONS ----
    checks = []
    def check(cond, label):
        checks.append((bool(cond), label))

    check(set(FEATURES + [TARGET, SPLIT]) == set(gold.columns),
          f"gold has exactly 19 features + target + split (got {len(gold.columns)} cols)")
    check(sorted(gold[SPLIT].unique()) == sorted(["train", "dev", "test"]), "split tags are train/dev/test")

    # exclusions: D10 / D12 / D3 / identifiers truly absent
    for forbidden in ("Claim_Status", "Patient_Age", "Patient_Gender",
                      "Claim_Submission_Date", "Claim_ID", "Provider_ID", "Days_Between_Service_and_Claim"):
        check(forbidden not in gold.columns, f"excluded per D10/D12/D3: '{forbidden}' not in gold")

    # split sizes + balance
    n = len(gold)
    tr, dv, te = (len(gold[gold[SPLIT] == s]) for s in ("train", "dev", "test"))
    check(tr == 8000 and dv == 1000 and te == 1000, f"split sizes 8000/1000/1000 (got {tr}/{dv}/{te})")
    for s in ("train", "dev", "test"):
        r = gold.loc[gold[SPLIT] == s, "Is_Fraud"].mean()
        check(0.075 < r < 0.10, f"split '{s}' fraud rate {r:.4f} in ~8.3% band")

    # velocity 'fast'=<=6 reconciles the 100%-under-6-days story (D14)
    fraud_days = raw.loc[raw["Is_Fraud"] == 1, "Days_Between_Service_and_Claim"]
    fraud_pct_under6 = (fraud_days <= 6).mean()
    check(fraud_pct_under6 >= 0.999, f"D14: 100% of FRAUD claims <=6 days (got {fraud_pct_under6:.5f})")

    # amounts log1p'd (D13): skew collapsed vs raw
    raw_claim_skew = raw["Claim_Amount"].skew()
    gold_claim_skew = gold["Claim_Amount"].skew()
    check(gold_claim_skew < 0.3, f"log1p tames Claim_Amount skew {raw_claim_skew:.3f}->{gold_claim_skew:.3f} (<0.3)")

    # zero nulls across the matrix
    check(gold.isnull().sum().sum() == 0, "gold matrix has zero nulls")
    check(len(gold) == 10000, f"row count preserved 10000 (got {len(gold)})")

    # ---- manifest ----
    manifest = {
        "rows": len(gold),
        "cols": len(gold.columns),
        "sha256": sha256_of_file(csv_path),
        "features": FEATURES,
        "target": TARGET,
        "split": SPLIT,
        "split_counts": {"train": int(tr), "dev": int(dv), "test": int(te)},
        "fraud_rate": float(gold["Is_Fraud"].mean()),
        "source": "silver-equivalent (local) -> gold tier",
    }
    with open(os.path.join(args.out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    # ---- REPORT ----
    failed = [lbl for ok, lbl in checks if not ok]
    print(f"GOLD TIER PROOF (evidence_pass5)")
    print(f"  input rows              : {len(raw)}")
    print(f"  gold shape              : {gold.shape}  (19 features + {TARGET} + {SPLIT})")
    print(f"  split sizes             : train={tr} dev={dv} test={te}")
    print(f"  fraud rate (overall)    : {gold['Is_Fraud'].mean():.4f}")
    print(f"  fraud by split          : " + ", ".join(
        f"{s}={gold.loc[gold[SPLIT]==s,'Is_Fraud'].mean():.4f}" for s in ("train", "dev", "test")))
    print(f"  100% fraud <=6d (D14)     : {(raw.loc[raw['Is_Fraud']==1,'Days_Between_Service_and_Claim']<=6).mean():.4f}")
    print(f"  Claim_Amount skew       : {raw_claim_skew:.3f} -> {gold_claim_skew:.3f} (log1p, D13)")
    print(f"  nulls in gold           : {int(gold.isnull().sum().sum())}")
    print(f"  checks passed           : {sum(ok for ok,_ in checks)}/{len(checks)}")
    print(f"  artifacts               : {csv_path}")
    print(f"                            {pqt_path}")
    print(f"  manifest sha256         : {manifest['sha256'][:16]}...")
    if failed:
        print("  CHECKS FAILED: " + "; ".join(failed))
        sys.exit(1)
    print("  GOLD CONTRACT VERIFIED")


if __name__ == "__main__":
    main()