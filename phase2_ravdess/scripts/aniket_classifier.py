#!/usr/bin/env python3
"""
aniket_classifier.py — Phase 2 Step 5 (Aniket)

OWNER: Aniket
STEP : 5 (Day 5-6) — WAITS for Charan Step 4 (pair_scores.csv)

Train a binary classifier (congruent vs incongruent) with leave-one-actor-out
24-fold cross-validation and 4 ablation configurations.

MODEL: sklearn LogisticRegression with class_weight='balanced'
       StandardScaler fit ONLY on training fold (no leakage).

CV: sklearn LeaveOneGroupOut, groups = actor.

ABLATION CONFIGURATIONS:
    A:  jsd_audio_video                                                            (1 feature)
    B:  jsd_text_audio, jsd_text_video, jsd_audio_video                            (3 features)
    C:  B + p_audio_* (4) + p_video_* (4)                                          (11 features)
    D:  C + p_text_* (4) + intensity + gender (one-hot)                            (17 features)

INPUT:
    phase2_ravdess/scores/pair_scores.csv  (from Charan Step 4)

OUTPUTS:
    phase2_ravdess/predictions/predictions.csv       — per-row predictions
        Columns: pair_id, label, pred_prob, pred_label, config, fold_actor

    phase2_ravdess/predictions/experiment_log.csv    — per-config summary
        Columns: config, auc_mean, auc_std, f1_mean, precision_mean, recall_mean

RUN:
    python3 phase2_ravdess/scripts/aniket_classifier.py \\
        --scores          phase2_ravdess/scores/pair_scores.csv \\
        --out_predictions phase2_ravdess/predictions/predictions.csv \\
        --out_experiment  phase2_ravdess/predictions/experiment_log.csv

DOWNSTREAM CONSUMER: Aaditya Step 6 (evaluation + plots)
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

np.random.seed(42)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EMOTION_NAMES = ["happy", "angry", "sad", "neutral"]

FEATURES_A = ["jsd_audio_video"]
FEATURES_B = ["jsd_text_audio", "jsd_text_video", "jsd_audio_video"]
FEATURES_C = FEATURES_B + [f"p_audio_{e}" for e in EMOTION_NAMES] \
                       + [f"p_video_{e}" for e in EMOTION_NAMES]
FEATURES_D = FEATURES_C + [f"p_text_{e}" for e in EMOTION_NAMES] \
                       + ["intensity", "gender_female"]  # gender one-hot: 1 if female

CONFIG_FEATURES = {
    "A": FEATURES_A,
    "B": FEATURES_B,
    "C": FEATURES_C,
    "D": FEATURES_D,
}

CONFIGS = ["A", "B", "C", "D"]
EXPECTED_FOLDS = 24  # 24 actors in RAVDESS

PREDICTION_COLUMNS  = ["pair_id", "label", "pred_prob", "pred_label",
                       "config", "fold_actor"]
EXPERIMENT_COLUMNS  = ["config", "auc_mean", "auc_std",
                       "f1_mean", "precision_mean", "recall_mean"]


# ---------------------------------------------------------------------------
# Feature preparation
# ---------------------------------------------------------------------------

def prepare_features(scores_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add any engineered / encoded features. Currently:
      - gender_female: 1 if gender == 'female', else 0 (one-hot encoding).
    Everything else lives in scores_df already.
    """
    df = scores_df.copy()
    if "gender" in df.columns:
        df["gender_female"] = (df["gender"].astype(str).str.lower() == "female").astype(int)
    else:
        df["gender_female"] = 0
    return df


# ---------------------------------------------------------------------------
# Training / CV
# ---------------------------------------------------------------------------

def run_one_config(df: pd.DataFrame, config: str) -> tuple[pd.DataFrame, dict]:
    """
    Train + evaluate one ablation config with LOAO CV.

    Returns:
        preds_df: per-row predictions with columns PREDICTION_COLUMNS
        summary : dict with auc_mean, auc_std, f1_mean, precision_mean, recall_mean
    """
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import LeaveOneGroupOut
        from sklearn.preprocessing import StandardScaler
        from sklearn.metrics import (
            roc_auc_score, f1_score, precision_score, recall_score,
        )
    except ImportError:
        print("ERROR: install scikit-learn: pip install scikit-learn", file=sys.stderr)
        sys.exit(1)

    feature_cols = CONFIG_FEATURES[config]
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        print(f"ERROR: config {config} missing columns {missing}", file=sys.stderr)
        sys.exit(1)

    X_all    = df[feature_cols].values.astype(np.float64)
    y_all    = df["label"].values.astype(int)
    groups   = df["actor"].values.astype(int)
    pair_ids = df["pair_id"].values

    logo = LeaveOneGroupOut()

    auc_list, f1_list, p_list, r_list = [], [], [], []
    pred_rows = []

    for train_idx, test_idx in logo.split(X_all, y_all, groups):
        scaler  = StandardScaler()
        X_train = scaler.fit_transform(X_all[train_idx])
        X_test  = scaler.transform(X_all[test_idx])

        clf = LogisticRegression(
            class_weight="balanced", max_iter=2000, random_state=42,
        )
        clf.fit(X_train, y_all[train_idx])

        prob = clf.predict_proba(X_test)[:, 1]
        pred = (prob >= 0.5).astype(int)

        fold_actor = int(groups[test_idx][0])
        for pid, yt, pp, pl in zip(pair_ids[test_idx], y_all[test_idx], prob, pred):
            pred_rows.append({
                "pair_id":    pid,
                "label":      int(yt),
                "pred_prob":  round(float(pp), 6),
                "pred_label": int(pl),
                "config":     config,
                "fold_actor": fold_actor,
            })

        # Skip folds where test set has only one class (AUC undefined)
        if len(set(y_all[test_idx])) > 1:
            auc_list.append(roc_auc_score(y_all[test_idx], prob))
        f1_list.append(f1_score(y_all[test_idx], pred, zero_division=0))
        p_list.append(precision_score(y_all[test_idx], pred, zero_division=0))
        r_list.append(recall_score(y_all[test_idx], pred, zero_division=0))

    summary = {
        "config":          config,
        "auc_mean":        round(float(np.mean(auc_list)),  4),
        "auc_std":         round(float(np.std(auc_list)),   4),
        "f1_mean":         round(float(np.mean(f1_list)),   4),
        "precision_mean":  round(float(np.mean(p_list)),    4),
        "recall_mean":     round(float(np.mean(r_list)),    4),
    }
    preds_df = pd.DataFrame(pred_rows, columns=PREDICTION_COLUMNS)
    return preds_df, summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(scores_path: str, out_predictions: str, out_experiment: str) -> None:
    """Run all 4 ablation configs and write predictions + experiment log."""
    if not os.path.isfile(scores_path):
        print(f"ERROR: scores CSV not found: {scores_path}", file=sys.stderr)
        sys.exit(1)

    scores_df = pd.read_csv(scores_path)
    print(f"  pair_scores: {len(scores_df)} rows  ({scores_path})")

    df = prepare_features(scores_df)

    for col in ("actor", "label", "pair_id"):
        if col not in df.columns:
            print(f"ERROR: pair_scores missing required column '{col}'", file=sys.stderr)
            sys.exit(1)

    all_preds: list[pd.DataFrame] = []
    summaries: list[dict] = []

    for cfg in CONFIGS:
        print(f"\n  Running config {cfg} ({len(CONFIG_FEATURES[cfg])} features)...")
        preds_df, summary = run_one_config(df, cfg)
        all_preds.append(preds_df)
        summaries.append(summary)
        print(f"    AUC={summary['auc_mean']:.4f} ± {summary['auc_std']:.4f}  "
              f"F1={summary['f1_mean']:.4f}  "
              f"P={summary['precision_mean']:.4f}  R={summary['recall_mean']:.4f}")

    predictions_df = pd.concat(all_preds, ignore_index=True)
    experiment_df  = pd.DataFrame(summaries, columns=EXPERIMENT_COLUMNS)

    os.makedirs(os.path.dirname(os.path.abspath(out_predictions)), exist_ok=True)
    predictions_df.to_csv(out_predictions, index=False)
    experiment_df.to_csv(out_experiment, index=False)
    print(f"\nWROTE {out_predictions} {len(predictions_df)} rows")
    print(f"WROTE {out_experiment}  {len(experiment_df)} rows")
    validate(out_predictions, out_experiment)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate(predictions_path: str, experiment_path: str) -> None:
    """
    Assert:
      - predictions.csv covers all 4 configs x 24 folds
      - experiment_log.csv has exactly 4 rows (A-D), auc_mean > 0.5
      - no NaN, pred_prob in [0,1]
    """
    preds = pd.read_csv(predictions_path)
    exp   = pd.read_csv(experiment_path)

    print(f"\nVALIDATE {predictions_path}")
    print(f"  rows: {len(preds)}")

    assert preds.isna().sum().sum() == 0, "NaN values in predictions"
    assert list(preds.columns) == PREDICTION_COLUMNS, (
        f"predictions column mismatch: {list(preds.columns)}"
    )
    assert preds["pred_prob"].between(0.0, 1.0).all(), "pred_prob outside [0,1]"
    assert set(preds["config"].unique()) == set(CONFIGS), "configs missing"
    assert preds["fold_actor"].nunique() == EXPECTED_FOLDS, (
        f"expected {EXPECTED_FOLDS} folds, got {preds['fold_actor'].nunique()}"
    )

    print(f"\nVALIDATE {experiment_path}")
    assert len(exp) == len(CONFIGS), f"expected {len(CONFIGS)} rows, got {len(exp)}"
    assert list(exp.columns) == EXPERIMENT_COLUMNS, (
        f"experiment column mismatch: {list(exp.columns)}"
    )
    assert exp.isna().sum().sum() == 0, "NaN values in experiment_log"
    assert (exp["auc_mean"] > 0.5).all(), "at least one config has AUC <= 0.5 (suspicious)"

    print("  OK — 4 configs x 24 folds, all metrics valid, no NaN")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Leave-one-actor-out classifier + ablation study on RAVDESS pairs",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--scores",
        default="phase2_ravdess/scores/pair_scores.csv",
        help="Input pair_scores CSV from Charan Step 4",
    )
    parser.add_argument(
        "--out_predictions",
        default="phase2_ravdess/predictions/predictions.csv",
    )
    parser.add_argument(
        "--out_experiment",
        default="phase2_ravdess/predictions/experiment_log.csv",
    )
    parser.add_argument("--validate_only", action="store_true")
    args = parser.parse_args()

    if args.validate_only:
        validate(args.out_predictions, args.out_experiment)
        return

    print("START aniket_classifier")
    run(args.scores, args.out_predictions, args.out_experiment)
    print("DONE")


if __name__ == "__main__":
    main()
