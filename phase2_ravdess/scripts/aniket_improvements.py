#!/usr/bin/env python3
"""
aniket_improvements.py — Phase 2 Improvement Pipeline (Aniket)

Runs entirely from pair_scores.csv — no re-running teammates' scripts needed.
All calibration, feature engineering, balancing, and diagnostics are inline.

  1. Temperature calibration sweep (T=1.0,1.5,2.0,2.5,3.0) on audio+video probs
  2. Stratified balancing (100 pairs per emotion-pair cell)
  3. New configs: C2 (audio probs only, no video), E (+ per-emotion diffs + jsd_max)
  4. Drop-happy diagnostic on Config C

Outputs (does not overwrite originals):
  phase2_ravdess/emotions/audio_emotions_calibrated.csv
  phase2_ravdess/scores/pair_scores_calibrated.csv
  phase2_ravdess/scores/pair_scores_calibrated_balanced.csv
  phase2_ravdess/predictions/predictions_improved.csv
  phase2_ravdess/predictions/experiment_log_improved.csv

RUN:
    python3 phase2_ravdess/scripts/aniket_improvements.py
    python3 phase2_ravdess/scripts/aniket_improvements.py --T 2.5 --n_per_cell 100
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

np.random.seed(42)

# ---------------------------------------------------------------------------
# JSD math
# ---------------------------------------------------------------------------

def kl_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-10) -> float:
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    return float(np.sum(p * np.log(p / q)))


def js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    m = 0.5 * (p + q)
    js_raw = 0.5 * kl_divergence(p, m) + 0.5 * kl_divergence(q, m)
    return float(js_raw / np.log(2))


def _safe_norm(vec: np.ndarray) -> np.ndarray:
    total = vec.sum()
    return vec / total if total > 0 else np.ones(4) / 4


# ---------------------------------------------------------------------------
# Temperature calibration
# ---------------------------------------------------------------------------

def calibrate(probs: np.ndarray, T: float) -> np.ndarray:
    """
    Soften a peaked probability vector with temperature T.
    T=1.0 → identity. T>1.0 → flatter/softer. T=2.5 gave best results in sweep.
    """
    probs = np.clip(probs, 1e-10, 1.0)
    logits = np.log(probs)
    scaled = np.exp(logits / T)
    return scaled / scaled.sum()


def apply_calibration(df: pd.DataFrame, audio_cols: list[str],
                      video_cols: list[str], T: float) -> pd.DataFrame:
    """Return a copy of df with p_audio_* and p_video_* temperature-scaled."""
    df = df.copy()
    for idx in range(len(df)):
        a = calibrate(df.iloc[idx][audio_cols].values.astype(np.float64), T)
        v = calibrate(df.iloc[idx][video_cols].values.astype(np.float64), T)
        df.iloc[idx, df.columns.get_indexer(audio_cols)] = a
        df.iloc[idx, df.columns.get_indexer(video_cols)] = v
    return df


# ---------------------------------------------------------------------------
# JSD recomputation
# ---------------------------------------------------------------------------

AUDIO_COLS = ["p_audio_happy", "p_audio_angry", "p_audio_sad", "p_audio_neutral"]
VIDEO_COLS = ["p_video_happy", "p_video_angry", "p_video_sad", "p_video_neutral"]
TEXT_COLS  = ["p_text_happy",  "p_text_angry",  "p_text_sad",  "p_text_neutral"]


def recompute_jsd(df: pd.DataFrame) -> pd.DataFrame:
    """Recompute the 4 JSD columns from current p_audio_*, p_video_*, p_text_*."""
    df = df.copy()
    ta_list, tv_list, av_list, comp_list = [], [], [], []

    for _, row in df.iterrows():
        pt = _safe_norm(np.array([row[c] for c in TEXT_COLS],  dtype=np.float64))
        pa = _safe_norm(np.array([row[c] for c in AUDIO_COLS], dtype=np.float64))
        pv = _safe_norm(np.array([row[c] for c in VIDEO_COLS], dtype=np.float64))

        ta = js_divergence(pt, pa)
        tv = js_divergence(pt, pv)
        av = js_divergence(pa, pv)
        ta_list.append(round(ta, 6))
        tv_list.append(round(tv, 6))
        av_list.append(round(av, 6))
        comp_list.append(round((ta + tv + av) / 3.0, 6))

    df["jsd_text_audio"]  = ta_list
    df["jsd_text_video"]  = tv_list
    df["jsd_audio_video"] = av_list
    df["jsd_composite"]   = comp_list
    return df


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def add_diff_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add per-emotion absolute differences and jsd_max for Config E."""
    df = df.copy()
    for e in ["happy", "angry", "sad", "neutral"]:
        df[f"diff_{e}"] = (df[f"p_audio_{e}"] - df[f"p_video_{e}"]).abs()
    df["jsd_max"] = df[["jsd_text_audio", "jsd_text_video", "jsd_audio_video"]].max(axis=1)
    return df


def add_gender_feature(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["gender_female"] = (df["gender"].astype(str).str.lower() == "female").astype(int)
    return df


# ---------------------------------------------------------------------------
# Ablation config definitions
# ---------------------------------------------------------------------------

EMOTION_NAMES = ["happy", "angry", "sad", "neutral"]

CONFIGS = {
    "A":  ["jsd_audio_video"],
    "A2": ["jsd_text_audio"],
    "B":  ["jsd_text_audio", "jsd_text_video", "jsd_audio_video"],
    "B2": ["jsd_text_audio", "jsd_audio_video"],
    "C":  ["jsd_text_audio", "jsd_text_video", "jsd_audio_video"]
          + [f"p_audio_{e}" for e in EMOTION_NAMES]
          + [f"p_video_{e}" for e in EMOTION_NAMES],
    "C2": ["jsd_text_audio", "jsd_audio_video"]
          + [f"p_audio_{e}" for e in EMOTION_NAMES],
    "D":  ["jsd_text_audio", "jsd_text_video", "jsd_audio_video"]
          + [f"p_audio_{e}" for e in EMOTION_NAMES]
          + [f"p_video_{e}" for e in EMOTION_NAMES]
          + [f"p_text_{e}"  for e in EMOTION_NAMES]
          + ["intensity", "gender_female"],
    "E":  ["jsd_text_audio", "jsd_text_video", "jsd_audio_video"]
          + [f"p_audio_{e}" for e in EMOTION_NAMES]
          + [f"p_video_{e}" for e in EMOTION_NAMES]
          + ["diff_happy", "diff_angry", "diff_sad", "diff_neutral", "jsd_max"],
}

PREDICTION_COLUMNS = ["pair_id", "label", "pred_prob", "pred_label", "config", "fold_actor"]
EXPERIMENT_COLUMNS = ["config", "n_pairs", "auc_mean", "auc_std",
                      "f1_mean", "precision_mean", "recall_mean"]


# ---------------------------------------------------------------------------
# LOAO cross-validation
# ---------------------------------------------------------------------------

def run_loao(df: pd.DataFrame, config: str) -> tuple[pd.DataFrame, dict]:
    """Leave-one-actor-out CV for a single config. Returns (preds_df, summary)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import LeaveOneGroupOut
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score

    feature_cols = CONFIGS[config]
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        print(f"  ERROR config {config}: missing features {missing}", file=sys.stderr)
        return pd.DataFrame(columns=PREDICTION_COLUMNS), {}

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

        if len(set(y_all[test_idx])) > 1:
            auc_list.append(roc_auc_score(y_all[test_idx], prob))
        f1_list.append(f1_score(y_all[test_idx], pred, zero_division=0))
        p_list.append(precision_score(y_all[test_idx], pred, zero_division=0))
        r_list.append(recall_score(y_all[test_idx], pred, zero_division=0))

    summary = {
        "config":         config,
        "n_pairs":        len(df),
        "auc_mean":       round(float(np.mean(auc_list)),  4),
        "auc_std":        round(float(np.std(auc_list)),   4),
        "f1_mean":        round(float(np.mean(f1_list)),   4),
        "precision_mean": round(float(np.mean(p_list)),    4),
        "recall_mean":    round(float(np.mean(r_list)),    4),
    }
    return pd.DataFrame(pred_rows, columns=PREDICTION_COLUMNS), summary


# ---------------------------------------------------------------------------
# T-sweep: find best calibration temperature
# ---------------------------------------------------------------------------

def sweep_T(df: pd.DataFrame, T_values: list[float]) -> float:
    """Run Config C LOAO for each T, return the T with highest AUC."""
    print("\n  === Temperature calibration sweep (Config C) ===")
    best_T, best_auc = 1.0, 0.0
    for T in T_values:
        cal = apply_calibration(df, AUDIO_COLS, VIDEO_COLS, T)
        cal = recompute_jsd(cal)
        cal = add_diff_features(cal)
        _, summary = run_loao(cal, "C")
        auc = summary["auc_mean"]
        marker = " ←" if auc > best_auc else ""
        print(f"    T={T:.1f}  AUC={auc:.4f}{marker}")
        if auc > best_auc:
            best_auc = auc
            best_T   = T
    print(f"  Best T={best_T}  AUC={best_auc:.4f}")
    return best_T


# ---------------------------------------------------------------------------
# Stratified balancing
# ---------------------------------------------------------------------------

def stratified_balance(df: pd.DataFrame, n_per_cell: int = 100) -> pd.DataFrame:
    """Sample up to n_per_cell rows per (audio_emotion_4class, video_emotion_4class)."""
    balanced = (
        df.groupby(["audio_emotion_4class", "video_emotion_4class"], group_keys=False)
        .apply(lambda g: g.sample(n=min(len(g), n_per_cell), random_state=42))
        .reset_index(drop=True)
        .sample(frac=1, random_state=42)  # shuffle
        .reset_index(drop=True)
    )
    return balanced


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run(pair_scores_path: str, audio_emotions_path: str, T: float,
        n_per_cell: int, out_dir_emotions: str, out_dir_scores: str,
        out_dir_predictions: str) -> None:

    print(f"\nLoading pair_scores: {pair_scores_path}")
    df_orig = pd.read_csv(pair_scores_path)
    print(f"  {len(df_orig)} pairs  label dist: {df_orig['label'].value_counts().to_dict()}")

    df_orig = add_gender_feature(df_orig)

    # ------------------------------------------------------------------ #
    # Step 1: T-sweep → pick best T
    # ------------------------------------------------------------------ #
    T_values = [1.0, 1.5, 2.0, 2.5, 3.0]
    best_T = sweep_T(df_orig, T_values)
    if T != 0.0:
        print(f"  (--T {T} overrides sweep result {best_T})")
        best_T = T

    # ------------------------------------------------------------------ #
    # Step 2: Build calibrated pair_scores
    # ------------------------------------------------------------------ #
    print(f"\n  Calibrating probs at T={best_T} ...")
    df_cal = apply_calibration(df_orig, AUDIO_COLS, VIDEO_COLS, best_T)
    df_cal = recompute_jsd(df_cal)
    df_cal = add_diff_features(df_cal)

    cal_scores_path = os.path.join(out_dir_scores, "pair_scores_calibrated.csv")
    os.makedirs(out_dir_scores, exist_ok=True)
    df_cal.to_csv(cal_scores_path, index=False)
    print(f"  WROTE {cal_scores_path} {len(df_cal)} rows")

    # Also write calibrated audio_emotions (for completeness / Aaditya's plots)
    audio_df = pd.read_csv(audio_emotions_path)
    prob_cols_audio = ["p_happy", "p_angry", "p_sad", "p_neutral"]
    audio_cal = audio_df.copy()
    for idx in range(len(audio_cal)):
        audio_cal.iloc[idx, audio_cal.columns.get_indexer(prob_cols_audio)] = calibrate(
            audio_cal.iloc[idx][prob_cols_audio].values.astype(np.float64), best_T
        )
    cal_audio_path = os.path.join(out_dir_emotions, "audio_emotions_calibrated.csv")
    os.makedirs(out_dir_emotions, exist_ok=True)
    audio_cal.to_csv(cal_audio_path, index=False)
    print(f"  WROTE {cal_audio_path} {len(audio_cal)} rows")

    # ------------------------------------------------------------------ #
    # Step 3: Stratified balanced version
    # ------------------------------------------------------------------ #
    print(f"\n  Stratified balancing ({n_per_cell} per emotion-pair cell) ...")
    df_bal = stratified_balance(df_cal, n_per_cell)
    print(f"  {len(df_bal)} pairs after balancing  "
          f"label dist: {df_bal['label'].value_counts().to_dict()}")

    bal_path = os.path.join(out_dir_scores, "pair_scores_calibrated_balanced.csv")
    df_bal.to_csv(bal_path, index=False)
    print(f"  WROTE {bal_path} {len(df_bal)} rows")

    # ------------------------------------------------------------------ #
    # Step 4a: Main ablation on FULL calibrated data (primary result)
    # ------------------------------------------------------------------ #
    print("\n  === Main ablation (full calibrated, 10,848 pairs) ===")
    run_configs = ["A", "A2", "B", "B2", "C", "C2", "D", "E"]
    all_preds  = []
    summaries  = []

    for cfg in run_configs:
        print(f"    Config {cfg} ({len(CONFIGS[cfg])} features)...", end=" ", flush=True)
        preds_df, summary = run_loao(df_cal, cfg)
        if summary:
            all_preds.append(preds_df)
            summaries.append(summary)
            print(f"AUC={summary['auc_mean']:.4f} ± {summary['auc_std']:.4f}  "
                  f"F1={summary['f1_mean']:.4f}")

    # ------------------------------------------------------------------ #
    # Step 4b: Balanced ablation (Configs C and E only — informational)
    # ------------------------------------------------------------------ #
    print("\n  === Balanced ablation (configs C, E — 1,600 pairs) ===")
    for cfg in ["C", "E"]:
        print(f"    Config {cfg}_bal ({len(CONFIGS[cfg])} features)...", end=" ", flush=True)
        preds_df_bal, summary_bal = run_loao(df_bal, cfg)
        if summary_bal:
            summary_bal["config"] = f"{cfg}_bal"
            summary_bal["n_pairs"] = len(df_bal)
            all_preds.append(preds_df_bal)
            summaries.append(summary_bal)
            print(f"AUC={summary_bal['auc_mean']:.4f} ± {summary_bal['auc_std']:.4f}  "
                  f"F1={summary_bal['f1_mean']:.4f}")

    # ------------------------------------------------------------------ #
    # Step 5: Drop-happy diagnostic (Config C on calibrated full data)
    # ------------------------------------------------------------------ #
    print("\n  === Drop-happy diagnostic (Config C, calibrated full) ===")
    df_nohappy = df_cal[df_cal["audio_emotion_4class"] != "happy"].copy()
    print(f"  Pairs without happy-audio: {len(df_nohappy)} (was {len(df_cal)})")
    _, summary_nohappy = run_loao(df_nohappy, "C")
    if summary_nohappy:
        summary_nohappy["config"] = "C_no_happy"
        summary_nohappy["n_pairs"] = len(df_nohappy)
        summaries.append(summary_nohappy)
        print(f"  AUC={summary_nohappy['auc_mean']:.4f}  "
              f"(full calibrated C = {[s for s in summaries if s['config']=='C'][0]['auc_mean']:.4f})")

    # ------------------------------------------------------------------ #
    # Write outputs
    # ------------------------------------------------------------------ #
    os.makedirs(out_dir_predictions, exist_ok=True)

    predictions_df = pd.concat(all_preds, ignore_index=True)
    experiment_df  = pd.DataFrame(summaries, columns=EXPERIMENT_COLUMNS)

    pred_path = os.path.join(out_dir_predictions, "predictions_improved.csv")
    exp_path  = os.path.join(out_dir_predictions, "experiment_log_improved.csv")
    predictions_df.to_csv(pred_path, index=False)
    experiment_df.to_csv(exp_path, index=False)
    print(f"\nWROTE {pred_path} {len(predictions_df)} rows")
    print(f"WROTE {exp_path}  {len(experiment_df)} rows")

    # ------------------------------------------------------------------ #
    # Final comparison table
    # ------------------------------------------------------------------ #
    baseline = {
        "A": 0.5375, "B": 0.5738, "C": 0.6115, "D": 0.6106,
    }
    print("\n" + "=" * 70)
    print(f"  FINAL RESULTS  (T={best_T}, n_per_cell={n_per_cell})")
    print("=" * 70)
    print(f"  {'Config':<12} {'Features':>8}  {'AUC_new':>8}  {'AUC_orig':>9}  {'Delta':>7}")
    print("  " + "-" * 58)
    for s in summaries:
        cfg  = s["config"]
        auc  = s["auc_mean"]
        orig = baseline.get(cfg, float("nan"))
        delta = f"+{auc - orig:.4f}" if not pd.isna(orig) else "    new"
        print(f"  {cfg:<12} {len(CONFIGS.get(cfg, []))!s:>8}  {auc:>8.4f}  "
              f"{orig!s:>9}  {delta:>7}")
    print("=" * 70)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 2 improvement pipeline: calibration, balancing, new configs",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--scores",     default="phase2_ravdess/scores/pair_scores.csv")
    parser.add_argument("--audio_emo",  default="phase2_ravdess/emotions/audio_emotions.csv")
    parser.add_argument(
        "--T", type=float, default=0.0,
        help="Force a specific temperature (0 = auto-sweep)",
    )
    parser.add_argument("--n_per_cell", type=int, default=100,
                        help="Pairs per emotion-pair cell after stratified balancing")
    parser.add_argument("--out_emotions",    default="phase2_ravdess/emotions")
    parser.add_argument("--out_scores",      default="phase2_ravdess/scores")
    parser.add_argument("--out_predictions", default="phase2_ravdess/predictions")
    args = parser.parse_args()

    for p in (args.scores, args.audio_emo):
        if not os.path.isfile(p):
            print(f"ERROR: not found: {p}", file=sys.stderr)
            sys.exit(1)

    print("START aniket_improvements")
    run(
        pair_scores_path   = args.scores,
        audio_emotions_path= args.audio_emo,
        T                  = args.T,
        n_per_cell         = args.n_per_cell,
        out_dir_emotions   = args.out_emotions,
        out_dir_scores     = args.out_scores,
        out_dir_predictions= args.out_predictions,
    )
    print("\nDONE")


if __name__ == "__main__":
    main()
