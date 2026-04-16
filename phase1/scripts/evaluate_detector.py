#!/usr/bin/env python3
"""
evaluate_detector.py — Evaluation script (Post-annotation)

Computes ROC-AUC, Precision, Recall, F1 for the JS divergence incongruence
scorer against human annotations. Also computes Krippendorff's alpha for
inter-rater agreement.

Usage (after annotation is complete):
    python3 scripts/evaluate_detector.py \
        --annotations annotations/annotated_segments.csv \
        --scores      annotations/annotation_sample.csv \
        --out         results/evaluation_metrics.csv

Requirements:
    pip install scikit-learn krippendorff numpy pandas
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Krippendorff's alpha
# ---------------------------------------------------------------------------

def krippendorff_alpha(ratings: np.ndarray) -> float:
    """
    Compute Krippendorff's alpha for ordinal/nominal data.

    ratings: 2D array of shape (n_annotators, n_items)
             NaN = missing rating

    Returns alpha in [-1, 1]. Target > 0.6 for acceptable agreement.
    """
    n_annotators, n_items = ratings.shape

    # Coincidence matrix for binary labels {0, 1}
    labels = [0, 1]
    n_labels = len(labels)
    label_to_idx = {l: i for i, l in enumerate(labels)}

    # Build coincidence matrix
    coincidence = np.zeros((n_labels, n_labels))
    for item in range(n_items):
        item_ratings = ratings[:, item]
        valid = item_ratings[~np.isnan(item_ratings)]
        n_valid = len(valid)
        if n_valid < 2:
            continue
        for i in range(n_valid):
            for j in range(n_valid):
                if i == j:
                    continue
                ri = int(valid[i])
                rj = int(valid[j])
                if ri in label_to_idx and rj in label_to_idx:
                    coincidence[label_to_idx[ri], label_to_idx[rj]] += 1.0 / (n_valid - 1)

    # Observed disagreement
    n_total = coincidence.sum()
    if n_total == 0:
        return float("nan")

    o_disagreement = 0.0
    for i in range(n_labels):
        for j in range(n_labels):
            if i != j:
                o_disagreement += coincidence[i, j]
    o_disagreement /= n_total

    # Expected disagreement (binary: distance = |i-j|, 0 or 1)
    marginals = coincidence.sum(axis=1)
    e_disagreement = 0.0
    for i in range(n_labels):
        for j in range(n_labels):
            if i != j:
                e_disagreement += marginals[i] * marginals[j]
    e_disagreement /= (n_total * (n_total - 1))

    if e_disagreement == 0:
        return 1.0 if o_disagreement == 0 else 0.0

    alpha = 1.0 - (o_disagreement / e_disagreement)
    return float(alpha)


# ---------------------------------------------------------------------------
# Majority vote
# ---------------------------------------------------------------------------

def majority_vote(row: pd.Series) -> int:
    """Return majority label from annotator_1, annotator_2, annotator_3."""
    votes = []
    for col in ["annotator_1", "annotator_2", "annotator_3"]:
        val = row.get(col)
        try:
            votes.append(int(val))
        except (ValueError, TypeError):
            pass
    if not votes:
        return -1  # invalid
    return 1 if sum(votes) > len(votes) / 2 else 0


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(annotations_path: str, scores_path: str, out_path: str) -> None:
    try:
        from sklearn.metrics import (
            roc_auc_score, precision_score, recall_score,
            f1_score, classification_report, roc_curve,
        )
    except ImportError:
        print("ERROR: pip install scikit-learn", file=sys.stderr)
        sys.exit(1)

    # --- Load ---
    ann_df    = pd.read_csv(annotations_path)
    scores_df = pd.read_csv(scores_path)

    # Check annotation completeness
    for col in ("annotator_1", "annotator_2", "annotator_3"):
        if col not in ann_df.columns:
            print(f"ERROR: annotations CSV missing column '{col}'", file=sys.stderr)
            sys.exit(1)
        missing = ann_df[col].isna().sum() + (ann_df[col] == "").sum()
        if missing > 0:
            print(f"  WARNING: {missing} missing values in {col} — those rows will be excluded")

    # --- Majority vote & inter-rater agreement ---
    ann_df["majority_label"] = ann_df.apply(majority_vote, axis=1)
    valid_mask = ann_df["majority_label"].isin([0, 1])
    ann_df = ann_df[valid_mask].copy()
    print(f"  Valid annotated segments: {len(ann_df)}")

    # Krippendorff's alpha
    rating_matrix = ann_df[["annotator_1", "annotator_2", "annotator_3"]].apply(
        pd.to_numeric, errors="coerce"
    ).to_numpy().T  # shape (3, n_items)
    alpha = krippendorff_alpha(rating_matrix)
    print(f"  Krippendorff's alpha: {alpha:.4f}  ({'acceptable' if alpha >= 0.6 else 'low — consider re-annotation'})")

    # --- Merge with final_score ---
    merged = ann_df.merge(
        scores_df[["annotation_id", "final_score"]],
        on="annotation_id", how="inner"
    )
    if len(merged) == 0:
        print("ERROR: no rows matched between annotations and scores on annotation_id", file=sys.stderr)
        sys.exit(1)

    y_true  = merged["majority_label"].to_numpy()
    y_score = merged["final_score"].to_numpy()

    # Binary prediction at threshold 0.3 (tunable)
    threshold = 0.3
    y_pred = (y_score >= threshold).astype(int)

    # --- Metrics ---
    auc       = roc_auc_score(y_true, y_score)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall    = recall_score(y_true, y_pred, zero_division=0)
    f1        = f1_score(y_true, y_pred, zero_division=0)

    print(f"\n  === Evaluation Results ===")
    print(f"  Threshold              : {threshold}")
    print(f"  ROC-AUC                : {auc:.4f}  (target >= 0.75)")
    print(f"  Precision              : {precision:.4f}")
    print(f"  Recall                 : {recall:.4f}")
    print(f"  F1                     : {f1:.4f}")
    print(f"  Krippendorff's alpha   : {alpha:.4f}")
    print(f"\n  Classification report:")
    print(classification_report(y_true, y_pred, target_names=["aligned", "incongruent"]))

    # --- Save results ---
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    results = pd.DataFrame([{
        "model":              "JS_divergence",
        "threshold":          threshold,
        "roc_auc":            round(auc, 4),
        "precision":          round(precision, 4),
        "recall":             round(recall, 4),
        "f1":                 round(f1, 4),
        "krippendorff_alpha": round(alpha, 4),
        "n_segments":         len(merged),
        "n_incongruent_gt":   int(y_true.sum()),
        "text_model":         "j-hartmann/emotion-english-distilroberta-base",
        "audio_model":        "superb/wav2vec2-large-superb-er",
    }])
    results.to_csv(out_path, index=False)
    print(f"\nWROTE {out_path}")

    # Save ROC curve data
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    roc_df = pd.DataFrame({"fpr": fpr, "tpr": tpr, "threshold": thresholds})
    roc_path = out_path.replace(".csv", "_roc_curve.csv")
    roc_df.to_csv(roc_path, index=False)
    print(f"WROTE {roc_path}")

    # Save majority labels back to annotations file
    ann_out = ann_df.copy()
    ann_out.to_csv(annotations_path, index=False)
    print(f"WROTE majority_label back to {annotations_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate JS incongruence detector against human annotations"
    )
    parser.add_argument("--annotations", required=True, help="Path to annotated_segments.csv")
    parser.add_argument("--scores",      required=True, help="Path to annotation_sample.csv (has final_score)")
    parser.add_argument("--out",         required=True, help="Path to output evaluation_metrics.csv")
    args = parser.parse_args()

    print("START evaluation")
    evaluate(args.annotations, args.scores, args.out)
    print("DONE")


if __name__ == "__main__":
    main()
