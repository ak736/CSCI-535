#!/usr/bin/env python3
"""
run_video_emotion.py — Step 2b: Visual Emotion Modeling

Maps OpenFace/py-feat AU features to emotion probabilities using
a rule-based AU-to-emotion mapping.

Emotion mapping (based on FACS literature):
    happy:   AU06 (cheek raiser) + AU12 (lip corner puller)
    angry:   AU04 (brow lowerer) + AU17 (chin raiser) + AU25 (lips part)
    sad:     AU01 (inner brow raise) + AU04 (brow lowerer) + AU15 (lip corner depressor)
    neutral: default / low activation across all AUs

Applies softmax to produce probabilities, then renormalizes to ensure
exact sum of 1.0.

Usage:
    python scripts/run_video_emotion.py --meeting ES2002a

    # Batch all meetings:
    python scripts/run_video_emotion.py --all

Input:
    outputs/video_features/ESXXXX_video_features.csv

Output:
    outputs/video_emotions/ESXXXX_video_emotions.csv
    Columns: meeting_id, segment_id, p_happy, p_angry, p_sad, p_neutral
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

# Fixed seed
np.random.seed(42)

MEETINGS = [
    "ES2002a", "ES2002b", "ES2002c", "ES2002d",
    "ES2003a", "ES2003b", "ES2003c", "ES2003d",
    "ES2004a", "ES2004b", "ES2004c", "ES2004d",
]

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEATURES_DIR = os.path.join(BASE_DIR, "outputs", "video_features")
EMOTIONS_DIR = os.path.join(BASE_DIR, "outputs", "video_emotions")
SEGMENTS_DIR = os.path.join(BASE_DIR, "outputs", "segments")


def au_to_emotion_scores(row):
    """Map AU intensities to raw emotion scores.

    Based on FACS (Facial Action Coding System) literature:
    - Happy: AU06 + AU12 (Duchenne smile)
    - Angry: AU04 + AU17 + AU25 (furrowed brow, tensed chin, parted lips)
    - Sad: AU01 + AU04 + AU15 (raised inner brow, lowered brow, depressed lip corners)
    - Neutral: inverse of total AU activation (high when AUs are low)
    """
    au01 = float(row.get("AU01", 0))
    au02 = float(row.get("AU02", 0))
    au04 = float(row.get("AU04", 0))
    au06 = float(row.get("AU06", 0))
    au12 = float(row.get("AU12", 0))
    au15 = float(row.get("AU15", 0))
    au17 = float(row.get("AU17", 0))
    au25 = float(row.get("AU25", 0))

    # Raw scores (weighted sums of relevant AUs)
    happy_score = 0.5 * au06 + 0.5 * au12
    angry_score = 0.4 * au04 + 0.3 * au17 + 0.3 * au25
    sad_score = 0.4 * au01 + 0.3 * au04 + 0.3 * au15

    # Neutral score: based on low overall activation
    total_activation = au01 + au02 + au04 + au06 + au12 + au15 + au17 + au25
    neutral_score = max(0.0, 1.0 - 0.15 * total_activation)

    return happy_score, angry_score, sad_score, neutral_score


def softmax(scores):
    """Compute softmax of scores, numerically stable."""
    scores = np.array(scores, dtype=np.float64)
    scores = scores - np.max(scores)  # numerical stability
    exp_scores = np.exp(scores)
    probs = exp_scores / np.sum(exp_scores)
    return probs


def process_meeting(meeting_id):
    """Convert video features to emotion probabilities for one meeting."""
    features_path = os.path.join(FEATURES_DIR, f"{meeting_id}_video_features.csv")
    segments_path = os.path.join(SEGMENTS_DIR, f"{meeting_id}_segments.csv")

    if not os.path.exists(features_path):
        print(f"ERROR: {features_path} not found. Run extract_video_features.py first.", file=sys.stderr)
        sys.exit(1)

    if not os.path.exists(segments_path):
        print(f"ERROR: {segments_path} not found", file=sys.stderr)
        sys.exit(1)

    features_df = pd.read_csv(features_path)
    segments_df = pd.read_csv(segments_path)
    warnings_log = []

    rows = []
    for _, feat_row in features_df.iterrows():
        segment_id = int(feat_row["segment_id"])
        meeting = feat_row["meeting_id"]

        # Check if this is a no-face fallback row (all AUs == 0.0)
        au_values = [float(feat_row.get(col, 0)) for col in ["AU01", "AU02", "AU04", "AU06", "AU12", "AU15", "AU17", "AU25"]]
        is_no_face = all(v == 0.0 for v in au_values)

        if is_no_face:
            # Uniform distribution fallback
            rows.append({
                "meeting_id": meeting,
                "segment_id": segment_id,
                "p_happy": 0.25,
                "p_angry": 0.25,
                "p_sad": 0.25,
                "p_neutral": 0.25,
            })
            warnings_log.append(f"segment_id={segment_id}: no-face fallback [0.25, 0.25, 0.25, 0.25]")
            continue

        # Compute emotion scores from AUs
        happy, angry, sad, neutral = au_to_emotion_scores(feat_row)

        # Apply softmax
        probs = softmax([happy, angry, sad, neutral])

        # Explicit renormalization to ensure exact sum of 1.0
        prob_sum = probs[0] + probs[1] + probs[2] + probs[3]
        probs = probs / prob_sum

        rows.append({
            "meeting_id": meeting,
            "segment_id": segment_id,
            "p_happy": round(float(probs[0]), 6),
            "p_angry": round(float(probs[1]), 6),
            "p_sad": round(float(probs[2]), 6),
            "p_neutral": round(float(probs[3]), 6),
        })

    result_df = pd.DataFrame(rows)
    result_df["segment_id"] = result_df["segment_id"].astype(int)

    # Ensure exact column order
    result_df = result_df[["meeting_id", "segment_id", "p_happy", "p_angry", "p_sad", "p_neutral"]]

    # Final renormalization pass to guarantee sum == 1.0
    prob_cols = ["p_happy", "p_angry", "p_sad", "p_neutral"]
    row_sums = result_df[prob_cols].sum(axis=1)
    for col in prob_cols:
        result_df[col] = result_df[col] / row_sums

    # Validate against reference segments
    ref_segment_ids = set(segments_df["segment_id"].astype(int))
    our_segment_ids = set(result_df["segment_id"].astype(int))

    missing = ref_segment_ids - our_segment_ids
    extra = our_segment_ids - ref_segment_ids

    if missing:
        print(f"ERROR: {len(missing)} segment_ids missing from output: {sorted(missing)[:10]}...", file=sys.stderr)
        sys.exit(1)
    if extra:
        print(f"ERROR: {len(extra)} extra segment_ids in output: {sorted(extra)[:10]}...", file=sys.stderr)
        sys.exit(1)

    # Write warnings
    if warnings_log:
        os.makedirs(EMOTIONS_DIR, exist_ok=True)
        warn_path = os.path.join(EMOTIONS_DIR, "warnings.log")
        with open(warn_path, "a") as f:
            f.write(f"\n--- {meeting_id} ---\n")
            for w in warnings_log:
                f.write(f"  {w}\n")
        print(f"  {len(warnings_log)} no-face fallbacks logged")

    return result_df


def main():
    parser = argparse.ArgumentParser(description="Convert video features to emotion probabilities")
    parser.add_argument("--meeting", type=str, help="Single meeting ID")
    parser.add_argument("--all", action="store_true", help="Process all 12 meetings")
    args = parser.parse_args()

    if not args.meeting and not args.all:
        parser.error("Specify --meeting ESXXXX or --all")

    meetings = MEETINGS if args.all else [args.meeting]

    os.makedirs(EMOTIONS_DIR, exist_ok=True)

    for mid in meetings:
        print(f"\nProcessing {mid}...")
        result_df = process_meeting(mid)

        output_path = os.path.join(EMOTIONS_DIR, f"{mid}_video_emotions.csv")

        # Final validation
        nan_count = result_df.isna().sum().sum()
        if nan_count > 0:
            print(f"ERROR: {nan_count} NaN values in {mid}", file=sys.stderr)
            sys.exit(1)

        dupes = result_df.duplicated(subset=["meeting_id", "segment_id"], keep=False)
        if dupes.any():
            print(f"ERROR: Duplicate (meeting_id, segment_id) in {mid}", file=sys.stderr)
            sys.exit(1)

        # Check probability sums
        prob_cols = ["p_happy", "p_angry", "p_sad", "p_neutral"]
        sums = result_df[prob_cols].sum(axis=1)
        bad_sums = sums[~np.isclose(sums, 1.0, atol=1e-6)]
        if len(bad_sums) > 0:
            print(f"ERROR: {len(bad_sums)} rows don't sum to 1.0 in {mid}", file=sys.stderr)
            sys.exit(1)

        result_df.to_csv(output_path, index=False)
        print(f"WROTE {output_path} {len(result_df)} rows")


if __name__ == "__main__":
    main()
