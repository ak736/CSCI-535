#!/usr/bin/env python3
"""
validate_video_outputs.py — Standalone validation for video pipeline outputs

Verifies that outputs/video_emotions/ESXXXX_video_emotions.csv meets all
pipeline requirements before handoff to Aniket (Step 4).

Checks:
    1. File exists for each meeting
    2. Column names exactly: meeting_id, segment_id, p_happy, p_angry, p_sad, p_neutral
    3. segment_id dtype is integer
    4. meeting_id column is present and correct
    5. Every segment_id matches outputs/segments/ESXXXX_segments.csv (no missing, no extras)
    6. No duplicate (meeting_id, segment_id) pairs
    7. p_happy + p_angry + p_sad + p_neutral == 1.0 ± 1e-6 for every row
    8. No NaN or null values
    9. All probabilities are in [0, 1]

Usage:
    # Validate one meeting:
    python scripts/validate_video_outputs.py --meeting ES2002a

    # Validate all:
    python scripts/validate_video_outputs.py --all

Prints PASS or lists every failure with meeting_id and segment_id.
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

MEETINGS = [
    "ES2002a", "ES2002b", "ES2002c", "ES2002d",
    "ES2003a", "ES2003b", "ES2003c", "ES2003d",
    "ES2004a", "ES2004b", "ES2004c", "ES2004d",
]

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEGMENTS_DIR = os.path.join(BASE_DIR, "outputs", "segments")
EMOTIONS_DIR = os.path.join(BASE_DIR, "outputs", "video_emotions")
FEATURES_DIR = os.path.join(BASE_DIR, "outputs", "video_features")

REQUIRED_COLS = ["meeting_id", "segment_id", "p_happy", "p_angry", "p_sad", "p_neutral"]
PROB_COLS = ["p_happy", "p_angry", "p_sad", "p_neutral"]


def validate_meeting(meeting_id):
    """Validate all outputs for a single meeting. Returns list of failure messages."""
    failures = []

    # --- Check emotions file exists ---
    emotions_path = os.path.join(EMOTIONS_DIR, f"{meeting_id}_video_emotions.csv")
    if not os.path.exists(emotions_path):
        return [f"MISSING: {emotions_path}"]

    # --- Check features file exists ---
    features_path = os.path.join(FEATURES_DIR, f"{meeting_id}_video_features.csv")
    if not os.path.exists(features_path):
        failures.append(f"WARNING: {features_path} not found (intermediate file)")

    # --- Load data ---
    emotions_df = pd.read_csv(emotions_path)
    segments_path = os.path.join(SEGMENTS_DIR, f"{meeting_id}_segments.csv")
    if not os.path.exists(segments_path):
        return [f"MISSING: reference {segments_path}"]
    segments_df = pd.read_csv(segments_path)

    # --- Check columns ---
    actual_cols = list(emotions_df.columns)
    if actual_cols != REQUIRED_COLS:
        failures.append(f"COLUMNS: expected {REQUIRED_COLS}, got {actual_cols}")

    # --- Check meeting_id ---
    unique_meetings = emotions_df["meeting_id"].unique()
    if len(unique_meetings) != 1 or unique_meetings[0] != meeting_id:
        failures.append(f"MEETING_ID: expected all '{meeting_id}', got {unique_meetings}")

    # --- Check segment_id dtype ---
    if not np.issubdtype(emotions_df["segment_id"].dtype, np.integer):
        failures.append(f"DTYPE: segment_id is {emotions_df['segment_id'].dtype}, expected integer")

    # --- Check no NaN ---
    nan_count = emotions_df.isna().sum().sum()
    if nan_count > 0:
        nan_cols = emotions_df.columns[emotions_df.isna().any()].tolist()
        failures.append(f"NaN: {nan_count} NaN values in columns {nan_cols}")

    # --- Check no duplicates ---
    dupes = emotions_df.duplicated(subset=["meeting_id", "segment_id"], keep=False)
    if dupes.any():
        dupe_ids = emotions_df[dupes]["segment_id"].unique().tolist()
        failures.append(f"DUPLICATES: {len(dupe_ids)} duplicate segment_ids: {dupe_ids[:10]}")

    # --- Check segment_id coverage ---
    ref_ids = set(segments_df["segment_id"].astype(int))
    our_ids = set(emotions_df["segment_id"].astype(int))

    missing = ref_ids - our_ids
    extra = our_ids - ref_ids

    if missing:
        failures.append(f"MISSING SEGMENTS: {len(missing)} segment_ids absent: {sorted(missing)[:10]}...")
    if extra:
        failures.append(f"EXTRA SEGMENTS: {len(extra)} unexpected segment_ids: {sorted(extra)[:10]}...")

    # --- Check probabilities sum to 1.0 ---
    if all(col in emotions_df.columns for col in PROB_COLS):
        sums = emotions_df[PROB_COLS].sum(axis=1)
        bad_sums = sums[~np.isclose(sums, 1.0, atol=1e-6)]
        if len(bad_sums) > 0:
            worst = bad_sums.iloc[0]
            bad_idx = bad_sums.index[0]
            seg_id = emotions_df.loc[bad_idx, "segment_id"]
            failures.append(
                f"PROB SUM: {len(bad_sums)} rows don't sum to 1.0. "
                f"Example: segment_id={seg_id}, sum={worst:.8f}"
            )

        # Check all probabilities in [0, 1]
        for col in PROB_COLS:
            below = (emotions_df[col] < -1e-8).sum()
            above = (emotions_df[col] > 1.0 + 1e-8).sum()
            if below > 0:
                failures.append(f"RANGE: {below} negative values in {col}")
            if above > 0:
                failures.append(f"RANGE: {above} values > 1.0 in {col}")

    # --- Row count ---
    if len(emotions_df) != len(segments_df):
        failures.append(
            f"ROW COUNT: emotions has {len(emotions_df)} rows, "
            f"segments has {len(segments_df)} rows"
        )

    return failures


def main():
    parser = argparse.ArgumentParser(description="Validate video pipeline outputs")
    parser.add_argument("--meeting", type=str, help="Single meeting ID")
    parser.add_argument("--all", action="store_true", help="Validate all 12 meetings")
    args = parser.parse_args()

    if not args.meeting and not args.all:
        parser.error("Specify --meeting ESXXXX or --all")

    meetings = MEETINGS if args.all else [args.meeting]

    overall_pass = True
    total_failures = 0

    for mid in meetings:
        failures = validate_meeting(mid)
        if failures:
            overall_pass = False
            total_failures += len(failures)
            print(f"\nFAIL: {mid}")
            for f in failures:
                print(f"  - {f}")
        else:
            print(f"PASS: {mid}")

    print(f"\n{'='*40}")
    if overall_pass:
        print("ALL MEETINGS PASSED VALIDATION")
    else:
        print(f"VALIDATION FAILED: {total_failures} issue(s) across {len(meetings)} meeting(s)")
        sys.exit(1)


if __name__ == "__main__":
    main()
