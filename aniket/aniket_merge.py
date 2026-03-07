#!/usr/bin/env python3
"""
aniket_merge.py — Step 3 (Aniket)

Merges per-meeting diarization and transcript CSVs into a single merged CSV.

For each transcript segment, finds the diarization speaker with the greatest
time overlap and assigns that speaker_id. If no overlap exists (gap/silence),
falls back to the nearest speaker by midpoint distance — so no row is ever
left without a speaker.

Usage:
    python3 aniket/aniket_merge.py \
        --diar  outputs/diarization/ES2002a_diarization.csv \
        --trans outputs/transcripts/ES2002a_transcript.csv \
        --out   outputs/merged/ES2002a_merged.csv

Output columns (must match pipeline spec exactly):
    segment_id, start, end, speaker_id, transcript, asr_confidence
"""

import argparse
import os
import sys

import pandas as pd


# ---------------------------------------------------------------------------
# Overlap helper
# ---------------------------------------------------------------------------

def overlap(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    """Return the length of overlap between two intervals."""
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def assign_speaker(trans_row: pd.Series, diar_df: pd.DataFrame) -> str:
    """
    Assign speaker_id to a transcript segment via max time overlap.
    Falls back to nearest midpoint if no diarization segment overlaps.
    """
    t_start, t_end = trans_row["start"], trans_row["end"]

    overlaps = diar_df.apply(
        lambda row: overlap(t_start, t_end, row["start"], row["end"]), axis=1
    )

    best_idx = overlaps.idxmax()
    if overlaps[best_idx] > 0.0:
        return diar_df.loc[best_idx, "speaker_id"]

    # No overlap — fall back to nearest midpoint
    t_mid = (t_start + t_end) / 2.0
    diar_mids = (diar_df["start"] + diar_df["end"]) / 2.0
    nearest_idx = (diar_mids - t_mid).abs().idxmin()
    return diar_df.loc[nearest_idx, "speaker_id"]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def merge(diar_path: str, trans_path: str, out_path: str) -> None:
    # Validate inputs
    for path in (diar_path, trans_path):
        if not os.path.isfile(path):
            print(f"ERROR: input not found: {path}", file=sys.stderr)
            sys.exit(1)

    diar_df = pd.read_csv(diar_path)
    trans_df = pd.read_csv(trans_path)

    # Column checks — fail loudly if pipeline format is broken
    for col in ("segment_id", "start", "end", "speaker_id"):
        if col not in diar_df.columns:
            print(f"ERROR: diarization CSV missing column '{col}'", file=sys.stderr)
            sys.exit(1)
    for col in ("segment_id", "start", "end", "transcript", "asr_confidence"):
        if col not in trans_df.columns:
            print(f"ERROR: transcript CSV missing column '{col}'", file=sys.stderr)
            sys.exit(1)

    print(f"  diarization : {len(diar_df)} segments  ({diar_path})")
    print(f"  transcript  : {len(trans_df)} segments  ({trans_path})")

    # Assign speakers
    trans_df = trans_df.copy()
    trans_df["speaker_id"] = trans_df.apply(
        lambda row: assign_speaker(row, diar_df), axis=1
    )

    # Build output with correct column order and re-indexed segment_id
    merged = trans_df[["start", "end", "speaker_id", "transcript", "asr_confidence"]].copy()
    merged = merged.reset_index(drop=True)
    merged.insert(0, "segment_id", merged.index)

    # Write
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    merged.to_csv(out_path, index=False)
    print(f"WROTE {out_path} {len(merged)} rows")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge diarization + transcript CSVs by maximum time overlap."
    )
    parser.add_argument("--diar",  required=True, help="Path to diarization CSV")
    parser.add_argument("--trans", required=True, help="Path to transcript CSV")
    parser.add_argument("--out",   required=True, help="Path to output merged CSV")
    args = parser.parse_args()

    print(f"START merge: {args.diar} + {args.trans}")
    merge(args.diar, args.trans, args.out)
    print("DONE")


if __name__ == "__main__":
    main()
