#!/usr/bin/env python3
"""
charan_diarization.py — Step 1 of 7: Speaker Diarization
Produces a diarization CSV from a single .wav file using pyannote.audio.

Usage:
    python charan/charan_diarization.py \
        --audio Datasets/amicorpus/ES2002a/audio/ES2002a.Mix-Headset.wav \
        --output outputs/diarization/ES2002a_diarization.csv

    python charan/charan_diarization.py \
        --audio Datasets/amicorpus/ES2002a/audio/ES2002a.Mix-Headset.wav \
        --output outputs/diarization/ES2002a_diarization.csv \
        --hf_token YOUR_HUGGINGFACE_TOKEN

Quality report on existing CSV:
    python charan/charan_diarization.py \
        --quality_report outputs/diarization/ES2002a_diarization.csv
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
from pyannote.audio import Pipeline


# ---------------------------------------------------------------------------
# Device helpers
# ---------------------------------------------------------------------------

def _auto_device() -> str:
    """Detect the best available device: cuda > mps > cpu."""
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ---------------------------------------------------------------------------
# Diarization pipeline
# ---------------------------------------------------------------------------

def run_diarization(
    audio_path: str,
    output_path: str,
    hf_token: str | None,
    device: str,
) -> None:
    """Run pyannote speaker diarization and write CSV."""

    # Input validation
    if not os.path.isfile(audio_path):
        print(f"ERROR: Audio file not found: {audio_path}", file=sys.stderr)
        sys.exit(1)

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    print(f"Device: {device}")
    print(f"[1/3] Loading pyannote diarization pipeline")

    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        token=hf_token,
    )
    pipeline.to(torch.device(device))

    print(f"[2/3] Running diarization on: {audio_path}")
    result = pipeline(audio_path)

    # Use exclusive_speaker_diarization (no overlapping speech turns)
    annotation = result.exclusive_speaker_diarization

    # Build rows from diarization output — sorted by start time, no overlaps
    print("[3/3] Building CSV rows")
    rows = []
    for seg_id, (segment, _, speaker) in enumerate(annotation.itertracks(yield_label=True)):
        rows.append({
            "segment_id": seg_id,
            "start": round(segment.start, 2),
            "end": round(segment.end, 2),
            "speaker_id": speaker,
        })

    df = pd.DataFrame(rows, columns=["segment_id", "start", "end", "speaker_id"])

    # Sort by start time and re-index segment_id
    df = df.sort_values("start").reset_index(drop=True)
    df["segment_id"] = range(len(df))

    # Remove overlapping segments: if a segment starts before the previous one
    # ends, trim its start to the previous segment's end. If that makes it
    # zero-length or negative, drop it.
    cleaned = []
    prev_end = -1.0
    for _, row in df.iterrows():
        start = row["start"]
        end = row["end"]
        if start < prev_end:
            start = round(prev_end, 2)
        if start >= end:
            continue
        cleaned.append({
            "segment_id": 0,  # will be reassigned
            "start": start,
            "end": end,
            "speaker_id": row["speaker_id"],
        })
        prev_end = end

    df = pd.DataFrame(cleaned, columns=["segment_id", "start", "end", "speaker_id"])
    df["segment_id"] = range(len(df))

    df.to_csv(output_path, index=False)
    print(f"WROTE {output_path} {len(df)} rows")


# ---------------------------------------------------------------------------
# Quality report
# ---------------------------------------------------------------------------

def run_quality_report(csv_path: str) -> None:
    """Validate schema and print stats for an existing diarization CSV."""
    if not os.path.isfile(csv_path):
        print(f"ERROR: CSV not found: {csv_path}", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(csv_path)

    expected_cols = ["segment_id", "start", "end", "speaker_id"]
    assert list(df.columns) == expected_cols, \
        f"Column mismatch: got {list(df.columns)}, expected {expected_cols}"
    assert df["segment_id"].is_monotonic_increasing, \
        "segment_id is not monotonically increasing"
    assert df["segment_id"].iloc[0] == 0, \
        f"segment_id must start at 0, got {df['segment_id'].iloc[0]}"
    assert (df["start"] >= 0).all(), \
        "Negative start times found"
    assert (df["end"] > df["start"]).all(), \
        "end <= start in one or more rows"

    # Check no overlaps (each start >= previous end)
    starts = df["start"].values
    ends = df["end"].values
    for i in range(1, len(starts)):
        assert starts[i] >= ends[i - 1], \
            f"Overlap at rows {i-1}-{i}: prev end={ends[i-1]}, cur start={starts[i]}"

    # Check sorted by start
    assert (df["start"].diff().dropna() >= 0).all(), \
        "Rows are not sorted by start time"

    duration = df["end"].max() - df["start"].min()
    speakers = df["speaker_id"].nunique()
    print(f"PASS — {len(df)} segments, {speakers} speakers")
    print(f"  Duration span   : {duration:.2f}s")
    print(f"  Speakers        : {df['speaker_id'].unique().tolist()}")
    print(f"  Mean seg length : {(df['end'] - df['start']).mean():.2f}s")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Speaker diarization pipeline — Step 1 of 7",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--audio", type=str, help="Path to input .wav file")
    parser.add_argument("--output", type=str, help="Path to output CSV")
    parser.add_argument(
        "--hf_token", type=str, default=None,
        help="HuggingFace token for pyannote model access (or set HF_TOKEN env var)",
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Device: cuda / mps / cpu (auto-detected if omitted)",
    )
    parser.add_argument(
        "--quality_report", type=str,
        help="Path to existing diarization CSV to validate (skips diarization)",
    )

    args = parser.parse_args()

    if args.quality_report:
        run_quality_report(args.quality_report)
        return

    if not (args.audio and args.output):
        parser.print_help()
        sys.exit(1)

    device = args.device or _auto_device()
    hf_token = args.hf_token or os.environ.get("HF_TOKEN")

    run_diarization(
        audio_path=args.audio,
        output_path=args.output,
        hf_token=hf_token,
        device=device,
    )


if __name__ == "__main__":
    np.random.seed(42)
    torch.manual_seed(42)
    main()
