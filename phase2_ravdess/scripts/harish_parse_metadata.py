#!/usr/bin/env python3
"""
harish_parse_metadata.py — Phase 2 Step 1 (Harish)

Parse every RAVDESS audio-only .wav filename into a structured metadata
table. The RAVDESS filename encodes a 7-part identifier:

    MODALITY-CHANNEL-EMOTION-INTENSITY-STATEMENT-REPETITION-ACTOR
    e.g.  03-01-05-02-01-01-08.wav

INPUT:
    Datasets/ravdess/audio_speech/Actor_01/*.wav  through  Actor_24/*.wav
    (1440 files total — 24 actors x 60 clips each)

OUTPUT:
    phase2_ravdess/metadata/ravdess_metadata.csv

    Columns (exact order):
        clip_id, filename, actor, gender, emotion_code, emotion_label,
        emotion_4class, intensity, statement, repetition

RUN:
    python3 phase2_ravdess/scripts/harish_parse_metadata.py \\
        --audio_root Datasets/ravdess/audio_speech \\
        --out        phase2_ravdess/metadata/ravdess_metadata.csv
"""

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

np.random.seed(42)

# ---------------------------------------------------------------------------
# Constants — RAVDESS filename encoding
# ---------------------------------------------------------------------------

# RAVDESS emotion code → official RAVDESS label
EMOTION_LABEL = {
    1: "neutral",
    2: "calm",
    3: "happy",
    4: "sad",
    5: "angry",
    6: "fearful",
    7: "disgust",
    8: "surprised",
}

# RAVDESS emotion code → our 4-class label space E = {happy, angry, sad, neutral}
FOUR_CLASS = {
    1: "neutral",
    2: "neutral",
    3: "happy",
    4: "sad",
    5: "angry",
    6: "neutral",
    7: "neutral",
    8: "neutral",
}

FOUR_CLASS_SET = {"happy", "angry", "sad", "neutral"}

EXPECTED_ROWS = 1440

METADATA_COLUMNS = [
    "clip_id",
    "filename",
    "actor",
    "gender",
    "emotion_code",
    "emotion_label",
    "emotion_4class",
    "intensity",
    "statement",
    "repetition",
]


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def parse_filename(filename: str) -> dict:
    """
    Parse a RAVDESS audio filename like '03-01-05-02-01-01-08.wav' into
    its 7 integer fields and derived columns.

    Returns dict with: clip_id, filename, actor, gender, emotion_code,
    emotion_label, emotion_4class, intensity, statement, repetition.
    """
    base = filename.strip()
    if not base.lower().endswith(".wav"):
        raise ValueError(f"expected .wav file, got: {filename!r}")

    stem = base[:-4] if base.lower().endswith(".wav") else base
    parts = stem.split("-")
    if len(parts) != 7:
        raise ValueError(
            f"expected 7 hyphen-separated fields, got {len(parts)}: {filename!r}"
        )

    try:
        modality, channel, emotion_code, intensity, statement, repetition, actor = (
            int(parts[0]),
            int(parts[1]),
            int(parts[2]),
            int(parts[3]),
            int(parts[4]),
            int(parts[5]),
            int(parts[6]),
        )
    except ValueError as e:
        raise ValueError(f"non-integer field in filename {filename!r}: {e}") from e

    # Audio-only speech subset (RAVDESS convention)
    if modality != 3 or channel != 1:
        raise ValueError(
            f"expected modality=03 and channel=01 for audio_speech, got "
            f"{modality:02d}-{channel:02d} in {filename!r}"
        )

    if not 1 <= emotion_code <= 8:
        raise ValueError(f"emotion_code out of range 1..8: {emotion_code} in {filename!r}")
    if not 1 <= actor <= 24:
        raise ValueError(f"actor out of range 1..24: {actor} in {filename!r}")
    if intensity not in (1, 2):
        raise ValueError(f"intensity must be 1 or 2: {intensity} in {filename!r}")
    if statement not in (1, 2):
        raise ValueError(f"statement must be 1 or 2: {statement} in {filename!r}")
    if repetition not in (1, 2):
        raise ValueError(f"repetition must be 1 or 2: {repetition} in {filename!r}")

    # GUIDELINES_HARISH: odd actor = male, even = female
    gender = "male" if actor % 2 == 1 else "female"

    clip_id = (
        f"{actor:02d}_{emotion_code:02d}_{intensity:02d}_"
        f"{statement:02d}_{repetition:02d}"
    )

    return {
        "clip_id": clip_id,
        "filename": base,
        "actor": actor,
        "gender": gender,
        "emotion_code": emotion_code,
        "emotion_label": EMOTION_LABEL[emotion_code],
        "emotion_4class": FOUR_CLASS[emotion_code],
        "intensity": intensity,
        "statement": statement,
        "repetition": repetition,
    }


def run(audio_root: str, out_path: str) -> None:
    """Walk audio_root/Actor_XX/*.wav, parse every filename, write metadata CSV."""

    if not os.path.isdir(audio_root):
        print(f"ERROR: audio_root not found: {audio_root}", file=sys.stderr)
        sys.exit(1)

    wav_paths = sorted(glob.glob(os.path.join(audio_root, "Actor_*", "*.wav")))
    print(f"  Found {len(wav_paths)} .wav files under {audio_root}")

    if len(wav_paths) == 0:
        print(f"ERROR: no .wav files found under {audio_root}", file=sys.stderr)
        sys.exit(1)

    if len(wav_paths) != EXPECTED_ROWS:
        print(
            f"ERROR: expected {EXPECTED_ROWS} .wav files, found {len(wav_paths)}",
            file=sys.stderr,
        )
        sys.exit(1)

    rows: list[dict] = []
    for path in wav_paths:
        try:
            rows.append(parse_filename(os.path.basename(path)))
        except ValueError as e:
            print(f"ERROR parsing {path}: {e}", file=sys.stderr)
            sys.exit(1)

    df = pd.DataFrame(rows, columns=METADATA_COLUMNS)
    df = df.sort_values("clip_id", kind="stable").reset_index(drop=True)

    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"WROTE {out_path} {len(df)} rows")

    validate(out_path)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate(csv_path: str) -> None:
    """Assert row count, uniqueness, no NaN, label-space correctness."""
    df = pd.read_csv(csv_path)

    print(f"\nVALIDATE {csv_path}")
    print(f"  rows: {len(df)} (expected {EXPECTED_ROWS})")

    assert len(df) == EXPECTED_ROWS, f"row count {len(df)} != {EXPECTED_ROWS}"
    assert df["clip_id"].is_unique, "clip_id values are not unique"
    assert df.isna().sum().sum() == 0, "NaN values present"
    assert set(df["emotion_4class"].unique()).issubset(FOUR_CLASS_SET), (
        f"emotion_4class contains labels outside {FOUR_CLASS_SET}"
    )
    assert list(df.columns) == METADATA_COLUMNS, (
        f"column order mismatch: {list(df.columns)}"
    )

    print("  OK — 1440 rows, unique clip_ids, no NaN, 4-class labels valid")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Parse RAVDESS audio-only filenames into a metadata CSV",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--audio_root",
        default="Datasets/ravdess/audio_speech",
        help="Root directory containing Actor_01/ ... Actor_24/ .wav files",
    )
    parser.add_argument(
        "--out",
        default="phase2_ravdess/metadata/ravdess_metadata.csv",
        help="Output metadata CSV path",
    )
    parser.add_argument(
        "--validate_only",
        action="store_true",
        help="Validate an existing metadata CSV and exit",
    )
    args = parser.parse_args()

    if args.validate_only:
        validate(args.out)
        return

    print("START harish_parse_metadata")
    run(args.audio_root, args.out)
    print("DONE")


if __name__ == "__main__":
    main()
