#!/usr/bin/env python3
"""
charan_audio_emotion.py — Phase 2 Step 2a (Charan)

Run superb/wav2vec2-large-superb-er on every RAVDESS audio-only .wav clip
(1440 clips total) and output emotion probabilities over our 4-class space.
Each .wav is loaded whole, resampled to 16 kHz, and passed to the model.

INPUTS:
    phase2_ravdess/metadata/ravdess_metadata.csv
    Datasets/ravdess/audio_speech/Actor_XX/*.wav

OUTPUT:
    phase2_ravdess/emotions/audio_emotions.csv

    Columns:
        clip_id, actor, emotion_code, emotion_4class,
        p_happy, p_angry, p_sad, p_neutral

RUN:
    python3 phase2_ravdess/scripts/charan_audio_emotion.py \\
        --metadata   phase2_ravdess/metadata/ravdess_metadata.csv \\
        --audio_root Datasets/ravdess/audio_speech \\
        --out        phase2_ravdess/emotions/audio_emotions.csv
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

MODEL_ID = "superb/wav2vec2-large-superb-er"

# superb/wav2vec2-large-superb-er labels → our canonical 4-class space
SUPERB_TO_FOUR = {
    "hap": "happy",
    "ang": "angry",
    "sad": "sad",
    "neu": "neutral",
}

OUTPUT_LABELS = ["p_happy", "p_angry", "p_sad", "p_neutral"]

TARGET_SR = 16_000  # wav2vec2 expects 16 kHz

EXPECTED_ROWS = 1440

OUTPUT_COLUMNS = [
    "clip_id", "actor", "emotion_code", "emotion_4class",
] + OUTPUT_LABELS


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model():
    """
    Load superb/wav2vec2-large-superb-er via AutoFeatureExtractor +
    AutoModelForAudioClassification — bypasses HuggingFace pipeline and
    its torchcodec/FFmpeg dependency entirely.
    """
    try:
        import torch
        from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
    except ImportError:
        print(
            "ERROR: install transformers torch torchaudio: "
            "pip install transformers torch torchaudio",
            file=sys.stderr,
        )
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Loading model: {MODEL_ID}  (device: {device})")

    extractor = AutoFeatureExtractor.from_pretrained(MODEL_ID)
    model = AutoModelForAudioClassification.from_pretrained(MODEL_ID)
    model.to(device)
    model.eval()

    return extractor, model, device


def predict_probs(model_bundle, audio_clip: np.ndarray, sr: int = TARGET_SR) -> dict:
    """
    Run audio emotion classification on a numpy audio clip.
    Returns {p_happy, p_angry, p_sad, p_neutral} summing to 1.0.
    """
    import torch

    extractor, model, device = model_bundle

    inputs = extractor(audio_clip, sampling_rate=sr, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        logits = model(**inputs).logits

    scores = torch.softmax(logits, dim=-1).squeeze().cpu().tolist()
    id2label = model.config.id2label

    four_probs = {"happy": 0.0, "angry": 0.0, "sad": 0.0, "neutral": 0.0}

    for idx, score in enumerate(scores):
        label = id2label[idx].strip().lower()
        canonical = SUPERB_TO_FOUR.get(label)
        if canonical is not None:
            four_probs[canonical] += float(score)
        else:
            four_probs["neutral"] += float(score)

    total = sum(four_probs.values())
    if total <= 0:
        four_probs = {"happy": 0.25, "angry": 0.25, "sad": 0.25, "neutral": 0.25}
    else:
        four_probs = {k: v / total for k, v in four_probs.items()}

    return {
        "p_happy":   round(four_probs["happy"],   6),
        "p_angry":   round(four_probs["angry"],   6),
        "p_sad":     round(four_probs["sad"],     6),
        "p_neutral": round(four_probs["neutral"], 6),
    }


# ---------------------------------------------------------------------------
# RAVDESS-specific audio loading
# ---------------------------------------------------------------------------

def load_ravdess_wav(audio_root: str, filename: str, actor: int,
                     sr: int = TARGET_SR) -> np.ndarray:
    """
    Load a single RAVDESS .wav file at 16 kHz mono.

    Path format:  {audio_root}/Actor_{actor:02d}/{filename}
    """
    import librosa

    wav_path = os.path.join(audio_root, f"Actor_{actor:02d}", filename)
    if not os.path.isfile(wav_path):
        raise FileNotFoundError(wav_path)

    audio, _ = librosa.load(wav_path, sr=sr, mono=True)

    # Pad very short clips so the model doesn't fail
    min_samples = int(0.5 * sr)
    if len(audio) < min_samples:
        audio = np.pad(audio, (0, min_samples - len(audio)))

    return audio


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run(metadata_path: str, audio_root: str, out_path: str) -> None:
    """Load metadata, run model on every .wav, write audio_emotions.csv."""
    if not os.path.isfile(metadata_path):
        print(f"ERROR: metadata CSV not found: {metadata_path}", file=sys.stderr)
        sys.exit(1)
    if not os.path.isdir(audio_root):
        print(f"ERROR: audio_root not found: {audio_root}", file=sys.stderr)
        sys.exit(1)

    metadata_df = pd.read_csv(metadata_path)
    print(f"  metadata: {len(metadata_df)} rows  ({metadata_path})")

    model_bundle = load_model()

    rows = []
    for i, meta in metadata_df.iterrows():
        clip_id = meta["clip_id"]
        filename = meta["filename"]
        actor = int(meta["actor"])
        emo_code = int(meta["emotion_code"])
        emo_4 = meta["emotion_4class"]

        try:
            audio = load_ravdess_wav(audio_root, filename, actor)
            probs = predict_probs(model_bundle, audio)
        except Exception as e:
            print(f"  WARNING: clip_id={clip_id} failed: {e}")
            probs = {
                "p_happy": 0.25,
                "p_angry": 0.25,
                "p_sad": 0.25,
                "p_neutral": 0.25,
            }

        rows.append({
            "clip_id": clip_id,
            "actor": actor,
            "emotion_code": emo_code,
            "emotion_4class": emo_4,
            **probs,
        })

        if (i + 1) % 100 == 0:
            print(f"    processed {i + 1}/{len(metadata_df)} clips...")

    df = pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"WROTE {out_path} {len(df)} rows")
    validate(out_path)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate(csv_path: str) -> None:
    """Assert 1440 rows, unique clip_ids, probs sum to ~1.0, no NaN."""
    df = pd.read_csv(csv_path)

    print(f"\nVALIDATE {csv_path}")
    print(f"  rows: {len(df)} (expected {EXPECTED_ROWS})")

    assert len(df) == EXPECTED_ROWS, f"row count {len(df)} != {EXPECTED_ROWS}"
    assert df["clip_id"].is_unique, "clip_id values are not unique"
    assert df.isna().sum().sum() == 0, "NaN values present"
    assert list(df.columns) == OUTPUT_COLUMNS, f"column mismatch: {list(df.columns)}"

    sums = df[OUTPUT_LABELS].sum(axis=1)
    assert np.allclose(sums, 1.0, atol=1e-4), "probs do not sum to 1.0"

    print("  OK — 1440 rows, unique clip_ids, probs sum to 1.0, no NaN")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="RAVDESS audio emotion prediction (wav2vec2-large-superb-er)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--metadata",
        default="phase2_ravdess/metadata/ravdess_metadata.csv",
        help="Input metadata CSV from Harish Step 1",
    )
    parser.add_argument(
        "--audio_root",
        default="Datasets/ravdess/audio_speech",
        help="Root directory of RAVDESS audio_speech (Actor_01 ... Actor_24)",
    )
    parser.add_argument(
        "--out",
        default="phase2_ravdess/emotions/audio_emotions.csv",
        help="Output audio_emotions CSV path",
    )
    parser.add_argument("--validate_only", action="store_true")
    args = parser.parse_args()

    if args.validate_only:
        validate(args.out)
        return

    print("START charan_audio_emotion")
    run(args.metadata, args.audio_root, args.out)
    print("DONE")


if __name__ == "__main__":
    main()
