#!/usr/bin/env python3
"""
aniket_audio_emotion.py — Step 6 (Aniket)

For each segment in segments CSV, predicts audio emotion probabilities over:
    E = {happy, angry, sad, neutral}

Model: superb/wav2vec2-large-superb-er
  - Trained on SUPERB Speech Emotion Recognition benchmark
  - Output labels: hap, ang, sad, neu  →  direct 1:1 map to our label space
  - No remapping gymnastics needed

Speaker → Headset mapping:
  Instead of slicing from Mix-Headset.wav (which contains overlapping voices),
  we map each speaker_id to its individual Headset-X.wav using energy analysis:
    For each speaker's diarization segments, compute mean RMS energy across
    all headset files. The headset with highest energy for that speaker is
    that speaker's microphone. This is reliable because individual headsets
    capture their own speaker at ~10x the energy of distant voices.

  Fallback: if diarization CSV is not provided, use Mix-Headset.wav for all
  segments (noted as lower quality).

Usage:
    python3 aniket/aniket_audio_emotion.py \
        --segments  outputs/segments/ES2002a_segments.csv \
        --diar      outputs/diarization/ES2002a_diarization.csv \
        --audio_dir Datasets/amicorpus/ES2002a/audio \
        --meeting   ES2002a \
        --out       outputs/embeddings/ES2002a_audio_emotions.csv

Output columns (must match pipeline spec exactly):
    segment_id, p_happy, p_angry, p_sad, p_neutral
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

# Fixed random seed for reproducibility
np.random.seed(42)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_ID = "superb/wav2vec2-large-superb-er"

# superb/wav2vec2-large-superb-er label → our canonical label
# Labels in the model: hap, ang, sad, neu  (exact strings from model config)
SUPERB_TO_FOUR = {
    "hap": "happy",
    "ang": "angry",
    "sad": "sad",
    "neu": "neutral",
}

OUTPUT_LABELS = ["p_happy", "p_angry", "p_sad", "p_neutral"]

TARGET_SR = 16_000  # wav2vec2 expects 16 kHz


# ---------------------------------------------------------------------------
# Speaker → Headset mapping (energy-based)
# ---------------------------------------------------------------------------

def _rms_energy(audio: np.ndarray) -> float:
    """Root-mean-square energy of an audio array."""
    if len(audio) == 0:
        return 0.0
    return float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))


def build_speaker_headset_map(
    diar_df: pd.DataFrame,
    audio_dir: str,
    meeting: str,
    sr: int = TARGET_SR,
) -> dict[str, str]:
    """
    Map each speaker_id to the headset WAV file where they are loudest.

    Strategy:
      For each headset file, for each speaker's diarization segments,
      compute mean RMS energy. Assign each speaker to the headset with
      highest mean energy. Uses a greedy assignment (highest energy wins)
      — good enough for Phase 1.

    Returns:
        dict  speaker_id → absolute path to WAV file
    """
    import librosa

    # Find available headset files (Headset-0 through Headset-3)
    headset_paths = []
    for i in range(4):
        p = os.path.join(audio_dir, f"{meeting}.Headset-{i}.wav")
        if os.path.isfile(p):
            headset_paths.append((i, p))

    if not headset_paths:
        print("  WARNING: no Headset-X.wav files found — falling back to Mix-Headset")
        mix_path = os.path.join(audio_dir, f"{meeting}.Mix-Headset.wav")
        speakers = diar_df["speaker_id"].unique().tolist()
        return {spk: mix_path for spk in speakers}

    print(f"  Found {len(headset_paths)} headset files. Building speaker→headset map via energy analysis...")

    # Load all headset files once
    headset_audio: dict[int, np.ndarray] = {}
    for idx, path in headset_paths:
        audio, _ = librosa.load(path, sr=sr, mono=True)
        headset_audio[idx] = audio

    speakers = diar_df["speaker_id"].unique().tolist()

    # energy_matrix[speaker][headset_idx] = mean RMS energy
    energy_matrix: dict[str, dict[int, float]] = {spk: {} for spk in speakers}

    for spk in speakers:
        spk_segs = diar_df[diar_df["speaker_id"] == spk]
        for idx, _ in headset_paths:
            audio = headset_audio[idx]
            energies = []
            for _, row in spk_segs.iterrows():
                start_sample = int(float(row["start"]) * sr)
                end_sample = int(float(row["end"]) * sr)
                clip = audio[start_sample:end_sample]
                if len(clip) > 0:
                    energies.append(_rms_energy(clip))
            energy_matrix[spk][idx] = float(np.mean(energies)) if energies else 0.0

    # Greedy assignment: each speaker gets the headset where they are loudest
    # (allowing multiple speakers to share a headset if needed)
    speaker_map: dict[str, str] = {}
    for spk in speakers:
        best_idx = max(energy_matrix[spk], key=lambda i: energy_matrix[spk][i])
        best_path = dict(headset_paths)[best_idx]
        speaker_map[spk] = best_path
        print(f"    {spk} → Headset-{best_idx}  (energy={energy_matrix[spk][best_idx]:.4f})")

    return speaker_map


# ---------------------------------------------------------------------------
# Audio slicing
# ---------------------------------------------------------------------------

def slice_audio(
    wav_path: str,
    start: float,
    end: float,
    sr: int = TARGET_SR,
    cache: dict | None = None,
) -> np.ndarray:
    """Load a slice of a WAV file between start and end seconds."""
    import librosa

    if cache is not None and wav_path not in cache:
        audio, _ = librosa.load(wav_path, sr=sr, mono=True)
        cache[wav_path] = audio

    audio = cache[wav_path] if cache is not None else librosa.load(wav_path, sr=sr, mono=True)[0]

    start_sample = max(0, int(start * sr))
    end_sample = min(len(audio), int(end * sr))
    clip = audio[start_sample:end_sample]

    # Pad short clips to at least 0.5s so the model doesn't fail
    min_samples = int(0.5 * sr)
    if len(clip) < min_samples:
        clip = np.pad(clip, (0, min_samples - len(clip)))

    return clip


# ---------------------------------------------------------------------------
# Emotion prediction
# ---------------------------------------------------------------------------

def load_model():
    """
    Load superb/wav2vec2-large-superb-er directly via AutoFeatureExtractor +
    AutoModelForAudioClassification — bypasses the HuggingFace pipeline and
    its torchcodec/FFmpeg dependency entirely.
    """
    try:
        import torch
        from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
    except ImportError:
        print("ERROR: install transformers torch torchaudio: pip install transformers torch torchaudio", file=sys.stderr)
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Loading model: {MODEL_ID}  (device: {device})")

    extractor = AutoFeatureExtractor.from_pretrained(MODEL_ID)
    model = AutoModelForAudioClassification.from_pretrained(MODEL_ID)
    model.to(device)
    model.eval()

    return extractor, model, device


def predict_probs(model_bundle, audio_clip: np.ndarray, sr: int = TARGET_SR) -> dict[str, float]:
    """
    Run audio emotion classification on a numpy audio clip.
    Returns {p_happy, p_angry, p_sad, p_neutral} summing to 1.0.

    Uses AutoFeatureExtractor + AutoModelForAudioClassification directly —
    no torchcodec, no FFmpeg required.
    """
    import torch

    extractor, model, device = model_bundle

    inputs = extractor(audio_clip, sampling_rate=sr, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        logits = model(**inputs).logits

    scores = torch.softmax(logits, dim=-1).squeeze().cpu().tolist()
    id2label = model.config.id2label  # e.g. {0: "hap", 1: "ang", ...}

    four_probs = {"happy": 0.0, "angry": 0.0, "sad": 0.0, "neutral": 0.0}

    for idx, score in enumerate(scores):
        label = id2label[idx].strip().lower()
        canonical = SUPERB_TO_FOUR.get(label)
        if canonical is not None:
            four_probs[canonical] += float(score)
        else:
            four_probs["neutral"] += float(score)

    # Normalize (guard against float drift)
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
# Main
# ---------------------------------------------------------------------------

def run(
    segments_path: str,
    diar_path: str | None,
    audio_dir: str,
    meeting: str,
    out_path: str,
) -> None:
    import librosa  # noqa: F401 — ensure available before loading data

    # --- Validate inputs ---
    if not os.path.isfile(segments_path):
        print(f"ERROR: segments CSV not found: {segments_path}", file=sys.stderr)
        sys.exit(1)
    if not os.path.isdir(audio_dir):
        print(f"ERROR: audio directory not found: {audio_dir}", file=sys.stderr)
        sys.exit(1)

    segments_df = pd.read_csv(segments_path)
    for col in ("segment_id", "start", "end", "speaker_id"):
        if col not in segments_df.columns:
            print(f"ERROR: segments CSV missing column '{col}'", file=sys.stderr)
            sys.exit(1)

    print(f"  Segments   : {len(segments_df)} rows  ({segments_path})")

    # --- Speaker → headset mapping ---
    if diar_path and os.path.isfile(diar_path):
        diar_df = pd.read_csv(diar_path)
        speaker_map = build_speaker_headset_map(diar_df, audio_dir, meeting)
    else:
        print("  WARNING: no diarization CSV provided — using Mix-Headset for all segments")
        mix_path = os.path.join(audio_dir, f"{meeting}.Mix-Headset.wav")
        speakers = segments_df["speaker_id"].unique().tolist()
        speaker_map = {spk: mix_path for spk in speakers}

    # --- Load model ---
    pipe = load_model()

    # --- Process segments ---
    audio_cache: dict[str, np.ndarray] = {}
    rows = []

    for i, row in segments_df.iterrows():
        seg_id = int(row["segment_id"])
        start = float(row["start"])
        end = float(row["end"])
        speaker = row["speaker_id"]

        wav_path = speaker_map.get(speaker)
        if not wav_path or not os.path.isfile(wav_path):
            # Ultimate fallback
            wav_path = os.path.join(audio_dir, f"{meeting}.Mix-Headset.wav")

        clip = slice_audio(wav_path, start, end, cache=audio_cache)
        probs = predict_probs(pipe, clip)

        rows.append({"segment_id": seg_id, **probs})

        if (i + 1) % 50 == 0:
            print(f"    processed {i + 1}/{len(segments_df)} segments...")

    # --- Write output ---
    out_df = pd.DataFrame(rows, columns=["segment_id"] + OUTPUT_LABELS)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    out_df.to_csv(out_path, index=False)
    print(f"WROTE {out_path} {len(out_df)} rows")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audio emotion prediction per segment (superb/wav2vec2-large-superb-er)"
    )
    parser.add_argument("--segments",  required=True, help="Path to segments CSV")
    parser.add_argument("--diar",      default=None,  help="Path to diarization CSV (for speaker→headset mapping)")
    parser.add_argument("--audio_dir", required=True, help="Directory containing ESXXXX.Headset-X.wav files")
    parser.add_argument("--meeting",   required=True, help="Meeting ID e.g. ES2002a")
    parser.add_argument("--out",       required=True, help="Path to output audio_emotions CSV")
    args = parser.parse_args()

    print(f"START audio_emotion: {args.meeting}")
    run(args.segments, args.diar, args.audio_dir, args.meeting, args.out)
    print("DONE")


if __name__ == "__main__":
    main()
