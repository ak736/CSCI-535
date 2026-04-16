#!/usr/bin/env python3
"""
aaditya_whisperx.py — Step 2 of 7: WhisperX Transcription
Produces aligned transcript CSV from a single .wav file.

Device notes (macOS Apple Silicon):
  CTranslate2/faster-whisper does NOT support MPS — transcription always runs on CPU.
  PyTorch/Wav2Vec2 alignment DOES support MPS — alignment runs on MPS when available.
  Pass --device mps and the script handles the split automatically.

Usage (GPU / Linux):
    python scripts/aaditya_whisperx.py \
        --audio Datasets/amicorpus/ES2002a/audio/ES2002a.Mix-Headset.wav \
        --output outputs/transcripts/ES2002a_transcript.csv \
        --model large-v2 --device cuda --batch_size 16 --compute_type float16

Usage (macOS Apple Silicon):
    python scripts/aaditya_whisperx.py \
        --audio Datasets/amicorpus/ES2002a/audio/ES2002a.Mix-Headset.wav \
        --output outputs/transcripts/ES2002a_transcript.csv \
        --model small --device mps --batch_size 4

Usage (CPU fallback):
    python scripts/aaditya_whisperx.py \
        --audio Datasets/amicorpus/ES2002a/audio/ES2002a.Mix-Headset.wav \
        --output outputs/transcripts/ES2002a_transcript.csv \
        --model small --device cpu --batch_size 4 --compute_type int8

Quality report on existing CSV:
    python scripts/aaditya_whisperx.py \
        --quality_report outputs/transcripts/ES2002a_transcript.csv
"""

import argparse
import gc
import os
import sys

import numpy as np
import pandas as pd
import torch
import whisperx


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


def _auto_compute_type(device: str) -> str:
    """float16 for CUDA (fast GPU fp16); int8 for CPU/MPS (CTranslate2 quantisation)."""
    return "float16" if device == "cuda" else "int8"


def _auto_batch_size(device: str) -> int:
    """Larger batches on GPU; conservative on CPU/MPS."""
    return 16 if device == "cuda" else 4


def _transcription_device(device: str) -> str:
    """CTranslate2 (faster-whisper) only supports cpu and cuda — map mps to cpu."""
    return "cpu" if device == "mps" else device


def _clear_cache(device: str) -> None:
    """Free accelerator memory after each model stage."""
    if device == "cuda":
        torch.cuda.empty_cache()
    elif device == "mps":
        torch.mps.empty_cache()


# ---------------------------------------------------------------------------
# Transcription pipeline
# ---------------------------------------------------------------------------

def run_transcription(
    audio_path: str,
    output_path: str,
    model_name: str,
    device: str,
    batch_size: int,
    compute_type: str,
) -> None:
    """Full WhisperX pipeline: load → transcribe → align → write CSV."""

    # Input validation
    if not os.path.isfile(audio_path):
        print(f"ERROR: Audio file not found: {audio_path}", file=sys.stderr)
        sys.exit(1)

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    transcribe_device = _transcription_device(device)
    print(
        f"Device: {device}  "
        f"(transcription→{transcribe_device}, alignment→{device})"
    )

    # Stage 1: Load audio (whisperx resamples to 16 kHz mono internally)
    print(f"[1/4] Loading audio: {audio_path}")
    audio = whisperx.load_audio(audio_path)

    # Stage 2: Transcribe (faster-whisper backend + Silero VAD)
    print(
        f"[2/4] Transcribing  model={model_name}  "
        f"device={transcribe_device}  compute_type={compute_type}  "
        f"batch_size={batch_size}"
    )
    model = whisperx.load_model(
        model_name,
        transcribe_device,
        compute_type=compute_type,
    )
    result = model.transcribe(audio, batch_size=batch_size)
    language = result.get("language", "en")
    print(f"      Detected language: {language}  raw segments: {len(result['segments'])}")

    del model
    gc.collect()
    _clear_cache(device)

    # Stage 3: Forced alignment (Wav2Vec2 → word-level timestamps + CTC scores)
    print(f"[3/4] Aligning  model=Wav2Vec2  language={language}  device={device}")
    model_a, metadata = whisperx.load_align_model(language_code=language, device=device)
    result = whisperx.align(
        result["segments"],
        model_a,
        metadata,
        audio,
        device,
        return_char_alignments=False,
    )

    del model_a
    gc.collect()
    _clear_cache(device)

    # Stage 4: Aggregate per-segment confidence and build rows
    print("[4/4] Building CSV rows")
    rows = []
    for seg_id, seg in enumerate(result["segments"]):
        start = round(float(seg.get("start", 0.0)), 2)
        end   = round(float(seg.get("end",   0.0)), 2)
        text  = seg.get("text", "").strip()

        # asr_confidence = mean CTC score across all words in the segment.
        # Words that couldn't be aligned (numbers, symbols) have score ≈ 0;
        # missing "score" key also defaults to 0.0 per spec.
        words = seg.get("words", [])
        scores = [w.get("score", 0.0) for w in words]
        asr_confidence = round(float(np.mean(scores)), 4) if scores else 0.0

        rows.append({
            "segment_id":     seg_id,
            "start":          start,
            "end":            end,
            "transcript":     text,
            "asr_confidence": asr_confidence,
        })

    df = pd.DataFrame(
        rows,
        columns=["segment_id", "start", "end", "transcript", "asr_confidence"],
    )
    df.to_csv(output_path, index=False)
    print(f"WROTE {output_path} {len(df)} rows")


# ---------------------------------------------------------------------------
# Quality report
# ---------------------------------------------------------------------------

def run_quality_report(csv_path: str) -> None:
    """Validate schema and print stats for an existing transcript CSV."""
    if not os.path.isfile(csv_path):
        print(f"ERROR: CSV not found: {csv_path}", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(csv_path)

    expected_cols = ["segment_id", "start", "end", "transcript", "asr_confidence"]
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
    assert (df["asr_confidence"] >= 0).all() and (df["asr_confidence"] <= 1).all(), \
        "asr_confidence out of [0, 1]"

    duration = df["end"].max() - df["start"].min()
    print(f"PASS — {len(df)} segments, mean conf {df['asr_confidence'].mean():.3f}")
    print(f"  Duration span : {duration:.2f}s")
    print(f"  Min conf      : {df['asr_confidence'].min():.3f}")
    print(f"  Zero-conf segs: {(df['asr_confidence'] == 0.0).sum()}")
    print(f"  Empty text    : {(df['transcript'].str.strip() == '').sum()}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="WhisperX transcription pipeline — Step 2 of 7",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Transcription mode
    parser.add_argument("--audio",  type=str, help="Path to input .wav file")
    parser.add_argument("--output", type=str, help="Path to output CSV")
    parser.add_argument(
        "--model", type=str, default="large-v2",
        help="Whisper model size (large-v2 for quality; small for speed/testing)",
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Device: cuda / mps / cpu  (auto-detected if omitted)",
    )
    parser.add_argument(
        "--batch_size", type=int, default=None,
        help="Transcription batch size (auto: 16 for cuda, 4 for mps/cpu)",
    )
    parser.add_argument(
        "--compute_type", type=str, default=None,
        help="Quantisation: float16 (cuda), int8 (cpu/mps), float32 (auto if omitted)",
    )

    # Quality report mode
    parser.add_argument(
        "--quality_report", type=str,
        help="Path to existing transcript CSV to validate (skips transcription)",
    )

    args = parser.parse_args()

    if args.quality_report:
        run_quality_report(args.quality_report)
        return

    if not (args.audio and args.output):
        parser.print_help()
        sys.exit(1)

    device       = args.device       or _auto_device()
    batch_size   = args.batch_size   or _auto_batch_size(device)
    compute_type = args.compute_type or _auto_compute_type(device)

    run_transcription(
        audio_path=args.audio,
        output_path=args.output,
        model_name=args.model,
        device=device,
        batch_size=batch_size,
        compute_type=compute_type,
    )


if __name__ == "__main__":
    np.random.seed(42)
    torch.manual_seed(42)
    main()
