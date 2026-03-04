# Step 2: WhisperX Transcription — Aaditya

Pipeline position: **Step 1 (Diarization) → Step 2 (This) → Step 3 (Merge)**

---

## What This Does

Takes a raw `.wav` meeting recording and produces a word-aligned transcript CSV with per-segment ASR confidence scores.

**WhisperX pipeline stages:**
1. Load audio and resample to 16 kHz mono
2. Transcribe using faster-whisper + Silero VAD (batched)
3. Forced alignment with Wav2Vec2 → word-level timestamps + CTC scores
4. Aggregate mean CTC score per segment → `asr_confidence`
5. Write CSV

---

## Files

| File | Description |
|------|-------------|
| `aaditya/aaditya_whisperx.py` | Main Python script — full WhisperX pipeline |
| `scripts/run_whisperx.sh` | Shell wrapper (team interface) |

---

## Output Format

**Location:** `outputs/transcripts/ESXXXX_transcript.csv`

| Column | Type | Description |
|--------|------|-------------|
| `segment_id` | int | 0-indexed, monotonically increasing |
| `start` | float | Segment start time in seconds (2 decimal places) |
| `end` | float | Segment end time in seconds (2 decimal places) |
| `transcript` | str | Transcribed text |
| `asr_confidence` | float | 0.0–1.0, mean Wav2Vec2 CTC word alignment score |

**Example row:**
```
6,55.95,59.34,"Well, this is the kickoff meeting for our project.",0.7289
```

**Already produced:** `outputs/transcripts/ES2002a_transcript.csv` (200 segments, mean conf 0.708)

---

## How to Run

### Standard interface (use this)
```bash
bash scripts/run_whisperx.sh \
  Datasets/amicorpus/ES2002a/audio/ES2002a.Mix-Headset.wav \
  outputs/transcripts/ES2002a_transcript.csv
```
Auto-detects device (CUDA > MPS > CPU) and sets sensible defaults.

### Direct Python — GPU (Linux/cloud)
```bash
python aaditya/aaditya_whisperx.py \
    --audio Datasets/amicorpus/ES2002a/audio/ES2002a.Mix-Headset.wav \
    --output outputs/transcripts/ES2002a_transcript.csv \
    --model large-v2 --device cuda --batch_size 16 --compute_type float16
```

### Direct Python — macOS Apple Silicon
```bash
python aaditya/aaditya_whisperx.py \
    --audio Datasets/amicorpus/ES2002a/audio/ES2002a.Mix-Headset.wav \
    --output outputs/transcripts/ES2002a_transcript.csv \
    --model small --device mps --batch_size 4
```
> Note: CTranslate2 (faster-whisper) does not support MPS — transcription runs on CPU automatically, alignment runs on MPS.

### Direct Python — CPU fallback
```bash
python aaditya/aaditya_whisperx.py \
    --audio Datasets/amicorpus/ES2002a/audio/ES2002a.Mix-Headset.wav \
    --output outputs/transcripts/ES2002a_transcript.csv \
    --model small --device cpu --batch_size 4 --compute_type int8
```

### Validate an existing CSV
```bash
python aaditya/aaditya_whisperx.py \
    --quality_report outputs/transcripts/ES2002a_transcript.csv
```
Checks schema, monotonicity, time ordering, and confidence range. Prints stats on pass.

---

## CLI Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--audio` | — | Path to input `.wav` file |
| `--output` | — | Path to output CSV |
| `--model` | `large-v2` | Whisper model size (`large-v2` for quality, `small` for speed) |
| `--device` | auto | `cuda` / `mps` / `cpu` |
| `--batch_size` | auto | 16 for CUDA, 4 for MPS/CPU |
| `--compute_type` | auto | `float16` for CUDA, `int8` for CPU/MPS |
| `--quality_report` | — | Validate existing CSV instead of transcribing |

---

## Environment Setup

```bash
cd CSCI-535
python -m venv venv
source venv/bin/activate
pip install whisperx torch torchaudio numpy pandas
```

---

## Notes for Downstream Steps

- **Aniket (Step 3 — Merge):** Join on overlapping time ranges with Charan's diarization. No `speaker` column here — that comes from diarization.
- **Harish (Step 4 — Segmentation):** Use `start`/`end` timestamps to create 3.0s windows with 1.5s hop.
- **Incongruence Scorer (Step 7):** `asr_confidence` is used as the confidence weight in the final scoring formula. Segments with unaligned words (numbers, symbols) will have lower scores — this is expected and handled downstream.

---

## Technical Notes

- Uses `Mix-Headset.wav` (not SDM arrays) — better quality for non-native speakers
- `large-v2` requires ~5 GB VRAM; model is freed between pipeline stages to avoid OOM
- Words that can't be aligned (numbers, punctuation) get `score ≈ 0` from Wav2Vec2 — this pulls `asr_confidence` down slightly but is expected behavior
- Random seeds fixed: `np.random.seed(42)`, `torch.manual_seed(42)`
