# README_TASKS.md — Split Tasks, Folders, and How to Use Them

## Purpose

This file explains what belongs in `models/` and `scripts/`, how to use the per-person `.txt` task files, and how to run the pipeline step-by-step.

---

## Folder Roles

### `Datasets/amicorpus/`

Raw AMI files. **READ-ONLY — do not edit.**

### `outputs/`

All pipeline outputs, organized into subfolders:

| Subfolder       | Contents                                |
| --------------- | --------------------------------------- |
| `diarization/`  | Speaker diarization CSVs                |
| `transcripts/`  | WhisperX transcript CSVs                |
| `merged/`       | Merged diarization + transcript CSVs    |
| `segments/`     | Fixed-length windowed segment CSVs      |
| `embeddings/`   | Text and audio emotion probability CSVs |
| `incongruence/` | Final scored output CSVs                |

---

### `models/`

Implementation of model logic used by the pipeline. Each file exposes a programmatic interface and a CLI entrypoint.

#### `models/text_emotion.py`

- **Function:** `predict_text_probs(transcript: str) -> List[float]`
- Accepts a single transcript string and returns `[p_happy, p_angry, p_sad, p_neutral]`
- CLI: processes a segments CSV and writes `outputs/embeddings/ESXXXX_text_emotions.csv`

#### `models/audio_emotion.py`

- **Function:** `predict_audio_probs(wav_path: str, start: float, end: float) -> List[float]`
- CLI: processes segment timestamps and writes `outputs/embeddings/ESXXXX_audio_emotions.csv`

#### `models/incongruence.py`

- **Function:** `compute_js(p: List[float], q: List[float]) -> float`
- **Function:** `compute_confidence_weight(asr_conf, audio_energy) -> float`
- CLI: takes text & audio emotion CSVs + segments CSV, outputs `outputs/incongruence/ESXXXX_scores.csv`

> **Implementation notes:**
>
> - All model scripts must accept input/output paths as CLI args.
> - Log model names, versions, and params in the script header on each run.

---

### `scripts/`

Orchestration scripts and helpers — these are the commands the team runs in order.

| Script                                        | Responsibility                                              |
| --------------------------------------------- | ----------------------------------------------------------- |
| `scripts/run_diarization.sh`                  | Run pyannote on a single meeting → `outputs/diarization/`   |
| `scripts/run_whisperx.sh`                     | Run WhisperX on a single meeting → `outputs/transcripts/`   |
| `scripts/merge_diarization_and_transcript.py` | Merge diarization + transcript CSVs → `outputs/merged/`     |
| `scripts/create_segments.py`                  | Create 3s/1.5s windows with filtering → `outputs/segments/` |
| `scripts/run_text_emotion.py`                 | Batch-run text emotion model over segments                  |
| `scripts/run_audio_emotion.py`                | Batch-run audio emotion model over segments                 |
| `scripts/compute_incongruence.py`             | Compute final incongruence scores                           |
| `scripts/quick_test.sh`                       | Demo pipeline on a single small meeting to validate config  |

> **Every script must:**
>
> - Validate that inputs exist before running
> - Log start/finish and output paths
> - Exit with non-zero on error

---

## How the `.txt` Task Files Are Used

Each person's `.txt` file (e.g. `charan/TASK_diarization.txt`) contains their exact instructions. Everyone should:

1. Read their assigned `.txt` file.
2. Run the corresponding `scripts/` wrapper — or produce output in the same CSV format.
3. Place outputs in `outputs/<folder>/` with the prescribed filename pattern (`ESXXXX_*`).

> ⚠️ Do **not** modify others' outputs. If a format issue arises, open an issue and tag the owner.

---

## Quick Run — Command Sequence (Exact Order)

Replace `ES2002a` with the target meeting ID.

**1. Diarization** _(Charan)_

```bash
bash scripts/run_diarization.sh \
  Datasets/amicorpus/ES2002a/audio/ES2002a.Mix-Headset.wav \
  outputs/diarization/ES2002a_diarization.csv
```

**2. Transcription** _(Aaditya)_

```bash
bash scripts/run_whisperx.sh \
  Datasets/amicorpus/ES2002a/audio/ES2002a.Mix-Headset.wav \
  outputs/transcripts/ES2002a_transcript.csv
```

**3. Merge** _(Aniket)_

```bash
python3 scripts/merge_diarization_and_transcript.py \
  --diar outputs/diarization/ES2002a_diarization.csv \
  --trans outputs/transcripts/ES2002a_transcript.csv \
  --out outputs/merged/ES2002a_merged.csv
```

**4. Create Segments** _(Harish)_

```bash
python3 scripts/create_segments.py \
  --merged outputs/merged/ES2002a_merged.csv \
  --out outputs/segments/ES2002a_segments.csv \
  --window 3.0 \
  --hop 1.5
```

**5. Text Emotion** _(Harish)_

```bash
python3 scripts/run_text_emotion.py \
  outputs/segments/ES2002a_segments.csv \
  outputs/embeddings/ES2002a_text_emotions.csv
```

**6. Audio Emotion** _(Audio owner)_

```bash
python3 scripts/run_audio_emotion.py \
  outputs/segments/ES2002a_segments.csv \
  outputs/embeddings/ES2002a_audio_emotions.csv
```

**7. Incongruence Scoring**

```bash
python3 scripts/compute_incongruence.py \
  --text outputs/embeddings/ES2002a_text_emotions.csv \
  --audio outputs/embeddings/ES2002a_audio_emotions.csv \
  --segments outputs/segments/ES2002a_segments.csv \
  --out outputs/incongruence/ES2002a_scores.csv
```

---

## File Name & Format Checklist

> Do not skip. Column names must match exactly — any mismatch halts the pipeline.

| File                                           | Required Columns                                                                        |
| ---------------------------------------------- | --------------------------------------------------------------------------------------- |
| `outputs/diarization/ESXXXX_diarization.csv`   | `segment_id, start, end, speaker_id`                                                    |
| `outputs/transcripts/ESXXXX_transcript.csv`    | `segment_id, start, end, transcript, asr_confidence`                                    |
| `outputs/merged/ESXXXX_merged.csv`             | `segment_id, start, end, speaker_id, transcript, asr_confidence`                        |
| `outputs/segments/ESXXXX_segments.csv`         | `segment_id, start, end, speaker_id, transcript, asr_confidence`                        |
| `outputs/embeddings/ESXXXX_text_emotions.csv`  | `segment_id, p_happy, p_angry, p_sad, p_neutral`                                        |
| `outputs/embeddings/ESXXXX_audio_emotions.csv` | `segment_id, p_happy, p_angry, p_sad, p_neutral`                                        |
| `outputs/incongruence/ESXXXX_scores.csv`       | `segment_id, start, end, speaker_id, transcript, incongruence_score, confidence_weight` |

> `segment_id` must align across emotion CSVs — rows are joined on this key.

---

## Environment & Dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` should include:

```
pyannote.audio
whisperx
transformers
torch
numpy
pandas
librosa
```

Add any additional libs used by your model scripts.

---

## Logging & Reproducibility

- Each script must print on success:
  ```
  WROTE <output_path> <num_rows> rows
  ```
- When running model inference, record the model ID and date in:
  ```
  results/experiment_log.csv
  ```
- Fix random seeds in all model scripts before running.

---

## Troubleshooting Rules

- If a script fails due to missing input — **do not** re-run upstream steps manually. Contact the step owner.
- Keep all raw files in `Datasets/` unchanged.
- If output column names change, the pipeline halts — revert or fix the downstream script immediately.
- Run `scripts/quick_test.sh` first on a small meeting to validate your setup before processing the full corpus.

---

## Link This File

Add the following line near the top of the main `README.md`:

```
> For task assignments and step-by-step commands, see [README_TASKS.md](./README_TASKS.md).
```
