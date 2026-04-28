# Beyond Fusion: Text vs Audio Incongruence (Phase 1)

**CSCI 535 — Multimodal Probabilistic Learning | University of Southern California**

Detect when a speaker's words emotionally disagree with their vocal tone using Jensen–Shannon divergence over emotion probability distributions.

---

## Team

| Name | Role |
|------|------|
| Aniket Kumar | Audio emotion model, merge pipeline, incongruence scoring |
| Harish Dukkipati | Segmentation, text emotion model |
| Charan Kumar D. | Speaker diarization |
| Aaditya Patil | WhisperX transcription, video emotion pipeline |

---

## Dataset

**AMI Meeting Corpus** — 12 meetings: ES2002a-d, ES2003a-d, ES2004a-d
Located at: `Datasets/amicorpus/<meeting_id>/audio/`

---

## Pipeline

```
Audio File
    ↓
[Step 1] Speaker Diarization  (charan_diarization.py)
    ↓
[Step 2] WhisperX Transcription  (aaditya_whisperx.py)
    ↓
[Step 3] Merge Diarization + Transcript  (aniket_merge.py)
    ↓
[Step 4/5] Fixed-Length Segmentation + Text Emotion  (harish_segmentation_and_text.py)
    ↓
[Step 6] Audio Emotion  (aniket_audio_emotion.py)
    ↓
[Step 7] JS Divergence Incongruence Scoring  (aniket_incongruence.py)
    ↓
outputs/incongruence/ESXXXX_scores.csv
```

---

## Scripts

| Step | Script | Output |
|------|--------|--------|
| 1 | `charan/charan_diarization.py` | `outputs/diarization/ESXXXX_diarization.csv` |
| 2 | `aaditya/aaditya_whisperx.py` | `outputs/transcripts/ESXXXX_transcript.csv` |
| 3 | `aniket/aniket_merge.py` | `outputs/merged/ESXXXX_merged.csv` |
| 4+5 | `harish/harish_segmentation_and_text.py` | `outputs/segments/`, `outputs/embeddings/ESXXXX_text_emotions.csv` |
| 6 | `aniket/aniket_audio_emotion.py` | `outputs/embeddings/ESXXXX_audio_emotions.csv` |
| 7 | `aniket/aniket_incongruence.py` | `outputs/incongruence/ESXXXX_scores.csv` |

Bash wrappers for batch runs are in `scripts/`.

---

## Technical Decisions

**Emotion label space:** `E = {happy, angry, sad, neutral}` — fixed across all modalities so JSD operates on aligned distributions.

**Text model:** `j-hartmann/emotion-english-distilroberta-base` — 7 raw labels collapsed to 4 (disgust/fear/surprise → neutral).

**Audio model:** `superb/wav2vec2-large-superb-er` — SUPERB labels hap/ang/sad/neu map 1:1 to our space. Loaded via `AutoFeatureExtractor + AutoModelForAudioClassification` (bypasses torchcodec/FFmpeg dependency).

**Speaker→headset mapping:** RMS energy analysis across Headset-0 to Headset-3 WAV files. Each speaker is assigned to the headset where they are loudest (~10× energy vs distant voices). Falls back to Mix-Headset if individual headsets unavailable.

**Segmentation:** 3s window, 1.5s hop, drop <3 words, drop multi-speaker windows.

**Incongruence score:**
```
M = 0.5 × (P_text + P_audio)
JS(P_text, P_audio) = 0.5 × KL(P_text ‖ M) + 0.5 × KL(P_audio ‖ M)
normalized to [0, 1] by dividing by log(2)

final_score = JS(P_text, P_audio) × asr_confidence
```

---

## Running the Pipeline

```bash
cd "CSCI 535 MPL/Implementation"
source venv/bin/activate

# Run all 12 meetings (Steps 1-7)
bash scripts/run_diarization.sh
bash scripts/run_whisperx.sh
bash scripts/run_merge.sh
python3 harish/harish_segmentation_and_text.py --merged ... --part ESXXXX --out_dir outputs
bash scripts/run_audio_emotion.sh
bash scripts/run_incongruence.sh
```

---

## Results

**12,076 segments scored** across all 12 meetings.

| Metric | Value |
|--------|-------|
| Mean final_score | 0.1600 |
| Max final_score | 0.7797 |
| High incongruence (≥ 0.5) | 638 segments (5.3%) |

---

## Annotation & Evaluation

**180 segments** hand-labeled by 3 annotators (majority vote ground truth).

| Label | Count | Share |
|-------|-------|-------|
| Aligned (0) | 162 | 89.4% |
| Incongruent (1) | 18 | 10.6% |

| Method | ROC-AUC | Precision | Recall | F1 | Krippendorff's α |
|--------|---------|-----------|--------|-----|-----------------|
| JS Divergence (ours) | **0.6351** | 0.1500 | 0.5000 | 0.2308 | **0.6222** |
| Valence Difference (baseline) | 0.5381 | 0.0545 | 0.1667 | 0.0822 | 0.6222 |

JS divergence outperforms the baseline by +0.097 AUC and +0.149 F1. Krippendorff's α = 0.622 (acceptable inter-rater agreement).

**Re-run evaluation:**
```bash
source venv/bin/activate

python3 scripts/evaluate_detector.py \
  --annotations annotations/annotated_segments.csv \
  --scores      annotations/annotation_sample.csv \
  --out         results/evaluation_metrics.csv

python3 scripts/evaluate_detector.py \
  --annotations annotations/annotated_segments.csv \
  --scores      annotations/baseline_scores.csv \
  --out         results/baseline_metrics.csv
```

---

## Video Emotion Pipeline (Aaditya)

Extends the incongruence scorer with a visual emotion modality from AMI per-participant closeup videos.

**Pipeline:** Closeup AVI → speaker-camera mapping → frame extraction → OpenFace AU features → FACS rule-based emotion probabilities

```python
happy_score   = 0.5 * AU06 + 0.5 * AU12
angry_score   = 0.4 * AU04 + 0.3 * AU17 + 0.3 * AU25
sad_score     = 0.4 * AU01 + 0.3 * AU04 + 0.3 * AU15
neutral_score = max(0.0, 1.0 - 0.15 * total_activation)
probs = softmax([happy, angry, sad, neutral])
```

| Script | Purpose |
|--------|---------|
| `scripts/map_speaker_camera.py` | Speaker-to-camera mapping via lip-movement variance |
| `scripts/extract_video_features.py` | Frame extraction + AU feature extraction (py-feat) |
| `scripts/run_video_emotion.py` | AU intensities → emotion probability distributions |
| `scripts/validate_video_outputs.py` | Validate output CSVs against reference segments |

All 12 meetings processed. Zero NaN, zero duplicates, all rows sum to 1.0.

---

## Environment

```bash
source venv/bin/activate   # always activate before running anything
```

- Python 3.11, torch 2.8.0, transformers 4.57.6, librosa, pandas, numpy, soundfile
- Video pipeline requires Python 3.10 venv (py-feat incompatible with 3.12)
