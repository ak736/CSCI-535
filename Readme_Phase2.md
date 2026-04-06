# Phase 2: Communication Instability Modeling

## Phase 1 Recap (What We Have)

Phase 1 built a complete text-audio incongruence pipeline on the AMI corpus (12 meetings, 12,076 segments). Every step is done and validated.

### Phase 1 Outputs (DO NOT MODIFY)

| Folder | File Pattern | Columns | Count |
|--------|-------------|---------|-------|
| `outputs/diarization/` | `ESXXXX_diarization.csv` | `segment_id, start, end, speaker_id` | 12 |
| `outputs/transcripts/` | `ESXXXX_transcript.csv` | `segment_id, start, end, transcript, asr_confidence` | 12 |
| `outputs/merged/` | `ESXXXX_merged.csv` | `segment_id, start, end, speaker_id, transcript, asr_confidence` | 12 |
| `outputs/segments/` | `ESXXXX_segments.csv` | `segment_id, start, end, speaker_id, transcript, asr_confidence` | 12 |
| `outputs/embeddings/` | `ESXXXX_text_emotions.csv` | `segment_id, p_happy, p_angry, p_sad, p_neutral` | 12 |
| `outputs/embeddings/` | `ESXXXX_audio_emotions.csv` | `segment_id, p_happy, p_angry, p_sad, p_neutral` | 12 |
| `outputs/incongruence/` | `ESXXXX_scores.csv` | `segment_id, start, end, speaker_id, transcript, incongruence_score, confidence_weight, final_score` | 12 |

### Phase 1 Results

| Metric | JS Divergence (Ours) | Valence Baseline |
|--------|---------------------|------------------|
| ROC-AUC | 0.6351 | 0.5381 |
| F1 | 0.2308 | 0.0822 |
| Krippendorff's alpha | 0.6222 | — |

- 180 segments annotated (3 annotators, majority vote)
- 162 aligned (89.4%), 18 incongruent (10.6%)
- Annotations in `annotations/annotated_segments.csv`

### Important: segment_id is Per-Meeting

`segment_id` resets to 0 in every meeting file. ES2002a has 0-495, ES2003a has 0-434, etc. All Phase 2 outputs **must include `meeting_id`** alongside `segment_id` to avoid collisions during cross-meeting merge.

---

## Step 0: Download Video Dataset

Before any Phase 2 work, download the AMI video files:

```bash
cd Datasets && bash dataset_phase2.sh
```

Downloads **84 video files** (7 per meeting × 12 meetings) into `Datasets/amicorpus/ESXXXX/video/`:
- `ESXXXX.Closeup1.avi` through `Closeup4.avi` — **individual participant cameras** (use these for OpenFace)
- `ESXXXX.PreferredOverview.avi`, `Corner.avi`, `Overhead.avi` — room cameras (not needed for emotion)

---

## Phase 2 Goal

Extend from single-modality mismatch (text vs audio) to full communication instability:

```
Communication Instability = Multimodal Mismatch + Interaction Imbalance
```

Phase 2 adds:
1. **Video emotion** (facial expressions via OpenFace)
2. **Interaction imbalance** (speaking dynamics from diarization)
3. **Instability-aware fusion model** (combine everything, ablation study)

---

## Pipeline

```
Phase 1 Outputs (frozen)
         │
         ├──────────────────────────────────┐
         ▼                                  ▼
┌────────────────────┐           ┌──────────────────────┐
│  Aaditya           │           │  Charan               │
│  Video Processing  │           │  Interaction Metrics  │
│  - Frame extract   │           │  - Gini coefficient   │
│  - OpenFace AUs    │           │  - Interruptions      │
│  - Visual emotion  │           │  - Dominance ratio    │
│  → P_video         │           │  - Turn entropy       │
└────────────────────┘           └──────────────────────┘
         │                                  │
         └──────────────┬───────────────────┘
                        ▼
             ┌─────────────────────┐
             │  Aniket             │
             │  Multimodal         │
             │  Incongruence       │
             │  (Text+Audio+Video) │
             └─────────────────────┘
                        │
                        ▼
             ┌─────────────────────┐
             │  Harish             │
             │  Feature Merge +    │
             │  Validation         │
             └─────────────────────┘
                        │
                        ▼
             ┌─────────────────────┐
             │  Aniket             │
             │  Fusion Model +     │
             │  Ablation Study     │
             └─────────────────────┘
                        │
                        ▼
             ┌─────────────────────┐
             │  Harish             │
             │  Evaluation + Plots │
             └─────────────────────┘
```

---

## Team Responsibilities

| Member  | Role                                   | Output Folder             |
|---------|----------------------------------------|---------------------------|
| Aaditya | Video processing + visual emotion      | `outputs/video_emotions/` |
| Charan  | Interaction imbalance metrics          | `outputs/imbalance/`      |
| Aniket  | Multimodal incongruence + fusion model | `outputs/predictions/`    |
| Harish  | Feature merging + evaluation + plots   | `outputs/final_dataset/`  |

---

## Execution Order

> **Strict sequential dependency. Do not skip steps.**

| Step | Owner          | Task                               | Depends On          |
|------|----------------|------------------------------------|---------------------|
| 0    | Aaditya        | Download AMI video files           | —                   |
| 1    | Aaditya        | Extract video frames               | Step 0              |
| 2    | Aaditya        | Run OpenFace + visual emotion      | Step 1              |
| 3    | Charan         | Compute interaction metrics        | Phase 1 diarization |
| 4    | Aniket         | Compute multimodal incongruence    | Steps 2, 3          |
| 5    | Harish         | Merge all features                 | Steps 2, 3, 4       |
| 6    | Aniket         | Train fusion model + ablation      | Step 5              |
| 7    | Harish         | Evaluate + generate plots          | Step 6              |

**Steps 1-2 (Aaditya) and Step 3 (Charan) can run in parallel.**

---

## Critical Rules

### 1. Compound Key: `meeting_id` + `segment_id`

Every Phase 2 output CSV **must include both columns**. Phase 1 files use `segment_id` alone (per-meeting), but the cross-meeting merge requires `meeting_id` to avoid collisions.

### 2. Emotion Label Space Must Match

```
E = {happy, angry, sad, neutral}
```

Aaditya's P_video must use `p_happy, p_angry, p_sad, p_neutral` — identical to text and audio. If the visual model outputs different labels, remap before saving.

### 3. Do Not Modify Phase 1 Outputs

All files in `outputs/diarization/`, `outputs/transcripts/`, `outputs/merged/`, `outputs/segments/`, `outputs/embeddings/`, and `outputs/incongruence/` are frozen. Read from them, never write to them.

---

## Output Folder Structure

```
outputs/
├── segments/               # Phase 1 — reference segments (FROZEN)
├── embeddings/             # Phase 1 — text/audio emotions (FROZEN)
├── incongruence/           # Phase 1 — text-audio scores (FROZEN)
├── video_features/         # Aaditya — raw OpenFace AU features
├── video_emotions/         # Aaditya — P_video per segment
├── imbalance/              # Charan  — interaction imbalance features
├── final_dataset/          # Harish  — merged full feature matrix
└── predictions/            # Aniket  — model outputs and experiment logs
```

---

## Known Constraints

### Video Data
Run `cd Datasets && bash dataset_phase2.sh` to download. Downloads into `Datasets/amicorpus/ESXXXX/video/`. The Closeup1-4 videos each capture one participant's face — Aaditya must determine which Closeup maps to which `SPEAKER_XX` (similar to the headset-speaker mapping from Phase 1).

### Limited Labels
Only 180 segments are human-annotated (18 incongruent). For the fusion model:
- Use logistic regression or small MLP — not enough data for complex models
- 5-fold cross-validation on these 180 segments
- Report AUC as primary metric (handles class imbalance better than F1)
- Consider threshold-based pseudo-labels from Phase 1 scores as additional weak supervision if needed

---

## Ablation Study Design

Train in this exact order to show incremental value of each modality:

| Config | Features |
|--------|----------|
| A — Baseline | P_text + P_audio |
| B — + Video | P_text + P_audio + P_video |
| C — + Incongruence | All above + JSD scores (text-audio, text-video, audio-video) |
| D — + Imbalance | All above + gini, interruptions, dominance_ratio, turn_entropy |
| E — Full Model | All features |

---

## Evaluation Metrics

- ROC-AUC (primary — handles class imbalance)
- Precision / Recall / F1
- Per-modality ablation delta
- Confusion matrix

---

## Deliverables

- [ ] AMI video downloaded and accessible (Aaditya)
- [ ] Visual emotion probabilities per segment (Aaditya)
- [ ] Interaction imbalance features per segment (Charan)
- [ ] Multimodal incongruence scores (Aniket)
- [ ] Merged final dataset (Harish)
- [ ] Fusion model + ablation study (Aniket)
- [ ] Evaluation metrics + plots (Harish)
- [ ] Final report
- [ ] Presentation

---

## File Ownership

| File / Script                        | Owner   |
|--------------------------------------|---------|
| `aaditya/aaditya_video_emotion.py`   | Aaditya |
| `scripts/run_openface.sh`            | Aaditya |
| `scripts/run_video_emotion.py`       | Aaditya |
| `charan/charan_interaction.py`       | Charan  |
| `scripts/run_interaction_metrics.py` | Charan  |
| `aniket/aniket_multimodal_incongruence.py` | Aniket |
| `aniket/aniket_fusion.py`            | Aniket  |
| `scripts/run_fusion.py`              | Aniket  |
| `scripts/merge_all_features.py`      | Harish  |
| `scripts/evaluate_detector.py`       | Harish  |
| `scripts/plot_results.py`            | Harish  |
