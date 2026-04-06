# Phase 2: Team Responsibilities & Implementation Guide

This document covers the full implementation breakdown for all four team members. Each section is self-contained — read your section, understand your inputs and outputs, and follow the validation checklist before handing off.

---

## Table of Contents

1. [Aaditya - Video Processing & Visual Emotion](#1-aaditya---video-processing--visual-emotion)
2. [Charan - Interaction Imbalance](#2-charan---interaction-imbalance)
3. [Aniket - Multimodal Incongruence & Fusion Model](#3-aniket---multimodal-incongruence--fusion-model)
4. [Harish - Data Integration & Evaluation](#4-harish---data-integration--evaluation)

---

## Step 0: Download Video Dataset (BEFORE ANYTHING ELSE)

Before starting any Phase 2 work, download the AMI video files:

```bash
cd Datasets
bash dataset_phase2.sh
```

This downloads **84 video files** (7 per meeting × 12 meetings) into `Datasets/amicorpus/ESXXXX/video/`:

| File | Description |
|------|-------------|
| `ESXXXX.Closeup1.avi` | Individual camera — participant 1 |
| `ESXXXX.Closeup2.avi` | Individual camera — participant 2 |
| `ESXXXX.Closeup3.avi` | Individual camera — participant 3 |
| `ESXXXX.Closeup4.avi` | Individual camera — participant 4 |
| `ESXXXX.PreferredOverview.avi` | Wide-angle overview |
| `ESXXXX.Corner.avi` | Corner camera |
| `ESXXXX.Overhead.avi` | Overhead camera |

**For OpenFace, use the Closeup videos** — each one captures a single participant's face. You will need to determine which Closeup (1-4) maps to which `SPEAKER_XX`, similar to how audio headsets were mapped to speakers via RMS energy in Phase 1.

Verify download completed:
```bash
ls Datasets/amicorpus/ES2002a/video/
# Should show 7 .avi files
```

---

## Critical Rules (Applies to Everyone)

### Rule 1: Compound Key

Every Phase 2 output CSV **must include both `meeting_id` and `segment_id`** columns.

Phase 1 files use `segment_id` alone, but it resets to 0 per meeting (ES2002a has 0-495, ES2003a has 0-434, etc.). Without `meeting_id`, the cross-meeting merge will produce wrong joins.

**How to add `meeting_id`:** Extract it from the filename. If you are processing `ES2002a_segments.csv`, every row in your output gets `meeting_id = ES2002a`.

### Rule 2: Validate Against Reference

Before handing off, verify your `segment_id` values match those in `outputs/segments/ESXXXX_segments.csv`. That file is the single source of truth for what segments exist.

### Rule 3: No NaN, No Duplicates

Every output must have zero NaN values and zero duplicate `(meeting_id, segment_id)` pairs.

---

## Phase 1 Output Reference (Read-Only)

These are the actual column names in the existing Phase 1 CSVs. Use these exact names when reading Phase 1 data.

| File | Columns |
|------|---------|
| `outputs/segments/ESXXXX_segments.csv` | `segment_id, start, end, speaker_id, transcript, asr_confidence` |
| `outputs/embeddings/ESXXXX_text_emotions.csv` | `segment_id, p_happy, p_angry, p_sad, p_neutral` |
| `outputs/embeddings/ESXXXX_audio_emotions.csv` | `segment_id, p_happy, p_angry, p_sad, p_neutral` |
| `outputs/incongruence/ESXXXX_scores.csv` | `segment_id, start, end, speaker_id, transcript, incongruence_score, confidence_weight, final_score` |
| `outputs/diarization/ESXXXX_diarization.csv` | `segment_id, start, end, speaker_id` |
| `outputs/merged/ESXXXX_merged.csv` | `segment_id, start, end, speaker_id, transcript, asr_confidence` |

**Note:** The Phase 1 incongruence file is named `ESXXXX_scores.csv` (not `ESXXXX_incongruence.csv`). The JSD value is in the `incongruence_score` column. The confidence-weighted version is `final_score`.

---

---

# 1. Aaditya - Video Processing & Visual Emotion

## Role

Build the **video modality pipeline**. Your output is `P_video`: per-segment emotion probability distributions extracted from meeting video using OpenFace. This feeds directly into Aniket's multimodal incongruence computation.

## Prerequisites

**Run `bash Datasets/dataset_phase2.sh` from the project root first** (or `cd Datasets && bash dataset_phase2.sh`). This downloads all AMI video files into `Datasets/amicorpus/ESXXXX/video/`.

After download, each meeting has 7 video files. **Use the Closeup videos for OpenFace:**

```
Datasets/amicorpus/ES2002a/video/
├── ES2002a.Closeup1.avi    ← participant 1 face
├── ES2002a.Closeup2.avi    ← participant 2 face
├── ES2002a.Closeup3.avi    ← participant 3 face
├── ES2002a.Closeup4.avi    ← participant 4 face
├── ES2002a.PreferredOverview.avi
├── ES2002a.Corner.avi
└── ES2002a.Overhead.avi
```

**Speaker-camera mapping:** You need to determine which Closeup (1-4) maps to which `SPEAKER_XX`. This is analogous to how Aniket mapped audio headsets to speakers in Phase 1 (via RMS energy analysis). Options: visual inspection, or use diarization timestamps to correlate who is speaking with which camera shows lip movement.

## Position in Pipeline

```
Datasets/amicorpus/ESXXXX/video/ESXXXX.Closeup{1-4}.avi
      │
      ▼
Step 1: Frame Extraction
      │
      ▼
Step 2: OpenFace Feature Extraction
      │
      ▼
      outputs/video_features/ESXXXX_video_features.csv
      │
      ▼
Step 3: Visual Emotion Modeling
      │
      ▼
      outputs/video_emotions/ESXXXX_video_emotions.csv   ← FINAL HANDOFF TO ANIKET
```

## Tasks

### Step 0 - Download AMI Video

```bash
cd Datasets && bash dataset_phase2.sh
```

Verify: `ls Datasets/amicorpus/ES2002a/video/` should show 7 `.avi` files.

### Step 0.5 - Speaker-Camera Mapping

Determine which `Closeup{1-4}` corresponds to which `SPEAKER_XX` for each meeting. Save this mapping — you'll need it to assign the correct face to each segment's speaker.

### Step 1 - Frame Extraction

Extract frames from the **Closeup videos** at a consistent frame rate. Align each frame to a `segment_id` using the segment timestamps in `outputs/segments/ESXXXX_segments.csv`.

- Use `Datasets/amicorpus/ESXXXX/video/ESXXXX.Closeup{N}.avi` where N is the camera for the segment's speaker
- Sample at 1 fps or keyframes only (balance coverage vs storage)
- For each frame, determine which segment it falls into using `start` and `end` times from the segments file
- If multiple frames fall within one segment, aggregate features in Step 3

### Step 2 - OpenFace Feature Extraction

Run OpenFace on the extracted frames using `scripts/run_openface.sh`.

Features to extract:

- Action Units (AU): AU01, AU02, AU04, AU06, AU12, AU15, AU17, AU25
- Head pose: pitch, yaw, roll
- Gaze direction vectors

### Step 3 - Visual Emotion Modeling

Map OpenFace features to emotion probabilities using `scripts/run_video_emotion.py`.

Target emotion label space (**must match text and audio exactly**):

```
happy, angry, sad, neutral
```

Use either a pretrained AU-to-emotion mapping or a lightweight classifier trained on AffectNet/CK+. Do **not** use a label space that differs from text/audio — the incongruence computation requires identical categories.

## Input Files

| File | Description |
|------|-------------|
| `Datasets/amicorpus/ESXXXX/video/ESXXXX.Closeup{1-4}.avi` | Per-participant face camera (downloaded via `dataset_phase2.sh`) |
| `outputs/segments/ESXXXX_segments.csv` | Segment timestamps and IDs — columns: `segment_id, start, end, speaker_id, transcript, asr_confidence` |

## Output Files

### Intermediate: Raw Features

```
outputs/video_features/ESXXXX_video_features.csv
```

| Column | Type | Description |
|--------|------|-------------|
| `meeting_id` | string | e.g. `ES2002a` |
| `segment_id` | int | Must match Phase 1 segments exactly |
| `AU01` ... `AU25` | float | Action unit intensities |
| `head_pitch` | float | Head pose pitch |
| `head_yaw` | float | Head pose yaw |
| `head_roll` | float | Head pose roll |
| `gaze_x` | float | Gaze direction x |
| `gaze_y` | float | Gaze direction y |

### Final Handoff: Visual Emotion Probabilities

```
outputs/video_emotions/ESXXXX_video_emotions.csv
```

| Column | Type | Description |
|--------|------|-------------|
| `meeting_id` | string | e.g. `ES2002a` |
| `segment_id` | int | Must match Phase 1 segments exactly |
| `p_happy` | float | Probability, sums to 1 with others |
| `p_angry` | float | Probability |
| `p_sad` | float | Probability |
| `p_neutral` | float | Probability |

> `p_happy + p_angry + p_sad + p_neutral` must equal 1.0 for every row.

## Scripts

| Script | Purpose |
|--------|---------|
| `scripts/run_openface.sh` | Batch OpenFace extraction on video frames |
| `scripts/run_video_emotion.py` | Map AU features to emotion probabilities |

## Validation Checklist

- [ ] AMI video files downloaded for all 12 meetings
- [ ] Every `segment_id` in your output exists in `outputs/segments/ESXXXX_segments.csv`
- [ ] `meeting_id` column is present and correct in every output file
- [ ] No `(meeting_id, segment_id)` pair is missing or duplicated
- [ ] All four probability columns are present and sum to 1.0 per row
- [ ] Emotion column names match exactly: `p_happy`, `p_angry`, `p_sad`, `p_neutral`
- [ ] Tested on at least one full meeting before batch run
- [ ] No NaN or null values in any row

## Notes

- Run on one meeting end-to-end first. Confirm output format before batching all meetings.
- If OpenFace fails on a segment (no face detected), fill with uniform distribution `[0.25, 0.25, 0.25, 0.25]` and log the segment ID.
- Do not invent or modify `segment_id` values. Copy them directly from the segments file.
- The segments file has `segment_id` as integer (0, 1, 2, ...) — keep the same type in your output.

---

---

# 2. Charan - Interaction Imbalance

## Role

Compute **conversation dynamics features** from diarization data. Your output captures how balanced or imbalanced the speaking patterns are within each meeting segment. This feeds into the final feature merge alongside video and audio-text incongruence.

## Position in Pipeline

```
outputs/segments/ESXXXX_segments.csv    (Phase 1 — segment boundaries + speaker)
outputs/diarization/ESXXXX_diarization.csv  (Phase 1 — raw diarization)
      │
      ▼
charan/charan_interaction.py
scripts/run_interaction_metrics.py
      │
      ▼
outputs/imbalance/ESXXXX_imbalance.csv   ← FINAL HANDOFF TO HARISH
```

## Tasks

### Task 1 - Speaking Time per Speaker

For each meeting, compute the total speaking time per speaker across all diarization segments.

- Use `start` and `end` from `outputs/diarization/ESXXXX_diarization.csv`
- Group by `speaker_id`
- Output: seconds spoken per speaker

### Task 2 - Gini Coefficient

Compute the Gini coefficient of speaking time distribution across speakers in the meeting.

- Gini = 0: perfectly equal participation
- Gini = 1: one speaker dominates entirely

This is a meeting-level metric. Broadcast it to all segments of that meeting (same value for every row of a given meeting).

### Task 3 - Interruption Detection

An interruption occurs when speaker B begins speaking before speaker A has finished, with a configurable overlap threshold (e.g. >= 200ms overlap).

- Use the raw diarization output (`outputs/diarization/ESXXXX_diarization.csv`) to find overlapping speaker turns
- For each segment in `outputs/segments/ESXXXX_segments.csv`, count how many interruptions occurred within that segment's `[start, end]` window

### Task 4 - Turn Statistics

Compute per segment:

- `dominance_ratio`: fraction of speaking time by the dominant speaker within a window around this segment
- `turn_entropy`: Shannon entropy of the turn distribution in a window around this segment (higher = more balanced)

Use a **30-second sliding window** centered on each segment's midpoint. Agree with the team before changing this.

## Input Files

| File | Actual Columns |
|------|----------------|
| `outputs/segments/ESXXXX_segments.csv` | `segment_id, start, end, speaker_id, transcript, asr_confidence` |
| `outputs/diarization/ESXXXX_diarization.csv` | `segment_id, start, end, speaker_id` |

**Note:** The `segment_id` in the diarization file is different from the `segment_id` in the segments file. Diarization has raw utterance-level IDs. Segments have windowed IDs. Use the **segments file** as your reference for output segment_ids. Use the **diarization file** for computing speaking time and overlaps.

## Output File

```
outputs/imbalance/ESXXXX_imbalance.csv
```

| Column | Type | Description |
|--------|------|-------------|
| `meeting_id` | string | e.g. `ES2002a` |
| `segment_id` | int | Must match Phase 1 segments exactly |
| `speaker_id` | string | Speaker for this segment (from segments file) |
| `gini` | float | Gini coefficient for the whole meeting (0 to 1) |
| `interruptions` | int | Interruption count in/around this segment |
| `dominance_ratio` | float | Fraction of speaking time by dominant speaker in window |
| `turn_entropy` | float | Shannon entropy of turn distribution in window |

## Script

```
charan/charan_interaction.py         — core logic
scripts/run_interaction_metrics.py   — batch runner
```

## Validation Checklist

- [ ] Every `segment_id` in your output exists in `outputs/segments/ESXXXX_segments.csv`
- [ ] `meeting_id` column is present and correct
- [ ] No `(meeting_id, segment_id)` pair is missing or duplicated
- [ ] `gini` is between 0.0 and 1.0 for all rows
- [ ] `dominance_ratio` is between 0.0 and 1.0 for all rows
- [ ] `turn_entropy` is non-negative for all rows
- [ ] No NaN or null values in any column
- [ ] Row count matches the segments file exactly per meeting

## Notes

- Gini is a meeting-level feature. All segments from the same meeting will have the same `gini` value. This is expected and correct.
- For turn entropy, use the 30s sliding window around each segment's timestamp, not a global per-meeting window.
- If a segment has only one speaker in its window, `turn_entropy` = 0. That is valid.
- Do not modify or reassign `segment_id` values. Copy them directly from the segments file.
- The diarization file uses its own `segment_id` numbering (raw utterances). Do not confuse it with the windowed segment IDs from the segments file.

---

---

# 3. Aniket - Multimodal Incongruence & Fusion Model

## Role

Two responsibilities in Phase 2:

1. **Multimodal incongruence** (Step 4): Extend text-audio JSD to all three modality pairs once Aaditya's video emotions are ready.
2. **Fusion model** (Step 6): Train the instability classifier on the merged feature matrix once Harish's merge is complete. Run the ablation study.

## Position in Pipeline

```
Step 4 — Multimodal Incongruence:

outputs/embeddings/ESXXXX_text_emotions.csv     (Phase 1)
outputs/embeddings/ESXXXX_audio_emotions.csv    (Phase 1)
outputs/video_emotions/ESXXXX_video_emotions.csv (Aaditya)
              │
              ▼
aniket/aniket_multimodal_incongruence.py
              │
              ▼
outputs/incongruence/ESXXXX_multimodal_incongruence.csv   ← HANDOFF TO HARISH

──────────────────────────────────────────────────────────

Step 6 — Fusion Model (after Harish merge):

outputs/final_dataset/ESXXXX_final.csv   (Harish)
              │
              ▼
aniket/aniket_fusion.py
scripts/run_fusion.py
              │
              ▼
outputs/predictions/predictions.csv           ← HANDOFF TO HARISH
results/experiment_log.csv
```

## Task 1: Multimodal Incongruence (Step 4)

### What to Build

Extend the Phase 1 text-audio JSD score to cover all three modality pairs.

**Important:** Phase 1 already produced `outputs/incongruence/ESXXXX_scores.csv` with the `incongruence_score` column (raw JSD between text and audio). You can reuse that value or recompute it — but the new output file should contain all three pairs plus the composite.

| Pair | Column Name |
|------|-------------|
| Text vs Audio | `jsd_text_audio` |
| Text vs Video | `jsd_text_video` |
| Audio vs Video | `jsd_audio_video` |
| Composite | `incongruence_composite` (mean of all three) |

### JSD Formula (same as Phase 1)

```
M = 0.5 * (P + Q)
JSD(P || Q) = 0.5 * KL(P || M) + 0.5 * KL(Q || M)
Normalized JSD = JSD / log(2)    → range [0, 1]
```

Add epsilon (1e-10) before computing KL to avoid log(0).

### Input

| File | Columns You Need |
|------|-----------------|
| `outputs/embeddings/ESXXXX_text_emotions.csv` | `segment_id, p_happy, p_angry, p_sad, p_neutral` |
| `outputs/embeddings/ESXXXX_audio_emotions.csv` | `segment_id, p_happy, p_angry, p_sad, p_neutral` |
| `outputs/video_emotions/ESXXXX_video_emotions.csv` | `meeting_id, segment_id, p_happy, p_angry, p_sad, p_neutral` |

**Note:** Phase 1 emotion files do not have `meeting_id`. Extract it from the filename when loading.

### Output

```
outputs/incongruence/ESXXXX_multimodal_incongruence.csv
```

| Column | Type | Description |
|--------|------|-------------|
| `meeting_id` | string | e.g. `ES2002a` |
| `segment_id` | int | Must match Phase 1 segments |
| `jsd_text_audio` | float | JSD between P_text and P_audio |
| `jsd_text_video` | float | JSD between P_text and P_video |
| `jsd_audio_video` | float | JSD between P_audio and P_video |
| `incongruence_composite` | float | Mean of all three JSD scores |

All JSD values must be in [0.0, 1.0].

**Note:** Do NOT overwrite the Phase 1 `ESXXXX_scores.csv` files. Write to a new filename: `ESXXXX_multimodal_incongruence.csv`.

## Task 2: Fusion Model (Step 6)

### Input

```
outputs/final_dataset/ESXXXX_final.csv   (from Harish)
```

Coordinate with Harish on exact column names before writing loading code.

### Ablation Configurations

Train in this exact order:

| Config | Features Used |
|--------|--------------|
| A — Baseline | `P_text + P_audio` (8 features) |
| B — + Video | `P_text + P_audio + P_video` (12 features) |
| C — + Incongruence | All above + 4 JSD scores (16 features) |
| D — + Imbalance | All above + 4 imbalance features (20 features) |
| E — Full Model | All features (20 features) |

### Model

Start simple. The goal is to show feature value, not model complexity.

- **Primary:** Logistic regression (good for 180 samples)
- **Secondary:** MLP with 2 hidden layers (64, 32) — only if logistic regression works
- Use 5-fold stratified cross-validation (stratify on label to handle 10% positive rate)
- Report mean and std of each metric across folds
- Normalize features with `StandardScaler` — fit on train fold only, transform test fold

### Labels

Use `annotations/annotated_segments.csv` — column `majority_label` (0 = aligned, 1 = incongruent). 180 labeled segments total.

To match annotation rows to features: join on `meeting_id` + `segment_id`. The annotation file has both columns.

### Output Files

```
outputs/predictions/predictions.csv
```

| Column | Description |
|--------|-------------|
| `meeting_id` | Meeting identifier |
| `segment_id` | Segment identifier |
| `label` | Ground truth (from annotations) |
| `pred_label` | Predicted label |
| `pred_prob` | Predicted probability for positive class |
| `config` | Ablation configuration name (A through E) |
| `fold` | CV fold number |

```
results/experiment_log.csv
```

| Column | Description |
|--------|-------------|
| `config` | Configuration name |
| `auc` | Mean ROC-AUC across folds |
| `auc_std` | Std of ROC-AUC |
| `f1` | Mean F1 score |
| `precision` | Mean precision |
| `recall` | Mean recall |

## Scripts

| Script | Purpose |
|--------|---------|
| `aniket/aniket_multimodal_incongruence.py` | Compute JSD for all three modality pairs |
| `aniket/aniket_fusion.py` | Fusion model + ablation logic |
| `scripts/run_fusion.py` | Run all ablation configurations |

## Validation Checklist

Before handing incongruence output to Harish:

- [ ] Every `segment_id` matches Phase 1 segments exactly
- [ ] `meeting_id` column present and correct
- [ ] No missing or duplicate `(meeting_id, segment_id)`
- [ ] All JSD values in [0.0, 1.0]
- [ ] No NaN values
- [ ] Phase 1 `ESXXXX_scores.csv` files are untouched

Before handing predictions to Harish:

- [ ] `predictions.csv` has rows for all 180 annotated segments x 5 configs
- [ ] `experiment_log.csv` has one row per config with all metrics
- [ ] All five ablation configurations (A through E) are present

## Notes

- Add epsilon (1e-10) before KL divergence to avoid log(0). Do not clip to zero.
- Normalize all features with `StandardScaler` before training. Fit on train split only.
- Phase 1 emotion CSVs do not have `meeting_id` — extract from filename (`ES2002a_text_emotions.csv` → `ES2002a`).
- The ablation delta (gain per added modality) is the most important result.

---

---

# 4. Harish - Data Integration & Evaluation

## Role

Merge all modality outputs into a single validated feature matrix, then run evaluation and generate all plots. You are the **last line of defense before modeling**. If your merge is wrong, everything downstream breaks silently.

## Position in Pipeline

```
outputs/embeddings/           (Phase 1 — text/audio emotions)
outputs/incongruence/         (Phase 1 — ESXXXX_scores.csv)
                              (Aniket — ESXXXX_multimodal_incongruence.csv)
outputs/video_emotions/       (Aaditya — P_video per segment)
outputs/imbalance/            (Charan  — interaction imbalance features)
annotations/annotated_segments.csv  (labels)
              │
              ▼
scripts/merge_all_features.py
              │
              ▼
outputs/final_dataset/final_dataset.csv   ← HANDOFF TO ANIKET

              [Aniket trains model]

outputs/predictions/predictions.csv       (Aniket)
              │
              ▼
scripts/evaluate_detector.py
scripts/plot_results.py
              │
              ▼
results/                    ← FINAL OUTPUT
```

## Tasks

### Task 1 - Feature Merge

Join all outputs on `(meeting_id, segment_id)` using an inner join. Every modality must be present for a segment to be included.

| Source | File Pattern | Key Columns to Extract |
|--------|-------------|----------------------|
| Text emotion | `outputs/embeddings/ESXXXX_text_emotions.csv` | `p_happy, p_angry, p_sad, p_neutral` |
| Audio emotion | `outputs/embeddings/ESXXXX_audio_emotions.csv` | `p_happy, p_angry, p_sad, p_neutral` |
| Video emotion | `outputs/video_emotions/ESXXXX_video_emotions.csv` | `p_happy, p_angry, p_sad, p_neutral` |
| Incongruence | `outputs/incongruence/ESXXXX_multimodal_incongruence.csv` | `jsd_text_audio, jsd_text_video, jsd_audio_video, incongruence_composite` |
| Imbalance | `outputs/imbalance/ESXXXX_imbalance.csv` | `gini, interruptions, dominance_ratio, turn_entropy` |
| Labels | `annotations/annotated_segments.csv` | `majority_label` |

**Important naming:** When merging emotion columns from three modalities, prefix them to avoid collision:

- Text: `p_text_happy`, `p_text_angry`, `p_text_sad`, `p_text_neutral`
- Audio: `p_audio_happy`, `p_audio_angry`, `p_audio_sad`, `p_audio_neutral`
- Video: `p_video_happy`, `p_video_angry`, `p_video_sad`, `p_video_neutral`

**Phase 1 emotion CSVs don't have `meeting_id`.** Extract it from the filename when loading.

> Do not begin the merge until all four inputs (video emotions, imbalance, multimodal incongruence) are confirmed ready.

### Task 2 - Alignment Validation

Before saving the merged file, run these checks:

- Row count is consistent across all input files for the same meeting
- No `(meeting_id, segment_id)` pair appears more than once
- No NaN values in any feature column
- Emotion probability columns sum to 1.0 per modality per row (within tolerance of 0.01)
- All expected columns are present

Log validation failures to `logs/merge_validation.log`. Do not silently drop rows.

### Task 3 - Evaluation

After Aniket returns predictions, compute:

- ROC-AUC (overall and per ablation config)
- Precision, Recall, F1 (at optimal threshold)
- Confusion matrix per config

### Task 4 - Plots

Generate the following and save to `results/plots/`:

- ROC curves (one per ablation configuration, overlaid)
- Precision-Recall curves
- Confusion matrix heatmap (for full model)
- Ablation delta bar chart: AUC gain from each added modality
- Feature importance bar chart (if model supports it)

## Output Files

### Merged Feature Matrix

```
outputs/final_dataset/final_dataset.csv
```

| Column | Source |
|--------|--------|
| `meeting_id` | Join key |
| `segment_id` | Join key |
| `p_text_happy`, `p_text_angry`, `p_text_sad`, `p_text_neutral` | Text emotion |
| `p_audio_happy`, `p_audio_angry`, `p_audio_sad`, `p_audio_neutral` | Audio emotion |
| `p_video_happy`, `p_video_angry`, `p_video_sad`, `p_video_neutral` | Video emotion |
| `jsd_text_audio`, `jsd_text_video`, `jsd_audio_video` | Incongruence |
| `incongruence_composite` | Incongruence |
| `gini`, `interruptions`, `dominance_ratio`, `turn_entropy` | Imbalance |
| `label` | Ground truth (from annotations, NaN if not annotated) |

### Evaluation Results

```
results/
├── phase2_metrics_summary.csv
├── phase2_experiment_log.csv
└── plots/
    ├── phase2_roc_curves.png
    ├── phase2_pr_curves.png
    ├── phase2_confusion_matrix.png
    ├── phase2_ablation_delta.png
    └── phase2_feature_importance.png
```

## Scripts

| Script | Purpose |
|--------|---------|
| `scripts/merge_all_features.py` | Inner join all modality outputs on `(meeting_id, segment_id)` |
| `scripts/evaluate_detector.py` | Compute metrics against ground truth |
| `scripts/plot_results.py` | Generate all result visualizations |

## Validation Checklist

Before handing merged data to Aniket:

- [ ] `meeting_id` and `segment_id` both present
- [ ] No duplicate `(meeting_id, segment_id)` rows
- [ ] Zero NaN values in feature columns
- [ ] All three P_text, P_audio, P_video distributions sum to ~1.0 per row
- [ ] All expected columns present with correct names (see table above)
- [ ] Validation log written to `logs/merge_validation.log`
- [ ] Labeled rows (from annotations) have `label` column populated

Before finalizing evaluation:

- [ ] Predictions file from Aniket loaded correctly
- [ ] Ground truth labels align on `(meeting_id, segment_id)`
- [ ] All plots saved and readable
- [ ] `phase2_metrics_summary.csv` has one row per ablation config

## Notes

- Use inner join only. If a segment is missing from any modality, exclude it and log it.
- Column naming must be exact — Aniket's fusion model loads columns by name.
- If row counts after merge drop by more than 10% compared to Phase 1 segments, flag it to the team.
- Phase 1 emotion CSVs use `p_happy`, `p_angry`, etc. — you must rename them to `p_text_happy`, `p_audio_happy`, `p_video_happy` during merge to avoid column name collisions.
- The existing Phase 1 `results/` files (`evaluation_metrics.csv`, etc.) should not be overwritten. Prefix Phase 2 results with `phase2_`.
