# Phase 2: Controlled Incongruence Detection on RAVDESS

> Drop this file + `dataset_ravdess.sh` into your repo. Create the `phase2_ravdess/` folder. Start.

---

## Before You Read Anything Else

**New folder.** All Phase 2 work goes in `phase2_ravdess/`. Phase 1 stays untouched.

**Old outputs are NOT reused.** Different dataset, different segments, different files. What IS reused: the same models (wav2vec2, distilroberta) and the same JSD math. You adapt the code, not copy the CSVs.

**No AMI work in Phase 2.** We focus entirely on RAVDESS. Phase 1 AMI results appear only in the final report as comparison numbers.

---

## Step 0: Everyone Downloads the Dataset

```bash
# Put dataset_ravdess.sh in Datasets/ folder, then:
cd Datasets
bash dataset_ravdess.sh
```

This downloads:

| What              | Size   | Files      | Path                                         |
| ----------------- | ------ | ---------- | -------------------------------------------- |
| Audio-only speech | 215 MB | 1440 .wav  | `Datasets/ravdess/audio_speech/Actor_XX/`    |
| Video speech      | ~12 GB | 2880 .mp4  | `Datasets/ravdess/video_speech/Actor_XX/`    |
| OpenFace tracking | ~50 MB | ~2452 .csv | `Datasets/ravdess/facial_tracking/Actor_XX/` |

**Verify after download:**

```bash
find Datasets/ravdess/audio_speech -name "*.wav" | wc -l     # must be 1440
find Datasets/ravdess/facial_tracking -name "*.csv" | wc -l  # must be ~2452
```

**If disk is tight:** You only strictly need audio (~215 MB) + tracking (~50 MB). Video .mp4 files are optional since we use the pre-extracted tracking CSVs.

---

## Why RAVDESS

Phase 1 proved the concept on AMI meetings (AUC 0.635). But AMI had problems: only 18 incongruent labels out of 180, messy meeting audio, and noisy ground truth (alpha 0.622).

RAVDESS is a controlled acted dataset: 24 actors, 2 fixed sentences, 8 emotions, ground-truth labels in the filenames. We synthetically construct congruent/incongruent pairs by mixing audio from one emotion with video from another. This gives us thousands of perfectly labeled samples instead of 180 noisy ones.

---

## Dataset Basics

**24 actors** (12 male, 12 female). **2 sentences:** "Kids are talking by the door" and "Dogs are sitting by the door." **8 emotions** at 2 intensity levels.

**Filename format:** `MODALITY-CHANNEL-EMOTION-INTENSITY-STATEMENT-REPETITION-ACTOR`

Example: `03-01-05-02-01-01-08.wav` = Audio-only (03), Speech (01), Angry (05), Strong (02), Statement 1 (01), Repetition 1 (01), Actor 08 (female).

**Emotion codes and our 4-class mapping:**

| Code | RAVDESS Label | Our Label |
| ---- | ------------- | --------- |
| 01   | neutral       | neutral   |
| 02   | calm          | neutral   |
| 03   | happy         | happy     |
| 04   | sad           | sad       |
| 05   | angry         | angry     |
| 06   | fearful       | neutral   |
| 07   | disgust       | neutral   |
| 08   | surprised     | neutral   |

Same 4-class space as Phase 1: `E = {happy, angry, sad, neutral}`.

**Pre-extracted OpenFace features:** The RAVDESS Facial Landmark Tracking dataset provides AU intensities, head pose, and gaze for every video trial. No need to run py-feat or OpenFace yourself.

---

## Folder Structure

Create this in your repo root:

```
phase2_ravdess/
├── scripts/
│   ├── harish_parse_metadata.py
│   ├── harish_text_emotion.py
│   ├── harish_build_pairs.py
│   ├── charan_audio_emotion.py
│   ├── charan_jsd_scoring.py
│   ├── aaditya_video_emotion.py
│   ├── aaditya_evaluate_and_plot.py
│   └── aniket_classifier.py
│
├── metadata/
│   └── ravdess_metadata.csv              ← Harish writes here
│
├── emotions/
│   ├── text_emotions.csv                 ← Harish writes here
│   ├── audio_emotions.csv                ← Charan writes here
│   └── video_emotions.csv                ← Aaditya writes here
│
├── pairs/
│   └── incongruence_pairs.csv            ← Harish writes here
│
├── scores/
│   └── pair_scores.csv                   ← Charan writes here
│
├── predictions/
│   ├── predictions.csv                   ← Aniket writes here
│   └── experiment_log.csv                ← Aniket writes here
│
└── results/
    ├── evaluation_metrics.csv            ← Aaditya writes here
    └── plots/
        ├── roc_curves.png                ← Aaditya writes here
        ├── ablation_delta.png
        ├── emotion_pair_heatmap.png
        ├── confusion_matrix.png
        └── score_distribution.png
```

**Each person writes ONLY to their designated output folders. Read from anyone's folder, write only to yours.**

---

## Architecture Diagram

```
RAVDESS Clips (emotion-labeled, single-actor, controlled)
      │
      ├── Audio .wav ─────→ wav2vec2 ──────────→ P_audio     [Charan]
      │   (1440 clips)                                  │
      │                                                 │
      ├── Tracking CSVs ──→ AU mean-pool → softmax ──→ P_video    [Aaditya]
      │   (pre-extracted)                                │
      │                                                 │
      └── 2 fixed sentences → distilroberta ──────→ P_text      [Harish]
                                                        │
                                                        ▼
                              ┌──────────────────────────────┐
                              │  Pair Construction           │
                              │  audio(angry) + video(happy) │  [Harish]
                              │  → label = 1 (incongruent)   │
                              │  audio(sad) + video(sad)     │
                              │  → label = 0 (congruent)     │
                              └──────────────────────────────┘
                                                        │
                                                        ▼
                              ┌──────────────────────────────┐
                              │  JSD Scoring                 │
                              │  jsd_text_audio              │  [Charan]
                              │  jsd_text_video              │
                              │  jsd_audio_video             │
                              └──────────────────────────────┘
                                                        │
                                                        ▼
                              ┌──────────────────────────────┐
                              │  Classifier + Ablation       │
                              │  Leave-one-actor-out CV      │  [Aniket]
                              │  4 ablation configs          │
                              │  Logistic Regression         │
                              └──────────────────────────────┘
                                                        │
                                                        ▼
                              ┌──────────────────────────────┐
                              │  Evaluation + Plots          │  [Aaditya]
                              │  ROC, confusion, heatmap     │
                              └──────────────────────────────┘
```

---

## Work Assignment Summary

| Person      | Early Task (parallel, Day 1-3)           | Later Task (sequential, Day 4-7)     |
| ----------- | ---------------------------------------- | ------------------------------------ |
| **Harish**  | Step 1: metadata + Step 2c: text emotion | Step 3: build pairs + merge emotions |
| **Charan**  | Step 2a: audio emotion (1440 clips)      | Step 4: JSD scoring on all pairs     |
| **Aaditya** | Step 2b: video emotion (tracking CSVs)   | Step 6: evaluation metrics + plots   |
| **Aniket**  | Review + coordinate                      | Step 5: classifier + ablation study  |

---

## Execution Order and Dependencies

```
DAY 1:
  Harish ──→ Step 1: parse metadata ──→ metadata/ravdess_metadata.csv
                                              │
                                    Everyone reads this
                                              │
DAY 2-3 (ALL THREE IN PARALLEL — no dependencies on each other):
  Charan  ──→ Step 2a: audio emotion ──→ emotions/audio_emotions.csv
  Aaditya ──→ Step 2b: video emotion ──→ emotions/video_emotions.csv
  Harish  ──→ Step 2c: text emotion  ──→ emotions/text_emotions.csv
                    │         │         │
                    └─────────┴─────────┘
                              │
                    All three CSVs ready
                              │
DAY 4:                        ▼
  Harish  ──→ Step 3: build pairs ──→ pairs/incongruence_pairs.csv
                                              │
DAY 4-5:                                      ▼
  Charan  ──→ Step 4: JSD scores  ──→ scores/pair_scores.csv
                                              │
DAY 5-6:                                      ▼
  Aniket  ──→ Step 5: classifier  ──→ predictions/predictions.csv
                                      predictions/experiment_log.csv
                                              │
DAY 6-7:                                      ▼
  Aaditya ──→ Step 6: evaluate    ──→ results/evaluation_metrics.csv
                                      results/plots/*.png
```

**Who blocks whom:**

| Step         | Waits for          | Can start when                                       |
| ------------ | ------------------ | ---------------------------------------------------- |
| 2a (Charan)  | Step 1             | metadata CSV exists                                  |
| 2b (Aaditya) | Step 1             | metadata CSV exists                                  |
| 2c (Harish)  | Step 1             | metadata CSV exists (Harish does both, so immediate) |
| 3 (Harish)   | Steps 2a + 2b + 2c | All three emotion CSVs exist                         |
| 4 (Charan)   | Step 3             | pairs CSV exists                                     |
| 5 (Aniket)   | Step 4             | pair_scores CSV exists                               |
| 6 (Aaditya)  | Step 5             | predictions CSV + experiment_log exist               |

**Steps 2a, 2b, 2c are FULLY PARALLEL. No one waits for anyone else during Day 2-3.**

---

## Git Branch Strategy

```
main                ← Phase 1 code, frozen, do not push here directly
phase2/harish       ← metadata + text emotion + pairs
phase2/charan       ← audio emotion + JSD scoring
phase2/aaditya      ← video emotion + evaluation
phase2/aniket       ← classifier + ablation
```

Everyone writes to `phase2_ravdess/`. No merge conflicts because each person has their own scripts and output folders. Merge into main when everything is validated.

---

## Critical Rules (Everyone Must Follow)

### Rule 1: clip_id Is the Universal Join Key

Every CSV must have a `clip_id` column. Format:

```
{actor:02d}_{emotion_code:02d}_{intensity:02d}_{statement:02d}_{repetition:02d}
```

Example: `08_05_02_01_01` = Actor 08, Angry (05), Strong (02), Statement 1 (01), Repetition 1 (01).

This is how all files are joined. If your clip_id format is wrong, nothing downstream works.

### Rule 2: 4-Class Label Space

```
E = {happy, angry, sad, neutral}
```

Every emotion output must have exactly these columns: `p_happy, p_angry, p_sad, p_neutral`. They must sum to 1.0 per row. No exceptions.

### Rule 3: No NaN, No Duplicates

Zero NaN values. Zero duplicate clip_ids. Validate before handing off.

### Rule 4: Carry Ground Truth

Every CSV must include `emotion_code` (int, 1-8) and `emotion_4class` (string) so downstream steps can always verify against ground truth without re-joining metadata.

### Rule 5: Do Not Touch Phase 1

All files in `outputs/`, `results/`, `annotations/` are Phase 1 and frozen. Never write there.

---

## Detailed Task Specs

---

### HARISH — Steps 1, 2c, 3

#### Step 1: Parse Metadata (Day 1)

**What:** Read every .wav filename in `Datasets/ravdess/audio_speech/`, parse the 7-part identifier, build a metadata table.

**Input:** `Datasets/ravdess/audio_speech/Actor_01/*.wav` through `Actor_24/*.wav`

**Script:** `phase2_ravdess/scripts/harish_parse_metadata.py`

**Output:** `phase2_ravdess/metadata/ravdess_metadata.csv`

**Columns (exact names, exact order):**

| Column           | Type   | Example                    | Description           |
| ---------------- | ------ | -------------------------- | --------------------- |
| `clip_id`        | string | `08_05_02_01_01`           | Universal key         |
| `filename`       | string | `03-01-05-02-01-01-08.wav` | Original filename     |
| `actor`          | int    | 8                          | Actor number 1-24     |
| `gender`         | string | `female`                   | odd=male, even=female |
| `emotion_code`   | int    | 5                          | RAVDESS code 1-8      |
| `emotion_label`  | string | `angry`                    | RAVDESS name          |
| `emotion_4class` | string | `angry`                    | Our mapped label      |
| `intensity`      | int    | 2                          | 1=normal, 2=strong    |
| `statement`      | int    | 1                          | 1 or 2                |
| `repetition`     | int    | 1                          | 1 or 2                |

**Mapping logic:**

```python
EMOTION_LABEL = {1:"neutral", 2:"calm", 3:"happy", 4:"sad",
                 5:"angry", 6:"fearful", 7:"disgust", 8:"surprised"}

FOUR_CLASS    = {1:"neutral", 2:"neutral", 3:"happy", 4:"sad",
                 5:"angry", 6:"neutral", 7:"neutral", 8:"neutral"}

# clip_id construction:
clip_id = f"{actor:02d}_{emotion_code:02d}_{intensity:02d}_{statement:02d}_{repetition:02d}"
```

**Validation before handoff:**

- [ ] Exactly 1440 rows
- [ ] Every clip_id is unique
- [ ] emotion_4class is always one of: happy, angry, sad, neutral
- [ ] No NaN in any column

---

#### Step 2c: Text Emotion (Day 2)

**What:** Run `j-hartmann/emotion-english-distilroberta-base` on the two fixed sentences. Since every RAVDESS clip uses one of two sentences, we only need 2 predictions.

**Script:** `phase2_ravdess/scripts/harish_text_emotion.py`

**Output:** `phase2_ravdess/emotions/text_emotions.csv`

| Column      | Type   | Example                        |
| ----------- | ------ | ------------------------------ |
| `statement` | int    | 1                              |
| `sentence`  | string | `Kids are talking by the door` |
| `p_happy`   | float  | 0.05                           |
| `p_angry`   | float  | 0.02                           |
| `p_sad`     | float  | 0.03                           |
| `p_neutral` | float  | 0.90                           |

Exactly 2 rows. Probabilities sum to 1.0. Both sentences are neutral in content, so expect high p_neutral. This is correct.

**Use the same model and label mapping as Phase 1** (`harish/harish_segmentation_and_text.py` — copy the emotion prediction logic, remove the segmentation logic).

---

#### Step 3: Build Pairs + Merge Emotions (Day 4)

**What:** Combine all three emotion CSVs and construct synthetic congruent/incongruent pairs.

**Waits for:** Charan's `audio_emotions.csv`, Aaditya's `video_emotions.csv`, and your own `text_emotions.csv`.

**Script:** `phase2_ravdess/scripts/harish_build_pairs.py`

**Logic:**

For each combination of (actor, statement, intensity, repetition), you have multiple emotion clips. Pair them:

```python
# For actor=1, statement=1, intensity=1, repetition=1:
#   audio clips: neutral, calm, happy, sad, angry, fearful, disgust, surprised
#   video clips: neutral, calm, happy, sad, angry, fearful, disgust, surprised
#
# Congruent pairs: audio(happy) + video(happy), audio(sad) + video(sad), etc.
#   → label = 0
#
# Incongruent pairs: audio(happy) + video(angry), audio(sad) + video(happy), etc.
#   → label = 1
```

For each pair, include the emotion probabilities from all three modalities. Text probabilities are looked up by statement number from text_emotions.csv.

**Output:** `phase2_ravdess/pairs/incongruence_pairs.csv`

| Column                 | Type   | Description                                                    |
| ---------------------- | ------ | -------------------------------------------------------------- |
| `pair_id`              | string | Unique: `{actor}_{stmt}_{int}_{rep}_A{audio_emo}_V{video_emo}` |
| `actor`                | int    | Shared actor                                                   |
| `gender`               | string | male/female                                                    |
| `statement`            | int    | Shared statement                                               |
| `intensity`            | int    | Shared intensity                                               |
| `repetition`           | int    | Shared repetition                                              |
| `audio_clip_id`        | string | Audio source clip_id                                           |
| `video_clip_id`        | string | Video source clip_id                                           |
| `audio_emotion_4class` | string | Ground truth emotion of audio                                  |
| `video_emotion_4class` | string | Ground truth emotion of video                                  |
| `label`                | int    | 0=congruent, 1=incongruent                                     |
| `p_audio_happy`        | float  | Audio model prediction                                         |
| `p_audio_angry`        | float  | Audio model prediction                                         |
| `p_audio_sad`          | float  | Audio model prediction                                         |
| `p_audio_neutral`      | float  | Audio model prediction                                         |
| `p_video_happy`        | float  | Video model prediction                                         |
| `p_video_angry`        | float  | Video model prediction                                         |
| `p_video_sad`          | float  | Video model prediction                                         |
| `p_video_neutral`      | float  | Video model prediction                                         |
| `p_text_happy`         | float  | Text prediction (by statement)                                 |
| `p_text_angry`         | float  | Text prediction (by statement)                                 |
| `p_text_sad`           | float  | Text prediction (by statement)                                 |
| `p_text_neutral`       | float  | Text prediction (by statement)                                 |

**Important:** Only create pairs where BOTH the audio clip and video clip exist in their respective emotion CSVs. Join audio_emotions and video_emotions on matching (actor, statement, intensity, repetition) but allow different emotions.

**Validation before handoff:**

- [ ] Every pair_id is unique
- [ ] label = 0 when audio_emotion_4class == video_emotion_4class
- [ ] label = 1 when audio_emotion_4class != video_emotion_4class
- [ ] All 12 probability columns present, audio and video probs each sum to ~1.0
- [ ] No NaN
- [ ] Print label distribution: count of 0s and 1s

---

### CHARAN — Steps 2a, 4

#### Step 2a: Audio Emotion (Day 2-3)

**What:** Run `superb/wav2vec2-large-superb-er` on all 1440 audio-only clips.

**Input:**

- `phase2_ravdess/metadata/ravdess_metadata.csv` (from Harish Step 1)
- `Datasets/ravdess/audio_speech/Actor_XX/03-01-XX-XX-XX-XX-XX.wav`

**Script:** `phase2_ravdess/scripts/charan_audio_emotion.py`

**How to adapt from Phase 1:** Look at `aniket/aniket_audio_emotion.py`. The key change:

```python
# Phase 1 (meeting audio — complex):
#   speaker_map = build_speaker_headset_map(...)
#   clip = slice_audio(wav_path, start, end)

# Phase 2 (RAVDESS — simple):
#   clip, sr = librosa.load(wav_path, sr=16000, mono=True)
#   probs = predict_probs(model_bundle, clip)
```

No diarization. No headset mapping. No segmentation. Each .wav IS the segment.

**Model:** `superb/wav2vec2-large-superb-er` — same model ID, same label mapping as Phase 1.

The model outputs: `hap, ang, sad, neu` → map to `happy, angry, sad, neutral` (same `SUPERB_TO_FOUR` dict from Phase 1). Any extra labels fold into neutral. Renormalize to sum to 1.0.

**Output:** `phase2_ravdess/emotions/audio_emotions.csv`

| Column           | Type   | Example          |
| ---------------- | ------ | ---------------- |
| `clip_id`        | string | `08_05_02_01_01` |
| `actor`          | int    | 8                |
| `emotion_code`   | int    | 5                |
| `emotion_4class` | string | `angry`          |
| `p_happy`        | float  | 0.03             |
| `p_angry`        | float  | 0.82             |
| `p_sad`          | float  | 0.05             |
| `p_neutral`      | float  | 0.10             |

**Validation before handoff:**

- [ ] Exactly 1440 rows
- [ ] Every clip_id matches metadata CSV
- [ ] `p_happy + p_angry + p_sad + p_neutral == 1.0` for every row (tolerance 1e-4)
- [ ] No NaN
- [ ] Ground truth columns (emotion_code, emotion_4class) present

---

#### Step 4: JSD Scoring (Day 4-5)

**What:** Compute Jensen-Shannon divergence for every pair.

**Waits for:** Harish's `incongruence_pairs.csv` (Step 3).

**Input:** `phase2_ravdess/pairs/incongruence_pairs.csv`

**Script:** `phase2_ravdess/scripts/charan_jsd_scoring.py`

**How to adapt from Phase 1:** Copy the JSD function from `aniket/aniket_incongruence.py`:

```python
def jsd(p, q, eps=1e-10):
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    m = 0.5 * (p + q)
    kl_pm = np.sum(p * np.log(p / m))
    kl_qm = np.sum(q * np.log(q / m))
    return float(0.5 * kl_pm + 0.5 * kl_qm) / np.log(2)  # normalized to [0,1]
```

For each pair, compute:

| Column            | Formula               |
| ----------------- | --------------------- |
| `jsd_text_audio`  | jsd(P_text, P_audio)  |
| `jsd_text_video`  | jsd(P_text, P_video)  |
| `jsd_audio_video` | jsd(P_audio, P_video) |
| `jsd_composite`   | mean of all three     |

**Output:** `phase2_ravdess/scores/pair_scores.csv`

This is the pairs CSV with 4 extra columns appended: `jsd_text_audio, jsd_text_video, jsd_audio_video, jsd_composite`.

**Validation before handoff:**

- [ ] Same number of rows as pairs CSV
- [ ] All JSD values in [0.0, 1.0]
- [ ] No NaN
- [ ] pair_id column matches exactly

---

### AADITYA — Steps 2b, 6

#### Step 2b: Video Emotion (Day 2-3)

**What:** Map pre-extracted OpenFace AU features to emotion probabilities.

**Input:**

- `phase2_ravdess/metadata/ravdess_metadata.csv` (from Harish Step 1)
- `Datasets/ravdess/facial_tracking/Actor_XX/*.csv` (pre-extracted OpenFace output)

**Script:** `phase2_ravdess/scripts/aaditya_video_emotion.py`

**You do NOT need py-feat or OpenFace.** The tracking CSVs already contain frame-level AU intensities from OpenFace 2.1.0.

**Important — filename matching:** Audio file `03-01-05-02-01-01-08.wav` and tracking CSV `01-01-05-02-01-01-08.csv` share the same clip_id because everything except the modality prefix (03 vs 01) is identical. Use the metadata CSV to build clip_ids, then find the corresponding tracking CSV.

Tracking CSV filenames use modality=01 (AV). So for clip_id `08_05_02_01_01`, the tracking file is `01-01-05-02-01-01-08.csv` in `Actor_08/`.

**OpenFace tracking CSV columns to use:**

| Our name | OpenFace column | Description                      |
| -------- | --------------- | -------------------------------- |
| AU01     | `AU01_r`        | Inner brow raise (intensity 0-5) |
| AU02     | `AU02_r`        | Outer brow raise                 |
| AU04     | `AU04_r`        | Brow lowerer                     |
| AU06     | `AU06_r`        | Cheek raiser                     |
| AU12     | `AU12_r`        | Lip corner puller                |
| AU15     | `AU15_r`        | Lip corner depressor             |
| AU17     | `AU17_r`        | Chin raiser                      |
| AU25     | `AU25_r`        | Lips part                        |

**Processing per clip:**

1. Load the tracking CSV
2. Filter rows where `confidence > 0.5` and `success == 1` (OpenFace quality columns)
3. Mean-pool AU_r columns across all valid frames
4. Apply FACS emotion mapping (same as Phase 1 `scripts/run_video_emotion.py`):

```python
happy_score   = 0.5 * AU06_r + 0.5 * AU12_r
angry_score   = 0.4 * AU04_r + 0.3 * AU17_r + 0.3 * AU25_r
sad_score     = 0.4 * AU01_r + 0.3 * AU04_r + 0.3 * AU15_r
neutral_score = max(0.0, 1.0 - 0.15 * (AU01+AU02+AU04+AU06+AU12+AU15+AU17+AU25))
probs = softmax([happy_score, angry_score, sad_score, neutral_score])
```

If a tracking CSV is missing or has no valid frames, use uniform `[0.25, 0.25, 0.25, 0.25]` and log the clip_id.

**Output:** `phase2_ravdess/emotions/video_emotions.csv`

| Column           | Type   | Example          |
| ---------------- | ------ | ---------------- |
| `clip_id`        | string | `08_05_02_01_01` |
| `actor`          | int    | 8                |
| `emotion_code`   | int    | 5                |
| `emotion_4class` | string | `angry`          |
| `p_happy`        | float  | 0.10             |
| `p_angry`        | float  | 0.45             |
| `p_sad`          | float  | 0.15             |
| `p_neutral`      | float  | 0.30             |

**Validation before handoff:**

- [ ] One row per clip_id that exists in metadata
- [ ] clip_ids match metadata exactly
- [ ] Probabilities sum to 1.0 per row
- [ ] No NaN
- [ ] Log how many clips used uniform fallback

---

#### Step 6: Evaluation + Plots (Day 6-7)

**What:** Compute final metrics from Aniket's predictions and generate all plots.

**Waits for:** Aniket's `predictions.csv` and `experiment_log.csv` (Step 5).

**Input:**

- `phase2_ravdess/predictions/predictions.csv`
- `phase2_ravdess/predictions/experiment_log.csv`
- `phase2_ravdess/scores/pair_scores.csv` (for score distributions)

**Script:** `phase2_ravdess/scripts/aaditya_evaluate_and_plot.py`

**Metrics to compute per ablation config:**

- ROC-AUC (primary)
- Precision / Recall / F1 at optimal threshold (Youden's J)
- Confusion matrix

**Plots to generate** (save to `phase2_ravdess/results/plots/`):

| Filename                   | Description                                                      |
| -------------------------- | ---------------------------------------------------------------- |
| `roc_curves.png`           | Overlaid ROC curves for configs A-D, with AUC in legend          |
| `ablation_delta.png`       | Bar chart: AUC gain from each added feature group                |
| `emotion_pair_heatmap.png` | Heatmap: JSD score or accuracy by (audio_emotion, video_emotion) |
| `confusion_matrix.png`     | Binary confusion matrix for best config                          |
| `score_distribution.png`   | Violin/box plot: JSD scores for congruent vs incongruent pairs   |

**Output:** `phase2_ravdess/results/evaluation_metrics.csv`

| Column           | Type             |
| ---------------- | ---------------- |
| `config`         | string (A/B/C/D) |
| `auc_mean`       | float            |
| `auc_std`        | float            |
| `f1_mean`        | float            |
| `precision_mean` | float            |
| `recall_mean`    | float            |

**Use same matplotlib style as Phase 1** (`scripts/plot_results.py`): monospace for formulas, no icons, publication quality, JS_COLOR = "#2563EB", BASE_COLOR = "#DC2626".

---

### ANIKET — Step 5

#### Step 5: Classifier + Ablation (Day 5-6)

**What:** Train a binary classifier (congruent vs incongruent) with leave-one-actor-out cross-validation and 4 ablation configurations.

**Waits for:** Charan's `pair_scores.csv` (Step 4).

**Input:** `phase2_ravdess/scores/pair_scores.csv`

**Script:** `phase2_ravdess/scripts/aniket_classifier.py`

**Evaluation protocol:** Leave-one-actor-out CV (LOAO). 24 folds. Train on 23 actors, test on 1. This prevents any speaker-specific leakage.

```python
from sklearn.model_selection import LeaveOneGroupOut
logo = LeaveOneGroupOut()
groups = pair_scores["actor"].values
for train_idx, test_idx in logo.split(X, y, groups):
    # train on 23 actors, test on 1
```

**Ablation configurations:**

| Config | Features                                                      | Count |
| ------ | ------------------------------------------------------------- | ----- |
| A      | `jsd_audio_video` only                                        | 1     |
| B      | `jsd_text_audio`, `jsd_text_video`, `jsd_audio_video`         | 3     |
| C      | All JSD + `p_audio_*` (4) + `p_video_*` (4)                   | 11    |
| D      | All above + `p_text_*` (4) + `intensity` + `gender` (encoded) | 17    |

**Model:** Logistic regression (`sklearn.linear_model.LogisticRegression`). Normalize features with `StandardScaler` — fit on train fold only, transform test fold. Class weighting: `class_weight='balanced'` to handle any imbalance.

**Output 1:** `phase2_ravdess/predictions/predictions.csv`

| Column       | Type                                      |
| ------------ | ----------------------------------------- |
| `pair_id`    | string                                    |
| `label`      | int (ground truth)                        |
| `pred_prob`  | float (predicted probability for class 1) |
| `pred_label` | int (predicted at threshold 0.5)          |
| `config`     | string (A/B/C/D)                          |
| `fold_actor` | int (which actor was held out)            |

**Output 2:** `phase2_ravdess/predictions/experiment_log.csv`

| Column           | Type   |
| ---------------- | ------ |
| `config`         | string |
| `auc_mean`       | float  |
| `auc_std`        | float  |
| `f1_mean`        | float  |
| `precision_mean` | float  |
| `recall_mean`    | float  |

**Validation before handoff:**

- [ ] predictions.csv has rows for all pairs x 4 configs x 24 folds
- [ ] experiment_log.csv has exactly 4 rows (configs A-D)
- [ ] All metrics are reasonable (AUC > 0.5 at minimum)
- [ ] No NaN
- [ ] pred_prob values in [0.0, 1.0]

---

## Environment

Same venv as Phase 1. No new packages needed.

```bash
cd "CSCI 535 MPL/Implementation"
source venv/bin/activate
# torch, transformers, librosa, pandas, numpy, sklearn all available
```

Aaditya does NOT need `venv_video` or py-feat. The tracking CSVs are plain pandas-readable CSV files.

---

## Validation Checklist (Before Merging to Main)

| Check                                                       | Who verifies |
| ----------------------------------------------------------- | ------------ |
| metadata has 1440 rows, all clip_ids unique                 | Harish       |
| audio_emotions has 1440 rows, probs sum to 1.0              | Charan       |
| video_emotions has rows matching metadata, probs sum to 1.0 | Aaditya      |
| text_emotions has 2 rows, probs sum to 1.0                  | Harish       |
| incongruence_pairs label=0 iff audio_emo==video_emo         | Harish       |
| pair_scores JSD values in [0,1], no NaN                     | Charan       |
| predictions.csv has all 4 configs, 24 folds each            | Aniket       |
| evaluation_metrics.csv has 4 rows                           | Aaditya      |
| All plots render correctly                                  | Aaditya      |

---

## Final Report Narrative

1. **Phase 1 (AMI):** Proved JS divergence detects text-audio mismatch in real meetings. AUC 0.635 vs baseline 0.538. Limited by class imbalance (10% incongruent) and noisy ground truth (alpha 0.622).

2. **Phase 2 (RAVDESS):** Rigorous controlled validation. Thousands of labeled pairs, leave-one-actor-out 24-fold CV, per-emotion-pair analysis. Demonstrates which types of cross-modal mismatch are detectable and which are not.

---

## Citation

```
Livingstone SR, Russo FA (2018) The Ryerson Audio-Visual Database of Emotional
Speech and Song (RAVDESS). PLoS ONE 13(5): e0196391.
https://doi.org/10.1371/journal.pone.0196391

Swanson, Livingstone, & Russo (2019). RAVDESS Facial Landmark Tracking (v1.0.0).
Zenodo. http://doi.org/10.5281/zenodo.3255102
```
