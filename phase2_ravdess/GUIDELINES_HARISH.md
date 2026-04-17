# Phase 2 Guidelines — Harish

You own **3 tasks**: metadata parsing (Step 1), text emotion (Step 2c), and pair construction (Step 3).

You read ONLY from: `phase2_ravdess/metadata/`, `phase2_ravdess/emotions/`, `Datasets/ravdess/audio_speech/`.
You write ONLY to: `phase2_ravdess/metadata/`, `phase2_ravdess/emotions/text_emotions.csv`, `phase2_ravdess/pairs/`.

---

## Execution Order

| Day | Step                | Waits for                                | Script                      |
| --- | ------------------- | ---------------------------------------- | --------------------------- |
| 1   | Step 1 — metadata   | dataset download complete                | `harish_parse_metadata.py`  |
| 2   | Step 2c — text emo  | Step 1 done (no strict dep, but in order)| `harish_text_emotion.py`    |
| 4   | Step 3 — pairs      | Charan's audio + Aaditya's video + yours | `harish_build_pairs.py`     |

**CRITICAL: Do NOT start Step 3 until all three emotion CSVs exist.**
  - `phase2_ravdess/emotions/audio_emotions.csv`  (Charan)
  - `phase2_ravdess/emotions/video_emotions.csv`  (Aaditya)
  - `phase2_ravdess/emotions/text_emotions.csv`   (you)

---

## Step 1 — Parse Metadata

### Script
`phase2_ravdess/scripts/harish_parse_metadata.py`

### Input
`Datasets/ravdess/audio_speech/Actor_01/*.wav` through `Actor_24/*.wav` — **1440 .wav files total**.

Filename format (7 fields, hyphen-separated):
```
MODALITY-CHANNEL-EMOTION-INTENSITY-STATEMENT-REPETITION-ACTOR.wav
   03   -  01   -  05   -   02    -    01   -    01    -  08  .wav
```
Example: `03-01-05-02-01-01-08.wav` = Audio-only (03), Speech (01), Angry (05), Strong (02), Statement 1, Repetition 1, Actor 08.

### Output
`phase2_ravdess/metadata/ravdess_metadata.csv` — **1440 rows, 10 columns**:

| Column           | Type   | Example                    | Notes                         |
| ---------------- | ------ | -------------------------- | ----------------------------- |
| `clip_id`        | string | `08_05_02_01_01`           | Universal join key            |
| `filename`       | string | `03-01-05-02-01-01-08.wav` | Original basename             |
| `actor`          | int    | 8                          | 1–24                          |
| `gender`         | string | `female`                   | odd actor = male, even = female |
| `emotion_code`   | int    | 5                          | 1–8                           |
| `emotion_label`  | string | `angry`                    | RAVDESS label from dict       |
| `emotion_4class` | string | `angry`                    | Our 4-class mapping           |
| `intensity`      | int    | 2                          | 1=normal, 2=strong            |
| `statement`      | int    | 1                          | 1 or 2                        |
| `repetition`     | int    | 1                          | 1 or 2                        |

`clip_id` format (exact):
```python
f"{actor:02d}_{emotion_code:02d}_{intensity:02d}_{statement:02d}_{repetition:02d}"
```

4-class mapping:
```python
FOUR_CLASS = {1:"neutral", 2:"neutral", 3:"happy", 4:"sad",
              5:"angry", 6:"neutral", 7:"neutral", 8:"neutral"}
```

### Run
```bash
source venv/bin/activate
python3 phase2_ravdess/scripts/harish_parse_metadata.py --help
python3 phase2_ravdess/scripts/harish_parse_metadata.py \
    --audio_root Datasets/ravdess/audio_speech \
    --out        phase2_ravdess/metadata/ravdess_metadata.csv
```

### Validation checklist (before handoff)
- [ ] exactly **1440 rows**
- [ ] `clip_id` is unique across all rows
- [ ] `emotion_4class` only contains `{happy, angry, sad, neutral}`
- [ ] no NaN in any column
- [ ] column order matches the spec above

Use `--validate_only` after writing the CSV to re-check.

---

## Step 2c — Text Emotion (2 sentences only)

### Script
`phase2_ravdess/scripts/harish_text_emotion.py`

### Why only 2 rows?
Every RAVDESS clip uses one of two fixed sentences:
  - Statement 1: "Kids are talking by the door"
  - Statement 2: "Dogs are sitting by the door"

Model output depends only on text, not actor/emotion. Two predictions are enough — downstream, Step 3 looks them up by `statement`.

### Model
`j-hartmann/emotion-english-distilroberta-base` — **copy emotion-prediction logic from Phase 1**: [phase1/harish/harish_segmentation_and_text.py](../phase1/harish/harish_segmentation_and_text.py) (`run_part2_embeddings`). Use the exact `LABEL_TO_FOUR` mapping (7 model labels → 4-class).

### Output
`phase2_ravdess/emotions/text_emotions.csv` — **exactly 2 rows**:

| Column      | Type   | Example                      |
| ----------- | ------ | ---------------------------- |
| `statement` | int    | 1                            |
| `sentence`  | string | `Kids are talking by the door` |
| `p_happy`   | float  | 0.05                         |
| `p_angry`   | float  | 0.02                         |
| `p_sad`     | float  | 0.03                         |
| `p_neutral` | float  | 0.90                         |

Both sentences are neutral content, so expect high `p_neutral` — that is correct.

### Run
```bash
python3 phase2_ravdess/scripts/harish_text_emotion.py \
    --out phase2_ravdess/emotions/text_emotions.csv
```

### Validation
- [ ] exactly 2 rows
- [ ] `p_happy + p_angry + p_sad + p_neutral == 1.0` per row (tol 1e-4)
- [ ] no NaN

---

## Step 3 — Build Pairs (GATED on Charan + Aaditya)

### Script
`phase2_ravdess/scripts/harish_build_pairs.py`

### When to start
ONLY after ALL three CSVs exist:
```bash
ls -la phase2_ravdess/emotions/audio_emotions.csv   # Charan
ls -la phase2_ravdess/emotions/video_emotions.csv   # Aaditya
ls -la phase2_ravdess/emotions/text_emotions.csv    # You
```

### Logic
For each (actor, statement, intensity, repetition), there are multiple emotion clips. Construct all (audio_emo × video_emo) pairs:
  - `audio(happy) + video(happy)` → label=0 (congruent)
  - `audio(happy) + video(angry)` → label=1 (incongruent)

Attach the emotion probabilities from all three modalities. Text probs come from `text_emotions.csv` keyed by `statement`.

### Output
`phase2_ravdess/pairs/incongruence_pairs.csv` — columns in this exact order (23):
```
pair_id, actor, gender, statement, intensity, repetition,
audio_clip_id, video_clip_id,
audio_emotion_4class, video_emotion_4class, label,
p_audio_happy, p_audio_angry, p_audio_sad, p_audio_neutral,
p_video_happy, p_video_angry, p_video_sad, p_video_neutral,
p_text_happy,  p_text_angry,  p_text_sad,  p_text_neutral
```

`pair_id` format: `{actor:02d}_{stmt}_{int}_{rep}_A{audio_emo}_V{video_emo}` — must be unique.

### Run
```bash
python3 phase2_ravdess/scripts/harish_build_pairs.py \
    --metadata phase2_ravdess/metadata/ravdess_metadata.csv \
    --audio    phase2_ravdess/emotions/audio_emotions.csv \
    --video    phase2_ravdess/emotions/video_emotions.csv \
    --text     phase2_ravdess/emotions/text_emotions.csv \
    --out      phase2_ravdess/pairs/incongruence_pairs.csv
```

### Validation
- [ ] `pair_id` unique
- [ ] `label == 0` iff `audio_emotion_4class == video_emotion_4class`
- [ ] audio/video/text probability triples each sum to ~1.0
- [ ] no NaN
- [ ] print label distribution (counts of 0 and 1)

---

## Before Pushing

```bash
# Syntax check
python3 -c "import ast; ast.parse(open('phase2_ravdess/scripts/harish_parse_metadata.py').read())"
python3 -c "import ast; ast.parse(open('phase2_ravdess/scripts/harish_text_emotion.py').read())"
python3 -c "import ast; ast.parse(open('phase2_ravdess/scripts/harish_build_pairs.py').read())"

# Re-validate outputs
python3 phase2_ravdess/scripts/harish_parse_metadata.py --validate_only
python3 phase2_ravdess/scripts/harish_text_emotion.py   --validate_only
python3 phase2_ravdess/scripts/harish_build_pairs.py    --validate_only
```

Branch: `phase2/harish`.
