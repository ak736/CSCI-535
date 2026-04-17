# Phase 2 Guidelines — Charan

You own **2 tasks**: audio emotion on 1440 clips (Step 2a) and JSD scoring (Step 4).

You read ONLY from: `phase2_ravdess/metadata/`, `phase2_ravdess/pairs/`, `Datasets/ravdess/audio_speech/`.
You write ONLY to: `phase2_ravdess/emotions/audio_emotions.csv`, `phase2_ravdess/scores/`.

---

## Execution Order

| Day | Step                | Waits for                         | Script                      |
| --- | ------------------- | --------------------------------- | --------------------------- |
| 2-3 | Step 2a — audio emo | Harish Step 1 (metadata.csv)      | `charan_audio_emotion.py`   |
| 4-5 | Step 4 — JSD scores | Harish Step 3 (incongruence pairs) | `charan_jsd_scoring.py`     |

---

## Step 2a — Audio Emotion (1440 clips)

### Script
`phase2_ravdess/scripts/charan_audio_emotion.py`

### Input
- `phase2_ravdess/metadata/ravdess_metadata.csv` (from Harish Step 1 — **must exist first**)
- `Datasets/ravdess/audio_speech/Actor_XX/*.wav` — 1440 audio-only files

### Model — COPY from Phase 1
`superb/wav2vec2-large-superb-er` (same model, same label mapping as Phase 1).

Reference file: [phase1/aniket/aniket_audio_emotion.py](../phase1/aniket/aniket_audio_emotion.py)
- `load_model()` — verbatim copy already in your stub script
- `predict_probs()` — verbatim copy already in your stub script
- `SUPERB_TO_FOUR` dict (hap/ang/sad/neu → happy/angry/sad/neutral) — already copied

### Simplification vs Phase 1
Phase 1 was complicated because of meeting audio:
```python
# Phase 1 (meeting audio):
speaker_map = build_speaker_headset_map(...)  # energy-based speaker→headset
clip = slice_audio(wav_path, start, end)       # slice diarization segment
```

**Phase 2 is simple — no diarization, no headset mapping, no slicing:**
```python
# Phase 2 (RAVDESS):
audio, _ = librosa.load(wav_path, sr=16000, mono=True)   # the whole .wav IS the clip
probs = predict_probs(model_bundle, audio)
```

Each .wav is already a single-actor, single-emotion 3-5s clip.

### Output
`phase2_ravdess/emotions/audio_emotions.csv` — **1440 rows**:

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

### Run
```bash
source venv/bin/activate
python3 phase2_ravdess/scripts/charan_audio_emotion.py --help
python3 phase2_ravdess/scripts/charan_audio_emotion.py \
    --metadata   phase2_ravdess/metadata/ravdess_metadata.csv \
    --audio_root Datasets/ravdess/audio_speech \
    --out        phase2_ravdess/emotions/audio_emotions.csv
```

### Validation checklist
- [ ] **exactly 1440 rows**
- [ ] every `clip_id` matches the metadata CSV
- [ ] `p_happy + p_angry + p_sad + p_neutral == 1.0` per row (tol 1e-4)
- [ ] no NaN
- [ ] ground-truth columns (`emotion_code`, `emotion_4class`) preserved

---

## Step 4 — JSD Scoring

### Script
`phase2_ravdess/scripts/charan_jsd_scoring.py`

### When to start
Only after Harish writes `phase2_ravdess/pairs/incongruence_pairs.csv` (Step 3).

### Input
`phase2_ravdess/pairs/incongruence_pairs.csv`

### Math — COPY from Phase 1 (already in your stub)
`kl_divergence()` and `js_divergence()` copied **verbatim** from [phase1/aniket/aniket_incongruence.py](../phase1/aniket/aniket_incongruence.py). Do not rewrite these.

JSD is already normalized to [0, 1] by dividing by `log(2)`.

Compute per pair:
```
jsd_text_audio  = JS(P_text,  P_audio)
jsd_text_video  = JS(P_text,  P_video)
jsd_audio_video = JS(P_audio, P_video)
jsd_composite   = (jsd_text_audio + jsd_text_video + jsd_audio_video) / 3
```

### Output
`phase2_ravdess/scores/pair_scores.csv` — **same rows as pairs CSV**, with 4 extra columns appended:
```
jsd_text_audio, jsd_text_video, jsd_audio_video, jsd_composite
```

### Run
```bash
python3 phase2_ravdess/scripts/charan_jsd_scoring.py \
    --pairs phase2_ravdess/pairs/incongruence_pairs.csv \
    --out   phase2_ravdess/scores/pair_scores.csv
```

### Validation checklist
- [ ] same row count as pairs CSV
- [ ] all 4 JSD values in `[0.0, 1.0]`
- [ ] `pair_id` values unchanged (one-to-one with pairs input)
- [ ] no NaN

---

## Before Pushing

```bash
# Syntax check
python3 -c "import ast; ast.parse(open('phase2_ravdess/scripts/charan_audio_emotion.py').read())"
python3 -c "import ast; ast.parse(open('phase2_ravdess/scripts/charan_jsd_scoring.py').read())"

# Re-validate outputs
python3 phase2_ravdess/scripts/charan_audio_emotion.py --validate_only
python3 phase2_ravdess/scripts/charan_jsd_scoring.py   --validate_only
```

Branch: `phase2/charan`.
