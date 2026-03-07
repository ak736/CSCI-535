# Phase 1 Progress Report — Text vs Audio Incongruence Scorer

> For task assignments and step-by-step commands, see [README_TASKS.md](./Readme_Tasks.md)

---

## Project Goal

Detect when a speaker's words emotionally disagree with their vocal tone — **segment-level Text vs Audio Incongruence Scoring** over the AMI Meeting Corpus.

**Core idea:**

> Transcript: _"That's fine."_ + Tone: **Angry** → High incongruence score

---

## Dataset

**AMI Meeting Corpus** — 12 meeting recordings across 3 scenarios

| Series | Meetings   | Description |
| ------ | ---------- | ----------- |
| ES2002 | a, b, c, d | Scenario 1  |
| ES2003 | a, b, c, d | Scenario 2  |
| ES2004 | a, b, c, d | Scenario 3  |

Audio format: `ESXXXX.Mix-Headset.wav` + individual `ESXXXX.Headset-0/1/2/3.wav` per speaker.

---

## Pipeline Overview

```
Raw Audio (.wav)
      ↓
Step 1 · Speaker Diarization       (Charan)    — who spoke, when
      ↓
Step 2 · WhisperX Transcription    (Aaditya)   — what was said, when
      ↓
Step 3 · Merge Diarization+Transcript (Aniket) — who said what, when
      ↓
Step 4 · Fixed-Length Segmentation (Harish)    — 3s windows, 1.5s hop
      ↓
Step 5 · Text Emotion              (Harish)    — P_text = [p_happy, p_angry, p_sad, p_neutral]
      ↓
Step 6 · Audio Emotion             (Aniket)    — P_audio = [p_happy, p_angry, p_sad, p_neutral]
      ↓
Step 7 · JS Divergence Scoring     (Aniket)    — final_score = JS(P_text, P_audio) × asr_confidence
      ↓
outputs/incongruence/ESXXXX_scores.csv
```

---

## Implementation Status

| Step | Task                           | Owner   | Status      | Script                                   |
| ---- | ------------------------------ | ------- | ----------- | ---------------------------------------- |
| 1    | Speaker Diarization            | Charan  | ✅ Complete | `scripts/run_diarization.sh`             |
| 2    | WhisperX Transcription         | Aaditya | ✅ Complete | `scripts/run_whisperx.sh`                |
| 3    | Merge Diarization + Transcript | Aniket  | ✅ Complete | `scripts/run_merge.sh`                   |
| 4    | Fixed-Length Segmentation      | Harish  | ✅ Complete | `harish/harish_segmentation_and_text.py` |
| 5    | Text Emotion Prediction        | Harish  | ✅ Complete | `harish/harish_segmentation_and_text.py` |
| 6    | Audio Emotion Prediction       | Aniket  | ✅ Complete | `scripts/run_audio_emotion.sh`           |
| 7    | Incongruence Scoring           | Aniket  | ✅ Complete | `scripts/run_incongruence.sh`            |

**All 7 steps complete. Full pipeline runs end-to-end on all 12 meetings.**

---

## Step Details

### Step 1 — Speaker Diarization (Charan)

- **Tool:** pyannote.audio
- **Output:** `outputs/diarization/ESXXXX_diarization.csv`
- **Columns:** `segment_id, start, end, speaker_id`
- **Coverage:** 12/12 meetings

### Step 2 — Transcription (Aaditya)

- **Tool:** WhisperX (`large-v2` model, Wav2Vec2 forced alignment)
- **Output:** `outputs/transcripts/ESXXXX_transcript.csv`
- **Columns:** `segment_id, start, end, transcript, asr_confidence`
- **Coverage:** 12/12 meetings

### Step 3 — Merge (Aniket)

- **Logic:** For each transcript segment, assigns `speaker_id` from the diarization segment with maximum time overlap. Falls back to nearest midpoint on gaps.
- **Output:** `outputs/merged/ESXXXX_merged.csv`
- **Columns:** `segment_id, start, end, speaker_id, transcript, asr_confidence`
- **Coverage:** 12/12 meetings

### Step 4 — Segmentation (Harish)

- **Logic:** 3s sliding window, 1.5s hop. Filters: drop < 1.5s, < 3 words, multi-speaker windows.
- **Output:** `outputs/segments/ESXXXX_segments.csv`
- **Columns:** `segment_id, start, end, speaker_id, transcript, asr_confidence`
- **Coverage:** 12/12 meetings

### Step 5 — Text Emotion (Harish)

- **Model:** `j-hartmann/emotion-english-distilroberta-base`
- **Label mapping:** joy→happy, anger→angry, sadness→sad, neutral/disgust/fear/surprise→neutral → renormalize
- **Output:** `outputs/embeddings/ESXXXX_text_emotions.csv`
- **Columns:** `segment_id, p_happy, p_angry, p_sad, p_neutral`
- **Coverage:** 12/12 meetings

### Step 6 — Audio Emotion (Aniket)

- **Model:** `superb/wav2vec2-large-superb-er`
  - Outputs `hap, ang, sad, neu` — direct 1:1 map to our label space, no remapping needed
  - Trained on SUPERB Speech Emotion Recognition benchmark
- **Speaker → Headset mapping (energy-based):** For each `speaker_id`, computes mean RMS energy across all 4 headset files over diarization segments. Assigns speaker to the headset where they are loudest. Individual headsets capture own speaker at significantly higher energy than distant voices.
- **Output:** `outputs/embeddings/ESXXXX_audio_emotions.csv`
- **Columns:** `segment_id, p_happy, p_angry, p_sad, p_neutral`
- **Coverage:** 12/12 meetings

### Step 7 — Incongruence Scoring (Aniket)

- **Method:** Jensen–Shannon Divergence (normalized to [0,1])

```
M = 0.5 × (P_text + P_audio)
JS(P_text, P_audio) = 0.5 × KL(P_text ‖ M) + 0.5 × KL(P_audio ‖ M)
normalized JS = JS / log(2)

confidence_weight = asr_confidence
final_score = normalized_JS × confidence_weight
```

- **Output:** `outputs/incongruence/ESXXXX_scores.csv`
- **Columns:** `segment_id, start, end, speaker_id, transcript, incongruence_score, confidence_weight, final_score`
- **Coverage:** 12/12 meetings

---

## Results

### Per-Meeting Incongruence Scores

| Meeting   | Segments   | Mean Score | Max Score  |
| --------- | ---------- | ---------- | ---------- |
| ES2002a   | 496        | 0.1678     | 0.7715     |
| ES2002b   | 1,151      | 0.1723     | 0.7058     |
| ES2002c   | 1,281      | 0.1319     | 0.6973     |
| ES2002d   | 1,178      | 0.1454     | 0.7764     |
| ES2003a   | 435        | 0.1461     | 0.6503     |
| ES2003b   | 1,189      | 0.1230     | 0.6723     |
| ES2003c   | 1,269      | 0.1505     | 0.7507     |
| ES2003d   | 1,208      | 0.1635     | 0.7640     |
| ES2004a   | 460        | 0.2321     | 0.7467     |
| ES2004b   | 1,216      | 0.1926     | 0.7797     |
| ES2004c   | 1,203      | 0.1846     | 0.7429     |
| ES2004d   | 990        | 0.1511     | 0.7425     |
| **Total** | **11,076** | **0.1600** | **0.7797** |

### Overall Summary

| Metric                             | Value      |
| ---------------------------------- | ---------- |
| Total segments scored              | 12,076     |
| Overall mean `final_score`         | 0.1600     |
| Overall max `final_score`          | 0.7797     |
| High incongruence segments (≥ 0.5) | 638 (5.3%) |

### Top 5 Most Incongruent Segments (across all meetings)

| Speaker    | Transcript                                                       | Score     |
| ---------- | ---------------------------------------------------------------- | --------- |
| SPEAKER_03 | _"a channel. It's not when it finishes. It's not anything like"_ | **0.780** |
| SPEAKER_02 | _"can throw Homer when you're frustrated."_                      | **0.776** |
| SPEAKER_03 | _"Is that a whale? Yeah."_                                       | **0.772** |
| SPEAKER_00 | _"we had to get"_                                                | **0.764** |
| SPEAKER_01 | _"this thing, which I think will take"_                          | **0.753** |

These are short, emotionally ambiguous utterances — exactly the kind of segments expected to score high on text-audio incongruence.

---

## Output File Structure

```
outputs/
├── diarization/          ESXXXX_diarization.csv       (12 files)
├── transcripts/          ESXXXX_transcript.csv        (12 files)
├── merged/               ESXXXX_merged.csv            (12 files)
├── segments/             ESXXXX_segments.csv          (12 files)
├── embeddings/
│   ├──                   ESXXXX_text_emotions.csv     (12 files)
│   └──                   ESXXXX_audio_emotions.csv    (12 files)
└── incongruence/         ESXXXX_scores.csv            (12 files)
```

---

## Team Scripts

```
aaditya/aaditya_whisperx.py              Step 2 — WhisperX transcription logic
aniket/aniket_merge.py                   Step 3 — Diarization + transcript merge
aniket/aniket_audio_emotion.py           Step 6 — Audio emotion prediction
aniket/aniket_incongruence.py            Step 7 — JS divergence scoring
harish/harish_segmentation_and_text.py  Step 4+5 — Segmentation + text emotion
charan/charan_diarization.py             Step 1 — Speaker diarization

scripts/run_diarization.sh              Shell wrapper — Step 1
scripts/run_whisperx.sh                 Shell wrapper — Step 2
scripts/run_merge.sh                    Shell wrapper — Step 3
scripts/run_audio_emotion.sh            Shell wrapper — Step 6
scripts/run_incongruence.sh             Shell wrapper — Step 7
```

---

## Remaining Work (Phase 1 Completion)

- [ ] Human annotation of 200–300 segments (3 annotators, majority vote)
- [ ] Krippendorff's alpha inter-rater agreement
- [ ] Baseline comparison: JS Divergence vs Valence Difference
- [ ] Evaluation metrics: ROC-AUC, Precision, Recall, F1 (target AUC ≥ 0.75)
- [ ] `results/evaluation_metrics.csv`
- [ ] `results/experiment_log.csv` (model versions, seeds, hyperparameters)
- [ ] Short report summarizing results

---

## Environment

```bash
cd "CSCI 535 MPL/Implementation"
python3 -m venv venv
source venv/bin/activate
pip install whisperx torch torchaudio transformers librosa numpy pandas
```

> Always activate the venv before running any script. The bash wrappers in `scripts/` do this automatically.
