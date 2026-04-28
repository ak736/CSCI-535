# Beyond Fusion: Text vs Audio Incongruence (Phase 2)

**CSCI 535 — Multimodal Probabilistic Learning | University of Southern California**

Controlled evaluation of audio-video incongruence detection on the RAVDESS dataset using synthetic congruent/incongruent pairs, JSD scoring, and a leave-one-actor-out classifier ablation.

---

## Team

| Name | Role |
|------|------|
| Harish Dukkipati | Metadata parsing, text emotion, pair construction |
| Charan Kumar D. | Audio emotion, JSD scoring |
| Aaditya Patil | Video emotion (OpenFace AUs), evaluation + plots |
| Aniket Kumar | Classifier, temperature calibration, ablation, improvements |

---

## Dataset

**RAVDESS** — Ryerson Audio-Visual Database of Emotional Speech and Song
- 24 actors (12 male, 12 female)
- 8 emotions: neutral, calm, happy, sad, angry, fearful, disgust, surprised
- 1,440 audio-only clips (Actor_01 – Actor_24, 60 clips each)
- Pre-extracted OpenFace facial tracking CSVs (frame-level AU intensities)

Located at: `Datasets/ravdess/audio_speech/` and `Datasets/ravdess/facial_tracking/`

---

## Pipeline

```
RAVDESS Audio + Facial Tracking
    ↓
[Step 1] Metadata Parsing         (harish_parse_metadata.py)
    ↓
[Step 2a] Audio Emotion           (charan_audio_emotion.py)
[Step 2b] Video Emotion           (aaditya_video_emotion.py)
[Step 2c] Text Emotion            (harish_text_emotion.py)
    ↓
[Step 3] Pair Construction        (harish_build_pairs.py)
    ↓
[Step 4] JSD Scoring              (charan_jsd_scoring.py)
    ↓
[Step 5] Classifier + Ablation    (aniket_classifier.py)
    ↓
[Step 6] Evaluation + Plots       (aaditya_evaluate_and_plot.py)
    ↓
phase2_ravdess/results/
```

---

## Scripts

| Step | Script | Output |
|------|--------|--------|
| 1 | `scripts/harish_parse_metadata.py` | `metadata/ravdess_metadata.csv` |
| 2a | `scripts/charan_audio_emotion.py` | `emotions/audio_emotions.csv` |
| 2b | `scripts/aaditya_video_emotion.py` | `emotions/video_emotions.csv` |
| 2c | `scripts/harish_text_emotion.py` | `emotions/text_emotions.csv` |
| 3 | `scripts/harish_build_pairs.py` | `pairs/incongruence_pairs.csv` |
| 4 | `scripts/charan_jsd_scoring.py` | `scores/pair_scores.csv` |
| 5 | `scripts/aniket_classifier.py` | `predictions/predictions.csv`, `predictions/experiment_log.csv` |
| 5+ | `scripts/aniket_improvements.py` | `predictions/predictions_improved.csv`, `predictions/experiment_log_improved.csv` |
| 6 | `scripts/aaditya_evaluate_and_plot.py` | `results/evaluation_metrics_improved.csv`, `results/plots/` |

---

## Technical Decisions

**Emotion label space:** `E = {happy, angry, sad, neutral}` — 4-class mapping applied to all modalities.

**RAVDESS → 4-class mapping:**
- neutral (1), calm (2), fearful (6), disgust (7), surprised (8) → `neutral`
- happy (3) → `happy`
- sad (4) → `sad`
- angry (5) → `angry`

**Pair construction:** Intra-actor, cross-emotion. For each group of (actor, statement, intensity, repetition), cross-join all emotion clips from the same actor. Congruent pairs have matching audio and video emotion (label=0); incongruent pairs differ (label=1). Total: **10,848 pairs**.

**Text emotion:** `j-hartmann/emotion-english-distilroberta-base` on RAVDESS transcripts. 7 raw labels collapsed to 4 (disgust/fear/surprise → neutral). Note: all RAVDESS clips use 2 fixed statements, so text emotion varies minimally.

**Audio emotion:** `superb/wav2vec2-large-superb-er` — SUPERB labels hap/ang/sad/neu map 1:1 to our 4-class space. Loaded via `AutoFeatureExtractor + AutoModelForAudioClassification`.

**Video emotion:** OpenFace AU intensities from pre-extracted tracking CSVs. FACS-based weighted rules:
```
happy   = 0.5 × AU06 + 0.5 × AU12
angry   = 0.4 × AU04 + 0.3 × AU17 + 0.3 × AU25
sad     = 0.4 × AU01 + 0.3 × AU04 + 0.3 × AU15
neutral = max(0.0, 1.0 − 0.15 × total_activation)
probs   = softmax([happy, angry, sad, neutral])
```

**JSD scoring:**
```
M = 0.5 × (P + Q)
JS(P, Q) = 0.5 × KL(P ‖ M) + 0.5 × KL(Q ‖ M)
normalized to [0, 1] by dividing by log(2)
```
Three JSD scores computed per pair: jsd_audio_video, jsd_text_audio, jsd_text_video. Composite = 0.5 × jsd_audio_video + 0.25 × jsd_text_audio + 0.25 × jsd_text_video.

**Temperature calibration:** Audio and video probability vectors softened with T=2.5 before JSD computation. Improves AUC by +0.006 vs T=1.0.

**Classifier:** sklearn `LogisticRegression(class_weight='balanced')`. `StandardScaler` fit only on training fold. Cross-validation: leave-one-actor-out (LOAO), 24 folds.

---

## Ablation Configurations

| Config | Features | # Features |
|--------|----------|-----------|
| A | jsd_audio_video | 1 |
| B | 3 JSD scores (audio-video, text-audio, text-video) | 3 |
| C | B + p_audio_* (4) + p_video_* (4) | 11 |
| D | C + p_text_* (4) + intensity + gender | 17 |
| E | C + per-emotion abs diffs (4) + jsd_max | 16 |

---

## Results

**10,848 synthetic pairs** (from 1,440 RAVDESS clips across 24 actors).

| Config | AUC | F1 | Precision | Recall |
|--------|-----|----|-----------|--------|
| A | 0.554 | 0.517 | 0.651 | 0.448 |
| B | 0.577 | 0.551 | 0.641 | 0.498 |
| C | 0.618 | 0.570 | 0.696 | 0.513 |
| D | 0.617 | 0.564 | 0.694 | 0.507 |
| **E** | **0.621** | 0.554 | **0.704** | 0.487 |

Best configuration: **E** (AUC 0.621, Precision 70.4%, Recall 48.7%)

### Diagnostic findings

- **Happy-bias:** Classifier over-predicts congruent for happy pairs (RAVDESS actors perform happy with consistent arousal → peaked probs → low JSD). Dropping happy emotion from evaluation raises AUC to **0.652**.
- **Video modality drop:** Removing video probs from Config C2 drops AUC from 0.618 → 0.561 (−0.057), confirming video emotion contributes meaningfully.
- **Balanced subset (100 pairs/cell):** AUC slightly lower (0.562) due to reduced dataset size, not a model failure.

---

## Running the Pipeline

```bash
cd "CSCI 535 MPL/Implementation"
source venv/bin/activate

# Step 1 — metadata
python3 phase2_ravdess/scripts/harish_parse_metadata.py \
    --audio_root Datasets/ravdess/audio_speech \
    --out        phase2_ravdess/metadata/ravdess_metadata.csv

# Step 2a — audio emotion
python3 phase2_ravdess/scripts/charan_audio_emotion.py \
    --metadata phase2_ravdess/metadata/ravdess_metadata.csv \
    --audio_root Datasets/ravdess/audio_speech \
    --out      phase2_ravdess/emotions/audio_emotions.csv

# Step 2b — video emotion
python3 phase2_ravdess/scripts/aaditya_video_emotion.py \
    --metadata     phase2_ravdess/metadata/ravdess_metadata.csv \
    --tracking_root Datasets/ravdess/facial_tracking \
    --out          phase2_ravdess/emotions/video_emotions.csv

# Step 2c — text emotion
python3 phase2_ravdess/scripts/harish_text_emotion.py \
    --metadata phase2_ravdess/metadata/ravdess_metadata.csv \
    --out      phase2_ravdess/emotions/text_emotions.csv

# Step 3 — pair construction
python3 phase2_ravdess/scripts/harish_build_pairs.py \
    --metadata phase2_ravdess/metadata/ravdess_metadata.csv \
    --audio    phase2_ravdess/emotions/audio_emotions.csv \
    --video    phase2_ravdess/emotions/video_emotions.csv \
    --text     phase2_ravdess/emotions/text_emotions.csv \
    --out      phase2_ravdess/pairs/incongruence_pairs.csv

# Step 4 — JSD scoring
python3 phase2_ravdess/scripts/charan_jsd_scoring.py \
    --pairs phase2_ravdess/pairs/incongruence_pairs.csv \
    --out   phase2_ravdess/scores/pair_scores.csv

# Step 5 — classifier + ablation
python3 phase2_ravdess/scripts/aniket_classifier.py \
    --scores          phase2_ravdess/scores/pair_scores.csv \
    --out_predictions phase2_ravdess/predictions/predictions.csv \
    --out_experiment  phase2_ravdess/predictions/experiment_log.csv

# Step 5+ — improvements (temperature calibration, Config E)
python3 phase2_ravdess/scripts/aniket_improvements.py --T 2.5 --n_per_cell 100

# Step 6 — evaluation + plots
python3 phase2_ravdess/scripts/aaditya_evaluate_and_plot.py \
    --predictions    phase2_ravdess/predictions/predictions_improved.csv \
    --experiment_log phase2_ravdess/predictions/experiment_log_improved.csv \
    --pair_scores    phase2_ravdess/scores/pair_scores_calibrated.csv \
    --out_metrics    phase2_ravdess/results/evaluation_metrics_improved.csv \
    --plots_dir      phase2_ravdess/results/plots
```

---

## Environment

```bash
source venv/bin/activate   # always activate before running anything
```

- Python 3.11, torch 2.8.0, transformers 4.57.6, librosa, pandas, numpy, soundfile, scikit-learn, seaborn, matplotlib
