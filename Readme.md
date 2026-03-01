# Beyond Fusion: Text vs Audio Incongruence (Phase 1)

## Project Goal

Build a **segment-level Text vs Audio Incongruence Scorer** that detects when a speaker's words disagree with their vocal tone.

**Example:**

> Transcript: "That's fine."
> Tone: Angry
> → High incongruence

This is the first implementation phase of the broader Communication Instability Modeling project.

---

## Phase 1 Scope (Strict)

**We are building:**

- Speaker-aware segmentation
- Text emotion prediction
- Audio emotion prediction
- Jensen–Shannon (JS) divergence scorer
- CSV output with timestamps and scores

**We are NOT building:**

- Visual modeling
- Interaction imbalance modeling
- UI / dashboard
- Complex fusion architectures

---

## System Architecture

```
Audio File
    ↓
Speaker Diarization
    ↓
WhisperX Transcription (aligned timestamps)
    ↓
Fixed-Length Segmentation (3s window, 1.5s hop)
    ↓
Text Emotion Model → P_text       Audio Emotion Model → P_audio
    ↓                                       ↓
              Probability Calibration
                    ↓
              Confidence Weighting
                    ↓
              JS Divergence
                    ↓
          Incongruence Score
                    ↓
              CSV Output
```

---

## Emotion Label Space (Fixed)

For Phase 1, we use a **fixed, shared label space** across all modalities:

```
E = {happy, angry, sad, neutral}
```

All modalities **MUST** output a probability distribution over this exact set.

**Why this matters:**

- JS divergence requires aligned distributions — mismatched label spaces will silently produce wrong results.
- Enforcing this upfront prevents hidden mismatch errors downstream.

> ⚠️ Any pretrained model whose output space differs from `E` must be mapped or re-trained before use.

---

## Emotion Modeling

### Text Emotion Model

| Option             | Description                               |
| ------------------ | ----------------------------------------- |
| **A (fast start)** | Pretrained HuggingFace emotion classifier |
| **B**              | DistilBERT + small classifier head        |

**Output per segment:**

```
P_text = [p(happy), p(angry), p(sad), p(neutral)]
```

**Deliverable:**

```
models/text_emotion.py
```

---

### Audio Emotion Model

| Option | Description                              |
| ------ | ---------------------------------------- |
| **A**  | OpenSMILE features + logistic regression |
| **B**  | Wav2Vec2 embeddings + classifier         |

**Output per segment:**

```
P_audio = [p(happy), p(angry), p(sad), p(neutral)]
```

**Deliverable:**

```
models/audio_emotion.py
```

---

### Probability Calibration

Before computing JS divergence, both distributions must be calibrated to avoid overconfident predictions that artificially inflate divergence scores.

Apply one of:

- **Softmax temperature scaling** — smooths overconfident peaks
- **Output normalization** — re-normalize so probabilities sum to 1

```python
# Temperature scaling example (T > 1 softens, T < 1 sharpens)
P_calibrated = softmax(logits / T)
```

> ⚠️ Skipping calibration can cause the scorer to detect model overconfidence rather than genuine incongruence.

---

### Confidence Handling

Real systems degrade due to noisy ASR output or weak audio signals. Apply the following rules before scoring:

| Condition                  | Action                         |
| -------------------------- | ------------------------------ |
| ASR confidence < threshold | Downweight text modality       |
| Audio energy < threshold   | Mark segment as low-confidence |
| Both below threshold       | Skip segment entirely          |

**Weighted scoring formula:**

```
final_score = JS(P_text, P_audio) × confidence_weight
```

Where `confidence_weight ∈ [0, 1]` is derived from ASR and audio energy signals.

This prevents the scorer from detecting noise as incongruence.

---

## Implementation Steps

### Step 1: Dataset Setup

**Primary dataset:** AMI Meeting Corpus
**Backup:** IEMOCAP

**Tasks:**

- Download dataset
- Extract audio files
- Organize into `/data/raw/`

**Deliverable:**

```
data/raw/
```

---

### Step 2: Speaker Diarization

**Tool:** `pyannote.audio`

**Goal:** Identify speaker segments with timestamps.

**Output format:**

```
segment_id | start | end | speaker_id
```

**Deliverable:**

```
data/processed/diarization.csv
```

---

### Step 3: Transcription & Alignment

**Tool:** WhisperX

**Goal:** Generate word-aligned transcripts with timestamps and ASR confidence scores.

**Output format:**

```
segment_id | speaker | start | end | transcript | asr_confidence
```

**Deliverable:**

```
data/processed/transcripts.csv
```

---

### Step 4: Fixed-Length Segmentation

Create overlapping windows:

- **Window size:** 3 seconds
- **Hop size:** 1.5 seconds

**Segment Filtering Rules:**

| Rule                                      | Reason                                 |
| ----------------------------------------- | -------------------------------------- |
| Remove segments < 1.5 seconds             | Too short to extract reliable features |
| Remove segments with < 3 spoken words     | Insufficient text signal               |
| Remove multi-speaker overlapping segments | Diarization ambiguity                  |
| Keep only single-speaker windows          | Ensures clean per-speaker scoring      |

**Deliverable:**

```
data/processed/segments.csv
```

---

## Incongruence Scoring

### Jensen–Shannon Divergence (Exact Definition)

For distributions **P** (text) and **Q** (audio):

```
M = 0.5 × (P + Q)

JS(P, Q) = 0.5 × KL(P ‖ M) + 0.5 × KL(Q ‖ M)
```

Where KL divergence is computed using **natural log**.

**Range:** `[0, log(2)]` → normalized to `[0, 1]`

> JS divergence is symmetric and always finite, making it well-suited for comparing probability distributions.

---

### Scoring Pipeline

For each segment:

1. **Obtain** calibrated distributions from both models:

   ```
   P_text,  P_audio
   ```

2. **Apply** confidence weighting:

   ```
   confidence_weight = f(asr_confidence, audio_energy)
   ```

3. **Compute** JS divergence:

   ```
   JS(P_text, P_audio)
   ```

4. **Apply** confidence weight:

   ```
   final_score = JS(P_text, P_audio) × confidence_weight
   ```

5. **Save** output:
   ```
   timestamp | speaker | transcript | incongruence_score | confidence_weight
   ```

**Deliverable:**

```
models/incongruence.py
```

---

## Validation Plan

### Annotation Protocol

Annotators are presented with both the audio clip and the transcript simultaneously and asked:

> _"Do the speaker's words and tone express the same emotional intent?"_

**Labels:**

| Label | Meaning                               |
| ----- | ------------------------------------- |
| `0`   | Aligned — words and tone match        |
| `1`   | Incongruent — words and tone conflict |

**Process:**

- Sample **200–300 segments**
- **3 independent annotators** per segment
- Final label = **majority vote**
- Compute **Krippendorff's alpha** for inter-rater agreement

---

### Baseline Comparison

We compare two methods to establish whether JS divergence outperforms a simpler heuristic:

| Method                            | Description                                |
| --------------------------------- | ------------------------------------------ |
| **JS Divergence** (primary)       | Full distributional comparison over `E`    |
| **Valence Difference** (baseline) | `score = \|valence_text − valence_audio\|` |

Both methods are evaluated against human annotations. Results reported in `evaluation_metrics.csv`.

---

### Evaluation Metrics

| Metric               | Target   |
| -------------------- | -------- |
| ROC-AUC              | ≥ 0.75   |
| Precision            | Reported |
| Recall               | Reported |
| F1                   | Reported |
| Krippendorff's Alpha | Reported |

**Deliverable:**

```
results/evaluation_metrics.csv
```

---

## Reproducibility

To ensure experiments are replicable:

- Fix **random seeds** across all models and splits
- Log **model versions** and hyperparameters
- Save **all intermediate CSV outputs** at each pipeline stage
- Track experiments using **Weights & Biases** or a local CSV experiment log

```
results/
├── evaluation_metrics.csv
├── experiment_log.csv
└── annotations/
    └── annotated_segments.csv
```

---

## Repository Structure

```
project/
│
├── data/
│   ├── raw/
│   └── processed/
│       ├── diarization.csv
│       ├── transcripts.csv
│       └── segments.csv
│
├── models/
│   ├── text_emotion.py
│   ├── audio_emotion.py
│   └── incongruence.py
│
├── scripts/
│   └── run_pipeline.py
│
├── results/
│   ├── evaluation_metrics.csv
│   ├── experiment_log.csv
│   └── annotations/
│
└── README.md
```

---

## Timeline (Phase 1)

| Week       | Tasks                                                |
| ---------- | ---------------------------------------------------- |
| **Week 1** | Dataset setup, Diarization, Transcription            |
| **Week 2** | Segmentation, Text emotion model                     |
| **Week 3** | Audio emotion model, Calibration                     |
| **Week 4** | Incongruence scorer, Baseline comparison, Validation |

---

## Phase 1 Deliverables

- [ ] Working Text vs Audio Incongruence Scorer
- [ ] Fixed emotion label space `E` enforced across all models
- [ ] Calibrated probability outputs for text and audio
- [ ] Confidence-weighted scoring implemented
- [ ] 200+ human-annotated segments with Krippendorff's alpha
- [ ] Baseline comparison (JS Divergence vs. Valence Difference)
- [ ] Evaluation metrics (AUC, F1, Precision, Recall)
- [ ] CSV output with timestamped scores and confidence weights
- [ ] Reproducibility log (seeds, model versions, experiment tracking)
- [ ] Short report summarizing results
