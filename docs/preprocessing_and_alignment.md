# Preprocessing and Alignment Reference Guide

This document outlines the configuration adjustments, alignment strategies, and preprocessing steps required to prepare the original DailyTalk raw dataset for training.

---

## 1. Raw Dataset Verification

The original DailyTalk raw dataset is verified and located at:
`/home/antonio/Documents/MastersDegree/DailyTalk/raw_data/dailytalk`

### Structure
```
/home/antonio/Documents/MastersDegree/DailyTalk/raw_data/dailytalk/
├── metadata.json       # Dialogue context and emotion annotations
└── data/               # Dialouge directories numbered 0 to 2529
    ├── 0/
    │   ├── 0_1_d0.wav  # Turn audio
    │   └── 0_1_d0.txt  # Turn clean transcript
    ├── 1/
    ...
```

---

## 2. Configuration Adjustments

Before starting preprocessing, you must update the legacy paths inside `config/DailyTalk/preprocess.yaml`.

### Proposed Changes to `config/DailyTalk/preprocess.yaml`
Modify the `path` block to point to your workspace:

```yaml
path:
  corpus_path: "/home/antonio/Documents/MastersDegree/DailyTalk/raw_data/dailytalk" # Points to raw data
  sub_dir_name: "data"                                                             # Dialogue folder under raw data
  lexicon_path: "lexicon/librispeech-lexicon.txt"                                  # Standard English phoneme lexicon
  raw_path: "./raw_data/dailytalk"                                                 # Where prepared files are written (lowercase)
  preprocessed_path: "./preprocessed_data/DailyTalk"                               # Target directory for extracted features
```

> [!IMPORTANT]
> **Case Sensitivity**: The folder name under `raw_data/` is lowercase `dailytalk` on the Linux filesystem. Ensure `raw_path` is updated to `./raw_data/dailytalk` (changed from the original `./raw_data/DailyTalk`) to avoid file-not-found issues.

---

## 3. Preprocessing Pipeline

Preprocessing is a two-step script execution utilizing the `uv` package manager for fast and reproducible runs.

### Step 1: Alignment Preparation
Run the preparation script to trim/scale wavs and generate clean text `.lab` files:
```bash
uv run python3 prepare_align.py --dataset DailyTalk
```

*   **What it does**:
    1.  Loads each utterance audio file from `corpus_path/data`.
    2.  Cleans and normalizes the transcript (converting punctuation, numerals, etc.).
    3.  Scales the audio file to uniform volume and saves it to `raw_path/data` as `.wav`.
    4.  Saves the cleaned transcript to `raw_path/data` as a `.lab` file alongside the scaled audio.

---

### Step 2: Alignment Selection (Crucial Choice)

DailyTalk supports two alignment mechanisms. Your choice determines whether you need the **Montreal Forced Aligner (MFA)**.

#### Option A: Unsupervised Duration Modeling (Recommended)
*   **Description**: The model learns to align text and audio dynamically during training using variational/attention methods (`learn_alignment: True` in `model.yaml`).
*   **MFA Dependency**: **No MFA required!** You do not need to download or generate `.TextGrid` files.
*   **How to run**: `preprocess.py` will skip the TextGrid block and generate frame-level features (`mel_frame`, `pitch_frame`, `energy_frame`, `text_emb`, `attn_prior`) which are sufficient for unsupervised training.
*   **Result**: Faster, simpler pipeline. This matches the authors' official pretrained model setup.

#### Option B: Supervised Duration Modeling (Alternative)
*   **Description**: The model uses static phone-level durations extracted from a forced-aligner (`learn_alignment: False`).
*   **MFA Dependency**: **MFA is required.** You must have `.TextGrid` files located inside `preprocessed_data/DailyTalk/TextGrid/`.
*   **How to run**:
    1.  Download the pre-extracted alignments from the authors' [Google Drive Link](https://drive.google.com/drive/folders/1fizpyOiQ1lG2UDaMlXnT3Ll4_j6Xwg7K?usp=sharing).
    2.  Extract the downloaded zip directly to create the path `preprocessed_data/DailyTalk/TextGrid/`.
    3.  Alternately, run the Montreal Forced Aligner yourself using the `.wav` and `.lab` files generated in Step 1.

---

### Step 3: Feature Extraction
Run the feature extraction script to build mel-spectrograms, pitch contours, energy tracks, and text embeddings:
```bash
uv run python3 preprocess.py --dataset DailyTalk
```

*   **What it does**:
    1.  Extracts 80-channel Log-Mel spectrograms from the scaled audios.
    2.  Calculates fundamental frequency ($F_0$) pitch tracks using `pyworld.dio` and `pyworld.stonemask`.
    3.  Calculates signal energy.
    4.  Generates sentence text embeddings using the `SentenceTransformer` ('distiluse-base-multilingual-cased-v1') to capture dialog history.
    5.  Saves these features as binary `.npy` arrays inside `preprocessed_data/DailyTalk/`.
    6.  Computes and writes global statistics to `stats.json` for feature normalization during training.
    7.  Randomly splits dialogues into training and validation sets, writing `train_frame.txt` and `val_frame.txt`.
