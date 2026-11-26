# Feature Specification: Unified Preprocessing Pipeline

## Overview
Re-architect and unify the DailyTalk preprocessing pipeline into a modular sub-package (`src/dailytalk/preprocessor/`). The unified pipeline orchestrates raw audio resampling, language-driven text cleaning, acoustic feature extraction (Mel, Pitch, Energy, Durations, Sentence Embeddings), and speaker identity vector extraction (`spker_embed.npy` via DeepSpeaker) under a single orchestrator (`pipeline.py`).

---

## Target Architecture

```
src/dailytalk/preprocessor/
    ├── __init__.py                         # Exposes unified Preprocessor API
    ├── pipeline.py                         # Main Orchestrator / Pipeline Runner
    ├── preparation_and_cleaning.py         # Stage 1: Audio resampling (22.05kHz) & transcript lab preparation using PyAV
    ├── language_frontend.py                # Stage 2: Language-driven G2P, cleaners & symbol sequence mapping
    ├── feature_extractor.py                # Stage 3: Mel STFT, PyWorld Pitch (F0), Energy (C2), Durations & Text Embeddings
    └── speaker_embedder.py                 # Stage 4: DeepSpeaker 512-dim identity vector extraction
```

---

## Requirements with Traceable IDs

### REQ-PREP-001: Stage 1 - Preparation & Audio Resampling (`preparation_and_cleaning.py`)
- **ID**: REQ-PREP-001
- **Description**: Read raw dataset files, clean transcripts using text cleaners, resample audio to target sample rate (22.05 kHz) using PyAV (`av`), and save cleaned `.wav` and `.lab` files.
- **Acceptance Criteria**:
  - Resampling and volume normalization must use `av` (PyAV).
  - All function parameters must have explicit type annotations.

### REQ-PREP-002: Stage 2 - Language Frontend Integration (`language_frontend.py`)
- **ID**: REQ-PREP-002
- **Description**: Wrap `dailytalk.text` language adapters (`EnglishFrontend`, `PortugueseFrontend`) to clean input text and convert text to phonemes/symbol ID sequences based on configured dataset language (`en` or `pt`).
- **Acceptance Criteria**:
  - Dynamically load frontend based on `PreprocessConfig`.
  - Provide typed interfaces for cleaning and sequence conversion.

### REQ-PREP-003: Stage 3 - Feature Extraction Engine (`feature_extractor.py`)
- **ID**: REQ-PREP-003
- **Description**: Extract Mel-spectrograms (via `TacotronSTFT`), Pitch ($F_0$ via PyWorld), Energy ($C_2$ via STFT L2 norm), phoneme/frame durations, and sentence embeddings (`SentenceTransformer`).
- **Acceptance Criteria**:
  - Compute feature statistics (`stats.json` with mean/std for pitch, energy, duration).
  - Save tensors as `.npy` arrays under `preprocessed_path`.

### REQ-PREP-004: Stage 4 - Speaker Identity Embeddings (`speaker_embedder.py`)
- **ID**: REQ-PREP-004
- **Description**: Extract 512-dimensional speaker identity vectors using DeepSpeaker (`spker_embed.npy`).
- **Acceptance Criteria**:
  - Run DeepSpeaker on preprocessed audio clips and save embeddings under `spker_embed/`.
  - Handle optional disabling if `speaker_embedder: "none"`.

### REQ-PREP-005: Pipeline Orchestrator & CLI (`pipeline.py` & `preprocess.py`)
- **ID**: REQ-PREP-005
- **Description**: Provide `PreprocessorPipeline` orchestrator in `pipeline.py` that executes Stages 1 through 4 in sequence or individually via `--stage` CLI flag.
- **Acceptance Criteria**:
  - Use Pydantic `BaseModel` for pipeline stage configurations and state.
  - Expose clean CLI command via `dailytalk.preprocess`.

### REQ-PREP-006: Testing & Quality Gate
- **ID**: REQ-PREP-006
- **Description**: Add comprehensive unit tests in `tests/test_preprocessor.py`.
- **Acceptance Criteria**:
  - `uv run pytest` passes cleanly.
  - `uv run ty check` reports 0 errors.
  - `uv run ruff check` reports 0 lint errors.
