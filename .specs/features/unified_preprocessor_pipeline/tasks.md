# Tasks: Unified Preprocessing Pipeline

- [ ] **Task 1: Stage 1 - PyAV Audio Resampling & Cleaning (`preparation_and_cleaning.py`)**
  - Implement PyAV (`av`) audio resampler and volume normalizer.
  - Implement `prepare_raw_data()` to clean text and generate `.lab` and `.wav` files.
  - Verification: Unit test PyAV audio loading and resampling.

- [ ] **Task 2: Stage 2 - Language Frontend Module (`language_frontend.py`)**
  - Implement `LanguageFrontendProcessor` wrapping `dailytalk.text` language adapters.
  - Verification: Unit test English and Portuguese frontend text processing.

- [ ] **Task 3: Stage 3 - Feature Extraction Engine (`feature_extractor.py`)**
  - Implement `AcousticFeatureExtractor` for STFT Mel, PyWorld Pitch F0, Energy C2, Durations, and `stats.json`.
  - Verification: Unit test feature extraction outputs.

- [ ] **Task 4: Stage 4 - Speaker Identity Embedder (`speaker_embedder.py`)**
  - Implement `SpeakerEmbedder` for DeepSpeaker 512-dim identity vector generation.
  - Verification: Unit test DeepSpeaker embedding extraction.

- [ ] **Task 5: Pipeline Orchestrator & API (`pipeline.py` & `__init__.py`)**
  - Implement `PreprocessorPipeline` and Pydantic `PipelineResult` model.
  - Expose clean `Preprocessor` API in `__init__.py`.
  - Verification: Unit test `PreprocessorPipeline` stage execution.

- [ ] **Task 6: Refactor CLI Entrypoints (`preprocess.py` & `prepare_align.py`)**
  - Update `preprocess.py` Typer CLI to run `PreprocessorPipeline`.
  - Verification: Test CLI help and stage flags.

- [ ] **Task 7: Quality Gate & Full Integration Tests (`tests/test_preprocessor.py`)**
  - Run full test suite with `uv run pytest`.
  - Run static type checker with `uv run ty check`.
  - Run linter with `uv run ruff check`.
