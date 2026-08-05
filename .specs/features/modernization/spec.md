# Modernize DailyTalk Specification

## Problem Statement

The legacy DailyTalk repository is built on a deprecated stack (Python 3.8, PyTorch 1.7, TensorFlow 2.5, NumPy 1.x, pydub) which fails to import, typecheck, or execute on modern Python versions (Python 3.14+) and modern GPUs (Nvidia T4, L4, RTX PRO 6000). To run training and preprocessing reliably and verify correctness, the codebase must be modernized to Python 3.14 compatible packages, type-safe structures using Pydantic, and covered with robust unit tests.

## Goals

- [ ] Successful typechecking (`ty check`) and linting (`ruff check`) across all files in `src/dailytalk` and `tests/`.
- [ ] Comprehensive unit tests in `tests/` covering configurations, audio, STFT, text utilities, and datasets, ensuring 100% passing rate.
- [ ] Mathematical and functional equivalence of modern STFT and audio prep compared to original.
- [ ] Safe training execution with PyTorch 2.6+ on modern GPUs.

## Out of Scope

| Feature | Reason |
| ------- | ------ |
| Downloading the whole raw dataset from Drive | Raw dataset is already present at `/home/antonio/Documents/MastersDegree/DailyTalk/raw_data/dailytalk` |
| Upgrading DeepSpeaker pre-trained checkpoint formats manually | We will support the existing format via backward-compatible keras/tensorflow loaders |
| Replacing MFA binary itself | Montreal Forced Aligner is an external command-line tool, out of scope for pure Python modernization. Unsupervised alignment mode avoids MFA entirely. |

---

## Assumptions & Open Questions

| Assumption / decision | Chosen default | Rationale | Confirmed? |
| --------------------- | -------------- | --------- | ---------- |
| PyAV vs Pydub | Avoid using `pydub` (March 2021) and prioritize `pyav` if audio decoding is needed, though we will primarily use `librosa`/`soundfile` which are actively maintained. | To follow user preference for modern/maintained audio packages. | yes |
| Pydantic BaseModels | Wrap YAML configurations inside Pydantic BaseModels for loading and validation. | To comply with the "Type with pydantic.BaseModels" constraint. | yes |
| Unsupervised alignment fallback | Default to learn_alignment=True (Unsupervised) since it does not require external MFA setups. | Simpler execution flow and matches pretrained weights. | yes |

---

## User Stories

### P1: Modernize Dependencies and Environment ⭐ MVP

**User Story**: As a developer/researcher, I want the DailyTalk repository to be installable and runnable under Python 3.14 using `uv` so that I can execute the code on modern GPUs.

**Why P1**: Essential first step. Without this, no imports or scripts work.

**Acceptance Criteria**:
1. WHEN running `uv run ty check` THEN it SHALL compile and typecheck without import or syntax errors.
2. WHEN running `uv run ruff check` THEN it SHALL run cleanly without major errors (or with configured exceptions).
3. WHEN running Python 3.14 interpreter THEN all major imports (`torch`, `tensorflow`, `librosa`, `pydantic`, `pyav`) SHALL import successfully.

**Independent Test**: Run `uv run ty check` and verify it reports 0 failures or only explicit ignored diagnostics.

---

### P2: Refactor Audio & STFT Processing

**User Story**: As a deep learning practitioner, I want `audio/stft.py` and processing scripts to use modern PyTorch STFT and complex number APIs without `torch.autograd.Variable` so that it runs fast and cleanly on PyTorch 2.6+.

**Why P2**: The legacy STFT implementation uses deprecated functions and custom manual complex math which is slow and throws deprecation warnings.

**Acceptance Criteria**:
1. WHEN initializing `TacotronSTFT` THEN it SHALL register correct mel filters.
2. WHEN calculating `mel_spectrogram` THEN system SHALL return mathematically equivalent spectrograms.
3. WHEN performing inverse STFT THEN system SHALL not use `torch.autograd.Variable`.

**Independent Test**: Write a unit test that takes dummy audio, transforms it to mel-spectrogram, reconstructs it, and checks values.

---

### P3: Pydantic Configuration Models

**User Story**: As a developer, I want all configuration files (preprocess.yaml, model.yaml, train.yaml) to be validated using Pydantic BaseModels so that any malformed config is caught early and has type-hinting support.

**Why P3**: Standardizes configs and complies with typing constraints.

**Acceptance Criteria**:
1. WHEN loading configurations THEN system SHALL parse and validate them using `pydantic.BaseModel`.
2. WHEN config is invalid THEN system SHALL raise a validation error explaining the missing/wrong fields.

**Independent Test**: Unit test in `tests/test_config.py` validating correct configuration parsing and raising error on invalid YAML files.

---

### P4: Codebase-wide NumPy 2.x and TensorFlow Modernization

**User Story**: As a researcher, I want all legacy numpy types (`np.float`, `np.int`) and Keras/TensorFlow imports to be updated so that the codebase doesn't crash at startup.

**Why P4**: NumPy 2.0 completely removed these deprecated aliases, and legacy Keras imports are incompatible with modern TensorFlow.

**Acceptance Criteria**:
1. WHEN running preprocess or training scripts THEN system SHALL not raise `AttributeError` for `np.float` or `np.int`.
2. WHEN DeepSpeaker is initialized THEN it SHALL load and run layers successfully on TensorFlow 2.18+.

**Independent Test**: Unit test testing DeepSpeaker model creation and layer compilation.

---

### P5: Extensive Unit Tests

**User Story**: As a project maintainer, I want comprehensive unit tests in the `tests/` folder running under `pytest` so that any regression is caught automatically.

**Why P5**: The main goal of the user request is: "generate unit tests as much as possible."

**Acceptance Criteria**:
1. WHEN running `uv run pytest` THEN all written unit tests SHALL pass cleanly.
2. Unit tests SHALL cover: Config models, Text utilities, STFT transforms, Audio processing, Dataset loading, and DeepSpeaker layers.

**Independent Test**: Run `uv run pytest` and verify it runs and reports tests passing.

---

## Edge Cases

- WHEN audio contains silence or all zeros THEN STFT/Griffin-Lim SHALL not divide by zero or raise NaNs.
- WHEN a text contains unexpected non-ASCII characters THEN text cleaners SHALL handle them gracefully without throwing exceptions.
- WHEN a configuration YAML is completely empty THEN Pydantic BaseModel SHALL raise a clearValidationError.

---

## Requirement Traceability

Each requirement gets a unique ID for tracking across design, tasks, and validation.

| Requirement ID | Story | Phase | Status |
| -------------- | ----- | ----- | ------ |
| MOD-01 | P1: Modernize Environment | Specify | Pending |
| MOD-02 | P1: Fix Import Errors | Specify | Pending |
| MOD-03 | P2: Refactor STFT to Modern APIs | Specify | Pending |
| MOD-04 | P3: Pydantic Config Models | Specify | Pending |
| MOD-05 | P4: NumPy 2.x Cleanup | Specify | Pending |
| MOD-06 | P4: TensorFlow/Keras Upgrade | Specify | Pending |
| MOD-07 | P5: Unit Tests Coverage | Specify | Pending |

**Coverage**: 7 total, 0 mapped to tasks, 7 unmapped ⚠️

---

## Success Criteria

- [ ] `uv run ty check` executes with zero errors.
- [ ] `uv run ruff check` executes with zero errors (or configured ignores).
- [ ] `uv run pytest` executes with all tests passing (at least 15+ unit tests covering different parts).
- [ ] Preprocessing prepare_align runs successfully on a subset or dummy of DailyTalk.
