# Modernize DailyTalk Tasks

## Execution Protocol (MANDATORY -- do not skip)

Implement these tasks with the `tlc-spec-driven` skill: **activate it by name and follow its Execute flow and Critical Rules.** Do not search for skill files by filesystem path. The skill is the source of truth for the full flow (per-task cycle, sub-agent delegation, adequacy review, Verifier, discrimination sensor).

**If the skill cannot be activated, STOP and tell the user — do not proceed without it.**

---

**Design**: `.specs/features/modernization/design.md`
**Status**: Draft

---

## Test Coverage Matrix

> Generated from codebase, project guidelines, and spec — confirm before Execute. Guidelines found: none — strong defaults applied (with pytest and ty).

| Code Layer | Required Test Type | Coverage Expectation | Location Pattern | Run Command |
| ---------- | ------------------ | -------------------- | ---------------- | ----------- |
| Configuration | unit | 1:1 schema verification, parses valid YAMLs, fails on invalid YAMLs | `tests/test_config.py` | `uv run pytest tests/test_config.py` |
| Text Processing | unit | Tests all cleaners, symbol mappings, and text_to_sequence conversions | `tests/test_text.py` | `uv run pytest tests/test_text.py` |
| STFT / Audio | unit | STFT and inverse STFT match, outputs valid mel spectrogram shape and values | `tests/test_audio.py` | `uv run pytest tests/test_audio.py` |
| Dataset | unit | Dataset class correctly reads filelists, returns formatted PyTorch tensors | `tests/test_dataset.py` | `uv run pytest tests/test_dataset.py` |
| DeepSpeaker | unit | Model instantiation, layer layout, and forward pass on dummy inputs | `tests/test_deepspeaker.py` | `uv run pytest tests/test_deepspeaker.py` |

---

## Gate Check Commands

> Generated from codebase — confirm before Execute.

| Gate Level | When to Use | Command |
| ---------- | ----------- | ------- |
| Quick | After modular task implementation | `uv run pytest tests/test_config.py tests/test_text.py` |
| Full | After any core system changes | `uv run pytest` |
| Build | Before finalizing a phase | `uv run ruff check src && uv run ty check && uv run pytest` |

---

## Execution Plan

Phases are ordered and run sequentially — each phase completes before the next begins, and tasks within a phase execute in order.

### Phase 1: Foundations and Configuration Validation

```
T1 → T2 → T3
```

### Phase 2: Core Refactoring (Imports, Audio, deepspeaker)

```
T4 → T5 → T6
```

### Phase 3: Validation and Verification

```
T7
```

---

## Task Breakdown

### T1: Implement Pydantic Configuration Models

**What**: Create Pydantic BaseModels in `src/dailytalk/config_models.py` for all YAML configurations and integrate into `utils/tools.py`.
**Where**: `src/dailytalk/config_models.py`, `src/dailytalk/utils/tools.py`
**Depends on**: None
**Reuses**: None
**Requirement**: MOD-04

**Done when**:
- [ ] `src/dailytalk/config_models.py` defines schemas for PreprocessConfig, ModelConfig, and TrainConfig.
- [ ] `get_configs_of` is updated to load configs via Pydantic schemas, with a `.dict()`/`.model_dump()` fallback so legacy dictionary accesses don't break.
- [ ] Unit tests in `tests/test_config.py` cover positive parsing and negative validation error cases.
- [ ] `uv run pytest tests/test_config.py` passes successfully (3+ tests).

**Tests**: unit
**Gate**: quick

---

### T2: Standardize Codebase to Package-Absolute Imports

**What**: Update all imports under `src/dailytalk` (like `from text import` or `from utils import`) to be package-absolute (`from dailytalk.text import` or `from dailytalk.utils import`).
**Where**: All files under `src/dailytalk/`
**Depends on**: T1
**Reuses**: Existing files
**Requirement**: MOD-02

**Done when**:
- [ ] All files under `src/dailytalk` compile without unresolved absolute imports.
- [ ] `uv run ty check` resolves first-party module imports cleanly.
- [ ] Unit tests in `tests/test_text.py` verify text converters and cleaners work under absolute imports.
- [ ] `uv run pytest tests/test_text.py` passes successfully (3+ tests).

**Tests**: unit
**Gate**: quick

---

### T3: Global NumPy 2.x Cleanup

**What**: Scan and replace legacy NumPy type aliases (`np.float`, `np.int`, `np.bool`, `np.object`) across the entire repository with standard types.
**Where**: All python files in the repository
**Depends on**: T2
**Reuses**: Existing files
**Requirement**: MOD-05

**Done when**:
- [ ] No occurrences of `np.float`, `np.int`, `np.bool`, `np.object` remain in `src/`.
- [ ] Ruff lint check runs cleanly with respect to NumPy types.

**Tests**: none
**Gate**: build

---

### T4: STFT Processing Modernization

**What**: Refactor `src/dailytalk/audio/stft.py` to remove deprecated `torch.autograd.Variable` and leverage standard modern PyTorch 2.x STFT APIs.
**Where**: `src/dailytalk/audio/stft.py`, `src/dailytalk/audio/audio_processing.py`
**Depends on**: T3
**Reuses**: Existing STFT basis algorithms
**Requirement**: MOD-03

**Done when**:
- [ ] `audio/stft.py` uses modern complex tensor STFT or standard float operations.
- [ ] `torch.autograd.Variable` is completely removed.
- [ ] Unit tests in `tests/test_audio.py` check TacotronSTFT mel spectrogram and audio reconstruction equivalence.
- [ ] `uv run pytest tests/test_audio.py` passes successfully (4+ tests).

**Tests**: unit
**Gate**: full

---

### T5: DeepSpeaker and Keras/TensorFlow 2.18+ Modernization

**What**: Modernize DeepSpeaker imports to support modern TensorFlow/Keras and write validation tests.
**Where**: `src/dailytalk/deepspeaker/`
**Depends on**: T3
**Reuses**: DeepSpeaker convolutional blocks
**Requirement**: MOD-06

**Done when**:
- [ ] DeepSpeaker files use standard `keras` or `tensorflow.keras` namespaces.
- [ ] Unused or broken imports/attributes (e.g. `_SixMetaPathImporter` or legacy backend hooks) are type-annotated with ignores or removed.
- [ ] Unit tests in `tests/test_deepspeaker.py` build the model and verify forward pass on dummy speech features.
- [ ] `uv run pytest tests/test_deepspeaker.py` passes successfully (2+ tests).

**Tests**: unit
**Gate**: full

---

### T6: Complete Dataset and Dataloader Unit Testing

**What**: Write comprehensive unit tests for dataset loading, verifying it parses the metadata/filelists correctly and returns valid PyTorch batches.
**Where**: `src/dailytalk/dataset.py`, `tests/test_dataset.py`
**Depends on**: T4
**Reuses**: `dataset.py` logic
**Requirement**: MOD-07

**Done when**:
- [ ] Unit tests in `tests/test_dataset.py` verify file list loading, feature padding, and batch collation.
- [ ] `uv run pytest tests/test_dataset.py` passes successfully (3+ tests).

**Tests**: unit
**Gate**: full

---

### T7: Full Codebase Lint and Typecheck Pass

**What**: Perform a complete pass to resolve any remaining Ruff and Ty errors.
**Where**: All files under `src/` and `tests/`
**Depends on**: T1, T2, T3, T4, T5, T6
**Requirement**: MOD-01, MOD-02

**Done when**:
- [ ] `uv run ruff check src` produces 0 errors.
- [ ] `uv run ty check` produces 0 diagnostics.
- [ ] `uv run pytest` runs and passes all 15+ unit tests across the test suites.

**Tests**: unit
**Gate**: build

---

## Phase Execution Map

```
Phase 1 → Phase 2 → Phase 3

Phase 1:  T1 ──→ T2 ──→ T3
Phase 2:  T4 ──→ T5 ──→ T6
Phase 3:  T7
```

---

## Task Granularity Check

| Task | Scope | Status |
| ---- | ----- | ------ |
| T1: Pydantic Configuration Models | 2 files, focused schemas | ✅ Granular |
| T2: Standardize Imports | 1 namespace refactoring | ✅ Granular |
| T3: Global NumPy 2.x Cleanup | Search and replace | ✅ Granular |
| T4: STFT Processing Modernization | 2 files, focused DSP refactor | ✅ Granular |
| T5: DeepSpeaker Modernization | 1 sub-package | ✅ Granular |
| T6: Dataset Unit Testing | 1 test suite | ✅ Granular |
| T7: Lint/Typecheck Pass | Compliance verification | ✅ Granular |

---

## Diagram-Definition Cross-Check

| Task | Depends On (task body) | Diagram Shows | Status |
| ---- | ---------------------- | ------------- | ------ |
| T1 | None | None | ✅ Match |
| T2 | T1 | T1 | ✅ Match |
| T3 | T2 | T2 | ✅ Match |
| T4 | T3 | T3 | ✅ Match |
| T5 | T3 | T3 | ✅ Match |
| T6 | T4 | T4 | ✅ Match |
| T7 | T1, T2, T3, T4, T5, T6 | Phase 1 & 2 tasks | ✅ Match |

---

## Test Co-location Validation

| Task | Code Layer Created/Modified | Matrix Requires | Task Says | Status |
| ---- | --------------------------- | --------------- | --------- | ------ |
| T1 | Configuration | unit | unit | ✅ OK |
| T2 | Text Processing | unit | unit | ✅ OK |
| T3 | Entity/Config | none | none | ✅ OK |
| T4 | STFT / Audio | unit | unit | ✅ OK |
| T5 | DeepSpeaker | unit | unit | ✅ OK |
| T6 | Dataset | unit | unit | ✅ OK |
| T7 | Codebase | unit | unit | ✅ OK |
