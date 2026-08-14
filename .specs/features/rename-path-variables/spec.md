# Feature Specification: Rename Path Variables for Disambiguation

## Context & User Goal
The preprocessing path configuration in `config/DailyTalk/preprocess.yaml` previously used ambiguous names (`corpus_path`, `raw_path`, `preprocessed_path`). To make data pipeline semantics explicit, rename these fields across the entire codebase to:
- `corpus_path` ➔ `raw_corpus_path`
- `raw_path` ➔ `intermediate_data_path`
- `preprocessed_path` ➔ `preprocessed_data_path`

## Requirements

- **REQ-001 (Pydantic Schema Update)**: Update `PathConfig` in `src/dailytalk/config_models.py` to use `raw_corpus_path`, `intermediate_data_path`, and `preprocessed_data_path` fields using `pydantic.BaseModel`.
- **REQ-002 (YAML Configuration Update)**: Update `config/DailyTalk/preprocess.yaml` to define `raw_corpus_path`, `intermediate_data_path`, and `preprocessed_data_path` under `path`.
- **REQ-003 (Preprocessor Pipeline Update)**: Update `preparation_and_cleaning.py`, `pipeline.py`, and `dailytalk.py` to reference `raw_corpus_path`, `intermediate_data_path`, and `preprocessed_data_path`.
- **REQ-004 (Model, Dataset & Utilities Update)**: Update `dataset.py`, `tools.py`, `synthesize.py`, `CompTransTTS.py`, and `modules.py` to access the renamed path fields.
- **REQ-005 (Unit Test Alignment)**: Update unit tests in `tests/test_config.py`, `tests/test_dataset.py`, `tests/test_preprocessor.py`, and `tests/test_tools.py` to use the updated path attributes.
- **REQ-006 (Validation Gate)**: Ensure zero type errors with `uv run ty check`, zero lint errors with `uv run ruff check`, and 100% test pass with `uv run pytest`.

## Acceptance Criteria

- **AC-1**: `PreprocessConfig.path` has attributes `raw_corpus_path`, `intermediate_data_path`, and `preprocessed_data_path`.
- **AC-2**: `uv run ty check` completes with `All checks passed!`.
- **AC-3**: `uv run ruff check` completes with `All checks passed!`.
- **AC-4**: `uv run pytest` passes all tests.
