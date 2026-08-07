# Feature Specification: Typer CLI & Pydantic CLI Arguments

## Overview
Migrate all command-line interface (CLI) entrypoints (`train.py`, `preprocess.py`, `synthesize.py`, `prepare_align.py`) from legacy `argparse` to `typer`. Define strongly-typed Pydantic `BaseModel` classes for CLI arguments (`TrainArgs`, `PreprocessArgs`, `SynthesizeArgs`, `PrepareAlignArgs`) to replace raw `argparse.Namespace` references across the codebase.

---

## Requirements with Traceable IDs

### REQ-CLI-001: Pydantic CLI Argument Models
- **ID**: REQ-CLI-001
- **Description**: Define Pydantic `BaseModel` classes for CLI arguments in `src/dailytalk/cli_models.py` (or `src/dailytalk/config_models.py`).
- **Models**:
  - `TrainArgs(BaseModel)`: `dataset: str`, `use_amp: bool = False`, `restore_step: int = 0`.
  - `PreprocessArgs(BaseModel)`: `dataset: str`.
  - `PrepareAlignArgs(BaseModel)`: `dataset: str`.
  - `SynthesizeArgs(BaseModel)`: `restore_step: int`, `mode: Literal["batch", "single"]`, `dataset: str`, `source: str | None = None`, `text: str | None = None`, `speaker_id: str = "p225"`, `emotion_id: str = "happiness"`.
- **Acceptance Criteria**:
  - All models must inherit from `pydantic.BaseModel`.
  - Models must validate input types and provide default values matching previous CLI defaults.

### REQ-CLI-002: Refactor `train.py`
- **ID**: REQ-CLI-002
- **Description**: Replace `argparse` in `train.py` with `typer.Typer` or `typer.run`.
- **Signature Update**: Update `train` signature to accept `args: TrainArgs` instead of `args: argparse.Namespace`.
- **Acceptance Criteria**:
  - `train(rank: int, args: TrainArgs, configs: tuple[PreprocessConfig, ModelConfig, TrainConfig], batch_size: int, num_gpus: int)` accepts `TrainArgs`.
  - Running `python -m dailytalk.train --help` or `uv run python -m dailytalk.train --help` displays Typer help output.

### REQ-CLI-003: Refactor `preprocess.py`, `prepare_align.py`, and `synthesize.py`
- **ID**: REQ-CLI-003
- **Description**: Replace `argparse` with `typer` in `preprocess.py`, `prepare_align.py`, and `synthesize.py`.
- **Signature Updates**:
  - `synthesize.py`: `synthesize(device, model, args: SynthesizeArgs, configs, vocoder, loader, control_values)` accepts `SynthesizeArgs`.
- **Acceptance Criteria**:
  - All CLI entrypoints parse options using `typer` and convert parsed flags into their respective Pydantic argument model.

### REQ-CLI-004: Unit Testing & Verification
- **ID**: REQ-CLI-004
- **Description**: Add unit tests verifying `TrainArgs`, `PreprocessArgs`, `SynthesizeArgs`, and `PrepareAlignArgs` instantiations, type validations, and Typer CLI execution.
- **Acceptance Criteria**:
  - All unit tests in `tests/` pass cleanly with `uv run pytest`.
  - Static type checking with `uv run ty check` passes with 0 errors.
  - Linting with `uv run ruff check` passes cleanly.
