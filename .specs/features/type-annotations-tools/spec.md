# Feature Specification: Type Annotations for `tools.py`

## Requirements

- **REQ-001**: Type annotate `get_configs_of(dataset: str) -> tuple[PreprocessConfig, ModelConfig, TrainConfig]` in `src/dailytalk/utils/tools.py` using Pydantic `BaseModel` classes from `dailytalk.config_models`.
- **REQ-002**: Type annotate `get_variance_level(preprocess_config: PreprocessConfig, model_config: ModelConfig, data_loading: bool = True) -> tuple[str, str, str, str]` in `src/dailytalk/utils/tools.py`.
- **REQ-003**: Type annotate all utility and plotting functions in `src/dailytalk/utils/tools.py` (`get_phoneme_level_pitch`, `get_phoneme_level_energy`, `to_device`, `log`, `get_mask_from_lengths`, `expand`, `synth_one_sample`, `synth_samples`, `plot_mel`, `plot_alignment`, `plot_embedding`, `save_figure_to_numpy`, `pad_1D`, `pad_2D`, `pad_3D`, `pad`).
- **REQ-004**: In `synth_samples`, type `args` as `SynthesizeArgs` (Pydantic `BaseModel` from `dailytalk.cli_models`).
- **REQ-005**: Add comprehensive unit tests in `tests/test_tools.py` testing `get_configs_of`, `get_variance_level`, `pad_1D`, `pad_2D`, `pad_3D`, `pad`, `expand`, `get_mask_from_lengths`, and other utilities in `tools.py`.
- **REQ-006**: Ensure `.vscode/settings.json` is configured for `ty` and `ruff`, and that `uv run ty check`, `uv run ruff check`, and `uv run pytest` pass cleanly.

## Acceptance Criteria

1. `PreprocessConfig`, `ModelConfig`, and `TrainConfig` (Pydantic `BaseModel`s) are used as explicit type hints for configuration arguments in `tools.py`.
2. `ty check` identifies zero `Unknown` or type errors when importing and using `get_configs_of` and `get_variance_level` from `dailytalk.utils.tools`.
3. `tests/test_tools.py` passes all unit tests via `uv run pytest`.
4. `uv run ruff check src/dailytalk/utils/tools.py tests/test_tools.py` passes with zero issues.
