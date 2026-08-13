# STATE

## Decisions

### AD-001
- **Decision**: Upgrade Python requirement to >=3.14, configure Ruff and Ty in pyproject.toml, and use Pydantic BaseModels for configuration validation and type safety.
- **Reason**: To comply with user requirements for Python 3.14, actively maintained packages, robust configuration loading, and strict static analysis via Ty and Ruff.
- **Trade-off**: Requires upgrading and adapting all configurations, legacy dependencies, and code patterns to strictly typechecked Pydantic structures and Python 3.14 features.
- **Scope**: Repository-wide environment, configurations, and static checking.
- **Date**: 2026-07-22
- **Status**: active

### AD-002
- **Decision**: Refactor deprecated Python, PyTorch, and NumPy interfaces (such as torch.autograd.Variable, np.float, and importlib/six monkey-patches).
- **Reason**: Python 3.14, PyTorch 2.6+, and NumPy 2.x completely removed these legacy APIs, causing immediate import or runtime crashes.
- **Trade-off**: Requires modifying legacy training and preprocessing modules to use standard, native modern APIs (e.g. native complex torch STFT and modern PyTorch AMP/autocast).
- **Scope**: Training, preprocessing, model, and audio modules.
- **Date**: 2026-07-22
- **Status**: active

### AD-003
- **Decision**: Create comprehensive unit tests in the tests/ directory and strictly run them via `uv run pytest`.
- **Reason**: To satisfy the main goal of generating unit tests as much as possible and guaranteeing correctness of updated STFT, audio preprocessing, and configuration parsing.
- **Trade-off**: Testing ML training requires mocking or small-data fixtures, so unit tests must focus on robust modular coverage of utilities, config parsers, STFT transforms, and dataset loading.
- **Scope**: tests/ directory
- **Date**: 2026-07-22
- **Status**: active

### AD-004
- **Decision**: Add full type annotations to `src/dailytalk/utils/tools.py` using Pydantic `BaseModel` config types (`PreprocessConfig`, `ModelConfig`, `TrainConfig`, `SynthesizeArgs`) and configure `.vscode/settings.json` for `ty` and `ruff`.
- **Reason**: Eliminates `Unknown` type inference issues across modules including `preprocess.py` and provides strict static type safety.
- **Scope**: `src/dailytalk/utils/tools.py`, `.vscode/settings.json`, and `tests/test_tools.py`.
- **Date**: 2026-08-13
- **Status**: active

## Handoff

- **Feature**: Type annotation to `tools.py` (`type-annotations-tools`)
- **Phase / Task**: Phase 4: Execute & Verify
- **Completed**:
  - Configured `.vscode/settings.json` for `ty` and `ruff` format-on-save.
  - Type-annotated all functions in `src/dailytalk/utils/tools.py` using Pydantic `BaseModel` classes (`PreprocessConfig`, `ModelConfig`, `TrainConfig`, `SynthesizeArgs`).
  - Added unit test suite in `tests/test_tools.py`.
  - Verified `uv run ty check` (All checks passed!), `uv run ruff check` (All checks passed!), and `uv run pytest` (37/37 tests passed!).
- **In-progress**: none
- **Next step**: Complete. All tests and static checks pass cleanly.
- **Blockers**: none
- **Uncommitted files**: none
- **Branch**: main
