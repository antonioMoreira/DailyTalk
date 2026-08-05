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

## Handoff

- **Feature**: Modernization to Python 3.14 and test generation
- **Phase / Task**: Phase 1: Specify Feature
- **Completed**: none
- **In-progress**: Creating .specs/features/modernization/spec.md
- **Next step**: Complete spec.md and present the specification to the user.
- **Blockers**: none
- **Uncommitted files**: .specs/STATE.md
- **Branch**: main
