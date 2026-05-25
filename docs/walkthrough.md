# Walkthrough - Configuration for `uv`

We have successfully configured the repository to use `uv` as its primary Python environment manager. This setup guarantees highly performant and reproducible builds and executions.

## Changes Made

1. **Created `.python-version`**:
   - Specified `3.8.20` as the Python version for this project.
   - This version ensures compatibility with older pre-compiled wheels for key libraries (e.g. `numba==0.48`, `numpy==1.19.2`, and older deep-learning packages).

2. **Created `pyproject.toml`**:
   - Converted the legacy `requirements.txt` into a modern PEP 621 compliant `pyproject.toml`.
   - Handled special compatibility constraints required by Python 3.8 and old package versions:
     - **`protobuf < 3.20`**: Pinned to resolve a deep conflict with `tensorflow==2.5.1` (where modern Protobuf v4+ or v5+ causes descriptor creation `TypeError` errors).
     - **`sentence-transformers < 2.0.0`** and **`transformers < 4.16.0`**: Pinned to avoid newer releases that import `torch.utils._pytree`, which does not exist in PyTorch `1.7.0`.
     - Standardized all deep learning and audio processing requirements from the original project.

3. **Generated `uv.lock`**:
   - Resolved all 96 dependencies successfully and created a lockfile (`uv.lock`) for reproducible setups.

4. **Created local Virtual Environment (`.venv`)**:
   - Created and synchronized a clean `.venv` containing all 96 packages.

---

## Verification Results

All core operations were verified using `uv run`:

### 1. PyTorch & TensorFlow Import Verification
Running `uv run python -c` successfully imported both backend libraries:
```bash
$ uv run python -c "import torch; print('Torch:', torch.__version__)" && uv run python -c "import tensorflow as tf; print('TF:', tf.__version__)"
Torch: 1.7.0
TF: 2.5.1
```

### 2. Preprocessor Dry-run Check
Running the preprocessor help command executes perfectly without import errors:
```bash
$ uv run python preprocess.py --help
usage: preprocess.py [-h] --dataset DATASET

optional arguments:
  -h, --help         show this help message and exit
  --dataset DATASET  name of dataset
```

### 3. Training Script Dry-run Check
Running the training script help command executes successfully:
```bash
$ uv run python train.py --help
usage: train.py [-h] [--use_amp] [--restore_step RESTORE_STEP] --dataset
                DATASET

optional arguments:
  -h, --help            show this help message and exit
  --use_amp
  --restore_step RESTORE_STEP
  --dataset DATASET     name of dataset
```

---

## How to use the new environment

You can run commands using the `uv` environment instantly without manual activation:

* **To run scripts**:
  ```bash
  uv run python <script.py> [arguments]
  ```
* **To add a dependency**:
  ```bash
  uv add <dependency-name>
  ```
* **To sync the environment**:
  ```bash
  uv sync
  ```
