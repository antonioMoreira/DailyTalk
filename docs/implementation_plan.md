# Implementation Plan - Modernizing Repository to Python 3.14

This plan outlines the steps required to modernize the DailyTalk codebase from its legacy stack (Python 3.8, PyTorch 1.7.0, TensorFlow 2.5.1) to **Python 3.14** and its compatible modern ecosystem (PyTorch 2.6+, TensorFlow 2.18+, NumPy 2.x+, Librosa 0.10.x+, Numba 0.61+).

---

## Goal Description
The objective is to upgrade the Python version of this repository to 3.14 while maintaining the mathematical correctness and execution capabilities of the Text-to-Speech training, preprocessing, and inference pipelines. 

Because Python 3.14 is very modern and introduces deep C-API and bytecode modifications, all major packages must be upgraded to their latest versions. Consequently, we must refactor deprecated APIs and breaking changes across the codebase.

---

## User Review Required

> [!WARNING]
> **Pretrained Weight Compatibility**: 
> Upgrading TensorFlow (and Keras) to v2.18+ (which defaults to Keras 3) might cause issues when loading older Keras v2 checkpoint weights (`.h5`) for the DeepSpeaker model. We may need to use Keras conversion utilities or run the backend in a Keras 2 compat mode if available.

> [!IMPORTANT]
> **Python 3.14 Wheel Availability**:
> At the time of this setup, some packages (specifically `numba` and CUDA-enabled `torch`) may still have experimental Python 3.14 wheels on PyPI. Depending on the exact system environment, compiling some dependencies from source might be required unless we fallback to the highest stable Python version available (e.g., Python 3.13) which has full stable precompiled wheel coverage.

---

## Proposed Changes

### 1. Environment & Dependency Layer

#### [MODIFY] [pyproject.toml](file:///home/antonio/Documents/MastersDegree/DailyTalk/pyproject.toml)
#### [MODIFY] [.python-version](file:///home/antonio/Documents/MastersDegree/DailyTalk/.python-version)

- Update `.python-version` to `3.14`.
- Elevate Python constraint in `pyproject.toml` to: `requires-python = ">=3.14"`.
- Upgrade the entire dependency block to modern, Python 3.14-compatible versions:
  ```toml
  dependencies = [
      "product_key_memory >= 0.2.0",
      "local_attention >= 1.9.0",
      "rotary_embedding_torch >= 0.6.0",
      "python_speech_features >= 0.6",
      "pandas >= 2.2.0",
      "tensorflow >= 2.18.0",
      "g2p_en >= 2.1.0",
      "inflect >= 7.0.0",
      "librosa >= 0.10.1",
      "matplotlib >= 3.9.0",
      "numba >= 0.61.0",
      "numpy >= 2.0.0",
      "pypinyin >= 0.50.0",
      "pyworld >= 0.3.4",
      "pyyaml >= 6.0.1",
      "scikit-learn >= 1.5.0",
      "scipy >= 1.13.0",
      "soundfile >= 0.12.1",
      "tensorboard >= 2.18.0",
      "tgt >= 1.4.4",
      "torch >= 2.6.0",
      "tqdm >= 4.66.0",
      "unidecode >= 1.3.8",
      "pillow >= 10.3.0",
      "einops >= 0.8.0",
      "sentence-transformers >= 3.0.0",
      "transformers >= 4.45.0",
  ]
  ```

---

### 2. Audio Processing & STFT Refactoring

#### [MODIFY] [audio/stft.py](file:///home/antonio/Documents/MastersDegree/DailyTalk/audio/stft.py)
#### [MODIFY] [audio/audio_processing.py](file:///home/antonio/Documents/MastersDegree/DailyTalk/audio/audio_processing.py)

- **Remove `torch.autograd.Variable`**: Replace with standard `torch.Tensor` operations since `Variable` is long deprecated and raises warnings or errors in newer PyTorch versions.
- **Refactor STFT Math**:
  - Legacy `audio/stft.py` manually calculates real/imaginary magnitudes and phases using float buffers.
  - Modernize this to use PyTorch's native complex numbers and `torch.stft(..., return_complex=True)`.
- **Adapt to Librosa 0.10+**:
  - Standardize signature of `librosa.filters.mel` (update arguments to match current API, ensuring keyword-only constraints are met).
  - Update usage of `librosa.util.pad_center` which was changed or shifted across versions.

---

### 3. Speaker Embedder & TensorFlow/Keras Modernization

#### [MODIFY] [deepspeaker/conv_models.py](file:///home/antonio/Documents/MastersDegree/DailyTalk/deepspeaker/conv_models.py)
#### [MODIFY] [deepspeaker/embedding.py](file:///home/antonio/Documents/MastersDegree/DailyTalk/deepspeaker/embedding.py)

- **Keras 3 Compatibility**:
  - Replace `import tensorflow.keras.backend as K` with the standardized, backend-agnostic `import keras` or newer TensorFlow APIs.
  - Modernize model layers instantiation to ensure compatibility with Keras 3.x (e.g., handling keyword arguments in layers).
  - Add explicit exception/compatibility block for weight loading (`load_weights`) to support reading older model formats.

---

### 4. Codebase-wide NumPy 2.0 Modernization

#### [MODIFY] Multiple Files (`dataset.py`, `preprocessor/preprocessor.py`, `train.py`, etc.)

- **Eliminate Deprecated NumPy Aliases**:
  - Conduct a global search and replace for legacy aliases:
    - Replace `np.float` $\rightarrow$ `float` or `np.float64`
    - Replace `np.int` $\rightarrow$ `int` or `np.int64`
    - Replace `np.bool` $\rightarrow$ `bool`
    - Replace `np.object` $\rightarrow$ `object`
  - This avoids instant crashes with NumPy 2.x which completely removed these deprecated namespaces.

---

## Verification Plan

Because we are changing core math libraries, we must execute a strict, step-by-step verification pipeline to guarantee results remain identical to the legacy setup.

### Automated Tests
1. **Verification Command**:
   ```bash
   uv run python -c "import torch; import tensorflow; import numba; print('All imported successfully')"
   ```
2. **Preprocessing Dry-Run**:
   Ensure speech extraction pipelines compile and execute:
   ```bash
   uv run python preprocess.py --dataset DailyTalk
   ```
3. **Training Iteration Check**:
   Validate that the backward pass, optimizers, and gradient steps run properly on PyTorch 2.6+:
   ```bash
   uv run python train.py --dataset DailyTalk --restore_step 0
   ```

### Manual Verification
- Check generated audio files in the preprocessed outputs to verify that modern `librosa` and `pyworld` calculations yield mathematically equivalent features compared to the reference data.
