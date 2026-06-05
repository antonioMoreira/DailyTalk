# Implementation Plan - English Preprocessing and Alignment

## Phase 1: Path & Compatibility Configuration
- [x] Task: Update `config/DailyTalk/preprocess.yaml` raw data path to point to case-sensitive `./raw_data/dailytalk` and output directory to `./raw_data/preprocessed_data/DailyTalk`.
- [x] Task: Resolve required NLTK English model files (`averaged_perceptron_tagger_eng`) for G2P tokenization.

## Phase 2: Librosa & STFT Modern API Adaptation
- [x] Task: Fix deprecated positional parameter usage in `librosa.load` in `preprocessor/dailytalk.py` and `preprocessor/preprocessor.py`.
- [x] Task: Fix deprecated parameters in STFT window centering functions (`pad_center`).

## Phase 3: Robust Feature Scaling Statistics
- [x] Task: Implement safe scaler checks in `preprocessor/preprocessor.py` to prevent `AttributeError` from unfitted StandardScalers in unsupervised duration mode.
- [x] Task: Ensure empty directories (such as phone-level directories when unsupervised duration modeling is active) do not crash feature normalizations.

## Phase 4: Verification & Execution
- [x] Task: Execute the full preprocessing pipeline using `preprocess.py --dataset DailyTalk`.
- [x] Task: Verify successful creation of `stats.json`, `train_frame.txt`, and `val_frame.txt`.
