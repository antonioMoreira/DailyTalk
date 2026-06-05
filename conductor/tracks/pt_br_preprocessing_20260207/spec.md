# Specification: English Text Preprocessing & Alignment for DailyTalk

## Context
The goal is to execute the preprocessing and alignment steps for the original English DailyTalk dataset. The pipeline should use the English `g2p_en` grapheme-to-phoneme library and perform feature extraction (Log-Mel spectrograms, F0 pitch contours, signal energy, sentence-level transformer embeddings, and alignment priors) with unsupervised duration modeling (where `learn_alignment: True` is configured, bypassing supervised MFA/TextGrid alignments). The output files must be written under the `raw_data/preprocessed_data/DailyTalk` folder.

## Requirements

### 1. English Text Normalization & Phonemization
- Ensure that the text preprocessing pipeline correctly handles the original English transcripts from DailyTalk.
- Use `g2p_en` for English text-to-phoneme mapping.
- Automatically resolve all necessary NLTK corpora requirements (`averaged_perceptron_tagger_eng`) in a non-interactive manner.

### 2. Audio Loading Compatibility
- Ensure full compatibility of audio loading functions with modern Librosa APIs (e.g., using explicit parameter keywords like `sr=sampling_rate` instead of deprecated positional parameters).
- Correct window centering functions in STFT processing (`pad_center` with keyword parameter `size`).

### 3. Graceful Scaling and Stat Reduction
- Support unsupervised alignment flow when no supervised `.TextGrid` files exist.
- Update the preprocessing feature scaling/statistics step to gracefully handle unfitted phone-level scalers (e.g. `StandardScaler` objects for phone-level statistics which are not used in unsupervised training) without raising attribute errors.
- Ensure statistical calculations write successfully to `stats.json`, `train_frame.txt`, and `val_frame.txt`.

### 4. Path Organization
- Write all preprocessed data into `./raw_data/preprocessed_data/DailyTalk` and read raw data from the lowercase path `./raw_data/dailytalk`.

## Verification
- Preprocessing completes without errors and outputs all extracted features (mel spectrograms, pitch/energy contours, embeddings, priors) to `raw_data/preprocessed_data/DailyTalk/`.
- Verifying files are successfully split into training and validation list files.
