# Tech Stack

## Programming Language
- **Python:** The primary language used for model implementation, data preprocessing, and experimentation.

## Deep Learning Frameworks
- **PyTorch (v1.7.0):** Used for the main TTS model implementation (CompTransTTS), training loops, and loss functions.
- **TensorFlow (v2.5.1):** Utilized for specific components or baseline comparisons (as indicated by the original project dependencies).

## Audio & Speech Processing
- **Librosa:** For audio loading, feature extraction (mel-spectrograms), and signal processing.
- **PyWorld:** A high-quality vocoder used for fundamental frequency (F0) and aperiodicity extraction.
- **SoundFile:** For reading and writing audio files with high precision.
- **HiFi-GAN:** The high-fidelity generative adversarial network used as the vocoder for waveform synthesis.

## Linguistic & Text Processing
- **g2p-en / pypinyin:** Libraries for Grapheme-to-Phoneme conversion (will be adapted/replaced for Brazilian Portuguese).
- **tgt (TextGrid Tools):** For handling Forced Alignment output (MFA) and phoneme-level timing.
- **inflect:** For text normalization (number-to-word conversion).

## Data Science & Visualization
- **NumPy & SciPy:** For numerical computation and advanced mathematical functions.
- **Pandas:** For dataset metadata management and structured data analysis.
- **Matplotlib:** For generating plots, including mel-spectrogram visualizations.
- **TensorBoard:** For logging training progress, loss curves, and audio samples.

## Utilities & Optimization
- **DeepSpeaker:** For extracting speaker embeddings in multi-speaker settings.
- **Einops:** For flexible and readable tensor operations.
- **PyYAML:** For managing configuration files (model, preprocess, train).
- **tqdm:** For progress bar visualization during long-running tasks.
