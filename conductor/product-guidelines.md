# Product Guidelines

## Documentation & Reporting
- **Academic Tone:** Maintain a formal, objective, and precise prose style in all documentation, mirroring the standards of peer-reviewed research papers in the speech synthesis field.
- **Precision in Terminology:** Use standardized technical terms (e.g., "non-autoregressive," "unsupervised duration modeling," "mel-spectrogram") consistently.
- **Detailed Citations:** Always cite the original DailyTalk paper and the MuPe Life Stories Dataset when referencing methodologies or training data.

## Evaluation & Metrics
- **Metric Standardization:** Report experimental results using a standardized set of tables. Essential metrics include Mean Opinion Score (MOS) for naturalness, Mel-Cepstral Distortion (MCD) for spectral accuracy, and specific conversational metrics as defined in the DailyTalk baseline.
- **Reproducibility Details:** Include exhaustive details for every experiment, such as hyperparameters, seed values, hardware used, and the specific version of the MuPe-derived dataset.
- **Visual & Audio Artifacts:** Accompany metric tables with comparative mel-spectrogram plots and links to synthesized audio samples to provide a holistic view of model performance.

## Linguistic Adaptation (Brazilian Portuguese)
- **PT-BR Validation:** Ensure all text normalization (numbers, abbreviations, punctuation) and G2P (grapheme-to-phoneme) rules align with standard Brazilian Portuguese linguistic conventions.
- **Accent & Dialect Documentation:** Explicitly document the regional focus of the training data from the MuPe dataset. If the model is multi-accent, describe the strategies used for accent embedding or conditioning.
- **Phonetic Fidelity:** Prioritize the accuracy of Brazilian Portuguese nasal vowels, diphthoms, and liquid consonants in the phoneme dictionary and alignment process.

## Research Ethics & Collaboration
- **Data Privacy:** Adhere to the terms of use for both the DailyTalk and MuPe Life Stories datasets, ensuring data privacy and ethical considerations are maintained.
- **Open Science:** Favor open-source tools and transparent methodologies to encourage community contributions and peer validation of the adapted pipeline.
