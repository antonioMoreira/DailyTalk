# Preprocessing Pipeline

```
    ┌─────────────────────────────────────────────────────────────────────────────┐
    │                       FULL PREPROCESSING PIPELINE                           │
    │                       ───────────────────────────                           │
    │  1. Stage 1: Preparation & Cleaning                                         │
    │     ├── src/dailytalk/prepare_align.py                                      │
    │     └── src/dailytalk/preprocessor/dailytalk.py                             │
    │         └── Cleans text (.lab), resamples audio (22.05kHz .wav).            │
    │                                                                             │
    │  2. Stage 2: Language Frontend & Text Processing                            │
    │     └── src/dailytalk/text/                                                 │
    │         └── Language adapters (English / Portuguese), G2P, symbol maps.     │
    │                                                                             │
    │  3. Stage 3: Feature Extraction Engine (The Core File)                      │
    │     └── src/dailytalk/preprocessor/preprocessor.py                          │
    │         ├── Mel-spectrogram calculation (TacotronSTFT)                      │
    │         ├── Pitch (F0) extraction (PyWorld Harvest)                         │
    │         ├── Energy (C2) calculation                                         │
    │         ├── TextGrid alignment parsing / Beta-binomial duration priors      │
    │         ├── Dialogue sentence embeddings (SentenceTransformer)              │
    │         └── Normalization (stats.json, train.txt, val.txt, speakers.json)   │
    │                                                                             │
    │  4. Auxiliary: Speaker Embedding Extraction                                 │
    │     └── src/dailytalk/deepspeaker/                                          │
    │         └── Extracts 512-dim DeepSpeaker identity vectors (spker_embed.npy) │
    │                                                                             │
    └─────────────────────────────────────────────────────────────────────────────┘

src/dailytalk/preprocessor/
    ├── __init__.py                         # Exposes unified Preprocessor API
    ├── pipeline.py                         # Main Orchestrator / Pipeline Runner
    ├── preparation_and_cleaning.py         # Stage 1: Audio resampling (22.05kHz) & transcript lab preparation
    ├── language_frontend.py                # Stage 2: Language-driven G2P, cleaners & symbol sequence mapping
    ├── feature_extractor.py                # Stage 3: Mel STFT, PyWorld Pitch (F0), Energy (C2) & Durations
    └── speaker_embedder.py                 # Stage 4: DeepSpeaker 512-dim identity vector extraction
```

