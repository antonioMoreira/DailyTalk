# DailyTalk Preprocessing Pipeline Flows

This document contains the ASCII flow diagrams for each stage of the `src/dailytalk/preprocess.py` pipeline.

---

## 1. Stage 1: Preparation (`stage=preparation`)

```
[ CLI Call: python -m dailytalk.preprocess --dataset DailyTalk --stage preparation ]
                                     |
                                     v
                  +------------------------------------+
                  |      get_configs_of("DailyTalk")   |
                  +------------------------------------+
                                     |
                                     v
           +---------------------------------------------------+
           | PreprocessorPipeline.run_stage_1_preparation()    |
           +---------------------------------------------------+
                                     |
                                     v
                       +---------------------------+
                       |    prepare_raw_data()     |
                       +---------------------------+
                                     |
    +--------------------------------+--------------------------------+
    | INPUT PATH:                                                     |
    | config.path.raw_corpus_path / config.path.sub_dir_name          |
    | (Raw audio: *.wav, Raw text: *.txt)                             |
    +--------------------------------+--------------------------------+
                                     |
                                     v
            +-------------------------------------------------+
            | STEP 1: Text Transcript Cleaning                |
            | - Reads raw text from *.txt                     |
            | - Applies clean_text() (english_cleaners)       |
            |   (lower, unidecode, number expansion, etc.)    |
            +-------------------------------------------------+
                                     |
                                     v
            +-------------------------------------------------+
            | STEP 2: PyAV Audio Resampling & Normalization   |
            | - Calls resample_audio_pyav() to 22050 Hz       |
            | - Peak volume normalization (max = 32768.0)     |
            | - Converts to 16-bit PCM waveform array         |
            +-------------------------------------------------+
                                     |
                                     v
    +--------------------------------+--------------------------------+
    | OUTPUT PATH:                                                    |
    | config.path.intermediate_data_path / config.path.sub_dir_name   |
    | - Cleaned transcript files: {dialog_id}/{basename}.lab          |
    | - Resampled 22.05kHz audio: {dialog_id}/{basename}.wav         |
    +--------------------------------+--------------------------------+
                                     |
                                     v
    [ Terminal Output: "Stage 1 (Preparation & Cleaning) complete: {'processed_count': N}" ]
```

---

## 2. Stage 3: Features (`stage=features`)

```
[ CLI Call: python -m dailytalk.preprocess --dataset DailyTalk --stage features ]
                                     |
                                     v
                  +------------------------------------+
                  |      get_configs_of("DailyTalk")   |
                  +------------------------------------+
                                     |
                                     v
          +-----------------------------------------------------+
          | PreprocessorPipeline.run_stage_3_feature_extraction()|
          +-----------------------------------------------------+
                                     |
    +--------------------------------+--------------------------------+
    | INPUT PATH:                                                     |
    | config.path.intermediate_data_path / config.path.sub_dir_name   |
    | ({dialog_id}/{basename}.wav & .lab)                             |
    +--------------------------------+--------------------------------+
                                     |
                                     v
              +---------------------------------------------+
              | AcousticFeatureExtractor                    |
              | Iterates through utterances in subdirectories|
              +---------------------------------------------+
                                     |
                                     v
            +-------------------------------------------------+
            | STEP 1: Spectral & Mel Extraction               |
            | - Torch STFT: 1024 win/fft, 256 hop             |
            | - Extract STFT magnitude & 80-channel Mel spec  |
            +-------------------------------------------------+
                                     |
                                     v
            +-------------------------------------------------+
            | STEP 2: Pitch & Energy Extraction               |
            | - PyWorld/librosa F0 pitch contour extraction   |
            | - L2 norm energy curve from STFT magnitude      |
            | - Phone-level averaging (if phoneme_level mode) |
            +-------------------------------------------------+
                                     |
                                     v
            +-------------------------------------------------+
            | STEP 3: Dataset Statistics Computation          |
            | - Aggregates pitch & energy across dataset      |
            | - Computes min, max, mean, std stats            |
            +-------------------------------------------------+
                                     |
                                     v
    +--------------------------------+--------------------------------+
    | OUTPUT PATH:                                                    |
    | config.path.preprocessed_data_path                              |
    | - mel_frame/{basename}.npy  (80-channel Mel spectrograms)      |
    | - pitch/{basename}.npy      (F0 contours / phone averages)   |
    | - energy/{basename}.npy     (Energy curves / phone averages) |
    | - stats.json                (Normalization metrics)         |
    +--------------------------------+--------------------------------+
                                     |
                                     v
    [ Terminal Output: "Stage 3 (Feature Extraction) complete: PipelineResult(...)" ]
```

---

## 3. Stage 4: Speaker Embedding (`stage=speaker`)

```
[ CLI Call: python -m dailytalk.preprocess --dataset DailyTalk --stage speaker ]
                                     |
                                     v
                  +------------------------------------+
                  |      get_configs_of("DailyTalk")   |
                  +------------------------------------+
                                     |
                                     v
          +-----------------------------------------------------+
          | PreprocessorPipeline.run_stage_4_speaker_embedding()|
          +-----------------------------------------------------+
                                     |
    +--------------------------------+--------------------------------+
    | CHECK CONFIG:                                                   |
    | If config.preprocessing.speaker_embedder == "none"              |
    | -> Return 0 embeddings                                          |
    +--------------------------------+--------------------------------+
                                     |
                                     v (If speaker_embedder != "none")
    +--------------------------------+--------------------------------+
    | INPUT PATH:                                                     |
    | config.path.intermediate_data_path / config.path.sub_dir_name   |
    | ({dialog_id}/{basename}.wav)                                    |
    +--------------------------------+--------------------------------+
                                     |
                                     v
            +-------------------------------------------------+
            | SpeakerEmbedder (DeepSpeaker Model)             |
            | - Lazy loads Keras/DeepSpeaker model checkpoint |
            |   (deep_speaker_model.h5)                       |
            +-------------------------------------------------+
                                     |
                                     v
            +-------------------------------------------------+
            | STEP 1: Audio Loading & Resampling              |
            | - Reads intermediate audio via PyAV             |
            | - Ensures 22.05 kHz sampling rate               |
            +-------------------------------------------------+
                                     |
                                     v
            +-------------------------------------------------+
            | STEP 2: Neural Embedding Inference              |
            | - Executes predict_embedding()                  |
            | - Outputs 512-dimensional speaker vector        |
            +-------------------------------------------------+
                                     |
                                     v
    +--------------------------------+--------------------------------+
    | OUTPUT PATH:                                                    |
    | config.path.preprocessed_data_path / spker_embed                |
    | - {basename}-spker_embed.npy  (512-dim float32 NumPy arrays)   |
    +--------------------------------+--------------------------------+
                                     |
                                     v
    [ Terminal Output: "Stage 4 (Speaker Embedding Extraction) complete: N embeddings" ]
```
