# Feature Specification: Training Pipeline Slicing & Language-Driven Study Plan

## Overview
This specification outlines a modular, language-driven Study Plan for modernizing the full training pipeline of **DailyTalk**. The pipeline is sliced into 8 distinct, self-contained steps. Each step is designed to be language-agnostic at its core while initially targeting English (`DailyTalk`), ensuring seamless adaptability to Portuguese (`pt-BR`) and other languages.

---

## Language Abstraction Strategy
To ensure the pipeline is robust for Portuguese and future multi-lingual datasets:
- **Language Adapters**: All text cleaners, number expanders, and phonemizers are encapsulated behind unified language interfaces (`LanguageAdapter` protocol).
- **Symbol Maps & Lexicons**: Phonetic alphabets (IPA / ARPAbet / Portuguese SAMPA), cleaners, and symbol vocabularies are dynamically loaded via dataset/language configs (`language: "en"` vs `language: "pt"`).
- **Audio & Feature Agnosticism**: Audio STFT, pitch/energy extraction, CTC aligner, and FastSpeech2 loss calculations remain 100% language-independent.

---

## 8-Step Pipeline Slicing Plan

```
Pipeline/
├── Step 1: Language-Driven Text Frontend & Phonemization
├── Step 2: Audio Loading & Acoustic Feature Extraction (STFT, Pitch, Energy)
├── Step 3: Alignment & Duration Model (CTC / MFA Aligner & Stats)
├── Step 4: Conversational & Speaker Conditioning (DeepSpeaker & Dialogue History)
├── Step 5: Dataset Engine, Collate Functions & Dynamic Batching
├── Step 6: Backbone Architecture & Modern Transformers (FlashAttention-2)
├── Step 7: Loss Functions, Fused Optimizers & Training Orchestration (AMP / W&B)
└── Step 8: Neural Vocoding, Evaluation & ONNX/TensorRT Model Export
```

---

## Requirements with Traceable IDs

### REQ-PIPE-001: Step 1 - Language-Driven Text Frontend & Phonemization
- **ID**: REQ-PIPE-001
- **Focus**: Text normalization, number expansion, G2P (Grapheme-to-Phoneme), and symbol maps.
- **English Implementation**: `g2p_en`, `inflect`, `cleaners.py`, `symbols.py` (ARPAbet/ASCII).
- **Portuguese Preparedness**: Interface for `g2p-pt` / `espeak-ng` (IPA / SAMPA), Portuguese number expander (`num2words`), accent/diacritic cleaners.
- **Deliverables**: Modular `dailytalk.text` frontend with dynamic language adapter loading.

### REQ-PIPE-002: Step 2 - Audio Loading & Acoustic Feature Extraction
- **ID**: REQ-PIPE-002
- **Focus**: Resampling, STFT Mel-spectrograms, Pitch ($F_0$) extraction, Energy ($C_2$).
- **Modernization**: Replace `pydub`/`librosa` loading with high-performance `pyav` / `soundfile`, native PyTorch complex STFT, and PyWorld / TorchAudio $F_0$ extraction.
- **Deliverables**: Language-agnostic, multi-threaded acoustic feature extraction pipeline.

### REQ-PIPE-003: Step 3 - Alignment & Duration Model
- **ID**: REQ-PIPE-003
- **Focus**: Phoneme-to-mel frame alignment, duration calculation, and feature normalization (`stats.json`).
- **Modernization**: CTC-based unsupervised aligner and TextGrid parser abstraction.
- **Deliverables**: Automated alignment duration extractor and variance feature statistical normalizer.

### REQ-PIPE-004: Step 4 - Conversational & Speaker Conditioning
- **ID**: REQ-PIPE-004
- **Focus**: Speaker verification embeddings (DeepSpeaker), emotion categories, and dialogue context encoder inputs.
- **Modernization**: Pure NumPy/PyTorch DeepSpeaker embedding extraction without TensorFlow runtime dependencies.
- **Deliverables**: Multi-speaker, multi-emotion, and dialogue turn history conditioning pipeline.

### REQ-PIPE-005: Step 5 - Dataset Engine, Collate Functions & Dynamic Batching
- **ID**: REQ-PIPE-005
- **Focus**: Metadata index generation (`train.txt`, `val.txt`, `speakers.json`, `emotions.json`), `Dataset` class, collate padding.
- **Modernization**: Pydantic validated dataset items, memory-efficient tensor collation, and dynamic sequence length sorting.
- **Deliverables**: Robust PyTorch `DataLoader` pipeline supporting arbitrary languages and multi-speaker datasets.

### REQ-PIPE-006: Step 6 - Backbone Architecture & Modern Transformers
- **ID**: REQ-PIPE-006
- **Focus**: FastSpeech2 Phoneme Encoder / Mel Decoder, Dialogue Context Encoder, Variance Adaptors.
- **Modernization**: Kernel fusion via PyTorch 2.x `torch.compile` and `F.scaled_dot_product_attention` (FlashAttention-2).
- **Deliverables**: High-performance, modern Transformer architecture preserving DailyTalk macro architecture.

### REQ-PIPE-007: Step 7 - Loss Functions, Fused Optimizers & Training Orchestration
- **ID**: REQ-PIPE-007
- **Focus**: `CompTransTTSLoss`, Scheduled AdamW optimizer, Distributed training (DDP / Accelerate / Lightning), Mixed Precision (BF16 / FP16).
- **Modernization**: Fused AdamW optimizer (`torch.optim.AdamW(fused=True)`), PyTorch AMP, Weights & Biases (W&B) logging for audio WAVs and alignment heatmaps.
- **Deliverables**: State-of-the-art multi-GPU training pipeline with real-time MLOps tracking.

### REQ-PIPE-008: Step 8 - Neural Vocoding, Evaluation & Model Export Engine
- **ID**: REQ-PIPE-008
- **Focus**: HiFi-GAN / BigVGAN vocoder integration, synthesis evaluation (`synthesize.py`), ONNX / TensorRT export.
- **Modernization**: `torch.onnx.export` with dynamic axes for FastSpeech2 + HiFi-GAN deployment on ONNX Runtime / TensorRT.
- **Deliverables**: End-to-end evaluation suite and deployment export engine.
