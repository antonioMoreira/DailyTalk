# Design Document: Unified Preprocessing Pipeline Architecture

## Component Design

### 1. `preparation_and_cleaning.py` (Stage 1)
High-performance PyAV (`av`) audio resampler and text transcript initializer.
```python
import av
import numpy as np
from dailytalk.config_models import PreprocessConfig

def resample_audio_pyav(input_path: str, output_path: str, target_sr: int, max_wav_value: float = 32768.0) -> np.ndarray:
    """Resample audio file to target_sr using PyAV and write output WAV."""
    ...

def prepare_raw_data(config: PreprocessConfig) -> dict[str, int]:
    """Clean text transcripts and resample raw audio files into .lab and .wav files."""
    ...
```

### 2. `language_frontend.py` (Stage 2)
Language-driven G2P, text cleaner, and symbol mapping interface.
```python
from dailytalk.text import LanguageFrontend, get_language_frontend

class LanguageFrontendProcessor:
    def __init__(self, language: str = "en", cleaner_names: list[str] | None = None) -> None:
        self.frontend: LanguageFrontend = get_language_frontend(language, cleaner_names)

    def clean_text(self, text: str) -> str: ...
    def text_to_sequence(self, text: str) -> list[int]: ...
    def sequence_to_text(self, sequence: list[int]) -> str: ...
```

### 3. `feature_extractor.py` (Stage 3)
Acoustic feature extraction engine (STFT Mel, PyWorld Pitch F0, Energy C2, Sentence Embeddings, Durations).
```python
import numpy as np
from dailytalk.config_models import PreprocessConfig, ModelConfig

class AcousticFeatureExtractor:
    def __init__(self, preprocess_config: PreprocessConfig, model_config: ModelConfig) -> None: ...
    def extract_mel(self, audio: np.ndarray) -> tuple[np.ndarray, np.ndarray]: ...
    def extract_pitch(self, audio: np.ndarray) -> np.ndarray: ...
    def extract_energy(self, stft_mag: np.ndarray) -> np.ndarray: ...
    def compute_normalization_stats(self, pitch_list: list[np.ndarray], energy_list: list[np.ndarray], duration_list: list[np.ndarray]) -> dict[str, Any]: ...
```

### 4. `speaker_embedder.py` (Stage 4)
DeepSpeaker 512-dim identity vector generator.
```python
import numpy as np

class SpeakerEmbedder:
    def __init__(self, model_path: str | None = None, use_cuda: bool = False) -> None: ...
    def extract_embedding(self, audio_path: str) -> np.ndarray: ...
```

### 5. `pipeline.py` (Orchestrator)
Unified orchestrator managing execution across all stages.
```python
from pydantic import BaseModel
from typing import Literal

class PipelineResult(BaseModel):
    dataset: str
    total_processed: int
    train_count: int
    val_size: int
    stats_saved: bool

class PreprocessorPipeline:
    def __init__(self, preprocess_config: PreprocessConfig, model_config: ModelConfig, train_config: TrainConfig) -> None: ...
    def run_stage(self, stage: Literal["all", "preparation", "text", "features", "speaker"]) -> PipelineResult: ...
```
