import numpy as np
import torch

from dailytalk.config_models import PreprocessConfig


class SpeakerEmbedder:
    """Stage 4: DeepSpeaker 512-dimensional speaker identity vector extractor."""

    def __init__(
        self,
        checkpoint_path: str = "deepspeaker/checkpoints/deep_speaker_model.h5",
        use_cuda: bool = False,
    ) -> None:
        self.checkpoint_path = checkpoint_path
        self.use_cuda = use_cuda and torch.cuda.is_available()
        self.model = None

    @classmethod
    def from_config(cls, config: PreprocessConfig) -> SpeakerEmbedder:
        ckpt_path = getattr(config.path, "speaker_embedder_path", "deepspeaker/checkpoints/deep_speaker_model.h5")
        use_cuda = config.preprocessing.speaker_embedder_cuda
        return cls(checkpoint_path=ckpt_path, use_cuda=use_cuda)

    def _lazy_init_model(self) -> None:
        if self.model is None:
            from dailytalk.deepspeaker.embedding import build_model
            self.model = build_model(self.checkpoint_path)

    def extract_embedding(self, audio: np.ndarray, sample_rate: int = 22050) -> np.ndarray:
        """Extract 512-dimensional speaker identity embedding from audio array."""
        self._lazy_init_model()
        from dailytalk.deepspeaker.embedding import predict_embedding
        embedding = predict_embedding(
            self.model, audio, sr=sample_rate, cuda=self.use_cuda
        )
        return np.squeeze(embedding).astype(np.float32)

    def extract_embedding_from_file(self, wav_path: str, sample_rate: int = 22050) -> np.ndarray:
        """Extract speaker embedding directly from audio file path using PyAV/Librosa."""
        from dailytalk.preprocessor.preparation_and_cleaning import resample_audio_pyav
        audio = resample_audio_pyav(wav_path, target_sr=sample_rate)
        return self.extract_embedding(audio, sample_rate=sample_rate)
