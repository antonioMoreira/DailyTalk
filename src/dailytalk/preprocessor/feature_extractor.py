
import numpy as np
import pyworld as pw
from scipy.stats import betabinom
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import StandardScaler

from dailytalk.audio import stft as AudioSTFT
from dailytalk.config_models import ModelConfig, PreprocessConfig
from dailytalk.utils.tools import get_phoneme_level_energy, get_phoneme_level_pitch


class AcousticFeatureExtractor:
    """Stage 3: Acoustic & Variance Feature Extraction Engine (Mel, Pitch, Energy, Durations, Embeddings)."""

    def __init__(self, preprocess_config: PreprocessConfig, model_config: ModelConfig) -> None:
        self.config = preprocess_config
        self.model_config = model_config
        self.sampling_rate = preprocess_config.preprocessing.audio.sampling_rate
        self.hop_length = preprocess_config.preprocessing.stft.hop_length
        self.filter_length = preprocess_config.preprocessing.stft.filter_length
        self.win_length = preprocess_config.preprocessing.stft.win_length
        self.n_mel_channels = preprocess_config.preprocessing.mel.n_mel_channels
        self.mel_fmin = preprocess_config.preprocessing.mel.mel_fmin
        self.mel_fmax = preprocess_config.preprocessing.mel.mel_fmax

        self.stft = AudioSTFT.TacotronSTFT(
            filter_length=self.filter_length,
            hop_length=self.hop_length,
            win_length=self.win_length,
            n_mel_channels=self.n_mel_channels,
            sampling_rate=self.sampling_rate,
            mel_fmin=self.mel_fmin,
            mel_fmax=self.mel_fmax,
        )

        self.pitch_phoneme_averaging = (
            preprocess_config.preprocessing.pitch.feature == "phoneme_level"
        )
        self.energy_phoneme_averaging = (
            preprocess_config.preprocessing.energy.feature == "phoneme_level"
        )
        self.pitch_normalization = preprocess_config.preprocessing.pitch.normalization
        self.energy_normalization = preprocess_config.preprocessing.energy.normalization
        self.beta_binomial_scaling_factor = (
            preprocess_config.preprocessing.duration.beta_binomial_scaling_factor
        )

        self._sentence_transformer: SentenceTransformer | None = None

    def get_sentence_transformer(self) -> SentenceTransformer:
        if self._sentence_transformer is None:
            self._sentence_transformer = SentenceTransformer("distiluse-base-multilingual-cased-v1")
        return self._sentence_transformer

    def extract_mel(self, audio: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Extract mel-spectrogram and STFT magnitudes using TacotronSTFT."""
        import torch

        max_val = float(np.max(np.abs(audio)))
        if max_val > 1.0:
            audio = audio / float(self.config.preprocessing.audio.max_wav_value)

        audio_tensor = torch.from_numpy(audio).float().unsqueeze(0)
        mel_tensor, energy_tensor = self.stft.mel_spectrogram(audio_tensor)
        mel = mel_tensor.squeeze(0).cpu().numpy()
        stft_mag = energy_tensor.squeeze(0).cpu().numpy()
        return mel, stft_mag

    def extract_pitch(
        self, audio: np.ndarray, duration: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Extract fundamental frequency (F0) pitch contour using PyWorld Harvest."""
        audio_double = audio.astype(np.float64)
        f0, _ = pw.harvest(  # type: ignore
            audio_double,
            self.sampling_rate,
            frame_period=self.hop_length / self.sampling_rate * 1000,
            f0_floor=71.0,
            f0_ceil=800.0,
        )
        pitch_frame = f0.astype(np.float32)

        if duration is not None and self.pitch_phoneme_averaging:
            pitch_phone = get_phoneme_level_pitch(duration, pitch_frame)
            return pitch_frame, pitch_phone
        return pitch_frame, pitch_frame

    def extract_energy(
        self, stft_mag: np.ndarray, duration: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Extract frame and phoneme-level energy contours from STFT magnitudes."""
        if stft_mag.ndim == 1:
            energy_frame = stft_mag.astype(np.float32)
        else:
            energy_frame = np.linalg.norm(stft_mag, axis=0).astype(np.float32)

        if duration is not None and self.energy_phoneme_averaging:
            energy_phone = get_phoneme_level_energy(duration, energy_frame)
            return energy_frame, energy_phone
        return energy_frame, energy_frame

    def extract_sentence_embedding(self, text: str) -> np.ndarray:
        """Extract 512-dim dialogue sentence embedding using SentenceTransformer."""
        model = self.get_sentence_transformer()
        embedding = model.encode(text)
        return np.array(embedding, dtype=np.float32)

    @staticmethod
    def calculate_beta_binomial_prior(
        phoneme_len: int, mel_len: int, scaling_factor: float = 1.0
    ) -> np.ndarray:
        """Calculate beta-binomial prior matrix for alignment duration estimation."""
        x = np.linspace(0, 1, phoneme_len)
        y = np.linspace(0, 1, mel_len)
        grid_x, grid_y = np.meshgrid(x, y, indexing="ij")
        prior = betabinom.pmf(
            np.round(grid_y * (mel_len - 1)),
            mel_len - 1,
            grid_x * scaling_factor + 1,
            (1 - grid_x) * scaling_factor + 1,
        )
        return prior.astype(np.float32)

    def compute_normalization_stats(
        self,
        pitch_list: list[np.ndarray],
        energy_list: list[np.ndarray],
    ) -> dict[str, list[float]]:
        """Compute mean and standard deviation for pitch and energy features."""
        pitch_scaler = StandardScaler()
        energy_scaler = StandardScaler()

        for p in pitch_list:
            nonzero_p = p[p > 0]
            if len(nonzero_p) > 0:
                pitch_scaler.partial_fit(nonzero_p.reshape(-1, 1))

        for e in energy_list:
            nonzero_e = e[e > 0]
            if len(nonzero_e) > 0:
                energy_scaler.partial_fit(nonzero_e.reshape(-1, 1))

        if hasattr(pitch_scaler, "mean_"):
            pitch_mean = float(np.asarray(pitch_scaler.mean_)[0])
            pitch_std = float(np.asarray(pitch_scaler.scale_)[0])
        else:
            pitch_mean = 0.0
            pitch_std = 1.0

        if hasattr(energy_scaler, "mean_"):
            energy_mean = float(np.asarray(energy_scaler.mean_)[0])
            energy_std = float(np.asarray(energy_scaler.scale_)[0])
        else:
            energy_mean = 0.0
            energy_std = 1.0

        return {
            "pitch": [pitch_mean, pitch_std],
            "energy": [energy_mean, energy_std],
        }
