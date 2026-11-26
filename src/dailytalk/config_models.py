from typing import Any

from pydantic import BaseModel, ConfigDict


class ConfigModel(BaseModel):
    """Base class providing dictionary-like access to Pydantic models for backward compatibility."""
    model_config = ConfigDict(extra="allow")

    def __getitem__(self, item: str) -> Any:
        try:
            return getattr(self, item)
        except AttributeError:
            raise KeyError(item) from None

    def __contains__(self, item: str) -> bool:
        return hasattr(self, item)

    def get(self, item: str, default: Any = None) -> Any:
        return getattr(self, item, default)


# === preprocess.yaml models ===

class PathConfig(ConfigModel):
    raw_corpus_path: str
    sub_dir_name: str
    lexicon_path: str
    intermediate_data_path: str
    preprocessed_data_path: str

class TextPreprocessingConfig(ConfigModel):
    text_cleaners: list[str]
    language: str

class AudioPreprocessingConfig(ConfigModel):
    trim_top_db: int
    sampling_rate: int
    max_wav_value: float

class STFTPreprocessingConfig(ConfigModel):
    filter_length: int
    hop_length: int
    win_length: int

class MelPreprocessingConfig(ConfigModel):
    n_mel_channels: int
    mel_fmin: int
    mel_fmax: int | None = None

class FeaturePreprocessingConfig(ConfigModel):
    feature: str
    normalization: bool

class DurationPreprocessingConfig(ConfigModel):
    beta_binomial_scaling_factor: float

class PreprocessingConfig(ConfigModel):
    speaker_embedder: str
    speaker_embedder_cuda: bool
    val_size: int
    text: TextPreprocessingConfig
    audio: AudioPreprocessingConfig
    stft: STFTPreprocessingConfig
    mel: MelPreprocessingConfig
    pitch: FeaturePreprocessingConfig
    energy: FeaturePreprocessingConfig
    duration: DurationPreprocessingConfig

class PreprocessConfig(ConfigModel):
    dataset: str
    path: PathConfig
    preprocessing: PreprocessingConfig


# === model.yaml models ===

class DurationModelingConfig(ConfigModel):
    learn_alignment: bool
    aligner_temperature: float

class HistoryEncoderConfig(ConfigModel):
    type: str
    text_emb_size: int
    max_history_len: int
    duration_max: int
    modal_hidden: int
    modal_layer: int
    modal_head: int
    modal_kernel_size: int
    modal_dropout: float
    cma_hidden: int
    cma_layer: int
    cma_head: int
    cma_filter_size: int
    cma_kernel_size: list[int]
    cma_dropout: float
    context_layer: int
    context_hidden: int
    context_dropout: float

class TransformerConfig(ConfigModel):
    encoder_layer: int
    encoder_head: int
    encoder_hidden: int
    decoder_layer: int
    decoder_head: int
    decoder_hidden: int
    conv_filter_size: int
    conv_kernel_size: list[int]
    encoder_dropout: float
    decoder_dropout: float

class ConformerConfig(ConfigModel):
    encoder_layer: int
    encoder_head: int
    encoder_hidden: int
    decoder_layer: int
    decoder_head: int
    decoder_hidden: int
    feed_forward_expansion_factor: int
    conv_expansion_factor: int
    conv_kernel_size: int
    half_step_residual: bool
    encoder_dropout: float
    decoder_dropout: float

class VariancePredictorConfig(ConfigModel):
    filter_size: int
    kernel_size: int
    dropout: float
    cond_dur_layer: int
    cond_dur_head: int
    cond_dur_hidden: int
    conv_filter_size: int
    conv_kernel_size: list[int]
    cond_dur_dropout: float

class VarianceEmbeddingConfig(ConfigModel):
    kernel_size: int
    pitch_quantization: str
    energy_quantization: str
    n_bins: int

class VocoderConfig(ConfigModel):
    model: str
    speaker: str

class ModelConfig(ConfigModel):
    block_type: str
    external_speaker_dim: int
    duration_modeling: DurationModelingConfig
    history_encoder: HistoryEncoderConfig
    transformer: TransformerConfig
    conformer: ConformerConfig
    variance_predictor: VariancePredictorConfig
    variance_embedding: VarianceEmbeddingConfig
    multi_speaker: bool
    multi_emotion: bool
    max_seq_len: int
    vocoder: VocoderConfig


# === train.yaml models ===

class DistConfig(ConfigModel):
    dist_backend: str
    dist_url: str
    world_size: int

class TrainPathConfig(ConfigModel):
    ckpt_path: str
    log_path: str
    result_path: str

class OptimizerConfig(ConfigModel):
    batch_size: int
    betas: list[float]
    eps: float
    weight_decay: float
    grad_clip_thresh: float
    grad_acc_step: int
    warm_up_step: int
    anneal_steps: list[int]
    anneal_rate: float

class LossConfig(ConfigModel):
    dur_loss: str
    lambda_ph_dur: float
    lambda_word_dur: float
    lambda_sent_dur: float

class StepConfig(ConfigModel):
    total_step: int
    log_step: int
    synth_step: int
    val_step: int
    save_step: int
    var_start_steps: int

class TrainDurationConfig(ConfigModel):
    binarization_start_steps: int
    binarization_loss_enable_steps: int
    binarization_loss_warmup_steps: int

class TrainConfig(ConfigModel):
    seed: int
    dist_config: DistConfig
    path: TrainPathConfig
    optimizer: OptimizerConfig
    loss: LossConfig
    step: StepConfig
    duration: TrainDurationConfig
