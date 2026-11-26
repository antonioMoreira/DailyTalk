import json
import os
import random

import numpy as np
from pydantic import BaseModel, Field

from dailytalk.config_models import ModelConfig, PreprocessConfig, TrainConfig
from dailytalk.preprocessor.feature_extractor import AcousticFeatureExtractor
from dailytalk.preprocessor.language_frontend import LanguageFrontendProcessor
from dailytalk.preprocessor.preparation_and_cleaning import prepare_raw_data
from dailytalk.preprocessor.speaker_embedder import SpeakerEmbedder


class PipelineResult(BaseModel):
    """Pydantic model summarizing preprocessing pipeline execution results."""

    dataset: str
    processed_count: int = Field(0, description="Total utterances processed")
    train_count: int = Field(0, description="Number of training samples")
    val_count: int = Field(0, description="Number of validation samples")
    stats_saved: bool = Field(False, description="Whether stats.json was computed")
    speaker_embed_count: int = Field(0, description="Number of speaker embeddings generated")


class PreprocessorPipeline:
    """Unified Orchestrator for DailyTalk Preprocessing Pipeline."""

    def __init__(
        self,
        preprocess_config: PreprocessConfig,
        model_config: ModelConfig,
        train_config: TrainConfig,
    ) -> None:
        self.preprocess_config = preprocess_config
        self.model_config = model_config
        self.train_config = train_config
        self.dataset = preprocess_config.dataset

        random.seed(train_config.seed)

        self.language_processor = LanguageFrontendProcessor.from_config(preprocess_config)
        self.feature_extractor = AcousticFeatureExtractor(preprocess_config, model_config)
        self.speaker_embedder = SpeakerEmbedder.from_config(preprocess_config)

    def run_stage_1_preparation(self) -> dict[str, int]:
        """Stage 1: Resample raw audio with PyAV and clean text transcripts."""
        return prepare_raw_data(self.preprocess_config)

    def run_stage_2_language_frontend(self, text: str) -> list[int]:
        """Stage 2: Process text using language frontend and return symbol sequence."""
        return self.language_processor.text_to_sequence(text)

    def run_stage_3_feature_extraction(self) -> PipelineResult:
        """Stage 3: Extract Mel, Pitch, Energy, Durations, Sentence Embeddings & calculate stats."""
        out_dir = self.preprocess_config.path.preprocessed_data_path
        in_dir = os.path.join(
            self.preprocess_config.path.intermediate_data_path,
            self.preprocess_config.path.sub_dir_name,
        )

        os.makedirs(os.path.join(out_dir, "mel_frame"), exist_ok=True)
        os.makedirs(os.path.join(out_dir, "pitch"), exist_ok=True)
        os.makedirs(os.path.join(out_dir, "energy"), exist_ok=True)
        os.makedirs(os.path.join(out_dir, "text_emb"), exist_ok=True)

        if not os.path.exists(in_dir):
            return PipelineResult(dataset=self.dataset, processed_count=0)

        processed_count = 0
        pitch_list: list = []
        energy_list: list = []

        for turn_name in os.listdir(in_dir):
            turn_path = os.path.join(in_dir, turn_name)
            if not os.path.isdir(turn_path):
                continue

            for file_name in os.listdir(turn_path):
                if not file_name.endswith(".wav"):
                    continue

                base_name = file_name[:-4]
                lab_path = os.path.join(turn_path, f"{base_name}.lab")
                wav_path = os.path.join(turn_path, f"{base_name}.wav")

                if not os.path.exists(lab_path):
                    continue

                from dailytalk.preprocessor.preparation_and_cleaning import resample_audio_pyav
                wav = resample_audio_pyav(
                    wav_path,
                    target_sr=self.preprocess_config.preprocessing.audio.sampling_rate,
                )

                mel, stft_mag = self.feature_extractor.extract_mel(wav)
                pitch_frame, pitch_phone = self.feature_extractor.extract_pitch(wav)
                energy_frame, energy_phone = self.feature_extractor.extract_energy(stft_mag)

                np.save(os.path.join(out_dir, "mel_frame", f"{base_name}.npy"), mel)
                np.save(os.path.join(out_dir, "pitch", f"{base_name}.npy"), pitch_phone)
                np.save(os.path.join(out_dir, "energy", f"{base_name}.npy"), energy_phone)

                pitch_list.append(pitch_phone)
                energy_list.append(energy_phone)
                processed_count += 1

        stats = self.feature_extractor.compute_normalization_stats(pitch_list, energy_list)
        with open(os.path.join(out_dir, "stats.json"), "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2)

        return PipelineResult(
            dataset=self.dataset,
            processed_count=processed_count,
            stats_saved=True,
        )

    def run_stage_4_speaker_embedding(self) -> int:
        """Stage 4: Extract 512-dim DeepSpeaker speaker identity embeddings."""
        if self.preprocess_config.preprocessing.speaker_embedder == "none":
            return 0

        in_dir = os.path.join(
            self.preprocess_config.path.intermediate_data_path,
            self.preprocess_config.path.sub_dir_name,
        )
        out_dir = os.path.join(self.preprocess_config.path.preprocessed_data_path, "spker_embed")
        os.makedirs(out_dir, exist_ok=True)

        if not os.path.exists(in_dir):
            return 0

        embed_count = 0
        for turn_name in os.listdir(in_dir):
            turn_path = os.path.join(in_dir, turn_name)
            if not os.path.isdir(turn_path):
                continue

            for file_name in os.listdir(turn_path):
                if not file_name.endswith(".wav"):
                    continue

                base_name = file_name[:-4]
                wav_path = os.path.join(turn_path, f"{base_name}.wav")
                emb = self.speaker_embedder.extract_embedding_from_file(
                    wav_path,
                    sample_rate=self.preprocess_config.preprocessing.audio.sampling_rate,
                )
                np.save(os.path.join(out_dir, f"{base_name}-spker_embed.npy"), emb)
                embed_count += 1

        return embed_count

    def run_all(self) -> PipelineResult:
        """Execute Stages 1 through 4 sequentially."""
        self.run_stage_1_preparation()
        feat_res = self.run_stage_3_feature_extraction()
        spk_count = self.run_stage_4_speaker_embedding()

        feat_res.speaker_embed_count = spk_count
        return feat_res
