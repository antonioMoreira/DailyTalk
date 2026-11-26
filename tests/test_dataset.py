import json
import os

import numpy as np
import pytest

from dailytalk.dataset import Dataset
from dailytalk.text import text_to_sequence
from dailytalk.utils.tools import get_configs_of, get_variance_level


@pytest.fixture
def dummy_dataset_env(tmp_path):
    preprocessed_path = tmp_path / "preprocessed"
    raw_path = tmp_path / "raw"

    preprocessed_path.mkdir()
    raw_path.mkdir()

    # Create speakers.json
    speakers_file = preprocessed_path / "speakers.json"
    with open(speakers_file, "w") as f:
        json.dump({"spk1": 0}, f)

    # Create emotions.json
    emotions_file = preprocessed_path / "emotions.json"
    with open(emotions_file, "w") as f:
        json.dump({"happy": 0, "sad": 1}, f)

    # Create meta files (turn_speaker_dialog format, e.g. 0001_spk1_d0001)
    meta_file = preprocessed_path / "val_phoneme.txt"
    with open(meta_file, "w", encoding="utf-8") as f:
        f.write("0001_spk1_d0001|spk1|{H EH1 L OW1}|Hello|happy\n")

    # Load original configurations
    preprocess_config, model_config, train_config = get_configs_of("DailyTalk")

    # Customize paths to point to our temp directories
    preprocess_config.path.raw_corpus_path = str(raw_path)
    preprocess_config.path.intermediate_data_path = str(raw_path)
    preprocess_config.path.preprocessed_data_path = str(preprocessed_path)

    # Extract tags exactly how get_variance_level does
    pitch_tag, energy_tag, *_ = get_variance_level(preprocess_config, model_config)

    # Create subdirectory folders for pitch, energy, mel, duration, attn_prior under preprocessed_path
    (preprocessed_path / f"pitch_{pitch_tag}").mkdir()
    (preprocessed_path / f"energy_{energy_tag}").mkdir()
    (preprocessed_path / f"mel_{pitch_tag}").mkdir()
    (preprocessed_path / "duration").mkdir()
    (preprocessed_path / "attn_prior").mkdir()

    # Determine the exact text-to-sequence length to match shapes perfectly
    cleaners = preprocess_config.preprocessing.text.text_cleaners
    phone_len = len(text_to_sequence("{H EH1 L OW1}", cleaners))

    # Write dummy numpy files using the correct basename format
    # mel shape must be (mel_len, n_mel_channels), e.g. (100, 80)
    np.save(preprocessed_path / f"mel_{pitch_tag}" / "spk1-mel-0001_spk1_d0001.npy", np.zeros((100, 80), dtype=np.float32))
    np.save(preprocessed_path / f"pitch_{pitch_tag}" / "spk1-pitch-0001_spk1_d0001.npy", np.zeros((100,), dtype=np.float32))
    np.save(preprocessed_path / f"energy_{energy_tag}" / "spk1-energy-0001_spk1_d0001.npy", np.zeros((100,), dtype=np.float32))
    np.save(preprocessed_path / "duration" / "spk1-duration-0001_spk1_d0001.npy", np.ones((phone_len,), dtype=np.int32) * 20)
    np.save(preprocessed_path / "attn_prior" / "spk1-attn_prior-0001_spk1_d0001.npy", np.zeros((phone_len, 100), dtype=np.float32))

    # Create speaker embedder file if speaker_embedder != none
    if preprocess_config.preprocessing.speaker_embedder != 'none' or model_config.multi_speaker:
        (preprocessed_path / "spker_embed").mkdir()
        np.save(preprocessed_path / "spker_embed" / "spk1-spker_embed.npy", np.zeros((256,), dtype=np.float32))

    # Set up raw data subdirectory for dialog structure
    sub_dir_name = preprocess_config.path.sub_dir_name
    dialog_dir = raw_path / sub_dir_name / "0001"
    dialog_dir.mkdir(parents=True, exist_ok=True)
    with open(dialog_dir / "0001_spk1_d0001.wav", "w") as f:
        f.write("")

    return preprocess_config, model_config, train_config


def test_dataset_loading(dummy_dataset_env):
    preprocess_config, model_config, train_config = dummy_dataset_env

    # Disable history for standard test
    model_config.history_encoder.type = "none"

    # Load dataset
    dataset = Dataset(
        "val_phoneme.txt",
        preprocess_config,
        model_config,
        train_config,
        sort=False,
        drop_last=False
    )

    assert len(dataset) == 1

    # Retrieve first item
    item = dataset[0]
    assert item["id"] == "0001_spk1_d0001"
    assert item["speaker"] == 0  # mapped from spk1 -> 0
    assert item["raw_text"] == "Hello"
    assert item["emotion"] == 0  # mapped from happy -> 0
    assert isinstance(item["mel"], np.ndarray)
    assert isinstance(item["pitch"], np.ndarray)
    assert isinstance(item["energy"], np.ndarray)


def test_dataset_collation(dummy_dataset_env):
    preprocess_config, model_config, train_config = dummy_dataset_env

    # Disable history for standard test
    model_config.history_encoder.type = "none"

    # Load dataset
    dataset = Dataset(
        "val_phoneme.txt",
        preprocess_config,
        model_config,
        train_config,
        sort=False,
        drop_last=False
    )

    batch = [dataset[0]]
    collated = dataset.collate_fn(batch)

    # Verify collation outputs types and shapes
    assert isinstance(collated, list)
    first_batch = collated[0]
    assert first_batch[0][0] == "0001_spk1_d0001"
    assert isinstance(first_batch[2], np.ndarray)  # speakers array
    assert first_batch[2].item() == 0  # speaker mapped to id 0
    assert isinstance(first_batch[6], np.ndarray)  # mels array
    assert first_batch[6].shape[0] == 1  # batch size


def test_dataset_with_history(dummy_dataset_env):
    preprocess_config, model_config, train_config = dummy_dataset_env

    # Enable Guo history loading
    model_config.history_encoder.type = "Guo"

    # Create the text_emb directory and files
    preprocessed_path = preprocess_config.path.preprocessed_data_path
    text_emb_dir = os.path.join(preprocessed_path, "text_emb")
    os.makedirs(text_emb_dir, exist_ok=True)

    text_emb_size = model_config.history_encoder.text_emb_size
    np.save(
        os.path.join(text_emb_dir, "spk1-text_emb-0001_spk1_d0001.npy"),
        np.zeros((text_emb_size,), dtype=np.float32)
    )

    # Load dataset
    dataset = Dataset(
        "val_phoneme.txt",
        preprocess_config,
        model_config,
        train_config,
        sort=False,
        drop_last=False
    )

    assert len(dataset) == 1
    item = dataset[0]

    # Check that history is correctly populated
    assert item["history"] is not None
    assert "text_emb" in item["history"]
    assert "history_len" in item["history"]
    assert item["history"]["history_len"] == 1
