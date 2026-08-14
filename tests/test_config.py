import pytest
from pydantic import ValidationError

from dailytalk.config_models import PreprocessConfig
from dailytalk.utils.tools import get_configs_of


def test_load_configs_success():
    # Verify we can successfully load actual configurations from config/DailyTalk
    preprocess, model, train = get_configs_of("DailyTalk")

    assert preprocess.dataset == "DailyTalk"
    assert preprocess.path.sub_dir_name == "data"
    assert preprocess.preprocessing.audio.sampling_rate == 22050
    assert preprocess.preprocessing.text.language == "en"

    # Test dictionary-like access
    assert preprocess["dataset"] == "DailyTalk"
    assert preprocess["path"]["sub_dir_name"] == "data"
    assert preprocess["preprocessing"]["audio"]["sampling_rate"] == 22050

    assert model.block_type == "transformer"
    assert model.multi_speaker is True
    assert model["variance_predictor"]["filter_size"] == 256

    assert train.seed == 1234
    assert train.optimizer.batch_size == 16
    assert train["optimizer"]["batch_size"] == 16


def test_invalid_config_raises_validation_error():
    # Check that malformed dict raises validation error
    with pytest.raises(ValidationError):
        PreprocessConfig.model_validate({
            "dataset": "DailyTalk",
            "path": {
                "raw_corpus_path": "/path/to/corpus",
                # missing sub_dir_name, lexicon_path, intermediate_data_path, preprocessed_data_path
            },
            "preprocessing": {}
        })
