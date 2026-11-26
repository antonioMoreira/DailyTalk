import pytest
from pydantic import ValidationError
from typer.testing import CliRunner

from dailytalk.cli_models import (
    PrepareAlignArgs,
    PreprocessArgs,
    SynthesizeArgs,
    TrainArgs,
)
from dailytalk.preprocess import app as preprocess_app
from dailytalk.train import app as train_app

runner = CliRunner()


def test_train_args_validation():
    args = TrainArgs(dataset="DailyTalk", use_amp=True, restore_step=100)
    assert args.dataset == "DailyTalk"
    assert args.use_amp is True
    assert args.restore_step == 100

    # Test defaults
    default_args = TrainArgs(dataset="DailyTalk")
    assert default_args.use_amp is False
    assert default_args.restore_step == 0

    # Missing required field
    with pytest.raises(ValidationError):
        TrainArgs.model_validate({})


def test_preprocess_args_validation():
    args = PreprocessArgs(dataset="DailyTalk")
    assert args.dataset == "DailyTalk"


def test_prepare_align_args_validation():
    args = PrepareAlignArgs(dataset="DailyTalk")
    assert args.dataset == "DailyTalk"


def test_synthesize_args_validation():
    args = SynthesizeArgs(
        restore_step=50000,
        mode="single",
        dataset="DailyTalk",
        text="Hello world",
        speaker_id="p225",
        emotion_id="happiness",
    )
    assert args.restore_step == 50000
    assert args.mode == "single"
    assert args.dataset == "DailyTalk"
    assert args.text == "Hello world"
    assert args.pitch_control == 1.0


def test_typer_cli_help_commands():
    result = runner.invoke(train_app, ["--help"])
    assert result.exit_code == 0
    assert "--dataset" in result.stdout

    result = runner.invoke(preprocess_app, ["--help"])
    assert result.exit_code == 0
    assert "--dataset" in result.stdout
