from typer.testing import CliRunner

from dailytalk.cli_models import SynthesizeArgs
from dailytalk.synthesize import app, preprocess_english, read_lexicon
from dailytalk.utils.tools import get_configs_of

runner = CliRunner()


def test_read_lexicon(tmp_path):
    lex_file = tmp_path / "test_lexicon.txt"
    lex_file.write_text("hello HH AH0 L OW1\nworld W ER1 L D\n", encoding="utf-8")

    lexicon = read_lexicon(str(lex_file))
    assert "hello" in lexicon
    assert lexicon["hello"] == ["HH", "AH0", "L", "OW1"]
    assert "world" in lexicon
    assert lexicon["world"] == ["W", "ER1", "L", "D"]


def test_preprocess_english():
    preprocess_config, _, _ = get_configs_of("DailyTalk")
    sequence = preprocess_english("Hello world!", preprocess_config)
    assert len(sequence) > 0


def test_synthesize_args_validation():
    args = SynthesizeArgs(
        restore_step=900000,
        mode="batch",
        dataset="DailyTalk",
        source="val_frame.txt",
        text=None,
    )
    assert args.restore_step == 900000
    assert args.mode == "batch"
    assert args.dataset == "DailyTalk"


def test_synthesize_cli_invalid_mode():
    result = runner.invoke(app, ["--restore_step", "900000", "--mode", "invalid_mode", "--dataset", "DailyTalk"])
    assert result.exit_code != 0
