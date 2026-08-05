from dailytalk.text import sequence_to_text, sil_phonemes_ids, text_to_sequence
from dailytalk.text.symbols import symbols


def test_text_to_sequence_and_back():
    text = "Hello, world!"
    cleaners = ["english_cleaners"]
    seq = text_to_sequence(text, cleaners)

    assert isinstance(seq, list)
    assert len(seq) > 0
    assert all(isinstance(i, int) for i in seq)

    reconstructed = sequence_to_text(seq)
    # The cleaned text should contain symbols from English pronunciation cleaners
    assert len(reconstructed) > 0


def test_sil_phonemes_ids():
    sil_ids = sil_phonemes_ids()
    assert isinstance(sil_ids, list)
    assert len(sil_ids) > 0
    assert all(isinstance(i, int) for i in sil_ids)


def test_symbols_list():
    assert isinstance(symbols, list)
    assert len(symbols) > 0
    assert "sil" in symbols or "sp" in symbols or " " in symbols
