import pytest

from dailytalk.text import (
    EnglishFrontend,
    PortugueseFrontend,
    get_language_frontend,
    sequence_to_text,
    text_to_sequence,
)


def test_english_frontend():
    frontend = get_language_frontend("en", cleaner_names=["english_cleaners"])
    assert isinstance(frontend, EnglishFrontend)
    assert frontend.language_code == "en"

    # Test cleaning
    cleaned = frontend.clean_text("Hello $100 and Dr. Smith.")
    assert "one hundred dollars" in cleaned
    assert "doctor smith" in cleaned

    # Test text_to_sequence and sequence_to_text
    seq = frontend.text_to_sequence("Hello world.")
    assert len(seq) > 0
    assert isinstance(seq[0], int)

    reconstructed = frontend.sequence_to_text(seq)
    assert isinstance(reconstructed, str)


def test_portuguese_frontend():
    frontend = get_language_frontend("pt")
    assert isinstance(frontend, PortugueseFrontend)
    assert frontend.language_code == "pt"

    # Test Portuguese number expansion
    cleaned_num = frontend.clean_text("Temos 150 alunos.")
    assert "cento e cinquenta" in cleaned_num

    # Test Portuguese abbreviation expansion
    cleaned_abbr = frontend.clean_text("O Dr. Silva disse que vc viria.")
    assert "doutor" in cleaned_abbr
    assert "você" in cleaned_abbr

    # Test sequence conversion with diacritics
    text = "Olá, tudo bem com você?"
    seq = frontend.text_to_sequence(text)
    assert len(seq) > 0

    # Ensure symbols map exists and contains Portuguese characters
    symbols = frontend.get_symbols()
    assert "ã" in symbols
    assert "ç" in symbols
    assert "ê" in symbols


def test_factory_language_normalization():
    en_frontend = get_language_frontend("en-US")
    assert isinstance(en_frontend, EnglishFrontend)

    pt_frontend = get_language_frontend("pt-BR")
    assert isinstance(pt_frontend, PortugueseFrontend)

    with pytest.raises(ValueError) as exc_info:
        get_language_frontend("fr")
    assert "Unsupported language" in str(exc_info.value)


def test_backward_compatibility_functions():
    seq_en = text_to_sequence("Testing backward compatibility.", ["english_cleaners"], language="en")
    assert len(seq_en) > 0

    seq_pt = text_to_sequence("Testando compatibilidade.", [], language="pt")
    assert len(seq_pt) > 0

    rec_en = sequence_to_text(seq_en, language="en")
    assert isinstance(rec_en, str)
