"""Language-driven text processing and phonemization module."""
import re

from dailytalk.text import cleaners
from dailytalk.text.frontend import get_language_frontend
from dailytalk.text.languages.base import LanguageFrontend
from dailytalk.text.languages.english import EnglishFrontend
from dailytalk.text.languages.portuguese import PortugueseFrontend
from dailytalk.text.symbols import _silences, symbols

# Mappings from symbol to numeric ID and vice versa for backward compatibility:
_symbol_to_id = {s: i for i, s in enumerate(symbols)}
_id_to_symbol = dict(enumerate(symbols))

# Regular expression matching text enclosed in curly braces:
_curly_re = re.compile(r"(.*?)\{(.+?)\}(.*)")


def clean_text(text, cleaner_names, language="en"):
    """Clean text using language frontend."""
    frontend = get_language_frontend(language, cleaner_names)
    return frontend.clean_text(text)


_clean_text = clean_text


def text_to_sequence(text, cleaner_names, language="en"):
    """Converts a string of text to a sequence of IDs corresponding to symbols."""
    frontend = get_language_frontend(language, cleaner_names)
    return frontend.text_to_sequence(text)


def grapheme_to_phoneme(text, g2p):
    """Converts grapheme to phoneme with punctuation."""
    phones = []
    words = filter(None, re.split(r"([,;.\-\?\!\s+])", text))
    for w in words:
        phones += list(filter(lambda p: p != " ", g2p(w)))
    return phones


def sequence_to_text(sequence, language="en"):
    """Converts a sequence of IDs back to a string."""
    frontend = get_language_frontend(language)
    return frontend.sequence_to_text(sequence)


def sil_phonemes_ids():
    return [_symbol_to_id[sil] for sil in _silences if sil in _symbol_to_id]


__all__ = [
    "cleaners",
    "symbols",
    "clean_text",
    "_clean_text",
    "text_to_sequence",
    "sequence_to_text",
    "grapheme_to_phoneme",
    "sil_phonemes_ids",
    "get_language_frontend",
    "LanguageFrontend",
    "EnglishFrontend",
    "PortugueseFrontend",
]
