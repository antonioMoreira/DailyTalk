import re

from num2words import num2words

# Portuguese symbols set
_pad = "_"
_punctuation = "!'(),.:;? "
_special = "-"
_letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyzáàãâéêíóôõúçÁÀÃÂÉÊÍÓÔÕÚÇ"
_silences = ["@sp", "@spn", "@sil"]

# Portuguese IPA / SAMPA phonetic symbols
_portuguese_phonemes = [
    "@a", "@e", "@i", "@o", "@u",
    "@b", "@d", "@f", "@g", "@k", "@l", "@m", "@n", "@p", "@s", "@t", "@v", "@z",
    "@S", "@Z", "@L", "@N", "@R", "@J",
    "@an", "@en", "@in", "@on", "@un",
    "@e~", "@i~", "@o~", "@u~", "@a~"
]

portuguese_symbols = (
    [_pad]
    + list(_special)
    + list(_punctuation)
    + list(_letters)
    + _portuguese_phonemes
    + _silences
)

_symbol_to_id = {s: i for i, s in enumerate(portuguese_symbols)}
_id_to_symbol = dict(enumerate(portuguese_symbols))

_whitespace_re = re.compile(r"\s+")
_number_re = re.compile(r"\d+")

_pt_abbreviations = [
    (re.compile(r"\bsr\.", re.IGNORECASE), "senhor"),
    (re.compile(r"\bsra\.", re.IGNORECASE), "senhora"),
    (re.compile(r"\bdr\.", re.IGNORECASE), "doutor"),
    (re.compile(r"\bdra\.", re.IGNORECASE), "doutora"),
    (re.compile(r"\bprof\.", re.IGNORECASE), "professor"),
    (re.compile(r"\bprofa\.", re.IGNORECASE), "professora"),
    (re.compile(r"\bpág\.", re.IGNORECASE), "página"),
    (re.compile(r"\bpágs\.", re.IGNORECASE), "páginas"),
    (re.compile(r"\betc\.", re.IGNORECASE), "etcétera"),
    (re.compile(r"\bvc\b", re.IGNORECASE), "você"),
    (re.compile(r"\btb\b", re.IGNORECASE), "também"),
    (re.compile(r"\bpq\b", re.IGNORECASE), "porque"),
]


class PortugueseFrontend:
    """Portuguese language text frontend adapter using num2words and Portuguese cleaners."""

    language_code = "pt"

    def __init__(self):
        self.symbols = portuguese_symbols
        self.symbol_to_id_map = _symbol_to_id
        self.id_to_symbol_map = _id_to_symbol

    def clean_text(self, text: str) -> str:
        # Expand abbreviations
        for regex, replacement in _pt_abbreviations:
            text = re.sub(regex, replacement, text)

        # Expand numbers to Portuguese words
        text = re.sub(_number_re, lambda m: num2words(int(m.group(0)), lang="pt"), text)

        # Collapse whitespace
        text = re.sub(_whitespace_re, " ", text).strip()
        return text

    def phonemize(self, text: str) -> list[str]:
        # Clean text and split by character / phonemes
        cleaned = self.clean_text(text)
        return list(cleaned)

    def text_to_sequence(self, text: str) -> list[int]:
        cleaned = self.clean_text(text)
        return [
            self.symbol_to_id_map[s]
            for s in cleaned
            if s in self.symbol_to_id_map and s != "_" and s != "~"
        ]

    def sequence_to_text(self, sequence: list[int]) -> str:
        result = []
        for symbol_id in sequence:
            if symbol_id in self.id_to_symbol_map:
                s = self.id_to_symbol_map[symbol_id]
                if s.startswith("@"):
                    s = f"{{{s[1:]}}}"
                result.append(s)
        return "".join(result)

    def get_symbols(self) -> list[str]:
        return self.symbols

    def symbol_to_id(self, symbol: str) -> int:
        return self.symbol_to_id_map.get(symbol, 0)

    def id_to_symbol(self, symbol_id: int) -> str:
        return self.id_to_symbol_map.get(symbol_id, "_")
