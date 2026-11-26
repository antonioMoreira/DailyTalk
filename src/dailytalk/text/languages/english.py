import re

from dailytalk.text import cleaners
from dailytalk.text.symbols import symbols

_symbol_to_id = {s: i for i, s in enumerate(symbols)}
_id_to_symbol = dict(enumerate(symbols))
_curly_re = re.compile(r"(.*?)\{(.+?)\}(.*)")


class EnglishFrontend:
    """English language text frontend adapter using Tacotron cleaners and ARPAbet."""

    language_code = "en"

    def __init__(self, cleaner_names: list[str] | None = None):
        self.cleaner_names = cleaner_names or ["english_cleaners"]
        self.symbols = symbols
        self.symbol_to_id_map = _symbol_to_id
        self.id_to_symbol_map = _id_to_symbol

    def clean_text(self, text: str) -> str:
        for name in self.cleaner_names:
            cleaner = getattr(cleaners, name, None)
            if cleaner:
                text = cleaner(text)
        return text

    def phonemize(self, text: str) -> list[str]:
        # Using g2p_en if available, or basic word splitting
        try:
            from g2p_en import G2p

            g2p = G2p()
            phones = []
            words = filter(None, re.split(r"([,;.\-\?\!\s+])", text))
            for w in words:
                phones += list(filter(lambda p: p != " ", g2p(w)))
            return phones
        except Exception:
            return list(text)

    def text_to_sequence(self, text: str) -> list[int]:
        sequence = []
        while len(text):
            m = _curly_re.match(text)
            if not m:
                sequence += self._symbols_to_sequence(self.clean_text(text))
                break
            sequence += self._symbols_to_sequence(self.clean_text(m.group(1)))
            sequence += self._arpabet_to_sequence(m.group(2))
            text = m.group(3)
        return sequence

    def sequence_to_text(self, sequence: list[int]) -> str:
        result = ""
        for symbol_id in sequence:
            if symbol_id in self.id_to_symbol_map:
                s = self.id_to_symbol_map[symbol_id]
                if len(s) > 1 and s[0] == "@":
                    s = f"{{{s[1:]}}}"
                result += s
        return result.replace("}{", " ")

    def get_symbols(self) -> list[str]:
        return self.symbols

    def symbol_to_id(self, symbol: str) -> int:
        return self.symbol_to_id_map.get(symbol, 0)

    def id_to_symbol(self, symbol_id: int) -> str:
        return self.id_to_symbol_map.get(symbol_id, "_")

    def _symbols_to_sequence(self, syms: list[str] | str) -> list[int]:
        return [
            self.symbol_to_id_map[s]
            for s in syms
            if s in self.symbol_to_id_map and s != "_" and s != "~"
        ]

    def _arpabet_to_sequence(self, text: str) -> list[int]:
        return self._symbols_to_sequence(["@" + s for s in text.split()])
