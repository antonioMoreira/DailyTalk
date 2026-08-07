from typing import Protocol, runtime_checkable


@runtime_checkable
class LanguageFrontend(Protocol):
    """Protocol for language-specific text processing & phonemization adapters."""

    language_code: str

    def clean_text(self, text: str) -> str:
        """Clean and normalize raw input text."""
        ...

    def phonemize(self, text: str) -> list[str]:
        """Convert cleaned text into phonemes/symbols."""
        ...

    def text_to_sequence(self, text: str) -> list[int]:
        """Convert raw text or phoneme string into symbol ID sequence."""
        ...

    def sequence_to_text(self, sequence: list[int]) -> str:
        """Convert symbol ID sequence back to text representation."""
        ...

    def get_symbols(self) -> list[str]:
        """Get list of valid symbols for this language."""
        ...

    def symbol_to_id(self, symbol: str) -> int:
        """Get numeric ID for a given symbol."""
        ...

    def id_to_symbol(self, symbol_id: int) -> str:
        """Get symbol string for a given numeric ID."""
        ...
