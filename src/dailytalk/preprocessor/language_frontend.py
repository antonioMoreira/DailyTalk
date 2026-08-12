from dailytalk.config_models import PreprocessConfig
from dailytalk.text import LanguageFrontend, get_language_frontend


class LanguageFrontendProcessor:
    """Stage 2: Language Frontend processor for cleaning and converting text to symbol IDs."""

    def __init__(self, language: str = "en", cleaner_names: list[str] | None = None) -> None:
        self.language = language
        self.cleaner_names = cleaner_names
        self.frontend: LanguageFrontend = get_language_frontend(
            language=language, cleaner_names=cleaner_names
        )

    @classmethod
    def from_config(cls, config: PreprocessConfig) -> LanguageFrontendProcessor:
        return cls(
            language=config.preprocessing.text.language,
            cleaner_names=config.preprocessing.text.text_cleaners,
        )

    def clean_text(self, text: str) -> str:
        """Clean and normalize raw text according to language rules."""
        return self.frontend.clean_text(text)

    def text_to_sequence(self, text: str) -> list[int]:
        """Convert cleaned text to symbol ID sequence."""
        return self.frontend.text_to_sequence(text)

    def sequence_to_text(self, sequence: list[int]) -> str:
        """Convert symbol ID sequence back to string."""
        return self.frontend.sequence_to_text(sequence)

    def get_symbols(self) -> list[str]:
        """Retrieve symbol vocabulary for language."""
        return self.frontend.get_symbols()
