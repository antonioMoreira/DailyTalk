from dailytalk.text.languages.base import LanguageFrontend
from dailytalk.text.languages.english import EnglishFrontend
from dailytalk.text.languages.portuguese import PortugueseFrontend


def get_language_frontend(
    language: str = "en",
    cleaner_names: list[str] | None = None,
) -> LanguageFrontend:
    """Factory function to retrieve language-specific frontend adapter.

    Args:
        language: Language code or name (e.g. 'en', 'english', 'pt', 'pt-BR', 'portuguese').
        cleaner_names: Optional cleaner function names for English frontend.

    Returns:
        LanguageFrontend instance matching the specified language.
    """
    normalized = language.lower().replace("_", "-").split("-")[0]
    if normalized in ("en", "english"):
        return EnglishFrontend(cleaner_names=cleaner_names)
    elif normalized in ("pt", "portuguese"):
        return PortugueseFrontend()
    else:
        raise ValueError(
            f"Unsupported language: '{language}'. Supported languages: 'en' (English), 'pt' (Portuguese)."
        )
