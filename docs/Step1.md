# Step 1 Report: Language-Driven Text Frontend & Phonemization

## Executive Summary

Step 1 of the DailyTalk pipeline modernization introduces a **Language-Driven Text Frontend** that:
- abstracts text normalization
- cleaning
- abbreviation expansion
- number conversion
- grapheme-to-phoneme (G2P) sequence generation 
behind a clean, unified protocol interface (`LanguageFrontend`).

While the original DailyTalk repository hardcoded English Tacotron cleaners and ARPAbet dictionary mappings, the modernized frontend decouples language-specific logic into modular language adapters. The pipeline currently ships with:
1. **`EnglishFrontend`**: Maintains **100% backward compatibility** with the original DailyTalk English dataset (`DailyTalk`).
2. **`PortugueseFrontend`**: Adds native, language-driven support for **Portuguese (`pt-BR`)**, including `num2words` number normalization, Portuguese abbreviation expansion, diacritic preservation, and SAMPA/IPA symbol mapping.

---

## Architecture & Design Patterns

### 1. Unified Language Protocol (`LanguageFrontend`)
Located in [`src/dailytalk/text/languages/base.py`](file:///home/antonio/Documents/MastersDegree/DailyTalk/src/dailytalk/text/languages/base.py), the `LanguageFrontend` protocol defines the standard contract required by all current and future language adapters:

```python
from typing import Protocol, runtime_checkable

@runtime_checkable
class LanguageFrontend(Protocol):
    language_code: str

    def clean_text(self, text: str) -> str: ...
    def phonemize(self, text: str) -> list[str]: ...
    def text_to_sequence(self, text: str) -> list[int]: ...
    def sequence_to_text(self, sequence: list[int]) -> str: ...
    def get_symbols(self) -> list[str]: ...
    def symbol_to_id(self, symbol: str) -> int: ...
    def id_to_symbol(self, symbol_id: int) -> str: ...
```

### 2. Dynamic Language Factory (`get_language_frontend`)
Located in [`src/dailytalk/text/frontend.py`](file:///home/antonio/Documents/MastersDegree/DailyTalk/src/dailytalk/text/frontend.py), the factory function dynamically instantiates the appropriate language adapter based on dataset or pipeline configuration parameters:

```python
from dailytalk.text.frontend import get_language_frontend

# English Frontend
en_frontend = get_language_frontend("en")

# Portuguese Frontend
pt_frontend = get_language_frontend("pt-BR")
```

---

## Language Adapters Comparison

| Feature | `EnglishFrontend` ([`english.py`](file:///home/antonio/Documents/MastersDegree/DailyTalk/src/dailytalk/text/languages/english.py)) | `PortugueseFrontend` ([`portuguese.py`](file:///home/antonio/Documents/MastersDegree/DailyTalk/src/dailytalk/text/languages/portuguese.py)) |
| :--- | :--- | :--- |
| **Language Code** | `"en"` | `"pt"` |
| **Number Normalization** | `inflect` / `numbers.py` (e.g., "$100" ➔ "one hundred dollars") | `num2words(lang="pt")` (e.g., "150" ➔ "cento e cinquenta") |
| **Abbreviation Expansion** | English (`Mr.`, `Mrs.`, `Dr.`, `St.`, `Co.`) | Portuguese (`Sr.`, `Sra.`, `Dr.`, `Dra.`, `Prof.`, `vc`, `tb`) |
| **Character & Diacritics** | ASCII + Tacotron Cleaners | Full Portuguese Alphabet (`á`, `à`, `ã`, `â`, `é`, `ê`, `í`, `ó`, `ô`, `õ`, `ú`, `ç`) |
| **Phonetic Alphabet** | ARPAbet (`@HH`, `@AW1`, `@S`, `@T`) | Portuguese SAMPA/IPA (`@a~`, `@e~`, `@i~`, `@o~`, `@u~`, `@S`, `@Z`, `@L`, `@N`) |

---

## File Modifications & Created Modules

1. **`src/dailytalk/text/languages/base.py`** *(New)*: Defines `LanguageFrontend` runtime protocol.
2. **`src/dailytalk/text/languages/english.py`** *(New)*: Encapsulates English cleaners, CMUDict, and ARPAbet sequence mappings.
3. **`src/dailytalk/text/languages/portuguese.py`** *(New)*: Implements Portuguese cleaner pipeline, `num2words` integration, and SAMPA/IPA symbol mapping.
4. **`src/dailytalk/text/languages/__init__.py`** *(New)*: Package exports for language adapters.
5. **`src/dailytalk/text/frontend.py`** *(New)*: Provides `get_language_frontend()` factory function.
6. **`src/dailytalk/text/__init__.py`** *(Updated)*: Exposes `get_language_frontend`, `clean_text`, `_clean_text`, and backwards-compatible `text_to_sequence` accepting optional `language="en"` parameter.
7. **`tests/test_text_frontend.py`** *(New)*: Complete unit test suite verifying English/Portuguese frontends, number expansions, diacritics, and factory resolution.

---

## Verification & Quality Gate Results

### 1. Unit Tests (`uv run pytest`)
All 22 unit tests across the workspace pass cleanly:
```bash
$ uv run pytest
============================= test session starts ==============================
platform linux -- Python 3.14.6, pytest-9.1.1, pluggy-1.6.0
collected 22 items

tests/test_audio.py ...                                                  [ 13%]
tests/test_cli_models.py .....                                           [ 36%]
tests/test_config.py ..                                                  [ 45%]
tests/test_dataset.py ...                                                [ 59%]
tests/test_deepspeaker.py ..                                             [ 68%]
tests/test_text.py ...                                                   [ 81%]
tests/test_text_frontend.py ....                                         [100%]

======================= 22 passed in 13.61s ========================
```

### 2. Static Type Checker (`uv run ty check`)
Static type checking returns zero errors:
```bash
$ uv run ty check
All checks passed!
```

---

## Next Steps

With **Step 1 (Language-Driven Text Frontend & Phonemization)** fully completed, tested, and documented, the project is ready to move to:
- **Step 2**: Audio Loading & Acoustic Feature Extraction (STFT, Pitch, Energy extraction with `pyav` and native PyTorch complex STFT).
