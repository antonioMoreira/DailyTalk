# Brazilian Portuguese Preprocessing Analysis

This document evaluates the existing `text/` module structure of the DailyTalk repository and outlines the integration points for supporting Brazilian Portuguese (PT-BR) text preprocessing, normalization, and phonemization.

---

## 1. Existing `text/` Module Structure

The current text processing module is based on the Tacotron/FastSpeech2 text pipeline, designed primarily for English:

- **`text/__init__.py`**:
  - Provides the primary entry points: `text_to_sequence` (converts cleaned text to IDs) and `grapheme_to_phoneme` (converts text words into a list of phoneme strings using a `g2p` object).
  - Dynamically retrieves cleaners from `text/cleaners.py` using `getattr(cleaners, name)`.
- **`text/cleaners.py`**:
  - Defines various cleaning pipelines (e.g., `english_cleaners`, `basic_cleaners`, `transliteration_cleaners`).
  - `english_cleaners` applies abbreviation expansion (`expand_abbreviations`), number expansion (`expand_numbers` via `text.numbers.normalize_numbers`), and lowercasing.
- **`text/numbers.py`**:
  - English-specific number-to-words normalization using the `inflect` library.
- **`text/symbols.py`**:
  - Defines the allowed vocabulary/phoneme set (`symbols`).
  - Combines punctuation, characters, ARPAbet symbols (from `text/cmudict.py`), and Pinyin symbols (from `text/pinyin.py`).
- **`text/cmudict.py`**:
  - CMU Pronouncing Dictionary wrapper for English G2P.

---

## 2. Proposed Integration Points for PT-BR

To add PT-BR support while maintaining full backward compatibility with English pipelines:

### A. Automatic Cleaner Discovery
We will define PT-BR cleaners in a new module, `text/cleaners_ptbr.py`. To make them automatically accessible to `text/__init__.py`'s `_clean_text` function (which calls `getattr(cleaners, name)`):
- We will import our PT-BR cleaner(s) in `text/cleaners.py`:
  ```python
  from .cleaners_ptbr import ptbr_cleaners
  ```
- This ensures `getattr(cleaners, "ptbr_cleaners")` works flawlessly without modifying the dynamic dispatcher inside `text/__init__.py`.

### B. Language-Aware G2P Selection in Preprocessor
Currently, `preprocessor/preprocessor.py` hardcodes:
```python
from g2p_en import G2p
...
self.g2p = G2p()
```
We will modify the preprocessor to load the G2P based on `preprocess_config["preprocessing"]["text"]["language"]`:
- If language is `"pt"`, load our new custom pure-Python rule-based PT-BR G2P class: `g2p = G2P_PTBR()`.
- Else (default `"en"`), load `G2p()`.

### C. Phoneme Vocabulary Adaptation
We will update `text/symbols.py` to append Brazilian Portuguese IPA phonemes to the list of `symbols`. This allows the text-to-sequence conversion to correctly map PT-BR phonemes to unique integer IDs.

---

## 3. Benefits of this Design

1. **Non-Intrusive**: No legacy English modules or configurations are broken.
2. **Backward-Compatible**: English training remains fully functional and unaffected.
3. **Pure-Python & Robust**: By implementing a custom rule-based PT-BR G2P, we avoid installing external binaries (like `espeak-ng`), which can fail or be absent in remote/restricted GPU training environments.
