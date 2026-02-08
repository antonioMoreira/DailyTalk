# Specification: Brazilian Portuguese Text Preprocessing

## Context
The current DailyTalk pipeline uses `text/cleaners.py` (English-specific regex) and `text/cmudict.py` (English dictionary) for text processing. To support Brazilian Portuguese (PT-BR), we must adapt or replace these modules with language-appropriate tools. This track focuses on enabling the pipeline to accept PT-BR text input, normalize it (handle numbers, abbreviations, currency), and convert it to phonemes suitable for TTS model input.

## Requirements

### 1. Text Normalization
- Implement a `text/cleaners_ptbr.py` module.
- Support expansion of numbers (e.g., "123" -> "cento e vinte e três").
- Support expansion of common PT-BR abbreviations (e.g., "Sr." -> "Senhor", "Av." -> "Avenida").
- Handle specific punctuation and casing rules relevant to PT-BR.

### 2. Grapheme-to-Phoneme (G2P) Conversion
- Implement a `text/g2p_ptbr.py` module or integrate an external library (e.g., `phonemizer` with `espeak-ng` backend).
- Ensure the phoneme set is compatible with the model's symbol table or define a new symbol table.
- Ideally, output IPA (International Phonetic Alphabet) symbols to maintain consistency with modern TTS practices, or map to a simplified set if preferred.

### 3. Symbol Set Update
- Review `text/symbols.py`.
- Add necessary PT-BR phonemes if not already present.
- Ensure compatibility with existing special tokens (pads, eos).

### 4. Integration
- Update `preprocess.py` to allow selecting the PT-BR cleaner and G2P module via configuration or arguments.

## Constraints
- Must maintain the existing API structure where possible to minimize refactoring of `train.py` and `synthesize.py`.
- External dependencies (like `espeak-ng`) must be documented in `requirements.txt` or system setup instructions.

## Verification
- Unit tests for number expansion (e.g., verify "R$ 10,00" becomes "dez reais").
- Unit tests for phonemization (e.g., verify "casa" becomes /'ka.za/).
- End-to-end test: Run a sample PT-BR sentence through the full preprocessing pipeline and verify the output format matches the model's expected input (sequence of IDs).
