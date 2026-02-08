# Implementation Plan - Adapt text preprocessing and phonemization for Brazilian Portuguese

## Phase 1: Analysis & Dependency Selection
- [ ] Task: Evaluate current `text/` module structure and identify integration points for new language support.
- [ ] Task: Research and select the best G2P library for Brazilian Portuguese (e.g., `phonemizer` vs `g2p_en` adaptation).
    - [ ] Research: Compare accuracy and ease of integration.
    - [ ] Decision: Select library and document in `tech-stack.md`.
- [ ] Task: Conductor - User Manual Verification 'Analysis & Dependency Selection' (Protocol in workflow.md)

## Phase 2: Text Normalization Implementation
- [ ] Task: Implement `text/cleaners_ptbr.py`.
    - [ ] Sub-task: Create module file.
    - [ ] Sub-task: Implement number expansion logic (using `num2words` or similar).
    - [ ] Sub-task: Implement abbreviation expansion map.
    - [ ] Sub-task: Write unit tests for normalization.
- [ ] Task: Conductor - User Manual Verification 'Text Normalization Implementation' (Protocol in workflow.md)

## Phase 3: G2P & Symbol Adaptation
- [ ] Task: Implement `text/g2p_ptbr.py`.
    - [ ] Sub-task: specific wrapper for selected G2P library.
    - [ ] Sub-task: Ensure output format aligns with expected list of phonemes.
    - [ ] Sub-task: Write unit tests for G2P conversion.
- [ ] Task: Update `text/symbols.py` to include Brazilian Portuguese phonemes.
    - [ ] Sub-task: Define new symbol set if IPA is used.
    - [ ] Sub-task: Ensure backward compatibility or configuration switching.
- [ ] Task: Conductor - User Manual Verification 'G2P & Symbol Adaptation' (Protocol in workflow.md)

## Phase 4: Integration & Verification
- [ ] Task: Update `preprocess.py` (and potentially `config/DailyTalk/preprocess.yaml`) to use PT-BR modules.
    - [ ] Sub-task: Add configuration option for language/cleaner.
    - [ ] Sub-task: Connect new cleaner and G2P to the preprocessing pipeline.
- [ ] Task: Run end-to-end preprocessing test with sample MuPe data.
    - [ ] Sub-task: Create a small sample file.
    - [ ] Sub-task: Run `preprocess.py` on sample.
    - [ ] Sub-task: Verify output files (`.txt`, `.npy`, etc.) are generated correctly.
- [ ] Task: Conductor - User Manual Verification 'Integration & Verification' (Protocol in workflow.md)
