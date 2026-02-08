# Initial Concept
This repository is a fork of the original [DailyTalk](https://github.com/keonlee9420/DailyTalk) project. The core objective is to leverage the DailyTalk methodology—including its rules for dialogue sampling and recording—to a Brazilian Portuguese context. The training data is a custom dataset developed from the [MuPe Life Stories Dataset](https://aclanthology.org/2025.coling-main.407.pdf), following the same rigorous standards established in the original DailyTalk paper.

# Product Definition

## Vision
To adapt the DailyTalk conversational Text-to-Speech (TTS) pipeline for Brazilian Portuguese using a custom-built dataset derived from the MuPe Life Stories Dataset. This project aims to deliver high-quality, context-aware speech synthesis tailored to the linguistic and conversational nuances of Brazilian Portuguese.

## Core Objectives
- **Brazilian Portuguese Support:** Fully adapt the pipeline (text normalization, g2p, alignment) for the Brazilian Portuguese language.
- **MuPe-based Dataset Integration:** Integrate the specialized dataset created from MuPe Life Stories, ensuring it adheres to the DailyTalk dialogue structure and attributes.
- **Context-Aware Training:** Train the non-autoregressive TTS models to effectively utilize historical dialogue information in a Portuguese conversational setting.
- **Evaluation and Benchmarking:** Evaluate the trained models using the metrics proposed in the DailyTalk paper to establish a baseline for conversational TTS in Brazilian Portuguese.
- **Research Extension:** Provide a platform for further research into multilingual and language-specific conversational speech synthesis.

## Target Audience
- **Portuguese Speech Researchers:** Scientists focused on conversational and expressive TTS for Brazilian Portuguese.
- **MuPe Dataset Users:** Researchers utilizing the MuPe Life Stories Dataset for various NLP and speech tasks.
- **AI Developers in Brazil:** Developers building advanced, natural-sounding conversational agents for the Brazilian market.
