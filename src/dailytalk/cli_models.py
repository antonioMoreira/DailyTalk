from typing import Literal

from pydantic import BaseModel, Field


class TrainArgs(BaseModel):
    """Pydantic model representing training command-line arguments."""

    dataset: str = Field(..., description="Name of dataset")
    use_amp: bool = Field(False, description="Use automatic mixed precision")
    restore_step: int = Field(0, description="Step to restore checkpoint from")


class PreprocessArgs(BaseModel):
    """Pydantic model representing preprocessing command-line arguments."""

    dataset: str = Field(..., description="Name of dataset")


class PrepareAlignArgs(BaseModel):
    """Pydantic model representing alignment preparation command-line arguments."""

    dataset: str = Field(..., description="Name of dataset")


class SynthesizeArgs(BaseModel):
    """Pydantic model representing speech synthesis command-line arguments."""

    restore_step: int = Field(..., description="Step number of checkpoint to restore")
    mode: Literal["batch", "single"] = Field(
        ..., description="Synthesize a whole dataset ('batch') or a single sentence ('single')"
    )
    dataset: str = Field(..., description="Name of dataset")
    source: str | None = Field(
        None, description="Path to a source file with format like train.txt, for batch mode"
    )
    text: str | None = Field(
        None, description="Raw text to synthesize, for single-sentence mode"
    )
    speaker_id: str = Field(
        "p225", description="Speaker ID for multi-speaker synthesis, for single mode"
    )
    emotion_id: str = Field(
        "happiness", description="Emotion ID for multi-emotion synthesis, for single mode"
    )
    pitch_control: float = Field(
        1.0, description="Control pitch of whole utterance"
    )
    energy_control: float = Field(
        1.0, description="Control energy of whole utterance"
    )
    duration_control: float = Field(
        1.0, description="Control speaking rate speed of whole utterance"
    )
