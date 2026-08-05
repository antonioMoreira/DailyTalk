import numpy as np
import torch

from dailytalk.deepspeaker.batcher import OneHotSpeakers
from dailytalk.deepspeaker.conv_models import DeepSpeakerModel


def test_deepspeaker_model_forward():
    model = DeepSpeakerModel()
    model.eval()

    # Input shape: (B, T, F, 1) or (B, T, F)
    # B = 2, T = 100 frames, F = 64 MFCC features
    dummy_input = torch.randn(2, 100, 64)

    with torch.no_grad():
        output = model(dummy_input)

    assert output.shape[0] == 2
    # DeepSpeaker produces 512-dimensional speaker embeddings
    assert output.shape[1] == 512
    # DeepSpeaker embeddings should be L2 normalized (norm of each vector should be 1.0)
    norms = torch.linalg.norm(output, dim=1)
    assert torch.allclose(norms, torch.ones_like(norms), rtol=1e-5, atol=1e-5)


def test_one_hot_speakers():
    speakers = ["spk1", "spk2", "spk3"]
    oh_speakers = OneHotSpeakers(speakers)

    assert oh_speakers.get_speaker_from_index(0) == "spk1"
    assert oh_speakers.get_speaker_from_index(1) == "spk2"
    assert oh_speakers.get_speaker_from_index(2) == "spk3"

    one_hot_1 = oh_speakers.get_one_hot("spk1")
    assert np.allclose(one_hot_1, np.array([1.0, 0.0, 0.0]))

    one_hot_2 = oh_speakers.get_one_hot("spk2")
    assert np.allclose(one_hot_2, np.array([0.0, 1.0, 0.0]))
