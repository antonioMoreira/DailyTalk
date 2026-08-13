import matplotlib.pyplot as plt
import numpy as np
import torch

from dailytalk.config_models import ModelConfig, PreprocessConfig, TrainConfig
from dailytalk.utils.tools import (
    expand,
    get_configs_of,
    get_mask_from_lengths,
    get_phoneme_level_energy,
    get_phoneme_level_pitch,
    get_variance_level,
    pad,
    pad_1D,
    pad_2D,
    pad_3D,
    save_figure_to_numpy,
    to_device,
)


def test_get_configs_of():
    preprocess_config, model_config, train_config = get_configs_of("DailyTalk")
    assert isinstance(preprocess_config, PreprocessConfig)
    assert isinstance(model_config, ModelConfig)
    assert isinstance(train_config, TrainConfig)
    assert preprocess_config.dataset == "DailyTalk"


def test_get_variance_level():
    preprocess_config, model_config, _ = get_configs_of("DailyTalk")
    pitch_tag, energy_tag, pitch_level, energy_level = get_variance_level(
        preprocess_config, model_config, data_loading=True
    )
    assert pitch_tag in ["frame", "phone"]
    assert energy_tag in ["frame", "phone"]
    assert pitch_level in ["frame_level", "phoneme_level"]
    assert energy_level in ["frame_level", "phoneme_level"]


def test_get_phoneme_level_pitch_and_energy():
    duration = np.array([2, 3, 0, 1])
    pitch = np.array([100.0, 110.0, 120.0, 130.0, 140.0, 150.0])
    energy = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])

    ph_pitch = get_phoneme_level_pitch(duration, pitch)
    ph_energy = get_phoneme_level_energy(duration, energy)

    assert len(ph_pitch) == len(duration)
    assert len(ph_energy) == len(duration)
    assert ph_pitch[2] == 0.0
    assert ph_energy[2] == 0.0


def test_pad_1D():
    inputs = [np.array([1, 2]), np.array([3, 4, 5])]
    padded = pad_1D(inputs, PAD=0)
    assert padded.shape == (2, 3)
    assert np.array_equal(padded[0], np.array([1, 2, 0]))


def test_pad_2D():
    inputs = [np.zeros((2, 4)), np.zeros((3, 4))]
    padded = pad_2D(inputs)
    assert padded.shape == (2, 3, 4)


def test_pad_3D():
    inputs = [np.ones((2, 3)), np.ones((1, 2))]
    padded = pad_3D(inputs, B=2, T=3, L=3)
    assert padded.shape == (2, 3, 3)


def test_torch_pad():
    t1 = torch.tensor([1.0, 2.0])
    t2 = torch.tensor([3.0, 4.0, 5.0])
    padded = pad([t1, t2])
    assert padded.shape == (2, 3)
    assert torch.equal(padded[0], torch.tensor([1.0, 2.0, 0.0]))


def test_get_mask_from_lengths():
    lengths = torch.tensor([2, 4])
    mask = get_mask_from_lengths(lengths, max_len=4)
    assert mask.shape == (2, 4)
    assert torch.equal(mask[0], torch.tensor([False, False, True, True]))
    assert torch.equal(mask[1], torch.tensor([False, False, False, False]))


def test_expand():
    values = np.array([10, 20])
    durations = np.array([2, 3])
    expanded = expand(values, durations)
    assert np.array_equal(expanded, np.array([10, 10, 20, 20, 20]))


def test_save_figure_to_numpy():
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    img = save_figure_to_numpy(fig)
    plt.close(fig)
    assert isinstance(img, np.ndarray)
    assert img.ndim == 3
    assert img.shape[2] == 3


def test_to_device():
    # Test data length 9
    data_9 = (
        ["id1"],
        ["text1"],
        np.array([1]),
        np.array([[1, 2]]),
        np.array([2]),
        2,
        None,
        None,
        None,
    )
    res_9 = to_device(data_9, "cpu")
    assert isinstance(res_9[2], torch.Tensor)
    assert isinstance(res_9[3], torch.Tensor)
