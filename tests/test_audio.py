import torch

from dailytalk.audio.audio_processing import dynamic_range_compression, dynamic_range_decompression
from dailytalk.audio.stft import STFT, TacotronSTFT


def test_stft_forward_inverse():
    # Setup stft module
    filter_length = 1024
    hop_length = 256
    win_length = 1024

    stft = STFT(filter_length, hop_length, win_length)

    # Create dummy 1D audio signal (1 batch, 16000 samples)
    dummy_audio = torch.randn(1, 16000)

    # Forward transform
    magnitude, phase = stft.transform(dummy_audio)

    assert magnitude.shape[0] == 1
    assert magnitude.shape[1] == (filter_length // 2 + 1)
    assert phase.shape == magnitude.shape

    # Inverse transform
    reconstructed = stft.inverse(magnitude, phase)

    # Check shape
    assert reconstructed.shape[0] == 1
    # Check that they are reasonably close (due to window boundary trims)
    assert abs(reconstructed.shape[2] - dummy_audio.shape[1]) <= filter_length


def test_tacotron_stft_mel():
    # Create TacotronSTFT
    t_stft = TacotronSTFT(
        filter_length=1024,
        hop_length=256,
        win_length=1024,
        n_mel_channels=80,
        sampling_rate=22050,
        mel_fmin=0,
        mel_fmax=8000
    )

    # Create dummy 1D audio in range [-1, 1]
    dummy_audio = torch.sin(torch.linspace(0, 100, 16000)).unsqueeze(0)

    # Extract mel-spectrogram and energy
    mel, energy = t_stft.mel_spectrogram(dummy_audio)

    assert mel.shape[0] == 1
    assert mel.shape[1] == 80
    assert energy.shape[0] == 1
    assert energy.shape[1] == mel.shape[2]


def test_dynamic_range_compression_decompression():
    # Test that compression and decompression are mathematical inverses
    x = torch.rand(2, 10, 10) * 100 + 1e-5
    compressed = dynamic_range_compression(x)
    decompressed = dynamic_range_decompression(compressed)

    assert torch.allclose(x, decompressed, rtol=1e-4, atol=1e-4)
