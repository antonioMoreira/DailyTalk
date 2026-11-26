import os

import numpy as np

from dailytalk.preprocessor import (
    AcousticFeatureExtractor,
    LanguageFrontendProcessor,
    PipelineResult,
    PreprocessorPipeline,
    resample_audio_pyav,
)


def test_resample_audio_pyav(tmp_path):
    # Generate synthetic 16kHz sine wave audio
    sr_in = 16000
    sr_out = 22050
    t = np.linspace(0, 0.5, int(sr_in * 0.5), endpoint=False)
    sine = (0.5 * np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16)

    # Save initial input wav via av or scipy
    import av
    in_file = os.path.join(tmp_path, "input.wav")
    out_file = os.path.join(tmp_path, "resampled.wav")

    out_container = av.open(in_file, mode="w")
    out_stream = out_container.add_stream("pcm_s16le", rate=sr_in)
    out_stream.layout = "mono"
    frame = av.AudioFrame.from_ndarray(sine.reshape(1, -1), format="s16", layout="mono")
    frame.rate = sr_in

    for packet in out_stream.encode(frame):
        out_container.mux(packet)
    for packet in out_stream.encode(None):
        out_container.mux(packet)
    out_container.close()

    # Test PyAV resampler function
    resampled_wav = resample_audio_pyav(
        input_path=in_file,
        output_path=out_file,
        target_sr=sr_out,
    )

    assert os.path.exists(out_file)
    assert len(resampled_wav) > 0
    assert isinstance(resampled_wav, np.ndarray)


def test_language_frontend_processor():
    processor = LanguageFrontendProcessor(language="en", cleaner_names=["english_cleaners"])
    cleaned = processor.clean_text("Testing 100 dollars.")
    assert "one hundred dollars" in cleaned

    seq = processor.text_to_sequence(cleaned)
    assert len(seq) > 0

    pt_processor = LanguageFrontendProcessor(language="pt")
    cleaned_pt = pt_processor.clean_text("Preço é R$ 150.")
    assert "cento e cinquenta" in cleaned_pt


from dailytalk.utils.tools import get_configs_of


def test_acoustic_feature_extractor():
    preprocess_config, model_config, _ = get_configs_of("DailyTalk")
    extractor = AcousticFeatureExtractor(preprocess_config, model_config)

    audio = (np.random.randn(22050) * 0.1).astype(np.float32)
    mel, stft_mag = extractor.extract_mel(audio)
    assert mel.shape[0] == preprocess_config.preprocessing.mel.n_mel_channels
    assert stft_mag.shape[0] == mel.shape[1]

    pitch_frame, _ = extractor.extract_pitch(audio)
    assert len(pitch_frame) > 0

    energy_frame, _ = extractor.extract_energy(stft_mag)
    assert len(energy_frame) > 0

    stats = extractor.compute_normalization_stats([pitch_frame], [energy_frame])
    assert "pitch" in stats
    assert "energy" in stats


def test_preprocessor_pipeline_orchestrator():
    preprocess_config, model_config, train_config = get_configs_of("DailyTalk")
    pipeline = PreprocessorPipeline(preprocess_config, model_config, train_config)

    assert pipeline.dataset == preprocess_config.dataset
    res = PipelineResult(dataset="DailyTalk", processed_count=10, train_count=8, val_count=2)
    assert res.dataset == "DailyTalk"
    assert res.processed_count == 10
