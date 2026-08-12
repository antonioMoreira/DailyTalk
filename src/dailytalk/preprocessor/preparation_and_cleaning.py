import os

import av
import numpy as np

from dailytalk.config_models import PreprocessConfig
from dailytalk.text import clean_text


def resample_audio_pyav(
    input_path: str,
    output_path: str | None = None,
    target_sr: int = 22050,
    max_wav_value: float = 32768.0,
) -> np.ndarray:
    """Load audio file using PyAV, resample to target_sr, normalize volume, and optionally write WAV."""
    container = av.open(input_path)
    stream = container.streams.audio[0]
    resampler = av.AudioResampler(format="s16", layout="mono", rate=target_sr)

    audio_frames = []
    for frame in container.decode(stream):
        resampled_frames = resampler.resample(frame)
        for rf in resampled_frames:
            audio_frames.append(rf.to_ndarray())

    for rf in resampler.resample(None):
        audio_frames.append(rf.to_ndarray())

    container.close()

    if not audio_frames:
        raise ValueError(f"No audio frames decoded from {input_path}")

    audio_data = np.concatenate(audio_frames, axis=1).squeeze()
    wav = audio_data.astype(np.float32)

    max_val = np.max(np.abs(wav))
    if max_val > 0:
        wav = wav / max_val * max_wav_value

    wav_int16 = wav.astype(np.int16)

    if output_path is not None:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        out_container = av.open(output_path, mode="w")
        out_stream = out_container.add_stream("pcm_s16le", rate=target_sr)
        out_stream.layout = "mono"

        frame = av.AudioFrame.from_ndarray(
            wav_int16.reshape(1, -1), format="s16", layout="mono"
        )
        frame.rate = target_sr

        for packet in out_stream.encode(frame):
            out_container.mux(packet)
        for packet in out_stream.encode(None):
            out_container.mux(packet)
        out_container.close()

    return wav


def prepare_raw_data(config: PreprocessConfig) -> dict[str, int]:
    """Clean text transcripts and resample raw audio files into .lab and .wav files."""
    in_dir = config.path.corpus_path
    sub_dir = config.path.sub_dir_name
    out_dir = config.path.raw_path
    sampling_rate = config.preprocessing.audio.sampling_rate
    max_wav_value = config.preprocessing.audio.max_wav_value
    cleaners = config.preprocessing.text.text_cleaners
    language = config.preprocessing.text.language

    data_dir = os.path.join(in_dir, sub_dir)
    processed_count = 0

    if not os.path.exists(data_dir):
        return {"processed_count": 0}

    for turn_name in os.listdir(data_dir):
        turn_path = os.path.join(data_dir, turn_name)
        if not os.path.isdir(turn_path):
            continue

        for file_name in os.listdir(turn_path):
            if not file_name.endswith(".wav"):
                continue

            base_name = file_name[:-4]
            text_path = os.path.join(turn_path, f"{base_name}.txt")
            wav_path = os.path.join(turn_path, f"{base_name}.wav")

            if not os.path.exists(text_path):
                continue

            with open(text_path, encoding="utf-8") as f:
                text = f.readline().strip("\n")

            cleaned_text = clean_text(text, cleaners, language=language)

            out_turn_dir = os.path.join(out_dir, sub_dir, turn_name)
            os.makedirs(out_turn_dir, exist_ok=True)

            out_wav_path = os.path.join(out_turn_dir, f"{base_name}.wav")
            resample_audio_pyav(
                input_path=wav_path,
                output_path=out_wav_path,
                target_sr=sampling_rate,
                max_wav_value=max_wav_value,
            )

            out_lab_path = os.path.join(out_turn_dir, f"{base_name}.lab")
            with open(out_lab_path, "w", encoding="utf-8") as f:
                f.write(cleaned_text)

            processed_count += 1

    return {"processed_count": processed_count}
