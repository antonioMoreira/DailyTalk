# Monkey-patch six to support PEP 451 under Python 3.14+
try:
    from importlib.machinery import ModuleSpec

    import six

    # pyrefly: ignore [missing-attribute]
    if not hasattr(six._SixMetaPathImporter, "find_spec"):

        def find_spec(self, fullname, path, target=None):
            if fullname in self.known_modules:
                return ModuleSpec(fullname, self, is_package=self.is_package(fullname))
            return None

        # pyrefly: ignore [missing-attribute]
        six._SixMetaPathImporter.find_spec = find_spec  # type: ignore
except Exception:
    pass

import json
import os
import re
from string import punctuation

import numpy as np
import torch
import typer
from g2p_en import G2p
from torch.utils.data import DataLoader

from dailytalk.cli_models import SynthesizeArgs
from dailytalk.dataset import TextDataset
from dailytalk.text import text_to_sequence
from dailytalk.utils.model import get_model, get_vocoder
from dailytalk.utils.tools import get_configs_of, synth_samples, to_device


def read_lexicon(lex_path):
    lexicon = {}
    with open(lex_path) as f:
        for line in f:
            temp = re.split(r"\s+", line.strip("\n"))
            word = temp[0]
            phones = temp[1:]
            if word.lower() not in lexicon:
                lexicon[word.lower()] = phones
    return lexicon


def preprocess_english(text, preprocess_config):
    text = text.rstrip(punctuation)
    lexicon = read_lexicon(preprocess_config["path"]["lexicon_path"])

    g2p = G2p()
    phones = []
    words = re.split(r"([,;.\-\?\!\s+])", text)
    for w in words:
        if w.lower() in lexicon:
            phones += lexicon[w.lower()]
        else:
            phones += list(filter(lambda p: p != " ", g2p(w)))
    phones = "{" + "}{".join(phones) + "}"
    phones = re.sub(r"\{[^\w\s]?\}", "{sp}", phones)
    phones = phones.replace("}{", " ")

    print(f"Raw Text Sequence: {text}")
    print(f"Phoneme Sequence: {phones}")
    sequence = np.array(
        text_to_sequence(
            phones, preprocess_config["preprocessing"]["text"]["text_cleaners"]
        )
    )

    return np.array(sequence)


def synthesize(device, model, args: SynthesizeArgs, configs, vocoder, batchs, control_values):
    preprocess_config, model_config, train_config = configs
    pitch_control, energy_control, duration_control = control_values

    for batch in batchs:
        batch = to_device(batch, device)
        with torch.no_grad():
            # Forward
            output = model(
                *(batch[2:-3]),
                spker_embeds=batch[-3],
                emotions=batch[-2],
                history_info=batch[-1],
                p_control=pitch_control,
                e_control=energy_control,
                d_control=duration_control,
            )
            synth_samples(
                batch,
                output,
                vocoder,
                model_config,
                preprocess_config,
                train_config["path"]["result_path"],
                args,
            )


app = typer.Typer(help="Synthesize speech with DailyTalk.")


@app.command()
def main(
    restore_step: int = typer.Option(..., "--restore_step", help="Step number of checkpoint to restore"),
    mode: str = typer.Option(..., "--mode", help="Synthesize a whole dataset ('batch') or a single sentence ('single')"),
    dataset: str = typer.Option(..., "--dataset", "-d", help="Name of dataset"),
    source: str | None = typer.Option(None, "--source", help="Path to source file for batch mode"),
    text: str | None = typer.Option(None, "--text", help="Raw text to synthesize for single mode"),
    speaker_id: str = typer.Option("p225", "--speaker_id", help="Speaker ID for single mode"),
    emotion_id: str = typer.Option("happiness", "--emotion_id", help="Emotion ID for single mode"),
    pitch_control: float = typer.Option(1.0, "--pitch_control", help="Control pitch of utterance"),
    energy_control: float = typer.Option(1.0, "--energy_control", help="Control energy of utterance"),
    duration_control: float = typer.Option(1.0, "--duration_control", help="Control speaking rate speed"),
):
    if mode not in ("batch", "single"):
        raise typer.BadParameter("mode must be 'batch' or 'single'")
    args = SynthesizeArgs(
        restore_step=restore_step,
        mode=mode,  # type: ignore
        dataset=dataset,
        source=source,
        text=text,
        speaker_id=speaker_id,
        emotion_id=emotion_id,
        pitch_control=pitch_control,
        energy_control=energy_control,
        duration_control=duration_control,
    )

    # Check source texts
    if args.mode == "batch":
        assert args.source is not None and args.text is None
    if args.mode == "single":
        assert args.source is None and args.text is not None

    # Read Config
    preprocess_config, model_config, train_config = get_configs_of(args.dataset)
    configs = (preprocess_config, model_config, train_config)
    os.makedirs(
        os.path.join(train_config["path"]["result_path"], str(args.restore_step)),
        exist_ok=True,
    )

    # Set Device
    torch.manual_seed(train_config["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed(train_config["seed"])
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print("Device of CompTransTTS:", device)

    # Get model
    model = get_model(args, configs, device, train=False)

    # Load vocoder
    vocoder = get_vocoder(model_config, device)

    # Preprocess texts
    if args.mode == "batch":
        # Get dataset
        text_dataset = TextDataset(args.source, preprocess_config, model_config)
        batchs = DataLoader(
            text_dataset,
            batch_size=8,
            collate_fn=text_dataset.collate_fn,
        )
    if args.mode == "single":
        assert model_config["history_encoder"]["type"] == "none", (
            "Single inference is not supported for conversational TTS, currently"
        )
        ids = raw_texts = [args.text[:100]]  # type: ignore

        # Speaker Info
        load_spker_embed = (
            model_config["multi_speaker"]
            and preprocess_config["preprocessing"]["speaker_embedder"] != "none"
        )
        with open(
            os.path.join(
                preprocess_config["path"]["preprocessed_path"], "speakers.json"
            )
        ) as f:
            speaker_map = json.load(f)
        speakers = (
            np.array([speaker_map[args.speaker_id]])
            if model_config["multi_speaker"]
            else np.array([0])
        )  # single speaker is allocated 0
        spker_embed = (
            np.load(
                os.path.join(
                    preprocess_config["path"]["preprocessed_path"],
                    "spker_embed",
                    f"{args.speaker_id}-spker_embed.npy",
                )
            )
            if load_spker_embed
            else None
        )

        # Emotion Info
        emotions = None
        if model_config["multi_emotion"]:
            with open(
                os.path.join(
                    preprocess_config["path"]["preprocessed_path"], "emotions.json"
                )
            ) as f:
                emotion_map = json.load(f)
            emotions = np.array([emotion_map[args.emotion_id]])

        if preprocess_config["preprocessing"]["text"]["language"] == "en":
            texts = np.array([preprocess_english(args.text, preprocess_config)])
        else:
            raise NotImplementedError
        text_lens = np.array([len(texts[0])])
        batchs = [
            (
                ids,
                raw_texts,
                speakers,
                texts,
                text_lens,
                max(text_lens),
                spker_embed,
                emotions,
            )
        ]

    control_values = args.pitch_control, args.energy_control, args.duration_control

    synthesize(device, model, args, configs, vocoder, batchs, control_values)


if __name__ == "__main__":
    app()
