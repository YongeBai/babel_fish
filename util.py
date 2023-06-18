import json
import os
import re
import subprocess
import sys

import numpy as np
from deep_translator import GoogleTranslator as Translator
from TTS.config import load_config, register_config


CORES = str(os.cpu_count())
PATH_TO_MODEL = "models/ggml-base.bin"
PATH_TO_WHISPER = "./whisper.cpp"
PATH_TO_UROMAN = "./uroman/bin"
THRESHOLD = 35


def is_silent(audio: np.ndarray, threshold: int = THRESHOLD) -> bool:
    rms = np.sqrt(np.mean(np.square(audio)))
    return rms < threshold


def transcribe_audio(path_to_recording: str) -> str:
    try:
        subprocess.run(
            ["make", "main"],
            check=True,
            cwd=PATH_TO_WHISPER,
            stdout=subprocess.DEVNULL,
        )
        process = subprocess.Popen(
            [
                "./main",
                "-m", PATH_TO_MODEL,
                "-t", CORES,
                "-l", "auto",
                "-f", path_to_recording,
            ],
            cwd=PATH_TO_WHISPER,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        output = process.communicate()[0].decode("utf-8")
        return re.sub(
            r"\[\d{2}:\d{2}:\d{2}\.\d{3} --> \d{2}:\d{2}:\d{2}\.\d{3}\]", "", output
        ).strip()

    except subprocess.CalledProcessError as e:
        print(f"Error: {e.stderr.decode('utf-8')}")
        sys.exit(1)


def translate_text(text: str) -> str:
    source_language = "auto"
    target_language = "en"
    translation = Translator(source=source_language, target=target_language).translate(
        text
    )
    return translation


def uromanize(text: str) -> str:
    process = subprocess.run(
        ["perl", "uroman.pl"],
        check=True,
        input=text.encode("utf-8"),
        cwd=PATH_TO_UROMAN,
        stdout=subprocess.PIPE,
    )
    return process.stdout.decode("utf-8")


def update_config(config_path: str, coqpit: dict):
    config_base = load_config(config_path)
    config = register_config(config_base.model)
    config = load_config(config_path)

    for key, value in coqpit.items():
        setattr(config, key, value)

    with open(config_path, "w") as f:
        json.dump(config.to_dict(), f, indent=2)
