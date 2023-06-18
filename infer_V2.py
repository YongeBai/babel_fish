import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Tuple

from TTS.api import TTS
import numpy as np
import sounddevice as sd
import soundfile as sf
import torch

from util import is_silent, transcribe_audio, translate_text, update_config, uromanize

def record_audio(dataset_path: str, num_generated: int) -> str:
    audio_blocks = []
    fs = 16000  # Sample rate (Hz)
    initial_block_duration = 3  # Longer first block
    first_block = int(fs * initial_block_duration)

    print("Recording...")

    recording = sd.rec(first_block, samplerate=fs, channels=1, dtype=np.int16)
    audio_blocks.append(recording)
    sd.wait()

    block_duration = 0.5
    num_blocks = int(fs * block_duration)
    while True:
        recording = sd.rec(num_blocks, samplerate=fs, channels=1, dtype=np.int16)
        audio_blocks.append(recording)
        sd.wait()

        if is_silent(recording):
            print("Done recording")
            break

    audio = np.concatenate(audio_blocks)
    filename = f"{dataset_path}/wavs/recording_{num_generated}.wav"
    sf.write(filename, audio, fs)
    return os.getcwd() + "/" + filename

def format_dataset(dataset_path: str, path_to_recording: str, transcript: str) -> None:

    audio_filename = os.path.basename(path_to_recording).split(".")[0]
    transcript_filename = audio_filename + ".txt"
    romanized_transcript = uromanize(transcript).replace("\n"," ")

    with open(os.path.join(dataset_path, transcript_filename), "w") as f:
        f.write(f"{audio_filename}|{romanized_transcript}\n")

def train_model(config_path: str, restore_path: str, coqpit: dict):
    train_tts_path = "./.venv/lib64/python3.10/site-packages/TTS/bin/train_tts.py"

    update_config(config_path, coqpit)

    with open(config_path, "r") as f:
        updated_config = json.load(f)

    subprocess.run(
        ["python", train_tts_path,
        "--config", config_path,
        "--restore_path", restore_path],
        check=True,
        stdout=subprocess.DEVNULL
        )
    output_path = updated_config.get("output_path")
    run_name = updated_config.get("run_name")

    # Generate the path to the trained model
    trained_model_path = os.path.join(output_path, run_name, "best_model.pth")
    config_path = os.path.join(output_path, run_name, "config.json")
    return trained_model_path, config_path

def synthesize_speech(text: str, model_path: str, config_path: str, output_path: str):
    subprocess.run(
        ["tts", 
         "--text", text,
         "--model_path", model_path,
         "--config_path", config_path,
         "--out_path", output_path],
        check=True,
        stdout=subprocess.DEVNULL
        )

    audio_data, samplerate = sf.read(output_path)

    sd.stop()
    sd.play(audio_data, samplerate)
    print("\nPlaying audio")
    sd.wait()

if __name__ == "__main__":
    num_generated = 0
    DATASET_PATH = "dataset"
    OUTPUT_PATH = "cloned_recordings"
    while True:
        try:
            path_to_recording = record_audio(DATASET_PATH, num_generated)
            transcript = transcribe_audio(path_to_recording)
            print(f"Transcript: {transcript}")
            translated_text = translate_text(transcript)
            print(f"Translation: {translated_text}")

            format_dataset(DATASET_PATH, path_to_recording, transcript)
            path_to_model, path_to_configs = train_model(
                config_path="./tts_models/tts_models--en--ljspeech--glow-tts/config.json",
                restore_path="./tts_models/tts_models--en--ljspeech--glow-tts/model_file.pth",
                coqpit = {
                    "output_path": "./tts_models/trained_models",
                    "phoneme_cache_path": "./tts_models/phoneme_cache_path",
                    "run_name": "finetune",
                    "run_eval": False,
                    "eval_split_max_size": 1,
                    "eval_split_size": 1,
                    "epochs": 1,
                    "batch_size": 1,
                    "lr": 10e-8,
                    "save_checkpoints": False,
                    "datasets": [
                        {   
                            "name": "dataset",
                            "path": "./dataset",
                            "meta_file_train": f"recording_{num_generated}.txt",
                            "formatter": "ljspeech",
                        }
                                ]
                        }

                        )
            output = OUTPUT_PATH + f"/output_{num_generated}.wav"
            synthesize_speech(translated_text, path_to_model, path_to_configs, output)
            num_generated += 1

        except KeyboardInterrupt:
            print("\nExiting")
            break
