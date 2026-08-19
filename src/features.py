"""
Feature extraction for RAVDESS speech-emotion audio.

RAVDESS filenames encode their own labels, e.g. 03-01-06-01-02-01-12.wav:
  modality-vocalchannel-EMOTION-intensity-statement-repetition-actor
See https://zenodo.org/records/1188976 for the full filename convention.

Two feature sets are extracted per clip:
  - HEURISTIC_FEATURES: mean pitch (F0) and RMS energy only - a human-readable
    baseline (loud + high pitch != specific emotion, but arousal-ish).
  - MODEL_FEATURES: 40 MFCCs (mean + std) + pitch + energy - what the trained
    classifier uses.
"""

from pathlib import Path

import librosa
import numpy as np

EMOTION_MAP = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised",
}

SAMPLE_RATE = 22050


def parse_label(wav_path: Path) -> dict:
    parts = wav_path.stem.split("-")
    emotion_code = parts[2]
    actor = int(parts[6])
    return {
        "emotion": EMOTION_MAP[emotion_code],
        "actor": actor,
        "gender": "female" if actor % 2 == 0 else "male",
    }


def extract_features(wav_path: Path) -> dict:
    """Load one clip and return both the heuristic-baseline features and
    the fuller feature set used by the trained model."""
    y, sr = librosa.load(wav_path, sr=SAMPLE_RATE)
    y, _ = librosa.effects.trim(y, top_db=25)

    rms = float(np.mean(librosa.feature.rms(y=y)))

    # piptrack (STFT-magnitude pitch tracking) instead of the much slower,
    # more precise pyin - accurate enough for a pitch/energy heuristic and
    # ~50x faster, which matters at 1,440 files.
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr, fmin=75, fmax=600)
    idx = magnitudes.argmax(axis=0)
    frame_pitches = pitches[idx, np.arange(pitches.shape[1])]
    voiced_f0 = frame_pitches[frame_pitches > 0]
    mean_pitch = float(np.mean(voiced_f0)) if voiced_f0.size else 0.0
    pitch_std = float(np.std(voiced_f0)) if voiced_f0.size else 0.0

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    mfcc_mean = mfcc.mean(axis=1)
    mfcc_std = mfcc.std(axis=1)

    zcr = float(np.mean(librosa.feature.zero_crossing_rate(y)))

    features = {
        "rms_energy": rms,
        "mean_pitch": mean_pitch,
        "pitch_std": pitch_std,
        "zcr": zcr,
    }
    for i, v in enumerate(mfcc_mean):
        features[f"mfcc_mean_{i}"] = float(v)
    for i, v in enumerate(mfcc_std):
        features[f"mfcc_std_{i}"] = float(v)
    return features


HEURISTIC_FEATURES = ["mean_pitch", "rms_energy"]
MODEL_FEATURES = (
    ["mean_pitch", "pitch_std", "rms_energy", "zcr"]
    + [f"mfcc_mean_{i}" for i in range(40)]
    + [f"mfcc_std_{i}" for i in range(40)]
)
