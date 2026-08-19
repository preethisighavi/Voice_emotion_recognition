# Submission README

## Track Chosen

Track C — Tiny Model/Eval.

## What I Built

A voice emotion classifier (RandomForest on MFCC + pitch + energy) trained on RAVDESS via `voice_emotion_recognition.ipynb` in Colab, plus a Streamlit app with two tabs: **Try it** (pick or upload a clip, see the prediction next to a pitch/energy heuristic) and **Model eval** (accuracy, confusion matrix, per-emotion recall, disagreements). Model: 54.6% vs. heuristic's 32.5% (8 classes, 12.5% chance). Built to show whether the complexity earns its keep, not just report a number.

## Who It Is For

Someone deciding if a simple heuristic is "good enough" before reaching for a trained model — the disagreement table makes that tradeoff concrete.

## Data Or Source Used

[RAVDESS](https://zenodo.org/records/1188976) speech audio (1,440 clips, 24 actors, 8 emotions), CC BY-NC-SA 4.0 by Livingstone & Russo (2018). Included directly in `data/raw/`, redistributed under that same license. 8 sample clips also ship in `sample-data/` for an instant demo.

## Assumptions I Made

- Held out 4 actors (2M/2F) entirely for testing, not a random row split — a row split would leak the same voice into train and test.
- Heuristic uses only mean pitch + RMS energy — an honest baseline, not a strawman.

## Data Issues Or Caveats I Noticed

- Actors are professional, reading the same two scripted sentences — accuracy here is an upper bound, not a field estimate.
- `neutral` has half the samples of other classes and the model's worst recall.
- 54.6% is solid for 8-way, speaker-independent, classical-feature classification, but nowhere near production-grade.

## What I Would Do Next

Add a confidence threshold ("uncertain" vs. a forced guess) and check whether errors cluster by actor/gender rather than emotion — that would point to speaker normalization, not a bigger model.
