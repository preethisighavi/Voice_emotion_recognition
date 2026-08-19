# AI Collaboration Note

## Did You Use AI?

Yes — Claude (in this CLI/agent environment) built the pipeline and app; I directed scope, dataset choice, and the modeling decisions.

## How You Used It

I picked the project direction: voice emotion recognition felt like a good fit because I wanted to build something that would actually show well as a live Streamlit app, not just a static notebook — building Streamlit dashboards is something I enjoy. When I asked Claude to find a small public dataset, I already had RAVDESS in mind as a strong candidate — 1,440 clips across 24 actors and 8 emotions is small enough to iterate on quickly but diverse enough for a real speaker-independent evaluation, rather than something with only 1-2 speakers. Claude wrote the feature extraction, training, and Streamlit code, but I made the calls on the things that actually determine whether the eval means anything: accuracy computed on held-out *actors* rather than random rows, restricting the heuristic baseline to features a person could eyeball (pitch + energy) so it's a fair floor and not a strawman, and framing the eval around whether the model beats that baseline rather than just reporting an accuracy number in isolation.

## One Prompt, Workflow, Or Moment That Helped

Feature extraction originally used `librosa.pyin` for pitch tracking. Claude ran a timed test on a handful of files before committing to the full 1,440-file run and found it was taking 8.5 seconds per file — over 3 hours total. It switched to `librosa.piptrack` instead, an STFT-based method that's far faster and still accurate enough for a heuristic feature, cutting the run to about 6 minutes. I confirmed that tradeoff was reasonable — piptrack's lower precision doesn't matter here since pitch is only feeding a coarse heuristic baseline, not the actual model — rather than just accepting a faster number without checking what was being traded away.

## One Thing You Verified Or Decided Yourself

I trained the model myself, end to end, in Colab, rather than shipping the weights from Claude's earlier smoke-test run — so I actually understand what's going into the RandomForest and can defend the 54.6%-vs-32.5% result if asked, instead of just repeating a number I didn't produce. I also spent real time manually verifying individual predictions: matching uploaded clips back to their exact RAVDESS source files by byte size, confirming the true label, and checking whether the model, the heuristic, or both got it right — rather than trusting the aggregate accuracy figure at face value.
