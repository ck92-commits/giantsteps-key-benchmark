#!/usr/bin/env python3
"""Run the Cratefox key CNN over a folder of audio and write a predictions CSV
that grade.py can score. Optional mode: needs the model files in ../models/
(see cnn/README.md). Without them, use grade.py with your own detector's CSV.

Decoding: uses the soundfile package if installed (libsndfile 1.1+ reads mp3),
otherwise falls back to the ffmpeg command-line tool.

Usage:
  python3 cnn/predict.py --audio path/to/audio --out predictions.csv
"""
import argparse
import csv
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

import keycnn_standalone as keycnn

AUDIO_EXTS = {".mp3", ".flac", ".m4a", ".wav", ".aiff", ".ogg"}


def decode(path):
    """Return (mono_float_array, sample_rate) or (None, 0)."""
    try:
        import soundfile as sf
        y, sr = sf.read(path, dtype="float64", always_2d=True)
        return y.mean(axis=1), sr
    except Exception:
        pass
    try:
        sr = 44100
        raw = subprocess.run(
            ["ffmpeg", "-v", "quiet", "-i", path, "-ac", "1", "-ar", str(sr),
             "-f", "f32le", "-"], capture_output=True, check=True).stdout
        return np.frombuffer(raw, dtype=np.float32).astype(np.float64), sr
    except Exception:
        return None, 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", required=True, help="folder of audio files")
    ap.add_argument("--out", default="predictions.csv")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    if not keycnn.available():
        sys.exit("model files not found: put key_cnn.onnx and key_cnn_cqt_kernel.npz "
                 "in models/ (see cnn/README.md), and pip install onnxruntime")

    files = sorted(f for f in os.listdir(args.audio)
                   if os.path.splitext(f)[1].lower() in AUDIO_EXTS)
    if args.limit:
        files = files[:args.limit]

    n_ok, n_err = 0, 0
    with open(args.out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["track_id", "filename", "key", "confidence"])
        for i, fn in enumerate(files, 1):
            y, sr = decode(os.path.join(args.audio, fn))
            r = keycnn.detect_key(y, sr) if y is not None and y.size else None
            if r is None:
                n_err += 1
                continue
            cam, _label, conf = r
            w.writerow([fn.split(".")[0], fn, cam, conf])
            n_ok += 1
            if i % 50 == 0:
                print(f"  {i}/{len(files)}")
    print(f"wrote {n_ok} predictions -> {args.out}  ({n_err} files could not be decoded)")


if __name__ == "__main__":
    main()
