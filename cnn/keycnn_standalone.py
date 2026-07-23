"""Standalone runner for the Cratefox key CNN (optional mode of this repo).

Reproduces the app's wired inference path: mono audio at 11025 Hz -> pure-numpy
constant-Q (l6_cqt) -> ONNX model -> Camelot key. Needs two files dropped into
../models/ from a Cratefox install:
  key_cnn.onnx              the model (24 classes: 12 major then 12 minor)
  key_cnn_cqt_kernel.npz    precomputed constant-Q FFT kernels

Measured through the app's end-to-end path on the GiantSteps key set graded
against the GiantSteps+ expert re-annotation (596 tracks):
exact 75.84% / would-clash 13.59%. This standalone path decodes audio slightly
differently from the app (see cnn/README.md), so expect results close to, but
not bit-identical with, that frozen number.

Requires: numpy, onnxruntime.
"""
from __future__ import annotations

import os

import numpy as np

import l6_cqt

_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(_DIR, "..", "models", "key_cnn.onnx")
KERNEL_PATH = os.path.join(_DIR, "..", "models", "key_cnn_cqt_kernel.npz")

SR = 11025                 # the model's fixed input sample rate
CROP_SEC = 120             # center-crop, matching training
_CORE = slice(14, 134)     # 120 core CQT bins (drop the margin)

# The 24 output classes in model order: 12 major (C..B) then 12 minor (C..B),
# mapped to Camelot. Same table the app derives from its key parser.
_NOTES = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]
CLASS_LABELS = [f"{n} major" for n in _NOTES] + [f"{n} minor" for n in _NOTES]
CLASS_CAMS = ["8B", "3B", "10B", "5B", "12B", "7B", "2B", "9B", "4B", "11B", "6B", "1B",
              "5A", "12A", "7A", "2A", "9A", "4A", "11A", "6A", "1A", "8A", "3A", "10A"]

_SESS = None


def available() -> bool:
    try:
        import onnxruntime  # noqa: F401
    except Exception:
        return False
    return os.path.exists(MODEL_PATH) and os.path.exists(KERNEL_PATH)


def _ensure_loaded():
    global _SESS
    if _SESS is not None:
        return _SESS
    import onnxruntime as ort
    l6_cqt.load_kernel(KERNEL_PATH)
    _SESS = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
    return _SESS


def _probs(out):
    """The shipped model emits a softmax-mean distribution; softmax raw logits
    if a logits model is dropped in instead. Robust to both."""
    out = np.asarray(out, dtype=np.float64).ravel()
    if out.size == 24 and out.min() >= -1e-6 and abs(out.sum() - 1.0) < 1e-2:
        return out
    z = out - out.max()
    e = np.exp(z)
    return e / (e.sum() + 1e-12)


def detect_key(y, sr=SR):
    """Predict (camelot, class_label, confidence_0_100) from a mono float array.
    Resamples to 11025 Hz and center-crops to 120 s, matching the app."""
    sess = _ensure_loaded()
    y = np.asarray(y, dtype=np.float64)
    if sr != SR and y.size:
        n_out = int(round(y.size * SR / sr))
        Y = np.fft.rfft(y)
        Yr = np.zeros(n_out // 2 + 1, dtype=complex)
        k = min(len(Y), len(Yr))
        Yr[:k] = Y[:k]
        y = np.fft.irfft(Yr, n=n_out) * (n_out / y.size)
    if y.size > CROP_SEC * SR:
        off = (y.size - CROP_SEC * SR) // 2
        y = y[off:off + CROP_SEC * SR]
    if y.size < SR:
        return None
    mag = l6_cqt.cqt_mag(y)
    x = l6_cqt.prep_x(mag[_CORE])[None, None].astype(np.float32)
    probs = _probs(sess.run(None, {"cqt": x})[0])
    i = int(probs.argmax())
    return CLASS_CAMS[i], CLASS_LABELS[i], int(round(100 * float(probs[i])))
