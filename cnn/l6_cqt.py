"""Pure-numpy reproduction of librosa.cqt for the frozen L6 key-CNN front-end.

No librosa, no scipy - numpy only, so it installs on bleeding-edge Python. The
per-octave FFT kernels are precomputed offline from librosa 0.11.0 and shipped as
l6_cqt_kernel.npz; here we reproduce librosa's recursive octave algorithm:
per-octave STFT (rectangular window, centered, zero-pad) -> kernel matmul, then
halve the signal (rFFT spectral truncation, the numpy analogue of soxr_hq that
was measured to match to ~6 nines cosine), then stack + sqrt(length) normalize.

Validated offline vs librosa.cqt: cos(|CQT|) ~0.99999, cos after log1p+standardize
~0.99999. This is the training front-end, reproduced.
"""
from __future__ import annotations

import os

import numpy as np

_KERNEL = None


def load_kernel(path: str | None = None):
    global _KERNEL
    if _KERNEL is not None:
        return _KERNEL
    if path is None:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "models", "key_cnn_cqt_kernel.npz")
    d = np.load(path)
    _KERNEL = {
        "n_octaves": int(d["n_octaves"]),
        "hop_ds": int(d["hop_ds"]),
        "n_bins": int(d["n_bins"]),
        "lengths": d["lengths"].astype(np.float64),
        "oct_nfft": [int(x) for x in d["oct_nfft"]],
        "basis": [d[f"basis_{i}"] for i in range(int(d["n_octaves"]))],
    }
    return _KERNEL


def _stft_ones(y, n_fft, hop):
    """librosa.stft(y, n_fft, hop, window='ones', center=True, pad_mode='constant')."""
    yp = np.pad(y, n_fft // 2, mode="constant")
    if len(yp) < n_fft:
        return np.zeros((n_fft // 2 + 1, 1), dtype=np.complex128)
    n_frames = 1 + (len(yp) - n_fft) // hop
    idx = np.arange(n_fft)[None, :] + hop * np.arange(n_frames)[:, None]
    frames = yp[idx]                                  # rectangular window (ones)
    return np.fft.rfft(frames, axis=1).T              # (n_fft//2+1, n_frames)


def _resample_half(y):
    """Downsample by 2 via rFFT spectral truncation with librosa scale=True
    (divide by sqrt(ratio)); measured to match soxr_hq to ~6 nines cosine."""
    n = len(y)
    n_out = int(np.ceil(n * 0.5))
    Y = np.fft.rfft(y)
    Yr = np.zeros(n_out // 2 + 1, dtype=complex)
    k = min(len(Y), len(Yr))
    Yr[:k] = Y[:k]
    out = np.fft.irfft(Yr, n=n_out) * (n_out / n)
    return out / np.sqrt(0.5)                          # scale=True


def _trim_stack(resp, n_bins):
    max_col = min(c.shape[-1] for c in resp)
    out = np.empty((n_bins, max_col), dtype=resp[0].dtype)
    end = n_bins
    for c in resp:
        n_oct = c.shape[0]
        if end < n_oct:
            out[:end, :] = c[-end:, :max_col]
        else:
            out[end - n_oct:end, :] = c[:, :max_col]
        end -= n_oct
    return out


def cqt_mag(y, kernel=None):
    """|CQT| of a mono float array at sr=11025 - reproduces np.abs(librosa.cqt(...,
    hop=2048, fmin=C2*2^(-14/24), n_bins=142, bins_per_octave=24)). Shape (142, T)."""
    k = kernel or load_kernel()
    my_y = np.asarray(y, dtype=np.float64)
    my_hop = k["hop_ds"]
    resp = []
    for i in range(k["n_octaves"]):
        D = _stft_ones(my_y, k["oct_nfft"][i], my_hop)
        resp.append(k["basis"][i] @ D)
        if my_hop % 2 == 0:
            my_hop //= 2
            my_y = _resample_half(my_y)
    V = _trim_stack(resp, k["n_bins"])
    V = V / np.sqrt(k["lengths"])[:, None]
    return np.abs(V).astype(np.float32)


def prep_x(mag):
    """Model input normalization (l6_eval.prep_x): log1p(100*mag) then standardize."""
    x = np.log1p(100.0 * mag.astype(np.float32))
    return (x - x.mean()) / (x.std() + 1e-5)
