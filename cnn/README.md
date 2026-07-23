# Optional: run the Cratefox key CNN

This folder reproduces the inference path of the key detector Cratefox ships:
mono audio at 11025 Hz, a pure-numpy constant-Q transform (`l6_cqt.py`, no
librosa or scipy), then a small ONNX model with 24 output classes (12 major,
12 minor, mapped to Camelot).

## What you need

Two files in `../models/`, taken from a Cratefox install:

- `key_cnn.onnx` (0.98 MB)
- `key_cnn_cqt_kernel.npz` (precomputed constant-Q FFT kernels)

Plus `pip install numpy onnxruntime`, and either `pip install soundfile`
(libsndfile 1.1+ decodes mp3) or the `ffmpeg` command-line tool for decoding.

## Run

```
python3 cnn/predict.py --audio audio --out cnn_predictions.csv
python3 grade.py --truth expert_truth.csv --pred cnn_predictions.csv
```

## Expected numbers

Measured through the app's end-to-end path on the 596 expert-labeled
GiantSteps tracks: 75.84% Camelot-exact, 13.59% would-clash. This standalone
runner uses the same CQT and the same model, but a different audio decode
chain than the app, so small decoder differences can move the result by a few
tenths of a percent. Inside the app repo, a ship-blocking parity test holds
the shipped model to the frozen score bit-for-bit.

If the model files are absent this mode simply reports that and exits; the
rest of the repo (grading any detector's CSV) does not need them.
