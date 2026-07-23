# models/

Drop-in location for the optional Cratefox CNN files (see `cnn/README.md`):

- `key_cnn.onnx`
- `key_cnn_cqt_kernel.npz`

They are not committed to this repository. Without them, `grade.py`,
`build_truth.py`, and `build_expert_truth.py` work fully; only
`cnn/predict.py` needs them.
