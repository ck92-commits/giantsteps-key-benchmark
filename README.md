# GiantSteps key benchmark harness

Scripts to measure a musical key detector (and optionally a BPM detector)
against the GiantSteps datasets, graded the way a DJ would grade it: exact
Camelot key, safe-to-mix rate, and clash risk. It downloads the audio from the
original academic sources, builds independent truth CSVs, and scores any
detector's output CSV. It also contains the frozen results of the Cratefox key
detector on this benchmark, published so anyone can re-run them.

This repo is maintained by [Cratefox](https://cratefox.app), which publishes
its accuracy numbers from this harness at
[cratefox.app/accuracy](https://cratefox.app/accuracy). The harness itself is
detector-agnostic: point it at any detector's CSV and you get the same report.

## The benchmark

- **GiantSteps key and tempo datasets**: 1225 real Beatport preview tracks
  with independent key, tempo, and genre labels (Knees et al., ISMIR 2015).
  Repos: [giantsteps-key-dataset](https://github.com/GiantSteps/giantsteps-key-dataset),
  [giantsteps-tempo-dataset](https://github.com/GiantSteps/giantsteps-tempo-dataset).
  604 tracks carry a key label, 664 carry a tempo label.
- **GiantSteps+ expert re-annotation**: Faraldo's expert re-annotation of the
  key tracks, [Zenodo record 4153506](https://zenodo.org/records/4153506).
  This is the answer key for the headline numbers. Each track's annotation is
  reduced to its dominant key (longest labelled span); 8 of the 604 have no
  mappable dominant key, so 596 tracks are scored.

**The audio is not in this repo and must not be redistributed here.** The
GiantSteps audio previews are not licensed for redistribution. `dl.sh`
downloads them track by track from the JKU academic mirror, exactly as the
dataset authors host them. The annotation files come from the original GitHub
repos and the Zenodo record. This repo contains only scripts, track id lists,
and frozen result records.

## Metrics

| Metric | Meaning |
|---|---|
| exact | detected Camelot key equals the truth key |
| safe to mix | exact, or the relative major/minor, or one step either way on the Camelot wheel |
| dangerous / clash risk | not safe to mix: trusting this key blindly would clash in a harmonic mix |
| bpm1x (strict) | within 1 BPM at the true 1x tempo; a half-time or double-time reading counts as fully wrong |
| octave-err | tracks matched only at 0.5x or 2x, i.e. halftime or doubletime flips |

The strict BPM metric is deliberately harsh: a 174 BPM drum and bass track
read as 87 counts as a miss even though pitch fader math still works.

## Reproduce the benchmark

```
git clone --depth 1 https://github.com/GiantSteps/giantsteps-key-dataset.git
git clone --depth 1 https://github.com/GiantSteps/giantsteps-tempo-dataset.git
bash dl.sh                    # audio from the JKU academic mirror (~1.8 GB)
python3 build_truth.py        # original labels -> giantsteps_truth.csv
```

For the expert answer key, download the GiantSteps+ re-annotation from
[Zenodo 4153506](https://zenodo.org/records/4153506), unpack it, then:

```
python3 build_expert_truth.py --annotations path/to/new_annotations \
    --genre-csv giantsteps_truth.csv    # -> expert_truth.csv (596 rows)
```

## Grade your own detector

Run your detector over `audio/`, write one CSV row per track with a
`filename` (or `track_id`) column and a `key` and/or `bpm` column, then:

```
python3 grade.py --truth expert_truth.csv --pred my_predictions.csv
```

Keys are accepted as Camelot (`8A`), Open Key (`1m`), or names (`F# minor`,
`Abm`). See `examples/predictions.example.csv`. `grade.py` needs only the
Python standard library and prints overall and per-genre exact / safe /
dangerous rates, strict BPM, octave errors, and a key miss breakdown
(relative, fifth, parallel, semitone, tritone).

## Run the Cratefox CNN (optional)

The key detector Cratefox ships is a small CNN (0.98 MB ONNX) over a
pure-numpy constant-Q transform. If you have the model files (see
`cnn/README.md`), you can reproduce its predictions here:

```
pip install numpy onnxruntime
python3 cnn/predict.py --audio audio --out cnn_predictions.csv
python3 grade.py --truth expert_truth.csv --pred cnn_predictions.csv
```

Without the model files, everything else in this repo still works; the
harness is for grading any detector, not just this one.

## Frozen results

Numbers below are verbatim from the frozen run records in `results/` and from
[cratefox.app/accuracy](https://cratefox.app/accuracy). They are measured, not
simulated.

### Key, graded on the GiantSteps+ expert re-annotation (596 tracks)

| Detector | Camelot-exact | Would clash |
|---|---|---|
| essentia (edmm profile) | 69.5% | 18.5% |
| first Cratefox CNN | 72.8% | 14.8% |
| shipped Cratefox CNN (2026-07-26, L6e) | 75.7% | 13.6% |

Safe to mix for the shipped model: 86.4% (100% minus clash risk).

A note on honesty: Cratefox's old public number was 69%, measured on a private
library and graded partly against tags the detector itself had written. That
is circular, so it was retired in favor of this benchmark. An earlier 64.9%
figure was graded against the original GiantSteps labels, an older answer key
that the expert re-annotation was made to correct, so it is not comparable to
the numbers above; that full 1225-track run is preserved verbatim in
`results/RESULTS_2026-07-07_final.txt`.

### Key by genre, essentia versus the shipped model (596 expert-labeled tracks)

Before is essentia edmm, after is the shipped model, the same one in the overall
table above (75.7% exact / 13.6% clash on all 596). Small buckets are noisy, so
the track count is shown. On trance and house the model trades some raw exact
for fewer clashes, so exact dips there while safe-to-mix still improves.

| Genre | Tracks | Key exact (before -> after) | Would clash (before -> after) |
|---|---|---|---|
| Progressive house | 88 | 71.6% -> 73.9% | 13.6% -> 5.7% |
| Tech house | 80 | 60.0% -> 70.0% | 26.2% -> 25.0% |
| Deep house | 76 | 76.3% -> 81.6% | 13.2% -> 11.8% |
| Trance | 58 | 75.9% -> 63.8% | 13.8% -> 12.1% |
| Electro house | 51 | 82.4% -> 88.2% | 9.8% -> 7.8% |
| House | 46 | 73.9% -> 71.7% | 21.7% -> 13.0% |
| Drum and bass | 36 | 61.1% -> 69.4% | 25.0% -> 19.4% |
| Techno | 33 | 51.5% -> 69.7% | 36.4% -> 27.3% |
| Dubstep | 22 | 54.5% -> 68.2% | 22.7% -> 18.2% |
| Electronica | 20 | 60.0% -> 80.0% | 25.0% -> 15.0% |
| Minimal | 11 | 63.6% -> 63.6% | 36.4% -> 36.4% |
| All 596 tracks | 596 | 69.5% -> 75.7% | 18.5% -> 13.6% |

Weak spots are shown, not hidden: minimal (n=11, small and noisy) got worse,
and tech house clash rose slightly. Small buckets are noisy; n is listed so
you can judge.

### Confidence gate (essentia cross-check, frozen 2026-07-15)

From `results/RESULTS_2026-07-15_gsplus.txt`: on the same 596 tracks, keys the
essentia detector was confident about (strength >= 0.70, 82.2% coverage)
scored 78.4% exact / 11.8% dangerous; the low-confidence remainder scored
28.3% exact / 49.1% dangerous. Low confidence really does predict the misses.
These figures describe the essentia cross-check, not the CNN's shown key.

### BPM (664 tempo-labeled tracks, strict 1x metric)

Neutral tempo fold: 49.8% strict-1x (frozen 2026-07-07 run). With the
genre-aware tempo level chooser (which only fires when a file carries a genre
tag): 49.8% -> 63.4% strict-1x, folded within 1 BPM 57.5% -> 67.3%
(re-measured 2026-07-15).

## Repo layout

```
dl.sh                    fetch the audio from the JKU mirror (id list: ids.txt)
build_truth.py           original GiantSteps labels -> giantsteps_truth.csv
build_expert_truth.py    GiantSteps+ re-annotation -> expert_truth.csv
grade.py                 score any predictions CSV (stdlib only)
examples/                predictions CSV example
cnn/                     optional: run the Cratefox key CNN (model not included)
models/                  drop-in location for the model files
results/                 frozen run records, kept verbatim
```

## Citations

- P. Knees, A. Faraldo, P. Herrera, R. Vogl, S. Boeck, F. Hoerschlaeger,
  M. Le Goff: "Two data sets for tempo estimation and key detection in
  electronic dance music annotated from user corrections", ISMIR 2015.
- A. Faraldo: GiantSteps+ key re-annotations,
  [Zenodo record 4153506](https://zenodo.org/records/4153506).

## License

The harness code in this repo is MIT licensed (see LICENSE). The GiantSteps
audio and annotations are NOT covered by this license; they belong to their
authors and are fetched from the original sources under their own terms.

If you re-run this benchmark and get different numbers, write
hey@cratefox.app and the correction will be published.
