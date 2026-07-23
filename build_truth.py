#!/usr/bin/env python3
"""Build the truth CSV from downloaded GiantSteps audio plus the datasets'
original key / tempo / genre annotations. Independent labels only.

Expects (relative to GIANTSTEPS_DIR, default the current directory):
  giantsteps-key-dataset/     git clone of github.com/GiantSteps/giantsteps-key-dataset
  giantsteps-tempo-dataset/   git clone of github.com/GiantSteps/giantsteps-tempo-dataset
  audio/                      the previews fetched by dl.sh

Writes giantsteps_truth.csv with one row per downloaded track that carries at
least one label. The key set carries no tempo and the tempo set carries no key,
so most rows have exactly one of key_truth / bpm_truth filled.
"""
import os, csv
from collections import Counter

BASE = os.environ.get("GIANTSTEPS_DIR", ".")
KEY = os.path.join(BASE, "giantsteps-key-dataset/annotations")
TEMPO = os.path.join(BASE, "giantsteps-tempo-dataset/annotations")
AUDIO = os.path.join(BASE, "audio")


def read1(path):
    try:
        with open(path) as f:
            return f.read().strip()
    except Exception:
        return ""


rows = []
for fn in sorted(os.listdir(AUDIO)):
    if not fn.endswith(".mp3"):
        continue
    tid = fn.split(".")[0]
    stem = f"{tid}.LOFI"
    key = read1(os.path.join(KEY, "key", stem + ".key"))
    bpm = read1(os.path.join(TEMPO, "tempo", stem + ".bpm"))
    genre = (read1(os.path.join(KEY, "genre", stem + ".genre"))
             or read1(os.path.join(TEMPO, "genre", stem + ".genre")) or "unknown")
    if not key and not bpm:
        continue
    rows.append(dict(track_id=tid, filename=fn, key_truth=key, bpm_truth=bpm,
                     genre=genre, source="giantsteps"))

out = os.path.join(BASE, "giantsteps_truth.csv")
with open(out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["track_id", "filename", "key_truth",
                                      "bpm_truth", "genre", "source"])
    w.writeheader()
    w.writerows(rows)

print(f"wrote {len(rows)} rows -> {out}")
print(f"  with key: {sum(1 for r in rows if r['key_truth'])}   with bpm: {sum(1 for r in rows if r['bpm_truth'])}")
print("  genres:", dict(Counter(r["genre"] for r in rows).most_common()))
