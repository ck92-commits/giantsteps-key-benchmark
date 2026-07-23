#!/usr/bin/env python3
"""Build the expert truth CSV from the GiantSteps+ key re-annotations
(Faraldo, Zenodo record 4153506).

GiantSteps+ re-annotates the GiantSteps key tracks with expert, time-spanned
key labels. This script reduces each track's annotation to a single dominant
key: the key with the longest total labelled span. Tracks whose dominant label
does not map to a Camelot key are skipped; on the 604-track key set that
leaves 596 scored tracks.

Usage:
  python3 build_expert_truth.py --annotations path/to/new_annotations
Writes expert_truth.csv next to this script (or --out).
"""
import argparse, csv, glob, json, os

from grade import to_camelot


def dominant_key(d):
    keys = d.get("key") or []
    if isinstance(keys, str):
        keys = [keys]
    st, en = d.get("start_times") or [], d.get("end_times") or []
    if keys and len(st) == len(keys) == len(en):
        spans = [max(0, (e if e > s else 120) - s) for s, e in zip(st, en)]
        return keys[spans.index(max(spans))]
    return keys[0] if keys else ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--annotations", required=True,
                    help="path to the GiantSteps+ new_annotations directory (Zenodo 4153506)")
    ap.add_argument("--out", default="expert_truth.csv")
    ap.add_argument("--genre-csv", default="",
                    help="optional giantsteps_truth.csv to copy genre labels from")
    args = ap.parse_args()

    genres = {}
    if args.genre_csv:
        with open(args.genre_csv, newline="") as fh:
            for r in csv.DictReader(fh):
                genres[r.get("track_id") or r.get("filename", "").split(".")[0]] = r.get("genre", "")

    rows, skipped = [], 0
    for f in sorted(glob.glob(os.path.join(args.annotations, "*.json"))):
        d = json.load(open(f))
        dom = dominant_key(d)
        cam = (to_camelot(dom)[0] or "").upper() if dom else ""
        if not cam:
            skipped += 1
            continue
        tid = str(d["id"])
        rows.append(dict(track_id=tid, filename=f"{tid}.LOFI.mp3",
                         key_truth=cam, bpm_truth="", genre=genres.get(tid, ""),
                         source="giantsteps-plus"))

    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["track_id", "filename", "key_truth",
                                           "bpm_truth", "genre", "source"])
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} rows -> {args.out}  ({skipped} annotations had no mappable dominant key)")


if __name__ == "__main__":
    main()
