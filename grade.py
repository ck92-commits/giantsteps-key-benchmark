#!/usr/bin/env python3
"""Grade ANY key/BPM detector's predictions against a GiantSteps truth CSV.

Standard library only. Works with any detector: run yours over the audio,
write one CSV row per track, and this script scores it with the same metrics
used in the frozen results in results/.

Truth CSV (from build_truth.py or build_expert_truth.py):
  track_id, filename, key_truth, bpm_truth, genre, source

Predictions CSV: needs a track_id or filename column, plus any of
  key   (Camelot "8A", Open Key "1m", or a name like "F# minor" / "Abm")
  bpm   (a number)
Column aliases accepted: key/camelot/pred_key/key_pred, bpm/tempo/pred_bpm/bpm_pred.

Metrics (identical to the frozen harness):
  exact       predicted Camelot equals the truth Camelot
  safe/mixable  exact, relative major/minor, or +-1 on the Camelot wheel
  dangerous   not mixable: would clash in a harmonic mix
  bpm1x       within 1 BPM at the true 1x tempo (a half/double-time reading counts
              as a MISS; this is the strict metric GiantSteps uses)
  octave-err  count of tracks matched only at 0.5x/2x (halftime/doubletime flips)

Usage:
  python3 grade.py --truth giantsteps_truth.csv --pred my_predictions.csv
"""
import argparse
import csv
import re

# ---------------------------------------------------------------------------
# Key normalization: anything -> Camelot
# ---------------------------------------------------------------------------

_CAMELOT = {
    # major (B side)
    "c": "8B", "c#": "3B", "db": "3B", "d": "10B", "d#": "5B", "eb": "5B",
    "e": "12B", "f": "7B", "f#": "2B", "gb": "2B", "g": "9B", "g#": "4B",
    "ab": "4B", "a": "11B", "a#": "6B", "bb": "6B", "b": "1B",
    # minor (A side)
    "cm": "5A", "c#m": "12A", "dbm": "12A", "dm": "7A", "d#m": "2A", "ebm": "2A",
    "em": "9A", "fm": "4A", "f#m": "11A", "gbm": "11A", "gm": "6A", "g#m": "1A",
    "abm": "1A", "am": "8A", "a#m": "3A", "bbm": "3A", "bm": "10A",
}
_CAM_LABEL = {v: k for k, v in _CAMELOT.items()}
_CAM_RE = re.compile(r"^\s*(\d{1,2})([ABab])\s*$")
_OPENKEY_RE = re.compile(r"^\s*(\d{1,2})([mdMD])\s*$")
_KEY_RE = re.compile(r"^\s*([a-gA-G])\s*([#♯b♭]?)\s*(.*)$")
_NOTE_PC = {"c": 0, "c#": 1, "db": 1, "d": 2, "d#": 3, "eb": 3, "e": 4, "f": 5,
            "f#": 6, "gb": 6, "g": 7, "g#": 8, "ab": 8, "a": 9, "a#": 10, "bb": 10, "b": 11}


def to_camelot(raw):
    """(camelot_code_or_None, cleaned_input) from any common key string."""
    if not raw:
        return None, ""
    s = raw.strip().replace("♯", "#").replace("♭", "b")
    m = _CAM_RE.match(s)                 # already Camelot, e.g. '8A'
    if m:
        return f"{int(m.group(1))}{m.group(2).upper()}", s
    m = _OPENKEY_RE.match(s)             # Open Key, e.g. '12m' / '1d'
    if m:
        n = int(m.group(1))
        if 1 <= n <= 12:
            return f"{((n + 6) % 12) + 1}{'A' if m.group(2).lower() == 'm' else 'B'}", s
    m = _KEY_RE.match(s)                 # standard, e.g. 'Abm', 'F# minor', 'A'
    if m:
        root = m.group(1).lower() + (m.group(2) or "")
        rest = m.group(3).lower()
        if rest.startswith("min") or rest == "m":
            minor = True
        elif rest.startswith("maj") or rest in ("", "d"):
            minor = False
        else:
            minor = "m" in rest
        cam = _CAMELOT.get(root + ("m" if minor else ""))
        if cam:
            return cam, s
    return None, s


def cam_parse(c):
    m = _CAM_RE.match((c or "").strip().upper())
    return (int(m.group(1)), m.group(2).upper()) if m else None


def camelot_compatible(a, b):
    """Harmonic-mixing neighbours on the Camelot wheel: identical, relative
    major/minor (same number, other letter), or +-1 around the wheel (same
    letter)."""
    pa, pb = cam_parse(a), cam_parse(b)
    if not pa or not pb:
        return False
    if pa == pb:
        return True
    if pa[0] == pb[0] and pa[1] != pb[1]:
        return True
    if pa[1] == pb[1] and abs(pa[0] - pb[0]) in (1, 11):
        return True
    return False


def cam_pcmode(cam):
    """(pitch_class 0-11, is_minor) for a Camelot code, or None."""
    lbl = _CAM_LABEL.get((cam or "").strip().upper())
    if not lbl:
        return None
    minor = lbl.endswith("m")
    note = lbl[:-1] if minor else lbl
    pc = _NOTE_PC.get(note)
    return None if pc is None else (pc, minor)


def miss_class(pred, gt):
    """Classify pred vs truth: exact / relative / fifth (all mixable)
    vs parallel / semitone / tritone / other (all dangerous)."""
    pp, gp = cam_parse(pred), cam_parse(gt)
    if not pp or not gp:
        return "no_key"
    if pp == gp:
        return "exact"
    if pp[0] == gp[0] and pp[1] != gp[1]:
        return "relative"
    if pp[1] == gp[1] and abs(pp[0] - gp[0]) in (1, 11):
        return "fifth"
    pm, gm = cam_pcmode(pred), cam_pcmode(gt)
    if pm and gm:
        dpc = (pm[0] - gm[0]) % 12
        if pm[0] == gm[0] and pm[1] != gm[1]:
            return "parallel"
        if dpc in (1, 11):
            return "semitone"
        if dpc == 6:
            return "tritone"
    return "other"


# ---------------------------------------------------------------------------
# BPM metrics
# ---------------------------------------------------------------------------

def bpm_strict_ok(pred, gt, tol=1.0):
    """Octave-STRICT: pred must match at 1x. Half/double time is a miss."""
    if pred is None or gt is None:
        return False
    return abs(pred - gt) <= tol


def bpm_folded_ok(pred, gt, tol=1.0):
    """Tolerant: also accepts a 0.5x/2x octave fold."""
    if pred is None or gt is None:
        return False
    for f in (1.0, 0.5, 2.0):
        if abs(pred * f - gt) <= tol or abs(pred - gt * f) <= tol:
            return True
    return False


# ---------------------------------------------------------------------------
# CSV plumbing
# ---------------------------------------------------------------------------

_KEY_COLS = ("key", "camelot", "pred_key", "key_pred", "detected_key")
_BPM_COLS = ("bpm", "tempo", "pred_bpm", "bpm_pred", "detected_bpm")
_ID_COLS = ("track_id", "id")
_FN_COLS = ("filename", "file", "path")


def _pick(row, names):
    for n in names:
        for k in row:
            if k and k.strip().lower() == n:
                v = (row[k] or "").strip()
                if v:
                    return v
    return ""


def _row_ids(row):
    tid = _pick(row, _ID_COLS)
    fn = _pick(row, _FN_COLS)
    if fn:
        fn = fn.replace("\\", "/").rsplit("/", 1)[-1]
        if not tid:
            tid = fn.split(".")[0]
    return tid, fn


def load_predictions(path):
    preds = {}
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            tid, fn = _row_ids(row)
            if not tid and not fn:
                continue
            cam = to_camelot(_pick(row, _KEY_COLS))[0]
            try:
                bpm = float(_pick(row, _BPM_COLS))
            except ValueError:
                bpm = None
            rec = {"cam": (cam or "").upper() or None, "bpm": bpm}
            if tid:
                preds[tid] = rec
            if fn:
                preds.setdefault(fn, rec)
    return preds


def pct(n, d):
    return f"{(100.0 * n / d):5.1f}%" if d else "  n/a"


def block(label, grp):
    if not grp:
        return
    kk = [r for r in grp if r["gt_cam"]]
    bb = [r for r in grp if r["gt_bpm"] is not None]
    nk, nb = len(kk), len(bb)
    e = sum(1 for r in kk if r["miss"] == "exact")
    mx = sum(1 for r in kk if r["miss"] in ("exact", "relative", "fifth"))
    b_str = sum(1 for r in bb if bpm_strict_ok(r["pred_bpm"], r["gt_bpm"]))
    b_fold = sum(1 for r in bb if bpm_folded_ok(r["pred_bpm"], r["gt_bpm"]))
    flips = b_fold - b_str
    kp = f"exact {pct(e, nk)}  safe {pct(mx, nk)}  dangerous {pct(nk - mx, nk)}" if nk \
        else "exact   n/a  safe   n/a  dangerous   n/a"
    bp = f"bpm1x {pct(b_str, nb)}  octave-err {flips:3d}" if nb else "bpm1x   n/a  octave-err   -"
    print(f"  {label:20} n={len(grp):4d} (k{nk:3d}/b{nb:3d})  {kp}  {bp}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--truth", required=True, help="truth CSV (build_truth.py / build_expert_truth.py)")
    ap.add_argument("--pred", required=True, help="your detector's predictions CSV")
    args = ap.parse_args()

    preds = load_predictions(args.pred)
    rows, unmatched = [], 0
    with open(args.truth, newline="") as fh:
        for row in csv.DictReader(fh):
            tid, fn = _row_ids(row)
            gt_cam = (to_camelot(_pick(row, ("key_truth", "camelot_truth", "key")))[0] or "").upper() or None
            try:
                gt_bpm = float(_pick(row, ("bpm_truth", "bpm")))
            except ValueError:
                gt_bpm = None
            if gt_cam is None and gt_bpm is None:
                continue
            p = preds.get(tid) or preds.get(fn)
            if p is None:
                unmatched += 1
                continue
            rows.append({
                "genre": (_pick(row, ("genre",)) or "unknown").lower(),
                "gt_cam": gt_cam, "gt_bpm": gt_bpm,
                "pred_cam": p["cam"], "pred_bpm": p["bpm"],
                "miss": miss_class(p["cam"], gt_cam) if gt_cam else None,
            })

    n = len(rows)
    print(f"GIANTSTEPS GRADE  truth={args.truth}  pred={args.pred}")
    print(f"  scored {n} tracks  ({unmatched} truth rows had no prediction)")
    if not n:
        return
    print("-" * 88)
    block("ALL", rows)
    for g in sorted(set(r["genre"] for r in rows)):
        block(g, [r for r in rows if r["genre"] == g])
    kk = [r for r in rows if r["gt_cam"]]
    if kk:
        bd = {k: sum(1 for r in kk if r["miss"] == k)
              for k in ("exact", "relative", "fifth", "parallel", "semitone", "tritone", "other", "no_key")}
        print("-" * 88)
        print(f"  key miss breakdown: relative={bd['relative']} fifth={bd['fifth']} "
              f"(both still mixable) | parallel={bd['parallel']} semitone={bd['semitone']} "
              f"tritone={bd['tritone']} other={bd['other']} no_key={bd['no_key']} (all dangerous)")


if __name__ == "__main__":
    main()
