#!/bin/bash
# Download the GiantSteps key+tempo audio previews from the JKU academic mirror.
# The audio is NOT part of this repository and is not redistributable here;
# this script fetches it from the original academic source, id by id.
# Usage: bash dl.sh            (downloads into ./audio, reads ./ids.txt)
#        GIANTSTEPS_DIR=/path bash dl.sh
cd "${GIANTSTEPS_DIR:-.}" || exit 1
mkdir -p audio
BASE=https://www.cp.jku.at/datasets/giantsteps/backup
ok=0; fail=0; skip=0; n=0
total=$(wc -l < ids.txt | tr -d ' ')
: > dl_fail.txt
while read -r id; do
  n=$((n+1))
  out="audio/${id}.LOFI.mp3"
  if [ -s "$out" ]; then skip=$((skip+1)); continue; fi
  if curl -s -m 90 -f -o "$out" "${BASE}/${id}.LOFI.mp3"; then
    ok=$((ok+1))
  else
    fail=$((fail+1)); rm -f "$out"; echo "$id" >> dl_fail.txt
  fi
  if [ $((n % 100)) -eq 0 ]; then echo "[$n/$total] ok=$ok fail=$fail skip=$skip"; fi
done < ids.txt
echo "DONE ok=$ok fail=$fail skip=$skip total=$total"
