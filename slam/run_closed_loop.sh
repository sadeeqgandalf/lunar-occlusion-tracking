#!/bin/bash
# Closed-loop missions for every brain at the three sun angles (lab server must be running: npm start).
#   bash slam/run_closed_loop.sh [seeds...]   default seed 1
set -e
for seed in ${@:-1}; do
  for sun in 30 7 2; do
    for brain in truth wheels orb; do
      out=slam/runs/closed/${brain}_seed${seed}_sun${sun}
      [ -f $out/result.json ] && continue
      node slam/drive_closed_loop.mjs --brain $brain --seed $seed --sun $sun --out $out > /dev/null 2>&1 || echo "FAILED $out"
      python3 -c "import json;r=json.load(open('$out/result.json'));print('$brain seed $seed sun $sun:', 'done' if r['missionDone'] else 'NOT DONE', 'home miss', r['homeMissM'], 'm | pose err rms', r['poseErrRmsM'], 'max', r['poseErrMaxM'], '| rock contact', r['rockBumps'], '| tracked', r.get('trackedPct'))"
    done
  done
done
