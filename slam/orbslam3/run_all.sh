#!/bin/bash
# Stereo ORB-SLAM3 on every recorded drive that has a right camera and no result yet, 2 at a time, then score.
#   bash slam/orbslam3/run_all.sh slam/runs/v6
cd "$(dirname "$0")/../.." && R=$1
ls -d $R/seed* | while read d; do [ -d $d/frames_right ] && [ ! -s $d/poses_orb.txt ] && echo $d; done \
  | xargs -P 2 -I{} sh -c 'echo "$(basename {}): $(docker run --rm -v $PWD/{}:/drive orbslam3 bash /run_in_container.sh | tail -1)"'
pytorch3d/.venv/bin/python -I slam/score_ate.py --est poses_orb.txt --se3 $R/seed* > $R/scores_orb_stereo.md
