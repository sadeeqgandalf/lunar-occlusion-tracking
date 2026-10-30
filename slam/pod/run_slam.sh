# Run VGGT-SLAM 2.0 on every run folder given (each holds frames/), saving poses_vggt.txt (+ _points.pcd) beside it.
#   bash run_slam.sh /workspace/sim/* /workspace/polar/runs/*
cd /root/VGGT-SLAM && source .venv/bin/activate
for R in "$@"; do
  [ -s $R/poses_vggt.txt ] && { echo "skip $(basename $R)"; continue; }
  python main.py --image_folder $R/frames --log_results --log_path $R/poses_vggt.txt > $R/vggt.log 2>&1 &
  pid=$!
  # main.py keeps its viewer open after writing results: stop it once the pose file exists (or it exits)
  for i in $(seq 1 600); do sleep 3; if [ -s $R/poses_vggt.txt ] || ! kill -0 $pid 2>/dev/null; then break; fi; done
  sleep 8; kill $pid 2>/dev/null; wait $pid 2>/dev/null
  [ -s $R/poses_vggt.txt ] && echo "done $(basename $R)" || echo "FAILED $(basename $R) (see vggt.log)"
done
