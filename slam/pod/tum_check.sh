# Accuracy check: reproduce VGGT-SLAM 2.0's own TUM RGB-D benchmark (their evals/eval_tum.sh settings and scoring:
# 9 freiburg1 sequences, evo_ape with Sim3 alignment) before trusting any lunar result.
# no set -e: one failing sequence must not stop the rest
cd /root/VGGT-SLAM && source .venv/bin/activate
uv pip install -q evo 2>/dev/null || pip install -q evo
mkdir -p /workspace/tum && cd /workspace/tum
SEQS="360 desk desk2 floor plant room rpy teddy xyz"
for s in $SEQS; do
  d=rgbd_dataset_freiburg1_$s
  [ -d $d ] || (curl -fsSL https://cvg.cit.tum.de/rgbd/dataset/freiburg1/$d.tgz | tar xz) &
done; wait
cd /root/VGGT-SLAM
for s in $SEQS; do
  d=/workspace/tum/rgbd_dataset_freiburg1_$s; log=$d/poses_vggt.txt
  [ -s $log ] || {
    python main.py --image_folder $d/rgb --max_loops 1 --min_disparity 50 --conf_threshold 25 --lc_thres 0.95 --submap_size 16 \
      --log_results --skip_dense_log --log_path $log > $d/vggt.log 2>&1 &
    pid=$!; for i in $(seq 1 600); do sleep 3; [ -s $log ] || kill -0 $pid 2>/dev/null || break; [ -s $log ] && break; done
    sleep 5; kill $pid 2>/dev/null; wait $pid 2>/dev/null || true; }
  r=$(evo_ape tum $d/groundtruth.txt $log -as 2>&1 | grep -m1 rmse | awk '{print $2}')
  echo "fr1_$s $r"
done | tee /workspace/tum/results.txt
echo TUM_DONE
