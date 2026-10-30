# NASA POLAR Traverse: monocular left camera, 11 positions 0..10 m, per view x traverse x exposure -> run folders.
# Truth = the dataset's refined (COLMAP) poses. LOCIDs count on across traverses/views (View 1 Traverse 2 starts at
# 22), so files are found by pattern and poses numbered from each traverse's first LOCID (left camera = even offset).
cd /workspace/polar && mkdir -p runs
for v in 1 2; do for t in 1 2 3 4 5 6; do for e in 025 100 400; do
  R=runs/V${v}_T${t}_${e}ms; mkdir -p $R/frames
  for m in $(seq 0 10); do ln -sf $(ls /workspace/polar/View$v/Traverse$t/$(printf %02dm $m)/loc*_camL_${e}ms.png) $R/frames/$(printf %06d $m).png; done
  awk -v OFS=' ' 'NR==2 { b = $1 } NR>1 && int($1 - b)%2==0 { print ($1 - b)/2, $2, $3, $4, $6, $7, $8, $5 }' View$v/Traverse$t/poses/refined_pose.txt > $R/poses_truth.txt
done; done; done
echo "prepared $(ls runs | wc -l) runs"
