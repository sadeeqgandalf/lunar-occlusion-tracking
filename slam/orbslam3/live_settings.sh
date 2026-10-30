#!/bin/bash
# Write ORB-SLAM3 stereo settings for the lab camera to $1: same recipe as run_in_container.sh (ideal pinhole,
# fx = (W/2) / tan(hfov/2), centre at the image middle, rectified pair), KITTI settings otherwise. Args: out W H HFOV_DEG BASELINE FPS
set -e
S=$1 W=$2 H=$3 HFOV=$4 B=$5 FPS=$6
FX=$(python3 -c "import math;print($W/2/math.tan(math.radians($HFOV/2)))")
cp /orb/Examples/Stereo/KITTI00-02.yaml $S
set_key() { sed -i -E "s|^($1:).*|\1 $2|" $S; grep -q "^$1:" $S || { echo "missing key $1"; exit 1; }; }
set_key Camera1.fx $FX; set_key Camera1.fy $FX; set_key Camera1.cx $(python3 -c "print($W/2-0.5)"); set_key Camera1.cy $(python3 -c "print($H/2-0.5)")
set_key Viewer.ViewpointY -100.0
set_key Camera.width $W; set_key Camera.height $H; set_key Camera.fps $FPS; set_key Stereo.b $B
