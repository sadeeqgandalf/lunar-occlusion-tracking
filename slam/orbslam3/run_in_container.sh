#!/bin/bash
# Inside the orbslam3 image: run stereo ORB-SLAM3 on one recorded drive mounted at /drive.
# Camera from meta.json: ideal pinhole, square pixels, fx = (W/2) / tan(hfov/2), centre at the image middle
# (OpenCV pixel centres: cx = W/2 - 0.5); no lens distortion; rectified pair, baseline stereoBaselineM.
# ORB-SLAM3's own KITTI settings, with only the camera values replaced (no tuning).
set -e
read W H FX B FPS < <(python3 -c "import json,math;m=json.load(open('/drive/meta.json'));print(m['width'],m['height'],m['width']/2/math.tan(math.radians(m['hfovDeg']/2)),m['stereoBaselineM'],m['fps'])")
S=/tmp/drive.yaml; cp /orb/Examples/Stereo/KITTI00-02.yaml $S
set_key() { sed -i -E "s|^($1:).*|\1 $2|" $S; grep -q "^$1:" $S || { echo "missing key $1"; exit 1; }; }
set_key Camera1.fx $FX; set_key Camera1.fy $FX; set_key Camera1.cx $(python3 -c "print($W/2-0.5)"); set_key Camera1.cy $(python3 -c "print($H/2-0.5)")
set_key Viewer.ViewpointY -100.0   # stock file has -100, which the settings parser rejects (not a real number)
set_key Camera.width $W; set_key Camera.height $H; set_key Camera.fps $FPS; set_key Stereo.b $B
mkdir -p /tmp/seq && ln -sfn /drive/frames /tmp/seq/image_0 && ln -sfn /drive/frames_right /tmp/seq/image_1
n=$(ls /drive/frames | wc -l); python3 -c "print('\n'.join(f'{i/$FPS:.6f}' for i in range($n)))" > /tmp/seq/times.txt
cd /tmp && /orb/Examples/Stereo/stereo_kitti /orb/Vocabulary/ORBvoc.txt $S /tmp/seq > /drive/orb_log.txt 2>&1 || echo "stereo_kitti exited $?"
# timestamp -> frame index; same columns as poses_vggt.txt
python3 -c "
import sys
for l in open('/tmp/CameraTrajectory_TUM.txt'):
    v = l.split(); print(round(float(v[0]) * $FPS), *v[1:])" > /drive/poses_orb.txt
echo "tracked $(wc -l < /drive/poses_orb.txt) of $n frames"
