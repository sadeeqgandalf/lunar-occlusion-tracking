# ORB-SLAM3 (Campos et al., IEEE T-RO 2021; GPLv3) in stereo mode, headless, CPU only.
#   docker build -t orbslam3 slam/orbslam3
FROM ubuntu:20.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends build-essential cmake git ca-certificates \
    libeigen3-dev libopencv-dev libboost-serialization-dev libssl-dev libglew-dev libgl1-mesa-dev libepoxy-dev && rm -rf /var/lib/apt/lists/*
RUN git clone --depth 1 -b v0.6 https://github.com/stevenlovegrove/Pangolin.git /pangolin && cd /pangolin && mkdir build && cd build \
    && cmake .. -DBUILD_EXAMPLES=OFF -DBUILD_TESTS=OFF && make -j$(nproc) install && ldconfig
RUN git clone --depth 1 https://github.com/UZ-SLAMLab/ORB_SLAM3.git /orb && cd /orb && git log -1 --format=%H \
    && sed -i 's/OpenCV 4.4/OpenCV 4.2/' CMakeLists.txt \
    && grep -q "STEREO,true" Examples/Stereo/stereo_kitti.cc && sed -i "s/STEREO,true/STEREO,false/" Examples/Stereo/stereo_kitti.cc \
    && sed -i -E 's/make -j[0-9]*/make -j2/' build.sh && ./build.sh
WORKDIR /orb
COPY fix_settings_print.pl /tmp/
# rectified stereo has no Camera 2 calibration: the stock settings printout dereferences it and crashes, so guard it.
# also save every tracked frame with its timestamp (TUM format skips frames where tracking was lost)
RUN grep -q 'SaveTrajectoryKITTI("CameraTrajectory.txt");' Examples/Stereo/stereo_kitti.cc \
    && sed -i 's|SLAM.SaveTrajectoryKITTI("CameraTrajectory.txt");|&\n    SLAM.SaveTrajectoryTUM("CameraTrajectory_TUM.txt");|' Examples/Stereo/stereo_kitti.cc \
    && perl -0p -i /tmp/fix_settings_print.pl src/Settings.cc \
    && cd build && make -j2 ORB_SLAM3 stereo_kitti
RUN apt-get update && apt-get install -y --no-install-recommends python3 && rm -rf /var/lib/apt/lists/*
COPY run_in_container.sh /run_in_container.sh
# live mode for closed-loop driving: frames in on stdin, a pose out per frame
COPY orb_live.cc /orb/Examples/Stereo/orb_live.cc
RUN printf '\nadd_executable(orb_live Examples/Stereo/orb_live.cc)\ntarget_link_libraries(orb_live ${PROJECT_NAME})\n' >> CMakeLists.txt \
    && cd build && cmake .. > /dev/null && make -j2 orb_live
COPY live_settings.sh /live_settings.sh
