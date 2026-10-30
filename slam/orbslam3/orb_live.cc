// Live stereo ORB-SLAM3 for closed-loop driving (slam/drive_closed_loop.mjs).
// stdin, one line per frame:  <time_s> <left.png> <right.png>
// stdout, one line per frame: POSE <time_s> <state> tx ty tz qx qy qz qw   (camera-to-world, ORB-SLAM3 frame:
//   the first camera is the origin, x right, y down, z forward). state: 2 = tracking OK, else not tracking.
// ORB-SLAM3 also prints its own messages to stdout, so pose lines carry the POSE prefix.
#include <iostream>
#include <sstream>
#include <opencv2/opencv.hpp>
#include "System.h"

int main(int argc, char **argv) {
  if (argc != 3) { std::cerr << "usage: orb_live vocabulary settings" << std::endl; return 1; }
  ORB_SLAM3::System SLAM(argv[1], argv[2], ORB_SLAM3::System::STEREO, false);
  std::string line;
  while (std::getline(std::cin, line)) {
    std::istringstream in(line);
    double t; std::string l, r;
    if (!(in >> t >> l >> r)) break;
    cv::Mat L = cv::imread(l, cv::IMREAD_UNCHANGED), R = cv::imread(r, cv::IMREAD_UNCHANGED);
    Sophus::SE3f Twc = SLAM.TrackStereo(L, R, t).inverse();
    int state = SLAM.GetTrackingState();
    Eigen::Vector3f p = Twc.translation();
    Eigen::Quaternionf q = Twc.unit_quaternion();
    std::cout << "POSE " << t << " " << state << " " << p.x() << " " << p.y() << " " << p.z() << " "
              << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << std::endl;
  }
  SLAM.Shutdown();
  return 0;
}
