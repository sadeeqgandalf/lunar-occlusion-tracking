# ORB-SLAM3 v1.0 Settings.cc printout reads Camera 2's calibration, which a rectified stereo pair (KITTI-style)
# never sets (uninitialised pointer -> segfault). Skip that printout for rectified cameras. Exactly one match.
my $n = s/if\(settings\.sensor_ == System::STEREO \|\| settings\.sensor_ == System::IMU_STEREO\)\{(\s*output << "\\t-Camera 2 parameters \()/if((settings.sensor_ == System::STEREO || settings.sensor_ == System::IMU_STEREO) && settings.cameraType_ != Settings::Rectified){$1/g;
die "expected 1 match, got $n\n" unless $n == 1;
