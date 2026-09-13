#pragma once

#include <Eigen/Eigen>

namespace noesis_phone {

inline Eigen::Matrix3d skew(const Eigen::Vector3d &p) {
  Eigen::Matrix3d result;
  result << 0, -p.z(), p.y(), p.z(), 0, -p.x(), -p.y(), p.x(), 0;
  return result;
}

inline Eigen::Matrix4d world_camera_transform(const Eigen::Matrix3d &R_GtoI,
                                             const Eigen::Vector3d &p_IinG,
                                             const Eigen::Matrix3d &R_ItoC,
                                             const Eigen::Vector3d &p_IinC) {
  Eigen::Matrix4d pose = Eigen::Matrix4d::Identity();
  pose.block<3, 3>(0, 0) = R_GtoI.transpose() * R_ItoC.transpose();
  pose.block<3, 1>(0, 3) = p_IinG - pose.block<3, 3>(0, 0) * p_IinC;
  return pose;
}

// Pinned PoseJPL::update has R' = Exp(-dtheta) R and additive position.
// Input order: IMU rotation, IMU world position, I-to-C rotation, I-in-C position.
// Output: camera rotation error in world coordinates, additive camera world position.
inline Eigen::Matrix<double, 6, 12> camera_pose_jacobian(const Eigen::Matrix3d &R_GtoI,
                                                       const Eigen::Matrix3d &R_ItoC,
                                                       const Eigen::Vector3d &p_IinC) {
  const Eigen::Matrix3d R_ItoG = R_GtoI.transpose();
  const Eigen::Matrix3d R_CtoG = R_ItoG * R_ItoC.transpose();
  const Eigen::Vector3d p_CinI = -R_ItoC.transpose() * p_IinC;
  Eigen::Matrix<double, 6, 12> J = Eigen::Matrix<double, 6, 12>::Zero();
  J.block<3, 3>(0, 0) = R_ItoG;
  J.block<3, 3>(3, 0) = -R_ItoG * skew(p_CinI);
  J.block<3, 3>(3, 3) = Eigen::Matrix3d::Identity();
  J.block<3, 3>(0, 6) = R_CtoG;
  J.block<3, 3>(3, 6) = R_CtoG * skew(p_IinC);
  J.block<3, 3>(3, 9) = -R_CtoG;
  return J;
}

} // namespace noesis_phone
