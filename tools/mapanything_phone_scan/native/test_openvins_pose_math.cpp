#include "openvins_pose_math.h"
#include "types/PoseJPL.h"

#include <iostream>
#include <stdexcept>

ov_type::PoseJPL pose(const Eigen::Matrix3d &rotation, const Eigen::Vector3d &position) {
  ov_type::PoseJPL result;
  Eigen::Matrix<double, 7, 1> value;
  value.head<4>() = ov_core::rot_2_quat(rotation);
  value.tail<3>() = position;
  result.set_value(value);
  return result;
}

int main() {
  try {
    const Eigen::Matrix3d R_GtoI = (Eigen::AngleAxisd(0.6, Eigen::Vector3d::UnitX()) *
                                   Eigen::AngleAxisd(-0.8, Eigen::Vector3d::UnitZ())).toRotationMatrix();
    for (double angle : {0.0, 0.3, -1.2}) {
      const Eigen::Matrix3d R_ItoC = (Eigen::AngleAxisd(angle, Eigen::Vector3d::UnitY()) *
                                     Eigen::AngleAxisd(0.5, Eigen::Vector3d::UnitX())).toRotationMatrix();
      const Eigen::Vector3d p_IinC(0.07, -0.03, 0.02);
      const Eigen::Vector3d p_IinG(1.2, -0.7, 2.1);
      const Eigen::Matrix4d nominal = noesis_phone::world_camera_transform(R_GtoI, p_IinG, R_ItoC, p_IinC);
      Eigen::Matrix<double, 6, 12> numerical;
      constexpr double step = 1e-6;
      for (int column = 0; column < 12; ++column) {
        Eigen::Matrix<double, 6, 2> errors;
        for (int direction = 0; direction < 2; ++direction) {
          auto imu = pose(R_GtoI, p_IinG);
          auto camera = pose(R_ItoC, p_IinC);
          Eigen::Matrix<double, 6, 1> perturb = Eigen::Matrix<double, 6, 1>::Zero();
          perturb(column % 6) = direction == 0 ? step : -step;
          if (column < 6) imu.update(perturb); else camera.update(perturb);
          const auto perturbed = noesis_phone::world_camera_transform(imu.Rot(), imu.pos(), camera.Rot(), camera.pos());
          const Eigen::AngleAxisd rotation_error(perturbed.block<3, 3>(0, 0) * nominal.block<3, 3>(0, 0).transpose());
          errors.block<3, 1>(0, direction) = rotation_error.axis() * rotation_error.angle();
          errors.block<3, 1>(3, direction) = perturbed.block<3, 1>(0, 3) - nominal.block<3, 1>(0, 3);
        }
        numerical.col(column) = (errors.col(0) - errors.col(1)) / (2 * step);
      }
      const auto analytical = noesis_phone::camera_pose_jacobian(R_GtoI, R_ItoC, p_IinC);
      const double error = (numerical - analytical).cwiseAbs().maxCoeff();
      if (error > 1e-7) throw std::runtime_error("camera-pose Jacobian disagrees with actual PoseJPL updates");
      std::cout << "rotation " << angle << ": max finite-difference error " << error << '\n';
    }
    return 0;
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
