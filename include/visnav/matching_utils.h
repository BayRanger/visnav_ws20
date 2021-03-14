/**
BSD 3-Clause License

Copyright (c) 2018, Vladyslav Usenko and Nikolaus Demmel.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#pragma once

#include <bitset>
#include <set>

#include <Eigen/Dense>
#include <sophus/se3.hpp>

#include <opengv/relative_pose/CentralRelativeAdapter.hpp>
#include <opengv/relative_pose/methods.hpp>
#include <opengv/sac/Ransac.hpp>
#include <opengv/sac_problems/relative_pose/CentralRelativePoseSacProblem.hpp>

#include <visnav/camera_models.h>
#include <visnav/common_types.h>

namespace visnav {

void computeEssential(const Sophus::SE3d& T_0_1, Eigen::Matrix3d& E) {
  const Eigen::Vector3d t_0_1 = T_0_1.translation();
  const Eigen::Matrix3d R_0_1 = T_0_1.rotationMatrix();

  /* ------------------------------ TASK SHEET 3 ------------------------------ */
  Eigen::Vector3d t_0_1_n = t_0_1.normalized();
  Eigen::Matrix3d t_0_1_n_hat;
  t_0_1_n_hat << 0, -t_0_1_n(2), t_0_1_n(1), t_0_1_n(2), 0, -t_0_1_n(0),
      -t_0_1_n(1), t_0_1_n(0), 0;

  E = t_0_1_n_hat * R_0_1;
  /* -------------------------------------------------------------------------- */

}

void findInliersEssential(const KeypointsData& kd1, const KeypointsData& kd2,
                          const std::shared_ptr<AbstractCamera<double>>& cam1,
                          const std::shared_ptr<AbstractCamera<double>>& cam2,
                          const Eigen::Matrix3d& E,
                          double epipolar_error_threshold, MatchData& md) {
  md.inliers.clear();

  for (size_t j = 0; j < md.matches.size(); j++) {
    const Eigen::Vector2d p0_2d = kd1.corners[md.matches[j].first];
    const Eigen::Vector2d p1_2d = kd2.corners[md.matches[j].second];

    /* ------------------------------ TASK SHEET 3 ------------------------------ */
    const double epipolar_error =
        abs(cam1->unproject(p0_2d).transpose() * E * cam2->unproject(p1_2d));
    if (epipolar_error <= epipolar_error_threshold)
      md.inliers.push_back(
          std::make_pair(md.matches[j].first, md.matches[j].second));
    /* -------------------------------------------------------------------------- */
    
  }
}

void findInliersRansac(const KeypointsData& kd1, const KeypointsData& kd2,
                       const std::shared_ptr<AbstractCamera<double>>& cam1,
                       const std::shared_ptr<AbstractCamera<double>>& cam2,
                       const double ransac_thresh, const int ransac_min_inliers,
                       MatchData& md) {
  md.inliers.clear();

  /* ------------------------------ TASK SHEET 3 ------------------------------ */
  using namespace opengv;

  bearingVectors_t vec_cam_1, vec_cam_2;
  for (const auto& pair : md.matches) {
    vec_cam_1.push_back(cam1->unproject(kd1.corners[pair.first]));
    vec_cam_2.push_back(cam2->unproject(kd2.corners[pair.second]));
  }

  relative_pose::CentralRelativeAdapter adapter(vec_cam_1, vec_cam_2);
  sac::Ransac<sac_problems::relative_pose::CentralRelativePoseSacProblem>
      ransac;
  std::shared_ptr<sac_problems::relative_pose::CentralRelativePoseSacProblem>
      relposeproblem_ptr(
          new sac_problems::relative_pose::CentralRelativePoseSacProblem(
              adapter, sac_problems::relative_pose::
                           CentralRelativePoseSacProblem::NISTER));

  ransac.sac_model_ = relposeproblem_ptr;
  ransac.threshold_ = ransac_thresh;
  if (!ransac.computeModel()) return;

  adapter.setR12(ransac.model_coefficients_.leftCols(3));
  adapter.sett12(ransac.model_coefficients_.rightCols(1));

  transformation_t nonlinear_transformation =
      relative_pose::optimize_nonlinear(adapter, ransac.inliers_);

  std::vector<int> inliers;
  relposeproblem_ptr->selectWithinDistance(nonlinear_transformation,
                                           ransac_thresh, inliers);

  md.T_i_j = Sophus::SE3d(nonlinear_transformation.leftCols(3),
                          nonlinear_transformation.rightCols(1).normalized());
  if (inliers.size() >= static_cast<size_t>(ransac_min_inliers))
    for (const auto inlier : inliers) md.inliers.push_back(md.matches[inlier]);
  /* -------------------------------------------------------------------------- */

}
}  // namespace visnav
