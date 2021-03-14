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

#include <set>

#include <visnav/common_types.h>

#include <visnav/calibration.h>

#include <opengv/absolute_pose/CentralAbsoluteAdapter.hpp>
#include <opengv/absolute_pose/methods.hpp>
#include <opengv/relative_pose/CentralRelativeAdapter.hpp>
#include <opengv/sac/Ransac.hpp>
#include <opengv/sac_problems/absolute_pose/AbsolutePoseSacProblem.hpp>
#include <opengv/triangulation/methods.hpp>

namespace visnav {

void project_landmarks(
    const Sophus::SE3d& current_pose,
    const std::shared_ptr<AbstractCamera<double>>& cam,
    const Landmarks& landmarks, const double cam_z_threshold,
    std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>>&
        projected_points,
    std::vector<TrackId>& projected_track_ids) {
  projected_points.clear();
  projected_track_ids.clear();

  /* ------------------------------ TASK SHEET 5 ------------------------------ */
  const auto T_c_w = current_pose.inverse();
  for (const auto& landmark : landmarks) {
    const auto p_c_3D = T_c_w * landmark.second.p;
    if (p_c_3D(2) >= cam_z_threshold) {
      const auto p_c_2D = cam->project(p_c_3D);
      if (0 <= p_c_2D(0) && p_c_2D(0) < cam->width() && 0 <= p_c_2D(1) &&
          p_c_2D(1) < cam->height()) {
        projected_points.push_back(p_c_2D);
        projected_track_ids.push_back(landmark.first);
      }
    }
  }
  /* -------------------------------------------------------------------------- */

}

void find_matches_landmarks(
    const KeypointsData& kdl, const Landmarks& landmarks,
    const Corners& feature_corners,
    const std::vector<Eigen::Vector2d,
                      Eigen::aligned_allocator<Eigen::Vector2d>>&
        projected_points,
    const std::vector<TrackId>& projected_track_ids,
    const double match_max_dist_2d, const int feature_match_threshold,
    const double feature_match_dist_2_best, LandmarkMatchData& md) {
  md.matches.clear();
  
  /* ------------------------------ TASK SHEET 5 ------------------------------ */
  for (size_t i = 0; i < kdl.corners.size(); ++i) {
    int best_dist = std::numeric_limits<int>::max(),
        best_dist_2 = std::numeric_limits<int>::max();
    TrackId best_idx = -1;

    for (size_t j = 0; j < projected_points.size(); ++j) {
      if ((kdl.corners[i] - projected_points[j]).norm() <= match_max_dist_2d) {
        int best_obs_dist = std::numeric_limits<int>::max();
        for (const auto& obs : landmarks.at(projected_track_ids.at(j)).obs) {
          const int dist =
              (kdl.corner_descriptors[i] ^
               feature_corners.at(obs.first).corner_descriptors[obs.second])
                  .count();
          if (dist < best_obs_dist) best_obs_dist = dist;
        }

        if (best_obs_dist <= best_dist) {
          best_dist_2 = best_dist;
          best_dist = best_obs_dist;
          best_idx = projected_track_ids.at(j);
        } else if (best_obs_dist < best_dist_2)
          best_dist_2 = best_obs_dist;
      }
    }
    if (best_dist < feature_match_threshold &&
        !(best_dist_2 < best_dist * feature_match_dist_2_best))
      md.matches.emplace_back(i, best_idx);
  }
  /* -------------------------------------------------------------------------- */

}

void localize_camera(const Sophus::SE3d& current_pose,
                     const std::shared_ptr<AbstractCamera<double>>& cam,
                     const KeypointsData& kdl, const Landmarks& landmarks,
                     const double reprojection_error_pnp_inlier_threshold_pixel,
                     LandmarkMatchData& md) {
  using namespace opengv;
  md.inliers.clear();

  // default to previous pose if not enough inliers
  md.T_w_c = current_pose;

  if (md.matches.size() < 4) {
    return;
  }

  /* ------------------------------ TASK SHEET 5 ------------------------------ */
  points_t vp3D;
  bearingVectors_t vp2D;

  for (const auto& match : md.matches) {
    vp3D.push_back(landmarks.at(match.second).p);
    vp2D.push_back(cam->unproject(kdl.corners.at(match.first)));
  }

  absolute_pose::CentralAbsoluteAdapter adapter(vp2D, vp3D);

  sac::Ransac<sac_problems::absolute_pose::AbsolutePoseSacProblem> ransac;
  std::shared_ptr<sac_problems::absolute_pose::AbsolutePoseSacProblem>
      absposeproblem_ptr(
          new sac_problems::absolute_pose::AbsolutePoseSacProblem(
              adapter,
              sac_problems::absolute_pose::AbsolutePoseSacProblem::KNEIP));

  ransac.sac_model_ = absposeproblem_ptr;
  ransac.threshold_ =
      1. -
      std::cos(std::atan(reprojection_error_pnp_inlier_threshold_pixel / 500.));
  ransac.computeModel();

  adapter.setR(ransac.model_coefficients_.leftCols(3));
  adapter.sett(ransac.model_coefficients_.rightCols(1));

  transformation_t nonlinear_transformation =
      absolute_pose::optimize_nonlinear(adapter, ransac.inliers_);

  std::vector<int> inliers;
  absposeproblem_ptr->selectWithinDistance(nonlinear_transformation,
                                           ransac.threshold_, inliers);

  md.inliers.reserve(inliers.size());
  for (const auto inlier : inliers) md.inliers.push_back(md.matches.at(inlier));
  md.T_w_c = Sophus::SE3d(nonlinear_transformation.leftCols(3),
                          nonlinear_transformation.rightCols(1));
  /* -------------------------------------------------------------------------- */
  
}

void add_new_landmarks(const FrameCamId fcidl, const FrameCamId fcidr,
                       const KeypointsData& kdl, const KeypointsData& kdr,
                       const Calibration& calib_cam, const MatchData& md_stereo,
                       const LandmarkMatchData& md, Landmarks& landmarks,
                       TrackId& next_landmark_id) {
  // input should be stereo pair
  assert(fcidl.cam_id == 0);
  assert(fcidr.cam_id == 1);

  const Sophus::SE3d T_0_1 = calib_cam.T_i_c[0].inverse() * calib_cam.T_i_c[1];
  const Eigen::Vector3d t_0_1 = T_0_1.translation();
  const Eigen::Matrix3d R_0_1 = T_0_1.rotationMatrix();

  /* ------------------------------ TASK SHEET 5 ------------------------------ */
  std::unordered_map<FeatureId, TrackId> left_inliers;
  for (const auto& inlier : md.inliers) {
    landmarks[inlier.second].obs[fcidl] = inlier.first;
    left_inliers[inlier.first] = inlier.second;
  }

  std::vector<size_t> not_in_map_indices;
  for (size_t i = 0; i < md_stereo.inliers.size(); ++i) {
    const auto& left_idx = md_stereo.inliers[i].first;
    const auto& right_idx = md_stereo.inliers[i].second;

    if (left_inliers.find(left_idx) != left_inliers.end())
      landmarks[left_inliers[left_idx]].obs[fcidr] = right_idx;
    else
      not_in_map_indices.push_back(i);
  }

  opengv::bearingVectors_t vec_cam_0, vec_cam_1;
  for (const auto& idx : not_in_map_indices) {
    const auto& left_idx = md_stereo.inliers[idx].first;
    const auto& right_idx = md_stereo.inliers[idx].second;

    vec_cam_0.push_back(
        calib_cam.intrinsics[fcidl.cam_id]->unproject(kdl.corners[left_idx]));
    vec_cam_1.push_back(
        calib_cam.intrinsics[fcidr.cam_id]->unproject(kdr.corners[right_idx]));
  }

  opengv::relative_pose::CentralRelativeAdapter adapter(vec_cam_0, vec_cam_1,
                                                        t_0_1, R_0_1);

  for (size_t i = 0; i < not_in_map_indices.size(); ++i) {
    const auto track_idx = next_landmark_id++;
    landmarks[track_idx].p =
        md.T_w_c * opengv::triangulation::triangulate(adapter, i);

    landmarks[track_idx].obs.emplace(
        fcidl, md_stereo.inliers[not_in_map_indices[i]].first);
    landmarks[track_idx].obs.emplace(
        fcidr, md_stereo.inliers[not_in_map_indices[i]].second);
  }
  /* -------------------------------------------------------------------------- */

}

void remove_old_keyframes(const FrameCamId fcidl, const int max_num_kfs,
                          Cameras& cameras, Landmarks& landmarks,
                          Landmarks& old_landmarks,
                          std::set<FrameId>& kf_frames) {
  kf_frames.emplace(fcidl.frame_id);

  /* ------------------------------ TASK SHEET 5 ------------------------------ */
  std::set<FrameCamId> removing_kfs;
  {
    auto it = kf_frames.begin();
    while (it != kf_frames.end() && kf_frames.size() > max_num_kfs) {
      removing_kfs.emplace(*it, 0);
      removing_kfs.emplace(*it, 1);
      it = kf_frames.erase(it);
    }
  }

  for (const auto& kf : removing_kfs) cameras.erase(kf);

  {
    auto it = landmarks.begin();
    while (it != landmarks.end()) {
      for (const auto& kf : removing_kfs) it->second.obs.erase(kf);

      if (it->second.obs.size() == 0) {
        old_landmarks.insert(*it);
        it = landmarks.erase(it);
      } else
        it++;
    }
  }
  /* -------------------------------------------------------------------------- */

}
}  // namespace visnav
