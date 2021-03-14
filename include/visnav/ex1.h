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

#include <sophus/se3.hpp>

#include <visnav/common_types.h>

namespace visnav {

/* ------------------------------ TASK SHEET 1 ------------------------------ */
float eps = 1e-16;

template <class T>
Eigen::Matrix<T, 3, 3> hat(const Eigen::Matrix<T, 3, 1>& xi) {
  Eigen::Matrix<T, 3, 3> xi_hat;
  xi_hat << 0, -xi(2), xi(1), xi(2), 0, -xi(0), -xi(1), xi(0), 0;
  return xi_hat;
}

template <class T>
Eigen::Matrix<T, 3, 1> vee(const Eigen::Matrix<T, 3, 3>& mat) {
  Eigen::Matrix<T, 3, 1> w;
  w << mat.coeff(2, 1) - mat.coeff(1, 2), mat.coeff(0, 2) - mat.coeff(2, 0),
      mat.coeff(1, 0) - mat.coeff(0, 1);
  return w;
}

/* -------------------------------------------------------------------------- */


// Implement exp for SO(3)
template <class T>
Eigen::Matrix<T, 3, 3> user_implemented_expmap(
    const Eigen::Matrix<T, 3, 1>& xi) {

  /* ------------------------------ TASK SHEET 1 ------------------------------ */
  T xi_norm = xi.norm();
  Eigen::Matrix<T, 3, 3> xi_hat = hat(xi);

  T sin_term = 0, cos_term = 0;
  if (xi_norm < eps) {
    sin_term =
        T(1) - xi_norm * xi_norm * (T(1) / T(6) + xi_norm * xi_norm / T(120));
    cos_term = T(1) / T(2) -
               xi_norm * xi_norm * (T(1) / T(24) + xi_norm * xi_norm / T(720));
  } else {
    sin_term = sin(xi_norm) / xi_norm;
    cos_term = (T(1) - cos(xi_norm)) / (xi_norm * xi_norm);
  }
  return Eigen::Matrix<T, 3, 3>::Identity() + sin_term * xi_hat +
         cos_term * (xi_hat * xi_hat);
  /* -------------------------------------------------------------------------- */
  
}

// Implement log for SO(3)
template <class T>
Eigen::Matrix<T, 3, 1> user_implemented_logmap(
    const Eigen::Matrix<T, 3, 3>& mat) {

  /* ------------------------------ TASK SHEET 1 ------------------------------ */
  T w_norm = acos((mat.trace() - 1) / 2);
  Eigen::Matrix<T, 3, 1> w = vee(mat);
  T sin_term = 0;
  if (w_norm < eps)
    sin_term = T(1) / T(2) +
               w_norm * w_norm *
                   (T(1) / T(12) +
                    w_norm * w_norm *
                        (T(7) / T(720) + w_norm * w_norm * T(31) / T(30240)));
  else
    sin_term = w_norm / (T(2) * sin(w_norm));
  return sin_term * w;
  /* -------------------------------------------------------------------------- */

}

// Implement exp for SE(3)
template <class T>
Eigen::Matrix<T, 4, 4> user_implemented_expmap(
    const Eigen::Matrix<T, 6, 1>& xi) {
  
  /* ------------------------------ TASK SHEET 1 ------------------------------ */
  Eigen::Matrix<T, 3, 1> w = xi.tail(3);
  Eigen::Matrix<T, 3, 1> v = xi.head(3);
  T w_norm = w.norm();
  auto w_hat = hat(w);

  T sin_term = 0, cos_term = 0;
  if (w_norm < eps) {
    sin_term = T(1) / T(6) +
               w_norm * w_norm * (-T(1) / T(120) + w_norm * w_norm / T(5040));
    cos_term = T(1) / T(2) +
               w_norm * w_norm * (-T(1) / T(24) + w_norm * w_norm / T(720));
  } else {
    sin_term = (w_norm - sin(w_norm)) / (w_norm * w_norm * w_norm);
    cos_term = (T(1) - cos(w_norm)) / (w_norm * w_norm);
  }
  Eigen::Matrix<T, 3, 3> J = Eigen::Matrix<T, 3, 3>::Identity() +
                             cos_term * w_hat + sin_term * (w_hat * w_hat);

  Eigen::Matrix<T, 4, 4> exp_xi = Eigen::Matrix<T, 4, 4>::Identity();
  exp_xi.block(0, 0, 3, 3) = user_implemented_expmap(w);
  exp_xi.col(3).head(3) = J * v;

  return exp_xi;
  /* -------------------------------------------------------------------------- */

}

// Implement log for SE(3)
template <class T>
Eigen::Matrix<T, 6, 1> user_implemented_logmap(
    const Eigen::Matrix<T, 4, 4>& mat) {

  /* ------------------------------ TASK SHEET 1 ------------------------------ */
  Eigen::Matrix<T, 3, 3> R = mat.block(0, 0, 3, 3);
  Eigen::Matrix<T, 3, 1> t = mat.col(3).head(3);
  Eigen::Matrix<T, 3, 1> w = user_implemented_logmap(R);
  Eigen::Matrix<T, 3, 3> w_hat = hat(w);
  T w_norm = w.norm();

  T norm_term = 0;
  if (w_norm < eps)
    norm_term = T(1) / T(12) +
                w_norm * w_norm * (T(1) / T(720) + w_norm * w_norm / T(30240));
  else
    norm_term = T(1) / (w_norm * w_norm) -
                (T(1) + cos(w_norm)) / (T(2) * w_norm * sin(w_norm));

  Eigen::Matrix<T, 3, 3> J_inverse = Eigen::Matrix<T, 3, 3>::Identity() -
                                     0.5 * w_hat + norm_term * (w_hat * w_hat);

  Eigen::Matrix<T, 6, 1> log_map;
  log_map.head(3) = J_inverse * t;
  log_map.tail(3) = w;

  return log_map;
  /* -------------------------------------------------------------------------- */

}

}  // namespace visnav
