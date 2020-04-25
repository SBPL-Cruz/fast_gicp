#ifndef FAST_GICP_CUDA_COMPUTE_DERIVATIVES_NN_CUH
#define FAST_GICP_CUDA_COMPUTE_DERIVATIVES_NN_CUH

#pragma once

#include <Eigen/Core>
#include <sophus/so3.hpp>

#include <thrust/remove.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

namespace fast_gicp {

namespace  {

struct compute_derivatives_nn_kernel {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  compute_derivatives_nn_kernel(const Eigen::Matrix<float, 6, 1>& x, 
                                const thrust::device_vector<Eigen::Vector3f>& target_points,
                                const thrust::device_vector<Eigen::Matrix3f>& target_covs,
                                const int corr_dist_threshold)
  : R(Sophus::SO3f::exp(x.head<3>()).matrix()),
    t(x.tail<3>()),
    target_points_ptr(target_points.data()),
    target_covs_ptr(target_covs.data()),
    corr_dist_threshold(corr_dist_threshold)
  {}

  // skew symmetric matrix
  __host__ __device__ Eigen::Matrix3f skew_symmetric(const Eigen::Vector3f& x) const {
    Eigen::Matrix3f skew = Eigen::Matrix3f::Zero();
    skew(0, 1) = -x[2];
    skew(0, 2) = x[1];
    skew(1, 0) = x[2];
    skew(1, 2) = -x[0];
    skew(2, 0) = -x[1];
    skew(2, 1) = x[0];

    return skew;
  }

  // calculate derivatives
  template<typename Tuple>
  __host__ __device__ void operator() (Tuple tuple) const {
    const Eigen::Vector3f& mean_A = thrust::get<0>(tuple);
    const Eigen::Matrix3f& cov_A = thrust::get<1>(tuple);
    const Eigen::Vector3f transed_mean_A = R * mean_A + t;

    int target_index = thrust::get<2>(tuple);
    float target_distance = thrust::get<3>(tuple);
    if (target_distance > corr_dist_threshold) {
      return;
    }

    const Eigen::Vector3f& mean_B = thrust::raw_pointer_cast(target_points_ptr)[target_index];
    const Eigen::Matrix3f& cov_B = thrust::raw_pointer_cast(target_covs_ptr)[target_index];

    Eigen::Matrix3f RCR = R * cov_A * R.transpose();
    Eigen::Matrix3f skew_mean_A = skew_symmetric(transed_mean_A);

    Eigen::Vector3f d = mean_B - transed_mean_A;
    Eigen::Matrix3f RCR_inv = (cov_B + RCR).inverse();

    Eigen::Vector3f& loss = thrust::get<4>(tuple);
    Eigen::Matrix<float, 3, 6, Eigen::RowMajor>& J = thrust::get<5>(tuple);

    loss = (RCR_inv * d);
    J.block<3, 3>(0, 0) = RCR_inv * skew_mean_A;
    J.block<3, 3>(0, 3) = -RCR_inv;
  }

  const Eigen::Matrix3f R;
  const Eigen::Vector3f t;
  const int corr_dist_threshold;
  thrust::device_ptr<const Eigen::Vector3f> target_points_ptr;
  thrust::device_ptr<const Eigen::Matrix3f> target_covs_ptr;


};

struct is_nan_kernel_nn {
  template<typename T>
  __host__ __device__ bool operator() (const T& x) const {
    return isnan(x.data()[0]);
  }
};

} // namespace


void compute_derivatives_nn(const thrust::device_vector<Eigen::Vector3f>& src_points, 
                            const thrust::device_vector<Eigen::Matrix3f>& src_covs, 
                            const thrust::device_vector<Eigen::Vector3f>& target_points, 
                            const thrust::device_vector<Eigen::Matrix3f>& target_covs,
                            const thrust::device_vector<int>& k_indices, 
                            const thrust::device_vector<float>& k_distances, 
                            const int corr_dist_threshold,
                            const Eigen::Matrix<float, 6, 1>& x, 
                            thrust::device_vector<Eigen::Vector3f>& losses, 
                            thrust::device_vector<Eigen::Matrix<float, 3, 6, Eigen::RowMajor>>& Js) {

  float nan = std::nanf("");
  losses.resize(src_points.size());
  Js.resize(src_points.size());
  thrust::fill(losses.begin(), losses.end(), Eigen::Vector3f::Constant(nan));
  thrust::fill(Js.begin(), Js.end(), Eigen::Matrix<float, 3, 6>::Constant(nan));

  thrust::for_each(
    thrust::make_zip_iterator(thrust::make_tuple(
      src_points.begin(), src_covs.begin(),
      k_indices.begin(), k_distances.begin(),
      losses.begin(), Js.begin()
    )),
    thrust::make_zip_iterator(thrust::make_tuple(
      src_points.end(), src_covs.end(),
      k_indices.end(), k_distances.end(),
      losses.end(), Js.end()
    )),
    compute_derivatives_nn_kernel(x, target_points, target_covs, corr_dist_threshold)
  );

  // erase invalid points
  losses.erase(thrust::remove_if(losses.begin(), losses.end(), is_nan_kernel_nn()), losses.end());
  Js.erase(thrust::remove_if(Js.begin(), Js.end(), is_nan_kernel_nn()), Js.end());
}

} // namespace fast_gicp


#endif