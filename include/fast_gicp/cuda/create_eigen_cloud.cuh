#ifndef FAST_GICP_CUDA_CREATE_EIGEN_CLOUD_CUH
#define FAST_GICP_CUDA_CREATE_EIGEN_CLOUD_CUH

#pragma once

#include <Eigen/Core>
#include <sophus/so3.hpp>

#include <thrust/remove.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

namespace fast_gicp {

namespace  {

struct create_eigen_cloud_kernel {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  create_eigen_cloud_kernel(const float* point_cloud, const int point_cloud_count)
  : device_point_count(point_cloud_count),
    device_point_cloud(point_cloud)
  {}

  // calculate derivatives
  template<typename Tuple>
  __host__ __device__ void operator() (Tuple tuple) const {
    const uint cloud_idx = thrust::get<0>(tuple);
    // printf("cloud_idx : %d\n", cloud_idx);
    Eigen::Vector3f& eigen_cloud_point = thrust::get<1>(tuple);

    eigen_cloud_point(0) = device_point_cloud[cloud_idx + 0*device_point_count];
    eigen_cloud_point(1) = device_point_cloud[cloud_idx + 1*device_point_count];
    eigen_cloud_point(2) = device_point_cloud[cloud_idx + 2*device_point_count];
  }

  const int device_point_count;
  const float* device_point_cloud;
};

struct extract_pose_index_kernel {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  extract_pose_index_kernel()
  {}

  // calculate derivatives
  template<typename Tuple>
  __host__ __device__ void operator() (Tuple tuple) const {
    const int pose_index_next = thrust::get<0>(tuple);
    const int pose_index = thrust::get<1>(tuple);

    const int uneq_index = thrust::get<2>(tuple);
    int& uneq_array_elem = thrust::get<3>(tuple);

    if (pose_index_next != pose_index) 
    {
      // Need to add one because next is 1 shifted
      uneq_array_elem = uneq_index + 1;
    }

  }

};

struct is_invalid_kernel {
  
  __host__ __device__ bool operator() (const int& x) const {
    return x == -1;
  }
};

} // namespace


void create_eigen_cloud(float* point_cloud,
                        const int src_points_count,
                        thrust::device_vector<Eigen::Vector3f>& src_points) {

  printf("Creating eigen clouds, point count : %d\n", src_points_count);

  src_points.resize(src_points_count);
  float nan = std::nanf("");
  thrust::fill(src_points.begin(), src_points.end(), Eigen::Vector3f::Constant(nan));

  thrust::counting_iterator<uint> first(0);
  thrust::counting_iterator<uint> last = first + src_points_count;

  thrust::for_each(
    thrust::make_zip_iterator( 
      thrust::make_tuple(first, src_points.begin()) 
    ), 
    thrust::make_zip_iterator( 
      thrust::make_tuple(last, src_points.end())
    ), 
    create_eigen_cloud_kernel(point_cloud, src_points_count)
  );
}

void extract_pose_indices(const thrust::device_vector<int>& cloud_pose_map_vec,
                        int cloud_point_count,
                        int num_poses,
                        thrust::device_vector<int>& unequal_indices) {
     
    thrust::device_vector<int> cloud_pose_map_next_vec;
    cloud_pose_map_next_vec.assign(cloud_pose_map_vec.begin() + 1, cloud_pose_map_vec.end());
    // Add last element to make length equal
    cloud_pose_map_next_vec.push_back(cloud_pose_map_vec.back());

    thrust::counting_iterator<uint> first(0);
    thrust::counting_iterator<uint> last = first + cloud_point_count;

    // thrust::device_vector<int> unequal_indices(cloud_point_count, -1);
    // unequal_indices.reset(new thrust::device_vector<int>(cloud_point_count));
    // thrust::fill(unequal_indices.begin(), unequal_indices.end(), -1);
    unequal_indices.resize(cloud_point_count, -1);

    thrust::for_each(
    thrust::make_zip_iterator( 
        thrust::make_tuple(cloud_pose_map_next_vec.begin(), cloud_pose_map_vec.begin(), first, unequal_indices.begin()) 
      ), 
      thrust::make_zip_iterator( 
        thrust::make_tuple(cloud_pose_map_next_vec.end(), cloud_pose_map_vec.end(), last, unequal_indices.end())
      ), 
      extract_pose_index_kernel()
    );

    unequal_indices.erase(thrust::remove_if(unequal_indices.begin(), unequal_indices.end(), is_invalid_kernel()), unequal_indices.end());
    unequal_indices.insert(unequal_indices.begin(), 0);
    unequal_indices.push_back(cloud_point_count);
    // thrust::copy(
    //   unequal_indices.begin(),
    //   unequal_indices.end(), 
    //   std::ostream_iterator<int>(std::cout, " ")
    // );
    // Unequal size should be one higher since it contains both lower and upper limit for every pose
    printf("Num poses : %d, size of unequal indices : %d\n", num_poses, unequal_indices.size());
} // namespace fast_gicp

}


#endif