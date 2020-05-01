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

  // Copy a raw 3D array of points into an eigen array of points
  template<typename Tuple>
  __host__ __device__ void operator() (Tuple tuple) const {
    const int cloud_idx = thrust::get<0>(tuple);
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

  // Calculate indices where array value is not equal to previous value
  template<typename Tuple>
  __host__ __device__ void operator() (Tuple tuple) const {
    const int pose_index_next = thrust::get<0>(tuple);
    const int pose_index = thrust::get<1>(tuple);

    const int uneq_index = thrust::get<2>(tuple);
    int& uneq_array_elem = thrust::get<3>(tuple);

    if (pose_index_next != pose_index) 
    {
      // Need to add one because next is 1 shifted, by adding one we get index in original point array
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

  thrust::counting_iterator<int> first(0);
  thrust::counting_iterator<int> last = first + src_points_count;

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
                        const int cloud_point_count,
                        const int num_poses,
                        thrust::device_vector<int>& unequal_indices,
                        int& max_pose_point_count) {
    /* Extract the ranges in cloud array for every pose to restrict nearest neighbour to within that range
     * cloud_pose_map_vec - contains mapping of every point to a pose index
     *
     */
    thrust::device_vector<int> cloud_pose_map_next_vec;
    cloud_pose_map_next_vec.assign(cloud_pose_map_vec.begin() + 1, cloud_pose_map_vec.end());
    // Add last element to make length equal
    cloud_pose_map_next_vec.push_back(cloud_pose_map_vec.back());

    thrust::counting_iterator<int> first(0);
    thrust::counting_iterator<int> last = first + cloud_point_count;

    unequal_indices.resize(cloud_point_count, -1);

    // Calculate the indices in cloud array where the pose index changes (not equal to previous value)
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

    // Calculate the maximum number of points from all poses
    thrust::device_vector<int> unequal_indices_prev(unequal_indices.begin(), unequal_indices.end()-1);
    thrust::device_vector<int> pose_point_count(unequal_indices_prev.size());
    thrust::transform(
        unequal_indices.begin() + 1, unequal_indices.end(), 
        unequal_indices_prev.begin(), pose_point_count.begin(), 
        thrust::minus<float>()
    );
    thrust::device_vector<int>::iterator max_elem_iter = thrust::max_element(pose_point_count.begin(), pose_point_count.end());

    max_pose_point_count = pose_point_count[max_elem_iter - pose_point_count.begin()];
    // thrust::copy(
    //   unequal_indices.begin() + 1,
    //   unequal_indices.end(), 
    //   std::ostream_iterator<int>(std::cout, " ")
    // );
    // printf("\n");

    // thrust::copy(
    //   unequal_indices_prev.begin(),
    //   unequal_indices_prev.end(), 
    //   std::ostream_iterator<int>(std::cout, " ")
    // );
    // printf("\n");
    // Unequal size should be one higher since it contains both lower and upper limit for every pose
    printf("Num poses : %d, size of unequal indices : %d, max_point_count : %d\n", num_poses, unequal_indices.size(), max_pose_point_count);
} // namespace fast_gicp

}


#endif