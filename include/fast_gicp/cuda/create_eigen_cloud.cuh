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

  extract_pose_index_kernel(thrust::device_vector<int>& unequal_indices)
  : unequal_indices_ptr(unequal_indices.data())
  {}

  // Calculate indices where array value is not equal to previous value
  template<typename Tuple>
  __host__ __device__ void operator() (Tuple tuple) const {
    const int pose_index_next = thrust::get<0>(tuple);
    const int pose_index = thrust::get<1>(tuple);

    const int uneq_index = thrust::get<2>(tuple);
    // int& uneq_array_elem = thrust::get<3>(tuple);

    if (pose_index_next != pose_index) 
    {
      // Need to add one because next is 1 shifted, by adding one we get index in original point array
      // uneq_array_elem = uneq_index + 1;
      unequal_indices_ptr[pose_index] = uneq_index + 1;
    }

  }

  thrust::device_ptr<int> unequal_indices_ptr;

};

struct is_invalid_kernel {
  
  __host__ __device__ bool operator() (const int& x) const {
    return x == -1;
  }
};

struct fix_empty_poses{

  fix_empty_poses(){}

  __host__ __device__
  int operator()(const int& end_i, const int& start_i) const
  {
    // For poses that are empty (-1 end indice), make end indice same as start indice
    if (end_i == -1)
    {
        return start_i;
    }
    else
    {
      return end_i;
    }
  }
};


struct fix_empty_poses_loop {
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  fix_empty_poses_loop(thrust::device_vector<int>& unequal_indices)
  : unequal_indices_ptr(unequal_indices.data())
  {}

  // Calculate indices where array value is not equal to previous value
  template<typename Tuple>
  __host__ __device__ void operator() (Tuple tuple) const {
    int uneq_index_i = thrust::get<1>(tuple);
    int& uneq_index_val = thrust::get<0>(tuple);
    if (uneq_index_val != -1) return;

    while(unequal_indices_ptr[uneq_index_i] == -1) uneq_index_i--;
    uneq_index_val =  unequal_indices_ptr[uneq_index_i];
  }

  thrust::device_ptr<int> unequal_indices_ptr;

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
    /* Extract the ranges in cloud array for every pose (or segmentation label) to restrict nearest neighbour to within that range
     * cloud_pose_map_vec - contains mapping of every point to a pose index
     * unequal_indices - output where index i denotes the start range and i + 1 denotes end range
     */
    // thrust::copy(
    //   cloud_pose_map_vec.begin() + 1,
    //   cloud_pose_map_vec.end(), 
    //   std::ostream_iterator<int>(std::cout, " ")
    // );
    // printf("\n");
    printf("extract_pose_indices()\n");
    printf("cloud_pose_map_vec() size : %d\n", cloud_pose_map_vec.size());
    printf("cloud_point_count : %d\n", cloud_point_count);
    thrust::device_vector<int> cloud_pose_map_next_vec;
    cloud_pose_map_next_vec.assign(cloud_pose_map_vec.begin() + 1, cloud_pose_map_vec.end());
    // Add last element to make length equal
    cloud_pose_map_next_vec.push_back(cloud_pose_map_vec.back());

    thrust::counting_iterator<int> first(0);
    thrust::counting_iterator<int> last = first + cloud_point_count;

    // unequal_indices.resize(cloud_point_count, -1);
    // Find max value in the mapping and use it to set the size of unequal indices
    thrust::device_vector<const int>::iterator iter = thrust::max_element(cloud_pose_map_vec.begin(), cloud_pose_map_vec.end());
    int max_val = *iter;
    printf("max index : %d\n", max_val);
    unequal_indices.resize(max_val, -1);

    // Calculate the end indices in cloud array where the pose index changes (not equal to previous value)
    // If a pose has zero points, the index will be set to -1
    thrust::for_each(
    thrust::make_zip_iterator( 
        thrust::make_tuple(cloud_pose_map_next_vec.begin(), cloud_pose_map_vec.begin(), first) 
      ), 
      thrust::make_zip_iterator( 
        thrust::make_tuple(cloud_pose_map_next_vec.end(), cloud_pose_map_vec.end(), last)
      ), 
      extract_pose_index_kernel(unequal_indices)
    );

    // unequal_indices.erase(thrust::remove_if(unequal_indices.begin(), unequal_indices.end(), is_invalid_kernel()), unequal_indices.end());
    unequal_indices.insert(unequal_indices.begin(), 0);
    unequal_indices.push_back(cloud_point_count);

    // thrust::copy(
    //   unequal_indices.begin(),
    //   unequal_indices.end(), 
    //   std::ostream_iterator<int>(std::cout, " ")
    // );
    // printf("\n");

    // For poses with no points, put the start indice in place of -1 at the end indice so that count is 0
    // thrust::transform(unequal_indices.begin(), unequal_indices.end(), unequal_indices.begin() + 1, unequal_indices.begin(), fix_empty_poses());
    last = first + unequal_indices.size();
    thrust::for_each(
    thrust::make_zip_iterator( 
        thrust::make_tuple(unequal_indices.begin(), first) 
      ), 
      thrust::make_zip_iterator( 
        thrust::make_tuple(unequal_indices.end(), last)
      ), 
      fix_empty_poses_loop(unequal_indices)
    );

    // thrust::copy(
    //   unequal_indices.begin(),
    //   unequal_indices.end(), 
    //   std::ostream_iterator<int>(std::cout, " ")
    // );
    // printf("\n");
    // printf("Num poses : %d, size of unequal indices : %d, max_point_count : %d\n", num_poses, unequal_indices.size(), max_pose_point_count);
    
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
    //   unequal_indices.begin(),
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