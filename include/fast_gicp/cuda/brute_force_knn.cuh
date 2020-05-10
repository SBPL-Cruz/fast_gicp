#ifndef FAST_GICP_CUDA_BRUTE_FORCE_KNN_CUH
#define FAST_GICP_CUDA_BRUTE_FORCE_KNN_CUH

#pragma once

#include <Eigen/Core>

#include <thrust/sequence.h>
#include <thrust/functional.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/iterator/zip_iterator.h>

#include <nvbio/basic/vector_view.h>
#include <nvbio/basic/priority_queue.h>

#include <sophus/so3.hpp>

#include <chrono>

namespace fast_gicp {
  struct untie_pair_second {
    __device__ int operator() (thrust::pair<float, int>& p) const {
      return p.second;
    }
  };

  struct untie_pair_first {
    __device__ float operator() (thrust::pair<float, int>& p) const {
      return p.first;
    }
  };

namespace {
  struct neighborsearch_kernel {
    neighborsearch_kernel(int k, 
                          const thrust::device_vector<Eigen::Vector3f>& target, 
                          thrust::device_vector<thrust::pair<float, int>>& k_neighbors,
                          const thrust::device_vector<int>& source_pose_index_range,
                          const thrust::device_vector<Eigen::Matrix<float, 6, 1>>& adjusted_x0s,
                          const thrust::device_vector<int>& pose_mask_icp)
        : k(k), 
          num_target_points(target.size()), 
          target_points_ptr(target.data()), 
          k_neighbors_ptr(k_neighbors.data()),
          source_pose_index_range_ptr(source_pose_index_range.data()),
          source_pose_index_range_size(source_pose_index_range.size()),
          adjusted_x0s_ptr(adjusted_x0s.data()),
          adjusted_x0s_size(adjusted_x0s.size()),
          pose_icp_mask_ptr(pose_mask_icp.data()),
          pose_icp_mask_size(pose_mask_icp.size())
    {}

    template<typename Tuple>
    __host__ __device__ void operator()(Tuple& idx_x) const {
      // threadIdx doesn't work because thrust split for_each in two loops
      int idx = thrust::get<0>(idx_x);
      Eigen::Vector3f x = thrust::get<1>(idx_x);

      if (pose_icp_mask_size > 0) {
        int point_pose_index = thrust::get<2>(idx_x);     
        // printf("point_pose_index : %d\n", point_pose_index); 
        // printf("pose_icp_mask_ptr : %d\n", pose_icp_mask_ptr[point_pose_index]); 
        // printf("pose_icp_mask : %d, point_pose_index : %d\n", pose_icp_mask_ptr[point_pose_index], point_pose_index);
        if (pose_icp_mask_ptr[point_pose_index] == 1) return;
      }
      // Make a queue and sort according to distance from each target point
      // target points buffer & nn output buffer
      const Eigen::Vector3f* pts = thrust::raw_pointer_cast(target_points_ptr);
      thrust::pair<float, int>* k_neighbors = thrust::raw_pointer_cast(k_neighbors_ptr) + idx * k;

      // priority queue
      struct compare_type {
        bool operator()(const thrust::pair<float, int>& lhs, const thrust::pair<float, int>& rhs) {
          return lhs.first < rhs.first;
        }
      };

      typedef nvbio::vector_view<thrust::pair<float, int>*> vector_type;
      typedef nvbio::priority_queue<thrust::pair<float, int>, vector_type, compare_type> queue_type;
      queue_type queue(vector_type(0, k_neighbors - 1));

      // for(int i = k; i < num_target_points; i++) {
      int low_i = 0;
      int high_i = num_target_points;
      // If pose range info is there
      if (source_pose_index_range_size > 0) {
        // Get pose index of current point
        int target_range_index = thrust::get<3>(idx_x);     
        // printf("target_range_index : %d\n", target_range_index);
        low_i  = source_pose_index_range_ptr[target_range_index];
        high_i = source_pose_index_range_ptr[target_range_index + 1];
        // printf("Setting index range for point : %d, %d\n", low_i, high_i);
      }

      if (adjusted_x0s_size > 0) {
        // Transform the point to the adjusted pose
        int point_pose_index = thrust::get<2>(idx_x);    
        // printf("point_pose_index : %d\n", point_pose_index); 
        const Eigen::Matrix<float, 6, 1>& T_x = adjusted_x0s_ptr[point_pose_index];
        Eigen::Matrix4f trans = Eigen::Matrix4f::Identity();
        trans.block<3, 3>(0, 0) = Sophus::SO3f::exp(T_x.head<3>()).matrix();
        trans.block<3, 1>(0, 3) = T_x.tail<3>();

        Eigen::Vector4f x_h;
        x_h[0] = x[0];
        x_h[1] = x[1];
        x_h[2] = x[2];
        x_h[3] = 1;
        x_h = trans * x_h;
        x = x_h.head<3>();
        // printf("x:%f, y:%f, z:%f\n", x[0], x[1], x[2]);
      }

      for(int i = low_i; i < low_i + k; i++) {
        float sq_dist = (pts[i] - x).squaredNorm();
        queue.push(thrust::make_pair(sq_dist, i));
      }
      for(int i = low_i + k; i < high_i; i++) {
        
        float sq_dist = (pts[i] - x).squaredNorm();
        if(sq_dist < queue.top().first) {
          queue.pop();
          queue.push(thrust::make_pair(sq_dist, i));
        }
      }
    }

    const int k;
    const int num_target_points;
    thrust::device_ptr<const Eigen::Vector3f> target_points_ptr;
    thrust::device_ptr<const int> source_pose_index_range_ptr;
    const int source_pose_index_range_size;

    thrust::device_ptr<thrust::pair<float, int>> k_neighbors_ptr;
    thrust::device_ptr<const Eigen::Matrix<float, 6, 1>> adjusted_x0s_ptr;
    const int adjusted_x0s_size;

    thrust::device_ptr<const int> pose_icp_mask_ptr;
    const int pose_icp_mask_size;
  };

  struct sorting_kernel {
    sorting_kernel(int k, thrust::device_vector<thrust::pair<float, int>>& k_neighbors) : k(k), k_neighbors_ptr(k_neighbors.data()) {}

    __host__ __device__ void operator()(int idx) const {
      // target points buffer & nn output buffer
      thrust::pair<float, int>* k_neighbors = thrust::raw_pointer_cast(k_neighbors_ptr) + idx * k;

      // priority queue
      struct compare_type {
        bool operator()(const thrust::pair<float, int>& lhs, const thrust::pair<float, int>& rhs) {
          return lhs.first < rhs.first;
        }
      };

      typedef nvbio::vector_view<thrust::pair<float, int>*> vector_type;
      typedef nvbio::priority_queue<thrust::pair<float, int>, vector_type, compare_type> queue_type;
      queue_type queue(vector_type(k, k_neighbors - 1));
      queue.m_size = k;

      for(int i = 0; i < k; i++) {
        thrust::pair<float, int> poped = queue.top();
        queue.pop();

        k_neighbors[k - i - 1] = poped;
      }
    }

    const int k;
    thrust::device_ptr<thrust::pair<float, int>> k_neighbors_ptr;
  };
}

static void brute_force_knn_search(const thrust::device_vector<Eigen::Vector3f>& source, 
                                   const thrust::device_vector<Eigen::Vector3f>& target, 
                                   int k, 
                                   thrust::device_vector<thrust::pair<float, int>>& k_neighbors, 
                                   const thrust::device_vector<int>& source_target_index_map = thrust::device_vector<int>(0),
                                   const thrust::device_vector<int>& target_index_range = thrust::device_vector<int>(0),
                                   const thrust::device_vector<int>& source_pose_map = thrust::device_vector<int>(0),
                                   const thrust::device_vector<Eigen::Matrix<float, 6, 1>> adjusted_x0s = thrust::device_vector<Eigen::Matrix<float, 6, 1>>(0),
                                   const thrust::device_vector<int>& pose_mask_icp = thrust::device_vector<int>(0),                                   
                                   bool do_sort=false) {
  /*
   * Below two can be provided to apply transformation before doing NN:
   *  - adjusted_x0s - transformation to apply to source point before computing distance
   *  - source_pose_map : the pose index of every point in source, maps it to a transform in adjusted_x0s
   * Below two can be provided to speed up KNN by searching segments of target array: 
   *  - source_target_index_map : a mapping from source point to a target index which determines the range in target to search over
   *  - target_index_range : target index range to use for every pose in source_pose_map
   *  - adjusted_x0s : the transform to apply to a point in source before computing distance
   */
  // assert(source_target_index_map.size() == source.size());
  // assert(source_pose_map.size() == source.size());
  std::chrono::time_point<std::chrono::system_clock> start, end;
  start = std::chrono::system_clock::now();
  printf("target size : %d\n", target.size());
  printf("pose_mask_icp size : %d\n", pose_mask_icp.size());
  printf("adjusted_x0s size : %d\n", adjusted_x0s.size());
  printf("target_index_range size : %d\n", target_index_range.size());
  printf("source_pose_map size : %d\n", source_pose_map.size());
  printf("source_target_index_map size : %d\n", source_target_index_map.size());
  // thrust::copy(
  //   pose_mask_icp.begin(),
  //   pose_mask_icp.end(), 
  //   std::ostream_iterator<int>(std::cout, " ")
  // );  
  // printf("\n");
  // thrust::copy(
  //   source_target_index_map.begin(),
  //   source_target_index_map.end(), 
  //   std::ostream_iterator<int>(std::cout, " ")
  // );  
  // printf("\n");
  // thrust::copy(
  //   target_index_range.begin(),
  //   target_index_range.end(), 
  //   std::ostream_iterator<int>(std::cout, " ")
  // );  
  // printf("\n");
  thrust::device_vector<int> d_indices(source.size());
  thrust::sequence(d_indices.begin(), d_indices.end());

  auto first = thrust::make_zip_iterator(thrust::make_tuple(d_indices.begin(), source.begin(), source_pose_map.begin(), source_target_index_map.begin()));
  auto last = thrust::make_zip_iterator(thrust::make_tuple(d_indices.end(), source.end(), source_pose_map.end(), source_target_index_map.end()));

  // nvbio::priority_queue requires (k + 1) working space
  if (k_neighbors.size() == 0) {
    k_neighbors.resize(source.size() * k, thrust::make_pair(-1.0f, -1));
  } else {
    thrust::fill(k_neighbors.begin(), k_neighbors.end(), thrust::make_pair(-1.0f, -1));
  }
  thrust::for_each(first, last, 
                   neighborsearch_kernel(k, target, k_neighbors, target_index_range, adjusted_x0s, pose_mask_icp));

  if(do_sort) {
    thrust::for_each(d_indices.begin(), d_indices.end(), sorting_kernel(k, k_neighbors));
  }
  end = std::chrono::system_clock::now();
  std::chrono::duration<double> elapsed_seconds = end-start;

  printf("brute_force_knn_search() done, took : %f\n", elapsed_seconds.count());
}

} // namespace fast_gicp


#endif