#ifndef FAST_GICP_FAST_GICP_CUDA_CORE_CUH
#define FAST_GICP_FAST_GICP_CUDA_CORE_CUH

#include <memory>
#include <vector>
#include <cmath>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <thrust/device_vector.h>

#include <fast_gicp/gicp/gicp_settings.hpp>

struct cublasContext;

namespace thrust {
template<typename T>
class device_allocator;

template<typename T, typename Alloc>
class device_vector;
}  // namespace thrust

namespace fast_gicp {

class GaussianVoxelMap;

class FastGICPCudaCore {
public:
  using Points = thrust::device_vector<Eigen::Vector3f, thrust::device_allocator<Eigen::Vector3f>>;
  using Indices = thrust::device_vector<int, thrust::device_allocator<int>>;
  using Matrices = thrust::device_vector<Eigen::Matrix3f, thrust::device_allocator<Eigen::Matrix3f>>;

  // EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  FastGICPCudaCore();
  ~FastGICPCudaCore();

  void set_resolution(double resolution);
  void set_max_iterations(int itr);
  void set_rotation_epsilon(double eps);
  void set_transformation_epsilon(double eps);
  void set_correspondence_randomness(int k);

  void swap_source_and_target();
  void set_source_cloud(const std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>>& cloud);
  void set_target_cloud(const std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>>& cloud);

  void set_source_neighbors(int k, const std::vector<int>& neighbors);
  void set_target_neighbors(int k, const std::vector<int>& neighbors);
  void find_source_neighbors(int k);
  void find_target_neighbors(int k);

  void calculate_source_covariances(RegularizationMethod method);
  void calculate_target_covariances(RegularizationMethod method);

  void create_target_voxelmap();

  bool optimize(Eigen::Isometry3f& estimated);
  bool optimize(const Eigen::Isometry3f& initial_guess, Eigen::Isometry3f& estimated);
  void set_source_cloud_multi(thrust::device_vector<Eigen::Vector3f>& source_cloud);
  void set_source_cloud_multi(float* point_cloud, int point_count);

  void set_target_cloud_multi(float* point_cloud, int point_count, int* target_cloud_label);
  void set_target_cloud_multi(thrust::device_vector<Eigen::Vector3f>& target_cloud, thrust::device_vector<int>& target_cloud_label);

  void find_source_neighbors_multi(int k, int* cloud_pose_map, int* source_pose_label_map, int num_poses);
  void find_source_neighbors_multi(int k, thrust::device_vector<int>& source_pose_map_vec, thrust::device_vector<int>& source_pose_label_map_vec, int num_poses);

  void find_target_neighbors_multi(int k, int num_poses);
  void set_input(float* source_cloud, 
                 int source_point_count, 
                 float* target_cloud, 
                 int target_point_count,
                 int* cloud_pose_map,
                 int* target_cloud_label,
                 int* source_pose_label_map,
                 int num_poses);
  void set_input(thrust::device_vector<Eigen::Vector3f>& source_cloud, 
                 thrust::device_vector<Eigen::Vector3f>& target_cloud, 
                 thrust::device_vector<int>& source_pose_map_ptr,
                 thrust::device_vector<int>& target_cloud_label_ptr,
                 thrust::device_vector<int>& source_pose_label_map_ptr,
                 int num_poses);

  bool optimize_multi(
                      // float* source_cloud, 
                      // int source_point_count, 
                      // float* target_cloud, 
                      // int target_point_count,
                      // int* cloud_pose_map,
                      // int* target_cloud_label,
                      // int* source_pose_label_map,
                      // int num_poses,
                      std::vector<Eigen::Isometry3f>& estimated);

private:
  bool is_converged(const Eigen::Matrix<float, 6, 1>& delta) const;

private:
  cublasContext* cublas_handle;
  
  double resolution;

  int max_iterations;
  double rotation_epsilon;
  double transformation_epsilon;
  int k_correspondences;
  int num_poses;

  std::unique_ptr<Points> source_points;
  std::unique_ptr<Points> target_points;

  std::unique_ptr<Indices> source_neighbors;
  std::unique_ptr<Indices> target_neighbors;

  std::unique_ptr<Matrices> source_covariances;
  std::unique_ptr<Matrices> target_covariances;

  std::unique_ptr<Indices> source_pose_indices; // point range for every pose
  std::unique_ptr<Indices> source_pose_map; // pose index for every point, should be sorted
  std::unique_ptr<Indices> source_label_map; // label index for every point, computed before icp
  int max_pose_point_count; // maximum number of points amongst all poses, used to decide stride value of batch GEMM multiplication

  std::unique_ptr<Indices> target_label_indices; // label index rabge for every kabek
  std::unique_ptr<Indices> target_label_map; // sorted label index for every point in target

  std::unique_ptr<GaussianVoxelMap> voxelmap;
};

}  // namespace fast_gicp

#endif