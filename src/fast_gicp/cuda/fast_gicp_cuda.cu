#include <fast_gicp/cuda/fast_gicp_cuda.cuh>

#include <sophus/so3.hpp>

#include <thrust/device_new.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <cusolverDn.h>
#include <cuda_runtime.h>

#include <fast_gicp/cuda/brute_force_knn.cuh>
#include <fast_gicp/cuda/covariance_estimation.cuh>
#include <fast_gicp/cuda/gaussian_voxelmap.cuh>
#include <fast_gicp/cuda/compute_derivatives.cuh>
#include <fast_gicp/cuda/create_eigen_cloud.cuh>
#include <fast_gicp/cuda/compute_derivatives_nn.cuh>

namespace fast_gicp {

FastGICPCudaCore::FastGICPCudaCore() {
  // warming up GPU
  cudaDeviceSynchronize();
  cublasCreate(&cublas_handle);
  resolution = 1.0;
  max_iterations = 64;
  rotation_epsilon = 2e-3;
  transformation_epsilon = 5e-4;
}
FastGICPCudaCore ::~FastGICPCudaCore() {
  cublasDestroy(cublas_handle);
}

void FastGICPCudaCore::set_resolution(double resolution) {
  this->resolution = resolution;
}

void FastGICPCudaCore::set_max_iterations(int itr) {
  this->max_iterations = itr;
}

void FastGICPCudaCore::set_rotation_epsilon(double eps) {
  this->rotation_epsilon = eps;
}

void FastGICPCudaCore::set_transformation_epsilon(double eps) {
  this->transformation_epsilon = eps;
}

void FastGICPCudaCore::set_correspondence_randomness(int k) {
  this->k_correspondences = k;
}

void FastGICPCudaCore::swap_source_and_target() {
  if(source_points && target_points) {
    source_points.swap(target_points);
  }
  if(source_neighbors && target_neighbors) {
    source_neighbors.swap(target_neighbors);
  }
  if(source_covariances && target_covariances) {
    source_covariances.swap(target_covariances);
  }

  if(!target_points || !target_covariances) {
    return;
  }

  if(!voxelmap) {
    voxelmap.reset(new GaussianVoxelMap(resolution));
  }
  voxelmap->create_voxelmap(*target_points, *target_covariances);
}

void FastGICPCudaCore::set_source_cloud(const std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>>& cloud) {
  thrust::host_vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>> points(cloud.begin(), cloud.end());
  source_points.reset(new Points(points));
}

void FastGICPCudaCore::set_target_cloud(const std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>>& cloud) {
  thrust::host_vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>> points(cloud.begin(), cloud.end());
  target_points.reset(new Points(points));
}


void FastGICPCudaCore::set_source_neighbors(int k, const std::vector<int>& neighbors) {
  assert(k * source_points->size() == neighbors.size());
  thrust::host_vector<int> k_neighbors(neighbors.begin(), neighbors.end());

  if(!source_neighbors) {
    source_neighbors.reset(new thrust::device_vector<int>());
  }

  *source_neighbors = k_neighbors;
}

void FastGICPCudaCore::set_target_neighbors(int k, const std::vector<int>& neighbors) {
  assert(k * target_points->size() == neighbors.size());
  thrust::host_vector<int> k_neighbors(neighbors.begin(), neighbors.end());

  if(!target_neighbors) {
    target_neighbors.reset(new thrust::device_vector<int>());
  }

  *target_neighbors = k_neighbors;
}

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

struct copy_label_functor{
  copy_label_functor(const thrust::device_vector<int>& src_pose_label)
  : src_pose_label_ptr(src_pose_label.data())
    {}
    __host__ __device__
    int operator()(const int& i) const
    {
        return src_pose_label_ptr[i];
    }
    thrust::device_ptr<const int> src_pose_label_ptr;
};

void FastGICPCudaCore::find_source_neighbors(int k) {
  assert(source_points);

  thrust::device_vector<thrust::pair<float, int>> k_neighbors;
  brute_force_knn_search(*source_points, *source_points, k, k_neighbors);

  if(!source_neighbors) {
    source_neighbors.reset(new thrust::device_vector<int>(k_neighbors.size()));
  } else {
    source_neighbors->resize(k_neighbors.size());
  }
  thrust::transform(k_neighbors.begin(), k_neighbors.end(), source_neighbors->begin(), untie_pair_second());
}

void FastGICPCudaCore::find_target_neighbors(int k) {
  assert(target_points);

  thrust::device_vector<thrust::pair<float, int>> k_neighbors;
  brute_force_knn_search(*target_points, *target_points, k, k_neighbors);

  if(!target_neighbors) {
    target_neighbors.reset(new thrust::device_vector<int>(k_neighbors.size()));
  } else {
    target_neighbors->resize(k_neighbors.size());
  }
  thrust::transform(k_neighbors.begin(), k_neighbors.end(), target_neighbors->begin(), untie_pair_second());
}

void FastGICPCudaCore::calculate_source_covariances(RegularizationMethod method) {
  assert(source_points && source_neighbors);
  int k = source_neighbors->size() / source_points->size();

  if(!source_covariances) {
    source_covariances.reset(new thrust::device_vector<Eigen::Matrix3f>(source_points->size()));
  }
  covariance_estimation(*source_points, k, *source_neighbors, *source_covariances, method);
}

void FastGICPCudaCore::calculate_target_covariances(RegularizationMethod method) {
  assert(target_points && target_neighbors);
  int k = target_neighbors->size() / target_points->size();

  if(!target_covariances) {
    target_covariances.reset(new thrust::device_vector<Eigen::Matrix3f>(target_points->size()));
  }
  covariance_estimation(*target_points, k, *target_neighbors, *target_covariances, method);
}

void FastGICPCudaCore::create_target_voxelmap() {
  assert(target_points && target_covariances);
  if(!voxelmap) {
    voxelmap.reset(new GaussianVoxelMap(resolution));
  }
  voxelmap->create_voxelmap(*target_points, *target_covariances);

  // cudaDeviceSynchronize();
}

bool FastGICPCudaCore::is_converged(const Eigen::Matrix<float, 6, 1>& delta) const {
  Eigen::Matrix3f R = Sophus::SO3f::exp(delta.head<3>()).matrix() - Eigen::Matrix3f::Identity();
  Eigen::Vector3f t = delta.tail<3>();

  Eigen::Matrix3f r_delta = 1.0 / rotation_epsilon * R.array().abs();
  Eigen::Vector3f t_delta = 1.0 / transformation_epsilon * t.array().abs();

  return std::max(r_delta.maxCoeff(), t_delta.maxCoeff()) < 1;
}

bool FastGICPCudaCore::optimize(Eigen::Isometry3f& estimated) {
  Eigen::Isometry3f initial_guess = Eigen::Isometry3f::Identity();
  return optimize(initial_guess, estimated);
}

bool FastGICPCudaCore::optimize(const Eigen::Isometry3f& initial_guess, Eigen::Isometry3f& estimated) {
  assert(source_points && source_covariances && voxelmap);

  Eigen::Matrix<float, 6, 1> x0;
  x0.head<3>() = Sophus::SO3f(initial_guess.linear()).log();
  x0.tail<3>() = initial_guess.translation();

  if(x0.head<3>().norm() < 1e-2) {
    x0.head<3>() = (Eigen::Vector3f::Random()).normalized() * 1e-2;
  }

  thrust::device_vector<Eigen::Vector3f> losses;                            // 3N error vector
  thrust::device_vector<Eigen::Matrix<float, 3, 6, Eigen::RowMajor>> Js;    // RowMajor 3Nx6 -> ColMajor 6x3N

  thrust::device_ptr<float> JJ_ptr = thrust::device_new<float>(6 * 6);
  thrust::device_ptr<float> J_loss_ptr = thrust::device_new<float>(6);

  bool converged = false;
  for(int i = 0; i < max_iterations; i++) {
    // Js is transpose of error jacobian
    compute_derivatives(*source_points, *source_covariances, 
      *target_points, *target_covariances, *voxelmap, x0, losses, Js);

    // gauss newton
    float alpha = 1.0f;
    float beta = 0.0f;

    int cols = 3 * losses.size();

    float* Js_ptr = thrust::reinterpret_pointer_cast<float*>(Js.data());
    // Computing J^T x J
    cublasSgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T, 6, 6, cols, &alpha, Js_ptr, 6, Js_ptr, 6, &beta, thrust::raw_pointer_cast(JJ_ptr), 6);

    // Computing J^T x L
    float* loss_ptr = thrust::reinterpret_pointer_cast<float*>(losses.data());
    cublasSgemv(cublas_handle, CUBLAS_OP_N, 6, cols, &alpha, Js_ptr, 6, loss_ptr, 1, &beta, thrust::raw_pointer_cast(J_loss_ptr), 1);

    Eigen::Matrix<float, 6, 6> JJ;
    cublasGetMatrix(6, 6, sizeof(float), thrust::raw_pointer_cast(JJ_ptr), 6, JJ.data(), 6);

    Eigen::Matrix<float, 6, 1> J_loss;
    cublasGetVector(6, sizeof(float), thrust::raw_pointer_cast(J_loss_ptr), 1, J_loss.data(), 1);

    Eigen::Matrix<float, 6, 1> delta = JJ.llt().solve(J_loss);

    // update parameters
    x0.head<3>() = (Sophus::SO3f::exp(-delta.head<3>()) * Sophus::SO3f::exp(x0.head<3>())).log();
    x0.tail<3>() -= delta.tail<3>();

    if(is_converged(delta)) {
      converged = true;
      break;
    }
  }

  estimated.setIdentity();
  estimated.linear() = Sophus::SO3f::exp(x0.head<3>()).matrix();
  estimated.translation() = x0.tail<3>();

  return converged;
}

void FastGICPCudaCore::set_source_cloud_multi(float* source_cloud, int source_point_count) {
  // float* device_point_cloud;
  // cudaMalloc(&device_point_cloud, 3 * point_count * sizeof(float));
  // cudaMemcpy(point_cloud, device_point_cloud, 3 * point_count * sizeof(float), cudaMemcpyHostToDevice);
  // create_eigen_cloud(device_point_cloud, point_count, *source_points);
  std::chrono::time_point<std::chrono::system_clock> start, end;
  start = std::chrono::system_clock::now();
  thrust::device_vector<Eigen::Vector3f> source_points_local;
  float* device_point_cloud;
  cudaMalloc(&device_point_cloud, 3 * source_point_count * sizeof(float));
  cudaMemcpy(device_point_cloud, source_cloud, 3 * source_point_count * sizeof(float), cudaMemcpyHostToDevice);
  create_eigen_cloud(device_point_cloud, source_point_count, source_points_local);

  end = std::chrono::system_clock::now();
  std::chrono::duration<double> elapsed_seconds = end-start;
  printf("*************set_source_cloud_multi time : %f*************\n", elapsed_seconds.count());

  // Store source points in class object
  thrust::host_vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>> points(source_points_local.begin(), source_points_local.end());
  source_points.reset(new Points(points));
}

void FastGICPCudaCore::set_target_cloud_multi(float* target_cloud, int target_point_count, int* target_cloud_label) {
  float* device_point_cloud;
  // cudaMalloc(&device_point_cloud, 3 * point_count * sizeof(float));
  // cudaMemcpy(device_point_cloud, point_cloud, 3 * point_count * sizeof(float), cudaMemcpyHostToDevice);
  // create_eigen_cloud(device_point_cloud, point_count, *target_points);

  thrust::device_vector<Eigen::Vector3f> target_points_local;
  cudaMalloc(&device_point_cloud, 3 * target_point_count * sizeof(float));
  cudaMemcpy(device_point_cloud, target_cloud, 3 * target_point_count * sizeof(float), cudaMemcpyHostToDevice);
  create_eigen_cloud(device_point_cloud, target_point_count, target_points_local);

  // Sort the target points according to increasing order of label, also sort the labels
  thrust::device_vector<int> target_label_map_vec(target_cloud_label, target_cloud_label + target_point_count);
  thrust::sort_by_key(target_label_map_vec.begin(), target_label_map_vec.end(), target_points_local.begin());
  // thrust::copy(
  //   target_label_map_vec.begin(),
  //   target_label_map_vec.end(), 
  //   std::ostream_iterator<int>(std::cout, " ")
  // );
  // thrust::copy(
  //   target_points_local.begin(),
  //   target_points_local.end(), 
  //   std::ostream_iterator<int>(std::cout, " ")
  // );
  thrust::host_vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>> points(target_points_local.begin(), target_points_local.end());
  target_points.reset(new Points(points));

  // Save sorted map to class object
  target_label_map.reset(new Indices(target_label_map_vec));
}

void FastGICPCudaCore::find_source_neighbors_multi(int k, int* source_pose_map_ptr, int* source_pose_label_map, int num_poses) {
  assert(source_points);
  int source_point_count = source_points->size();
  // For source KNN need to know range of pose indices in source cloud array
  thrust::device_vector<int> source_pose_map_vec(source_pose_map_ptr, source_pose_map_ptr + source_point_count);
  thrust::device_vector<int> pose_indices;
  extract_pose_indices(source_pose_map_vec, source_point_count, num_poses, pose_indices, max_pose_point_count);

  // For each point in source array, KNN will be done within the pose index range
  thrust::device_vector<thrust::pair<float, int>> k_neighbors;
  brute_force_knn_search(*source_points, *source_points, k, k_neighbors, source_pose_map_vec, pose_indices);

  if(!source_neighbors) {
    source_neighbors.reset(new thrust::device_vector<int>(k_neighbors.size()));
  } else {
    source_neighbors->resize(k_neighbors.size());
  }
  thrust::transform(k_neighbors.begin(), k_neighbors.end(), source_neighbors->begin(), untie_pair_second());

  // For each point in source, use corresponding point to pose map and pose to label map to create a direct source to label map for every point in source
  thrust::device_vector<int> source_pose_label_map_vec(source_pose_label_map, source_pose_label_map + num_poses);
  thrust::device_vector<int> source_label_map_vec(source_point_count);
  thrust::transform(
    source_pose_map_vec.begin(), source_pose_map_vec.end(), source_label_map_vec.begin(), copy_label_functor(source_pose_label_map_vec)
  );
  
  source_pose_indices.reset(new Indices(pose_indices));
  source_pose_map.reset(new Indices(source_pose_map_vec));
  source_label_map.reset(new Indices(source_label_map_vec));
  // thrust::copy(
  //   source_neighbors->begin(),
  //   source_neighbors->end(), 
  //   std::ostream_iterator<int>(std::cout, " ")
  // );
  // thrust::copy(
  //   source_label_map->begin(),
  //   source_label_map->end(), 
  //   std::ostream_iterator<int>(std::cout, " ")
  // );

}

void FastGICPCudaCore::find_target_neighbors_multi(int k, int num_poses) {
  assert(target_points);
  
  int target_point_count = target_points->size();

  thrust::device_vector<int> label_indices;
  int max_label_point_count;
  extract_pose_indices(*target_label_map, target_point_count, num_poses, label_indices, max_label_point_count);
  target_label_indices.reset(new Indices(label_indices));
  // thrust::copy(
  //   target_label_indices->begin(),
  //   target_label_indices->end(), 
  //   std::ostream_iterator<int>(std::cout, " ")
  // );

  thrust::device_vector<thrust::pair<float, int>> k_neighbors;
  // brute_force_knn_search(*target_points, *target_points, k, k_neighbors);
  brute_force_knn_search(*target_points, *target_points, k, k_neighbors, *target_label_map, label_indices);

  if(!target_neighbors) {
    target_neighbors.reset(new thrust::device_vector<int>(k_neighbors.size()));
  } else {
    target_neighbors->resize(k_neighbors.size());
  }
  thrust::transform(k_neighbors.begin(), k_neighbors.end(), target_neighbors->begin(), untie_pair_second());
}

bool FastGICPCudaCore::optimize_multi(float* source_cloud, 
                                      int source_point_count, 
                                      float* target_cloud, 
                                      int target_point_count,
                                      int* source_pose_map_ptr,
                                      int* target_cloud_label_ptr,
                                      int* source_pose_label_map_ptr,
                                      int num_poses,
                                      std::vector<Eigen::Isometry3f>& estimated) {
  /*
   * source_pose_map - map a point in source to its pose index
   * source_pose_label_map - map a pose in source to its label index
   * target_cloud_label - map a point in target to its label index (obtained from semantic segmentation)
   */

  std::chrono::time_point<std::chrono::system_clock> start, end;
  int solver_type = 1;
  start = std::chrono::system_clock::now();
  estimated.resize(num_poses, Eigen::Isometry3f::Identity());

  //// Source cloud, contains all points of all rendered poses
  set_source_cloud_multi(source_cloud, source_point_count);

  //// Below method finds neighbours, and various maps related to poses and stores them as class variables
  find_source_neighbors_multi(k_correspondences, source_pose_map_ptr, source_pose_label_map_ptr, num_poses);
  calculate_source_covariances(FROBENIUS);
  // thrust::copy(
  //   source_neighbors.begin(),
  //   source_neighbors.end(), 
  //   std::ostream_iterator<int>(std::cout, " ")
  // );

  //// Set target cloud
  set_target_cloud_multi(target_cloud, target_point_count, target_cloud_label_ptr);
  find_target_neighbors_multi(k_correspondences, num_poses);
  calculate_target_covariances(FROBENIUS);

  Eigen::Isometry3f initial_guess = Eigen::Isometry3f::Identity();
  std::cout << initial_guess.matrix() << std::endl;
  Eigen::Matrix<float, 6, 1> x0;
  x0.head<3>() = Sophus::SO3f(initial_guess.linear()).log();
  x0.tail<3>() = initial_guess.translation();

  if(x0.head<3>().norm() < 1e-2) {
    x0.head<3>() = (Eigen::Vector3f::Random()).normalized() * 1e-2;
  }
  thrust::device_vector<Eigen::Matrix<float, 6, 1>> adjusted_x0s(num_poses);
  thrust::fill(adjusted_x0s.begin(), adjusted_x0s.end(), x0);
  
  thrust::device_vector<int> mask_pose_icp(num_poses, 0); // stores if a pose should be masked from all kernels because it has converged or because loss became 0
  
  thrust::device_ptr<float> JJ_ptr = thrust::device_new<float>(6 * 6 * num_poses);
  thrust::device_ptr<float> J_loss_ptr = thrust::device_new<float>(6 * num_poses);
  
  thrust::host_vector<Eigen::Matrix<float, 6, 1>> adjusted_x0s_host(num_poses);
  thrust::host_vector<int> mask_pose_icp_host(num_poses);

  ///// Stuff for CUSOLVER
  cusolverDnHandle_t handle = NULL;
  cudaStream_t stream = NULL;
  cusolverStatus_t status = CUSOLVER_STATUS_SUCCESS;
  cudaError_t cudaStat1 = cudaSuccess;
  cudaError_t cudaStat2 = cudaSuccess;
  const cublasFillMode_t uplo = CUBLAS_FILL_MODE_LOWER;

  status = cusolverDnCreate(&handle);
  assert(CUSOLVER_STATUS_SUCCESS == status);

  cudaStat1 = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
  assert(cudaSuccess == cudaStat1);

  status = cusolverDnSetStream(handle, stream);
  assert(CUSOLVER_STATUS_SUCCESS == status);

  float *Aarray[num_poses];
  float *Barray[num_poses];
  float** d_Aarray = NULL;
  cudaMalloc ((void**)&d_Aarray, sizeof(float*) * num_poses);
  float** d_Barray = NULL;
  cudaStat2 = cudaMalloc ((void**)&d_Barray, sizeof(float*) * num_poses);

  for (int p = 0; p < num_poses; p++)
  {
    cudaStat1 = cudaMalloc ((void**)&Aarray[p], sizeof(float) * 6 * 6);
    cudaStat1 = cudaMalloc ((void**)&Barray[p], sizeof(float) * 6 * 1);
  }

  thrust::device_ptr<int> JJ_infoArray = thrust::device_new<int>(num_poses);

  // Dynamic variables, reassigned in loop
  thrust::device_vector<Eigen::Vector3f> losses(max_pose_point_count * num_poses);   // 3N error vector
  thrust::device_vector<Eigen::Matrix<float, 3, 6, Eigen::RowMajor>> Js(max_pose_point_count * num_poses);    // RowMajor 3Nx6 -> ColMajor 6x3N

  thrust::device_vector<thrust::pair<float, int>> k_neighbors(source_points->size());

  /***** ICP Iterate ******/
  for (int iter = 0; iter < max_iterations; iter++)
  {
    printf("ICP iteration : %d\n", iter);
    // Get NN and loss
    // NN should take x0 also
    // thrust::copy(
    //   source_label_map->begin(),
    //   source_label_map->end(), 
    //   std::ostream_iterator<int>(std::cout, " ")
    // );
    brute_force_knn_search(*source_points, 
                          *target_points, 
                          1, 
                          k_neighbors,
                          // thrust::device_vector<int>(0), // NN will not be segmentation specific
                          // thrust::device_vector<int>(0), // NN will not be segmentation specific
                          *source_label_map,
                          *target_label_indices,
                          *source_pose_map,
                          adjusted_x0s,
                          mask_pose_icp);
    

    thrust::device_vector<int> k_indices;
    thrust::device_vector<float> k_distances;
    k_indices.resize(k_neighbors.size());
    k_distances.resize(k_neighbors.size());
    thrust::transform(k_neighbors.begin(), k_neighbors.end(), k_indices.begin(), untie_pair_second());
    thrust::transform(k_neighbors.begin(), k_neighbors.end(), k_distances.begin(), untie_pair_first());
    // thrust::copy(
    //   k_indices.begin(),
    //   k_indices.end(), 
    //   std::ostream_iterator<int>(std::cout, " ")
    // );
    // printf("\n");
    // thrust::copy(
    //   k_distances.begin(),
    //   k_distances.end(), 
    //   std::ostream_iterator<float>(std::cout, " ")
    // );
    // printf("\n");

    compute_derivatives_nn(*source_points, *source_covariances, 
                          *target_points, *target_covariances,
                          k_indices, k_distances, 0.05,
                          *source_pose_map, *source_pose_indices, num_poses,
                          max_pose_point_count,
                          adjusted_x0s,
                          mask_pose_icp,
                          losses, 
                          Js);
    // thrust::copy(
    //   losses.begin(),
    //   losses.end(), 
    //   std::ostream_iterator<Eigen::Vector3f>(std::cout, " ")
    // );
    // printf("\\n");
    // thrust::copy(
    //   Js.begin(),
    //   Js.end(), 
    //   std::ostream_iterator<Eigen::Matrix<float, 3, 6, Eigen::RowMajor>>(std::cout, " ")
    // );
    // printf("\\n");

    // gauss newton
    float alpha = 1.0f;
    float beta = 0.0f;

    // int cols = 3 * losses.size();
    int cols = 3 * max_pose_point_count;
    float* Js_ptr = thrust::reinterpret_pointer_cast<float*>(Js.data());
    // Computing J x J^T (J is 3 * 1000 * num_poses X 6 in row major format)
    cublasSgemmStridedBatched(cublas_handle, 
                              CUBLAS_OP_N, CUBLAS_OP_T, 
                              6, 6, cols, 
                              &alpha, 
                              Js_ptr, 6, 18 * max_pose_point_count,
                              Js_ptr, 6, 18 * max_pose_point_count,
                              &beta, 
                              thrust::raw_pointer_cast(JJ_ptr), 6, 36,
                              num_poses);

    float* loss_ptr = thrust::reinterpret_pointer_cast<float*>(losses.data());

    // Computing J x L
    // typedef Matrix<float, 3, 1> Vector3f;
    cublasSgemmStridedBatched(cublas_handle, 
                              CUBLAS_OP_N, CUBLAS_OP_N, 
                              6, 1, cols, 
                              &alpha, 
                              Js_ptr, 6, 18 * max_pose_point_count,
                              loss_ptr, cols, 3 * max_pose_point_count,
                              &beta, 
                              thrust::raw_pointer_cast(J_loss_ptr), 6, 6,
                              num_poses);
    
    if (solver_type == 1)
    {
      adjusted_x0s_host = adjusted_x0s;
      mask_pose_icp_host = mask_pose_icp;
      // TODO : explore why is this needed
      for (int p = 0; p < num_poses; p++)
      {
        Aarray[p] = thrust::raw_pointer_cast(JJ_ptr) + 6 * 6 * p;
        Barray[p] = thrust::raw_pointer_cast(J_loss_ptr) + 6 * p;
      }
      cudaMemcpy(d_Aarray, Aarray, sizeof(float*) * num_poses, cudaMemcpyHostToDevice);
      // Do cholesky decomposition with lower trinagle of A
      status = cusolverDnSpotrfBatched(
          handle,
          uplo,
          6,
          d_Aarray,
          6,
          thrust::raw_pointer_cast(JJ_infoArray),
          num_poses);
      
      cudaStat2 = cudaMemcpy(d_Barray, Barray, sizeof(float*)*num_poses, cudaMemcpyHostToDevice);
      // Solve equation JJ^T X = JL
      status = cusolverDnSpotrsBatched(
        handle,
        uplo,
        6,
        1, /* only support rhs = 1*/
        d_Aarray,
        6,
        d_Barray,
        6,
        thrust::raw_pointer_cast(JJ_infoArray),
        num_poses);

      adjusted_x0s_host = adjusted_x0s;
      mask_pose_icp_host = mask_pose_icp;

      // TODO : can shift this to a kernel
      for (int i = 0; i < num_poses; i++)
      {
        if (mask_pose_icp_host[i] == 1) {
          continue;
        }
        Eigen::Matrix<float, 6, 1> delta_host;
        // std::cout << "Delta CUDA" << delta_host << std::endl;

        cublasGetMatrix(6, 1, sizeof(float), Barray[i], 6, delta_host.data(), 6);
        if (std::isnan(delta_host(0,0))) {
          mask_pose_icp_host[i] == 1;
          continue;
        } 

        if(!is_converged(delta_host)) {
          Eigen::Matrix<float, 6, 1> x0 = adjusted_x0s_host[i];
          adjusted_x0s_host[i].head<3>() = (Sophus::SO3f::exp(-delta_host.head<3>()) * Sophus::SO3f::exp(x0.head<3>())).log();
          adjusted_x0s_host[i].tail<3>() -= delta_host.tail<3>();
        }
        else
        {
          mask_pose_icp_host[i] = 1;
          printf("Pose ICP converged : %d, solver : %d\n", i, solver_type);
          std::cout << adjusted_x0s_host[i] << std::endl;
          Eigen::Matrix4f trans = Eigen::Matrix4f::Identity();
          trans.block<3, 3>(0, 0) = Sophus::SO3f::exp(adjusted_x0s_host[i].head<3>()).matrix();
          trans.block<3, 1>(0, 3) = adjusted_x0s_host[i].tail<3>();
          estimated[i].matrix() = trans;
        }
      }
      adjusted_x0s = adjusted_x0s_host;
      mask_pose_icp = mask_pose_icp_host;
    }
    else if (solver_type == 0)
    {
      adjusted_x0s_host = adjusted_x0s;
      mask_pose_icp_host = mask_pose_icp;

      for (int i = 0; i < num_poses; i++)
      {
        if (mask_pose_icp_host[i] == 1) {
          continue;
        }
        
        // std::cout << J_loss_ptr[i] << " ";
        // std::cout << "Doing pose " << i << std::endl;
        Eigen::Matrix<float, 6, 6> JJ;
        cublasGetMatrix(6, 6, sizeof(float), thrust::raw_pointer_cast(JJ_ptr) + 6 * 6 * i, 6, JJ.data(), 6);
        // std::cout << JJ << std::endl;
        // if (JJ.isZero()) continue;

        Eigen::Matrix<float, 6, 1> J_loss;
        cublasGetMatrix(6, 1, sizeof(float), thrust::raw_pointer_cast(J_loss_ptr) + 6 * i, 6, J_loss.data(), 6);
        // std::cout << J_loss << std::endl;

        if (JJ.isZero() || J_loss.isZero()) {
          mask_pose_icp_host[i] = 1;
          continue;
        }

        Eigen::Matrix<float, 6, 1> delta = JJ.llt().solve(J_loss);

        if(!is_converged(delta)) {
          Eigen::Matrix<float, 6, 1> x0 = adjusted_x0s_host[i];
          adjusted_x0s_host[i].head<3>() = (Sophus::SO3f::exp(-delta.head<3>()) * Sophus::SO3f::exp(x0.head<3>())).log();
          adjusted_x0s_host[i].tail<3>() -= delta.tail<3>();
        }
        else
        {
          mask_pose_icp_host[i] = 1;
          // if (iter == max_iterations-1)
          // {
          printf("Pose ICP converged : %d, solver : %d\n", i, solver_type);
          std::cout << adjusted_x0s_host[i] << std::endl;
          // Eigen::Isometry3f final_estimated;
          Eigen::Matrix4f trans = Eigen::Matrix4f::Identity();
          trans.block<3, 3>(0, 0) = Sophus::SO3f::exp(adjusted_x0s_host[i].head<3>()).matrix();
          trans.block<3, 1>(0, 3) = adjusted_x0s_host[i].tail<3>();
          estimated[i].matrix() = trans;
          // }
        }
      }
      adjusted_x0s = adjusted_x0s_host;
      mask_pose_icp = mask_pose_icp_host;
    }
  }
  cudaFree(d_Aarray);
  cudaFree(d_Barray);
  end = std::chrono::system_clock::now();
  std::chrono::duration<double> elapsed_seconds = end-start;
  printf("*************ICP time : %f*************\n", elapsed_seconds.count());

  return false;
} 


}  // namespace fast_gicp
