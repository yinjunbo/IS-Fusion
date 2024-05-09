// Modified from
// https://github.com/sshaoshuai/PCDet/blob/master/pcdet/ops/roiaware_pool3d/src/roiaware_pool3d_kernel.cu

#include <assert.h>
#include <torch/extension.h>
#include <torch/serialize/tensor.h>
#include <vector>
#include<stdio.h>

#define CHECK_CUDA(x) \
  TORCH_CHECK(x.device().is_cuda(), #x, " must be a CUDAtensor ")
#define CHECK_CONTIGUOUS(x) \
  TORCH_CHECK(x.is_contiguous(), #x, " must be contiguous ")
#define CHECK_INPUT(x) \
  CHECK_CUDA(x);       \
  CHECK_CONTIGUOUS(x)
// #define DEBUG
// #define CHECK_TYPE(x, dtype) \
//   TORCH_CHECK(x.dtype == dtype, #x, " must be ", #dtype, ", but got ", x.dtype)

void roiaware_pool3d_launcher(int boxes_num, int pts_num, int channels,
                              int max_pts_each_voxel, int out_x, int out_y,
                              int out_z, const float *rois, const float *pts,
                              const float *pts_feature, int *argmax,
                              int *pts_idx_of_voxels, float *pooled_features,
                              int *pooled_coors,
                              int pool_method,
                              int max_voxels
                              );

void roiaware_pool3d_backward_launcher(int boxes_num, int out_x, int out_y,
                                       int out_z, int channels,
                                       int max_pts_each_voxel,
                                       const int *pts_idx_of_voxels,
                                       const int *argmax, const float *grad_out,
                                       float *grad_in, int pool_method, int max_voxels);

int roiaware_pool3d_gpu(at::Tensor rois, at::Tensor pts, at::Tensor pts_feature,
                        at::Tensor argmax, at::Tensor pts_idx_of_voxels,
                        at::Tensor pooled_features, at::Tensor pooled_coors,
                        int pool_method, std::vector<int> roi_shape);

int roiaware_pool3d_gpu_backward(at::Tensor pts_idx_of_voxels,
                                 at::Tensor argmax, at::Tensor grad_out,
                                 at::Tensor grad_in, int pool_method, std::vector<int> roi_shape);

int points_in_boxes_cpu(at::Tensor boxes_tensor, at::Tensor pts_tensor,
                        at::Tensor pts_indices_tensor);

int points_in_boxes_gpu(at::Tensor boxes_tensor, at::Tensor pts_tensor,
                        at::Tensor box_idx_of_points_tensor);

int points_in_boxes_batch(at::Tensor boxes_tensor, at::Tensor pts_tensor,
                          at::Tensor box_idx_of_points_tensor);

int roiaware_pool3d_gpu(
  at::Tensor rois,
  at::Tensor pts,
  at::Tensor pts_feature,
  at::Tensor argmax,
  at::Tensor pts_idx_of_voxels,
  at::Tensor pooled_features,
  at::Tensor pooled_coors,
  int pool_method,
  std::vector<int> roi_shape
  ) {
  // params rois: (N, 7) [x, y, z, w, l, h, ry] in LiDAR coordinate
  // params pts: (npoints, 3) [x, y, z] in LiDAR coordinate
  // params pts_feature: (npoints, C)
  // params argmax: (N, max_voxels, C)
  // params pts_idx_of_voxels: (N, max_voxels, max_pts_each_voxel)
  // params pooled_features: (N, max_voxels, C)
  // params pooled_coors: (N, max_voxels, 3)
  // params pool_method: 0: max_pool 1: avg_pool

  CHECK_INPUT(rois);
  CHECK_INPUT(pts);
  CHECK_INPUT(pts_feature);
  CHECK_INPUT(argmax);
  CHECK_INPUT(pts_idx_of_voxels);
  CHECK_INPUT(pooled_features);
  CHECK_INPUT(pooled_coors);

  assert(rois.dtype == torch::kFloat32);
  assert(pts.dtype == torch::kFloat32);
  assert(pts_feature.dtype == torch::kFloat32);
  assert(pooled_feature.dtype == torch::kFloat32);
  assert(argmax.dtype == torch::kInt32);
  assert(pts_idx_of_voxels.dtype == torch::kInt32);
  assert(pooled_coors.dtype == torch::kInt32);

  #ifdef DEBUG
  printf("********* Line %d, File %s *********\n", __LINE__, __FILE__);
  #endif

  int boxes_num = rois.size(0);
  int pts_num = pts.size(0);
  int channels = pts_feature.size(1);
  int max_pts_each_voxel = pts_idx_of_voxels.size(2);  // index 0 is the counter
  int out_x = roi_shape[0];
  int out_y = roi_shape[1];
  int out_z = roi_shape[2];
  int max_voxels = pooled_features.size(1);
  assert((out_x < 256) && (out_y < 256) &&
         (out_z < 256));  // we encode index with 8bit
  #ifdef DEBUG
  printf("********* Line %d, File %s *********\n", __LINE__, __FILE__);
  #endif

  TORCH_CHECK(pooled_coors.size(2) == 3);
  TORCH_CHECK(pooled_features.size(2) == channels);

  const float *rois_data = rois.data_ptr<float>();
  const float *pts_data = pts.data_ptr<float>();
  const float *pts_feature_data = pts_feature.data_ptr<float>();
  int *argmax_data = argmax.data_ptr<int>();
  int *pts_idx_of_voxels_data = pts_idx_of_voxels.data_ptr<int>();
  float *pooled_features_data = pooled_features.data_ptr<float>();
  int *pooled_coors_data = pooled_coors.data_ptr<int>();

  roiaware_pool3d_launcher(
      boxes_num, pts_num, channels, max_pts_each_voxel, out_x, out_y, out_z,
      rois_data, pts_data, pts_feature_data, argmax_data,
      pts_idx_of_voxels_data, pooled_features_data,
      pooled_coors_data,
      pool_method, max_voxels);

  return 1;
}

int roiaware_pool3d_gpu_backward(
  at::Tensor pts_idx_of_voxels,
  at::Tensor argmax,
  at::Tensor grad_out,
  at::Tensor grad_in,
  int pool_method,
  std::vector<int> roi_shape
  ) {
  // params pts_idx_of_voxels: (N, max_voxels, max_pts_each_voxel)
  // params argmax: (N, max_voxels, C)
  // params grad_out: (N, max_voxels, C)
  // params grad_in: (npoints, C), return value
  // params pool_method: 0: max_pool 1: avg_pool

  CHECK_INPUT(pts_idx_of_voxels);
  CHECK_INPUT(argmax);
  CHECK_INPUT(grad_out);
  CHECK_INPUT(grad_in);

  int boxes_num = pts_idx_of_voxels.size(0);
  int out_x = roi_shape[0];
  int out_y = roi_shape[1];
  int out_z = roi_shape[2];
  int max_pts_each_voxel = pts_idx_of_voxels.size(2);  // index 0 is the counter
  int channels = grad_out.size(2);
  int max_voxels = grad_out.size(1);

  const int *pts_idx_of_voxels_data = pts_idx_of_voxels.data_ptr<int>();
  const int *argmax_data = argmax.data_ptr<int>();
  const float *grad_out_data = grad_out.data_ptr<float>();
  float *grad_in_data = grad_in.data_ptr<float>();

  roiaware_pool3d_backward_launcher(boxes_num, out_x, out_y, out_z, channels,
                                    max_pts_each_voxel, pts_idx_of_voxels_data,
                                    argmax_data, grad_out_data, grad_in_data,
                                    pool_method, max_voxels);

  return 1;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &roiaware_pool3d_gpu, "roiaware pool3d forward (CUDA)");
  m.def("backward", &roiaware_pool3d_gpu_backward,
        "roiaware pool3d backward (CUDA)");
  // m.def("points_in_boxes_gpu", &points_in_boxes_gpu,
  //       "points_in_boxes_gpu forward (CUDA)");
  // m.def("points_in_boxes_batch", &points_in_boxes_batch,
  //       "points_in_boxes_batch forward (CUDA)");
  // m.def("points_in_boxes_cpu", &points_in_boxes_cpu,
  //       "points_in_boxes_cpu forward (CPU)");
}
