// Modified from
// https://github.com/sshaoshuai/PCDet/blob/master/pcdet/ops/roiaware_pool3d/src/roiaware_pool3d_kernel.cu

#include <assert.h>
#include <torch/extension.h>
#include <torch/serialize/tensor.h>
#include <vector>

#define CHECK_CUDA(x) \
  TORCH_CHECK(x.device().is_cuda(), #x, " must be a CUDAtensor ")
#define CHECK_CONTIGUOUS(x) \
  TORCH_CHECK(x.is_contiguous(), #x, " must be contiguous ")
#define CHECK_INPUT(x) \
  CHECK_CUDA(x);       \
  CHECK_CONTIGUOUS(x)

void dynamic_point_pool_launcher(
  int boxes_num,
  int pts_num, 
  int max_num_pts_per_box,
  int max_all_pts_num,
  const float *rois,
  const float *pts,
  const float *extra_wlh,
  long *out_pts_idx,
  long *out_roi_idx,
  float *out_pts_feats,
  int info_dim
  );


void dynamic_point_pool_gpu(
  at::Tensor rois,
  at::Tensor pts,
  std::vector<float> extra_wlh,
  int max_num_pts_per_box,
  at::Tensor out_pts_idx,
  at::Tensor out_roi_idx,
  at::Tensor out_pts_feats
);


void dynamic_point_pool_gpu(
  at::Tensor rois,
  at::Tensor pts,
  std::vector<float> extra_wlh,
  int max_num_pts_per_box,
  at::Tensor out_pts_idx,
  at::Tensor out_roi_idx,
  at::Tensor out_pts_feats
) {
  // params rois: (N, 7) [x, y, z, w, l, h, ry] in LiDAR coordinate
  // params pts: (npoints, 3) [x, y, z] in LiDAR coordinate

  CHECK_INPUT(rois);
  CHECK_INPUT(pts);
  CHECK_INPUT(out_pts_idx);
  CHECK_INPUT(out_roi_idx);
  CHECK_INPUT(out_pts_feats);

  int boxes_num = rois.size(0);
  int pts_num = pts.size(0);
  int max_all_pts_num = out_pts_idx.size(0);
  // printf("boxes_num: %d, pts_num: %d, max_all_pts_num: %d \n", boxes_num, pts_num, max_all_pts_num);
  int info_dim = out_pts_feats.size(1);
  assert (info_dim == 13);

  const float *rois_data = rois.data_ptr<float>();
  const float *pts_data = pts.data_ptr<float>();

  long *out_pts_idx_data = out_pts_idx.data_ptr<long>();
  long *out_roi_idx_data = out_roi_idx.data_ptr<long>();
  float *out_pts_feats_data = out_pts_feats.data_ptr<float>();

  const float extra_wlh_array[3] = {extra_wlh[0], extra_wlh[1], extra_wlh[2]};

  dynamic_point_pool_launcher(
    boxes_num,
    pts_num, 
    max_num_pts_per_box,
    max_all_pts_num,
    rois_data,
    pts_data,
    extra_wlh_array,
    out_pts_idx_data,
    out_roi_idx_data,
    out_pts_feats_data,
    info_dim
  );

}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &dynamic_point_pool_gpu, "dynamic_point_pool_gpu forward (CUDA)");
}
