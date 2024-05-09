// Modified from
// https://github.com/sshaoshuai/PCDet/blob/master/pcdet/ops/roiaware_pool3d/src/roiaware_pool3d_kernel.cu

#include <assert.h>
#include <vector>
#include <math.h>
#include <stdio.h>
#include <torch/serialize/tensor.h>
#include <torch/extension.h>
#include <torch/types.h>
#include "cuda_fp16.h"
#include "../utils/error.cuh"

#define THREADS_PER_BLOCK 256
#define LARGE_NEG -10000
#define DIVUP(m, n) ((m) / (n) + ((m) % (n) > 0))

// #define DEBUG
// #define ASSERTION

__device__ inline void lidar_to_local_coords(float shift_x, float shift_y,
                                             float rz, float &local_x,
                                             float &local_y) {
  // should rotate pi/2 + alpha to translate LiDAR to local
  float rot_angle = rz + M_PI / 2;
  float cosa = cos(rot_angle), sina = sin(rot_angle);
  local_x = shift_x * cosa + shift_y * (-sina);
  local_y = shift_x * sina + shift_y * cosa;
}


__device__ inline int check_pt_in_box3d(const float *pt, const float *box3d,
                                        float &local_x, float &local_y,
                                        const float extra_w,
                                        const float extra_l,
                                        const float extra_h
                                        ) {
  // param pt: (x, y, z)
  // param box3d: (cx, cy, cz, w, l, h, rz) in LiDAR coordinate, cz in the
  // bottom center
  float x = pt[0], y = pt[1], z = pt[2];
  float cx = box3d[0], cy = box3d[1], cz = box3d[2];
  float w = box3d[3], l = box3d[4], h = box3d[5], rz = box3d[6];
  float large_w = w + extra_w;
  float large_l = l + extra_l;
  float large_h = h + extra_h;
  cz += h / 2.0;  // shift to the center since cz in box3d is the bottom center

  if (fabsf(z - cz) > large_h / 2.0) return 0;

  lidar_to_local_coords(x - cx, y - cy, rz, local_x, local_y);

  float in_flag = (local_x > -l / 2.0) & (local_x < l / 2.0) &
                  (local_y > -w / 2.0) & (local_y < w / 2.0) & (fabsf(z - cz) < h / 2.0);

  float in_large_flag = 
                  (local_x > -large_l / 2.0) & (local_x < large_l / 2.0) &
                  (local_y > -large_w / 2.0) & (local_y < large_w / 2.0);
  
  if (in_flag > 0) return 2;
  else if (in_large_flag > 0) return 1;
  else return 0;
  // 0: out of large box
  // 1: in large box but out of small box
  // 2: in small box
}

__global__ void generate_pts_mask_for_box3d(
  const int boxes_num, const int pts_num, const int max_pts_num, const int max_all_pts_num,
  const float *rois,
  const float *pts,
  int *pts_mask,
  float *pts_mask_info,
  int *in_box_counter,
  int *global_counter,
  const int info_dim,
  const float extra_w,
  const float extra_l,
  const float extra_h
) {

  // params rois: (N, 7) [x, y, z, w, l, h, rz] in LiDAR coordinate
  // params pts: (npoints, 3) [x, y, z]
  // counter: [num_box, ]
  // pts_mask [num_box, max_pts_num] 
  // pts_mask_info [num_box, max_pts_num, info_dim] 

  int pt_idx = blockIdx.x * blockDim.x + threadIdx.x;
  int box_idx = blockIdx.y;
  assert (box_idx < boxes_num);
  if (pt_idx >= pts_num) return;

  pts += pt_idx * 3;
  rois += box_idx * 7;

  float local_x = 0, local_y = 0;
  int cur_in_flag = check_pt_in_box3d(pts, rois, local_x, local_y, extra_w, extra_l, extra_h);

  if (cur_in_flag > 0) {

    float w = rois[3], l = rois[4], h = rois[5];
    float local_z = pts[2] - (rois[2] + h / 2.0); // to roi center

    int cnt = atomicAdd(&in_box_counter[box_idx], 1);
    if (cnt >= max_pts_num) return;

    int new_pt_idx = atomicAdd(&global_counter[0], 1);
    if (new_pt_idx >= max_all_pts_num) return;

    float off_x = local_x + l / 2;
    float off_y = local_y + w / 2;
    float off_z = local_z + h / 2;

    float off_x_2 = -local_x + l / 2;
    float off_y_2 = -local_y + w / 2;
    float off_z_2 = -local_z + h / 2;

    float is_in_margin = cur_in_flag == 1 ? 1 : 0;

    pts_mask[box_idx * max_pts_num * 2 + cnt * 2 + 0] = pt_idx;
    pts_mask[box_idx * max_pts_num * 2 + cnt * 2 + 1] = new_pt_idx;

    assert (info_dim == 13);
    assert (boxes_num * max_pts_num * info_dim > box_idx * max_pts_num * info_dim + cnt * info_dim + 12);

    pts_mask_info[box_idx * max_pts_num * info_dim + cnt * info_dim + 0] = pts[0];
    pts_mask_info[box_idx * max_pts_num * info_dim + cnt * info_dim + 1] = pts[1];
    pts_mask_info[box_idx * max_pts_num * info_dim + cnt * info_dim + 2] = pts[2];
    pts_mask_info[box_idx * max_pts_num * info_dim + cnt * info_dim + 3] = local_x;
    pts_mask_info[box_idx * max_pts_num * info_dim + cnt * info_dim + 4] = local_y;
    pts_mask_info[box_idx * max_pts_num * info_dim + cnt * info_dim + 5] = local_z;
    pts_mask_info[box_idx * max_pts_num * info_dim + cnt * info_dim + 6] = off_x;
    pts_mask_info[box_idx * max_pts_num * info_dim + cnt * info_dim + 7] = off_y;
    pts_mask_info[box_idx * max_pts_num * info_dim + cnt * info_dim + 8] = off_z;
    pts_mask_info[box_idx * max_pts_num * info_dim + cnt * info_dim + 9] = off_x_2;
    pts_mask_info[box_idx * max_pts_num * info_dim + cnt * info_dim + 10] = off_y_2;
    pts_mask_info[box_idx * max_pts_num * info_dim + cnt * info_dim + 11] = off_z_2;
    pts_mask_info[box_idx * max_pts_num * info_dim + cnt * info_dim + 12] = is_in_margin;


    #ifdef ASSERTION
    if (is_in_margin == 1){
      assert (
        (local_x <= -l/2 + 1e-4) | (local_x >= l/2 - 1e-4) | 
        (local_y <= -w/2 + 1e-4) | (local_y >= w/2 - 1e-4) |
        (local_z <= -h/2 + 1e-4) | (local_z >= h/2 - 1e-4)
      );
      assert (
        ((off_x <= 0 + 1e-4) && (off_x >= -extra_l/2-1e-4)) || ((off_x >= l - 1e-4) && (off_x <= l + extra_l/2 + 1e-4)) ||
        ((off_y <= 0 + 1e-4) && (off_y >= -extra_w/2-1e-4)) || ((off_y >= w - 1e-4) && (off_y <= w + extra_w/2 + 1e-4)) ||
        ((off_z <= 0 + 1e-4) && (off_z >= -extra_h/2-1e-4)) || ((off_z >= h - 1e-4) && (off_z <= h + extra_h/2 + 1e-4))
      );
      assert (
        ((off_x_2 <= 0 + 1e-4) && (off_x_2 >= -extra_l/2-1e-4)) || ((off_x_2 >= l - 1e-4) && (off_x_2 <= l + extra_l/2 + 1e-4)) ||
        ((off_y_2 <= 0 + 1e-4) && (off_y_2 >= -extra_w/2-1e-4)) || ((off_y_2 >= w - 1e-4) && (off_y_2 <= w + extra_w/2 + 1e-4)) ||
        ((off_z_2 <= 0 + 1e-4) && (off_z_2 >= -extra_h/2-1e-4)) || ((off_z_2 >= h - 1e-4) && (off_z_2 <= h + extra_h/2 + 1e-4))
      );
    }
    if (is_in_margin == 2){
      assert(off_x > 0 - 1e-5 && off_x_2 > 0 - 1e-5 && off_y > 0 - 1e-5 && off_y_2 > 0 - 1e-5 && off_z > 0 - 1e-5 && off_z_2 > 0 - 1e-5);
      assert(off_x < l + 1e-5 && off_x_2 < l + 1e-5 && off_y < w + 1e-5 && off_y_2 < w + 1e-5 && off_z < h + 1e-5 && off_z_2 < h + 1e-5);
    }
    #endif

  }
}

__global__ void collect_inside_points(
  const int boxes_num,
  const int pts_num,
  const int new_pts_num,
  const int max_num_pts_per_box,
  const int info_dim,
  const int *pts_mask,
  const float *pts_mask_info,
  long *out_pts_idx,
  long *out_roi_idx,
  float *out_pts_feats
) {

  int box_idx = blockIdx.y;
  int inbox_pt_idx = blockIdx.x * blockDim.x + threadIdx.x;

  assert (box_idx < boxes_num);
  if (inbox_pt_idx >= max_num_pts_per_box) return;

  int ori_pts_idx = pts_mask[box_idx * max_num_pts_per_box * 2 + inbox_pt_idx * 2 + 0];
  int new_pts_idx = pts_mask[box_idx * max_num_pts_per_box * 2 + inbox_pt_idx * 2 + 1];

  if (ori_pts_idx < 0) {
    assert (new_pts_idx < 0);
    return;
  }

  assert (new_pts_idx < new_pts_num && new_pts_idx >= 0);
  assert (ori_pts_idx < pts_num && ori_pts_idx >= 0);
  out_pts_idx[new_pts_idx] = ori_pts_idx;
  out_roi_idx[new_pts_idx] = box_idx;

  for (int i = 0; i < info_dim; i++){
    out_pts_feats[new_pts_idx * info_dim + i] = pts_mask_info[box_idx * max_num_pts_per_box * info_dim + inbox_pt_idx * info_dim + i];
  }

}



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
  ) {
  
  #ifdef DEBUG
  printf("pts_num: %d, boxes_num: %d, max_num: %d \n", pts_num, boxes_num, max_num_pts_per_box);
  #endif

  int *inbox_counter = NULL;
  CHECK_CALL(cudaMalloc(&inbox_counter,   boxes_num * sizeof(int)));
  CHECK_CALL(cudaMemset(inbox_counter, 0, boxes_num * sizeof(int)));

  int *global_counter = NULL;
  CHECK_CALL(cudaMalloc(&global_counter,   1 * sizeof(int)));
  CHECK_CALL(cudaMemset(global_counter, 0, 1 * sizeof(int)));

  int *pts_mask = NULL;
  CHECK_CALL(cudaMalloc(&pts_mask,    boxes_num * max_num_pts_per_box * 2 * sizeof(int)));
  CHECK_CALL(cudaMemset(pts_mask, -1, boxes_num * max_num_pts_per_box * 2 * sizeof(int)));

  float *pts_mask_info = NULL;
  CHECK_CALL(cudaMalloc(&pts_mask_info,    boxes_num * max_num_pts_per_box * info_dim * sizeof(float)));
  CHECK_CALL(cudaMemset(pts_mask_info, 0,  boxes_num * max_num_pts_per_box * info_dim * sizeof(float)));



  float extra_w = extra_wlh[0];
  float extra_l = extra_wlh[1];
  float extra_h = extra_wlh[2];
  dim3 blocks_mask(DIVUP(pts_num, THREADS_PER_BLOCK), boxes_num);
  dim3 threads(THREADS_PER_BLOCK);

  generate_pts_mask_for_box3d<<<blocks_mask, threads>>>(
    boxes_num, pts_num, max_num_pts_per_box, max_all_pts_num,
    rois,
    pts,
    pts_mask,
    pts_mask_info,
    inbox_counter,
    global_counter,
    info_dim,
    extra_w,
    extra_l,
    extra_h
  );
  #ifdef DEBUG
  printf("Finish mask generation.\n");
  CHECK_CALL(cudaGetLastError());
  CHECK_CALL(cudaDeviceSynchronize());
  #endif

  int new_pts_num[1];
  CHECK_CALL(cudaMemcpy(new_pts_num, global_counter, sizeof(int), cudaMemcpyDeviceToHost));

  dim3 block_collect(DIVUP(max_num_pts_per_box, THREADS_PER_BLOCK), boxes_num);
  dim3 thread_collect(THREADS_PER_BLOCK);


  collect_inside_points<<<block_collect, thread_collect>>>(
    boxes_num,
    pts_num,
    new_pts_num[0],
    max_num_pts_per_box,
    info_dim,
    pts_mask,
    pts_mask_info,
    out_pts_idx, out_roi_idx, out_pts_feats
  );


  cudaFree(inbox_counter);
  cudaFree(global_counter);
  cudaFree(pts_mask);
  cudaFree(pts_mask_info);

  #ifdef DEBUG
  CHECK_CALL(cudaGetLastError());
  CHECK_CALL(cudaDeviceSynchronize());
  #endif

  return;

}