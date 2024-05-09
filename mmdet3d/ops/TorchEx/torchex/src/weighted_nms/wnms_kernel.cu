// Modified from
// https://github.com/open-mmlab/OpenPCDet/blob/master/pcdet/ops/iou3d_nms/src/iou3d_nms_kernel.cu

/*
3D IoU Calculation and Rotated NMS(modified from 2D NMS written by others)
Written by Shaoshuai Shi
All Rights Reserved 2019-2020.
*/

#include <stdio.h>
#include <assert.h>
#include "../utils/error.cuh"
#include <vector>
// #define THREADS_PER_BLOCK 16
#define DIVUP(m, n) ((m) / (n) + ((m) % (n) > 0))

//#define DEBUG
const int THREADS_PER_BLOCK_NMS = sizeof(unsigned long long) * 8;
const float EPS = 1e-8;
struct Point {
  float x, y;
  __device__ Point() {}
  __device__ Point(double _x, double _y) { x = _x, y = _y; }

  __device__ void set(float _x, float _y) {
    x = _x;
    y = _y;
  }

  __device__ Point operator+(const Point &b) const {
    return Point(x + b.x, y + b.y);
  }

  __device__ Point operator-(const Point &b) const {
    return Point(x - b.x, y - b.y);
  }
};

__device__ inline float cross(const Point &a, const Point &b) {
  return a.x * b.y - a.y * b.x;
}

__device__ inline float cross(const Point &p1, const Point &p2,
                              const Point &p0) {
  return (p1.x - p0.x) * (p2.y - p0.y) - (p2.x - p0.x) * (p1.y - p0.y);
}

__device__ int check_rect_cross(const Point &p1, const Point &p2,
                                const Point &q1, const Point &q2) {
  int ret = min(p1.x, p2.x) <= max(q1.x, q2.x) &&
            min(q1.x, q2.x) <= max(p1.x, p2.x) &&
            min(p1.y, p2.y) <= max(q1.y, q2.y) &&
            min(q1.y, q2.y) <= max(p1.y, p2.y);
  return ret;
}

__device__ inline int check_in_box2d(const float *box, const Point &p) {
  // params: box (5) [x1, y1, x2, y2, angle]
  const float MARGIN = 1e-5;

  float center_x = (box[0] + box[2]) / 2;
  float center_y = (box[1] + box[3]) / 2;
  float angle_cos = cos(-box[4]),
        angle_sin =
            sin(-box[4]);  // rotate the point in the opposite direction of box
  float rot_x =
      (p.x - center_x) * angle_cos + (p.y - center_y) * angle_sin + center_x;
  float rot_y =
      -(p.x - center_x) * angle_sin + (p.y - center_y) * angle_cos + center_y;
#ifdef DEBUG
  printf("box: (%.3f, %.3f, %.3f, %.3f, %.3f)\n", box[0], box[1], box[2],
         box[3], box[4]);
  printf(
      "center: (%.3f, %.3f), cossin(%.3f, %.3f), src(%.3f, %.3f), rot(%.3f, "
      "%.3f)\n",
      center_x, center_y, angle_cos, angle_sin, p.x, p.y, rot_x, rot_y);
#endif
  return (rot_x > box[0] - MARGIN && rot_x < box[2] + MARGIN &&
          rot_y > box[1] - MARGIN && rot_y < box[3] + MARGIN);
}

__device__ inline int intersection(const Point &p1, const Point &p0,
                                   const Point &q1, const Point &q0,
                                   Point &ans) {
  // fast exclusion
  if (check_rect_cross(p0, p1, q0, q1) == 0) return 0;

  // check cross standing
  float s1 = cross(q0, p1, p0);
  float s2 = cross(p1, q1, p0);
  float s3 = cross(p0, q1, q0);
  float s4 = cross(q1, p1, q0);

  if (!(s1 * s2 > 0 && s3 * s4 > 0)) return 0;

  // calculate intersection of two lines
  float s5 = cross(q1, p1, p0);
  if (fabs(s5 - s1) > EPS) {
    ans.x = (s5 * q0.x - s1 * q1.x) / (s5 - s1);
    ans.y = (s5 * q0.y - s1 * q1.y) / (s5 - s1);

  } else {
    float a0 = p0.y - p1.y, b0 = p1.x - p0.x, c0 = p0.x * p1.y - p1.x * p0.y;
    float a1 = q0.y - q1.y, b1 = q1.x - q0.x, c1 = q0.x * q1.y - q1.x * q0.y;
    float D = a0 * b1 - a1 * b0;

    ans.x = (b0 * c1 - b1 * c0) / D;
    ans.y = (a1 * c0 - a0 * c1) / D;
  }

  return 1;
}

__device__ inline void rotate_around_center(const Point &center,
                                            const float angle_cos,
                                            const float angle_sin, Point &p) {
  float new_x =
      (p.x - center.x) * angle_cos + (p.y - center.y) * angle_sin + center.x;
  float new_y =
      -(p.x - center.x) * angle_sin + (p.y - center.y) * angle_cos + center.y;
  p.set(new_x, new_y);
}

__device__ inline int point_cmp(const Point &a, const Point &b,
                                const Point &center) {
  return atan2(a.y - center.y, a.x - center.x) >
         atan2(b.y - center.y, b.x - center.x);
}

__device__ inline float box_overlap(const float *box_a, const float *box_b) {
  // params: box_a (5) [x1, y1, x2, y2, angle]
  // params: box_b (5) [x1, y1, x2, y2, angle]

  float a_x1 = box_a[0], a_y1 = box_a[1], a_x2 = box_a[2], a_y2 = box_a[3],
        a_angle = box_a[4];
  float b_x1 = box_b[0], b_y1 = box_b[1], b_x2 = box_b[2], b_y2 = box_b[3],
        b_angle = box_b[4];

  Point center_a((a_x1 + a_x2) / 2, (a_y1 + a_y2) / 2);
  Point center_b((b_x1 + b_x2) / 2, (b_y1 + b_y2) / 2);
#ifdef DEBUG
  printf(
      "a: (%.3f, %.3f, %.3f, %.3f, %.3f), b: (%.3f, %.3f, %.3f, %.3f, %.3f)\n",
      a_x1, a_y1, a_x2, a_y2, a_angle, b_x1, b_y1, b_x2, b_y2, b_angle);
  printf("center a: (%.3f, %.3f), b: (%.3f, %.3f)\n", center_a.x, center_a.y,
         center_b.x, center_b.y);
#endif

  Point box_a_corners[5];
  box_a_corners[0].set(a_x1, a_y1);
  box_a_corners[1].set(a_x2, a_y1);
  box_a_corners[2].set(a_x2, a_y2);
  box_a_corners[3].set(a_x1, a_y2);

  Point box_b_corners[5];
  box_b_corners[0].set(b_x1, b_y1);
  box_b_corners[1].set(b_x2, b_y1);
  box_b_corners[2].set(b_x2, b_y2);
  box_b_corners[3].set(b_x1, b_y2);

  // get oriented corners
  float a_angle_cos = cos(a_angle), a_angle_sin = sin(a_angle);
  float b_angle_cos = cos(b_angle), b_angle_sin = sin(b_angle);

  for (int k = 0; k < 4; k++) {
#ifdef DEBUG
    printf("before corner %d: a(%.3f, %.3f), b(%.3f, %.3f) \n", k,
           box_a_corners[k].x, box_a_corners[k].y, box_b_corners[k].x,
           box_b_corners[k].y);
#endif
    rotate_around_center(center_a, a_angle_cos, a_angle_sin, box_a_corners[k]);
    rotate_around_center(center_b, b_angle_cos, b_angle_sin, box_b_corners[k]);
#ifdef DEBUG
    printf("corner %d: a(%.3f, %.3f), b(%.3f, %.3f) \n", k, box_a_corners[k].x,
           box_a_corners[k].y, box_b_corners[k].x, box_b_corners[k].y);
#endif
  }

  box_a_corners[4] = box_a_corners[0];
  box_b_corners[4] = box_b_corners[0];

  // get intersection of lines
  Point cross_points[16];
  Point poly_center;
  int cnt = 0, flag = 0;

  poly_center.set(0, 0);
  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 4; j++) {
      flag = intersection(box_a_corners[i + 1], box_a_corners[i],
                          box_b_corners[j + 1], box_b_corners[j],
                          cross_points[cnt]);
      if (flag) {
        poly_center = poly_center + cross_points[cnt];
        cnt++;
      }
    }
  }

  // check corners
  for (int k = 0; k < 4; k++) {
    if (check_in_box2d(box_a, box_b_corners[k])) {
      poly_center = poly_center + box_b_corners[k];
      cross_points[cnt] = box_b_corners[k];
      cnt++;
    }
    if (check_in_box2d(box_b, box_a_corners[k])) {
      poly_center = poly_center + box_a_corners[k];
      cross_points[cnt] = box_a_corners[k];
      cnt++;
    }
  }

  poly_center.x /= cnt;
  poly_center.y /= cnt;

  // sort the points of polygon
  Point temp;
  for (int j = 0; j < cnt - 1; j++) {
    for (int i = 0; i < cnt - j - 1; i++) {
      if (point_cmp(cross_points[i], cross_points[i + 1], poly_center)) {
        temp = cross_points[i];
        cross_points[i] = cross_points[i + 1];
        cross_points[i + 1] = temp;
      }
    }
  }

#ifdef DEBUG
  printf("cnt=%d\n", cnt);
  for (int i = 0; i < cnt; i++) {
    printf("All cross point %d: (%.3f, %.3f)\n", i, cross_points[i].x,
           cross_points[i].y);
  }
#endif

  // get the overlap areas
  float area = 0;
  for (int k = 0; k < cnt - 1; k++) {
    area += cross(cross_points[k] - cross_points[0],
                  cross_points[k + 1] - cross_points[0]);
  }

  return fabs(area) / 2.0;
}

__device__ inline float iou_bev(const float *box_a, const float *box_b) {
  // params: box_a (5) [x1, y1, x2, y2, angle]
  // params: box_b (5) [x1, y1, x2, y2, angle]
  float sa = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1]);
  float sb = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1]);
  float s_overlap = box_overlap(box_a, box_b);
  return s_overlap / fmaxf(sa + sb - s_overlap, EPS);
}


__global__ void wnms_kernel(const int boxes_num, const float nms_overlap_thresh, const float merge_thresh,
                           const float *boxes, unsigned long long *mask, unsigned long long *merge_mask) {
  // params: boxes (N, 5) [x1, y1, x2, y2, ry]
  // params: mask (N, N/THREADS_PER_BLOCK_NMS)

  const int row_start = blockIdx.y;
  const int col_start = blockIdx.x;

  // if (row_start > col_start) return;

  const int row_size = fminf(boxes_num - row_start * THREADS_PER_BLOCK_NMS,
                             THREADS_PER_BLOCK_NMS);
  const int col_size = fminf(boxes_num - col_start * THREADS_PER_BLOCK_NMS,
                             THREADS_PER_BLOCK_NMS);

  __shared__ float block_boxes[THREADS_PER_BLOCK_NMS * 5];

  if (threadIdx.x < col_size) {
    block_boxes[threadIdx.x * 5 + 0] =
        boxes[(THREADS_PER_BLOCK_NMS * col_start + threadIdx.x) * 5 + 0];
    block_boxes[threadIdx.x * 5 + 1] =
        boxes[(THREADS_PER_BLOCK_NMS * col_start + threadIdx.x) * 5 + 1];
    block_boxes[threadIdx.x * 5 + 2] =
        boxes[(THREADS_PER_BLOCK_NMS * col_start + threadIdx.x) * 5 + 2];
    block_boxes[threadIdx.x * 5 + 3] =
        boxes[(THREADS_PER_BLOCK_NMS * col_start + threadIdx.x) * 5 + 3];
    block_boxes[threadIdx.x * 5 + 4] =
        boxes[(THREADS_PER_BLOCK_NMS * col_start + threadIdx.x) * 5 + 4];
  }
  __syncthreads();

  if (threadIdx.x < row_size) {
    const int cur_box_idx = THREADS_PER_BLOCK_NMS * row_start + threadIdx.x;
    const float *cur_box = boxes + cur_box_idx * 5;

    int i = 0;
    unsigned long long t = 0;
    unsigned long long t_merge = 0;
    int start = 0;
    if (row_start == col_start) {
      start = threadIdx.x + 1;
    }
    for (i = start; i < col_size; i++) {
      float iou = iou_bev(cur_box, block_boxes + i * 5);
      if (iou > nms_overlap_thresh) {
        t |= 1ULL << i;
      }
      if (iou > merge_thresh) {
        t_merge |= 1ULL << i;
      }
    }
    const int col_blocks = DIVUP(boxes_num, THREADS_PER_BLOCK_NMS);
    mask[cur_box_idx * col_blocks + col_start] = t;
    merge_mask[cur_box_idx * col_blocks + col_start] = t_merge;
  }
}

__device__ inline float get_min_angle_diff(float angle_diff)
{
        angle_diff = fmodf(angle_diff, static_cast<float>(M_PI));
        if (angle_diff < M_PI/-2) return angle_diff+M_PI;
        else if (angle_diff > M_PI/2) return angle_diff-M_PI;
        else return angle_diff;
}
__device__ inline float get_old_angle_diff(float angle_diff){
  return fmodf(fabsf(angle_diff), float(2 * 3.1415926));
}

// __device__ inline float get_min_angle_diff(float angle_diff)
// {
//         angle_diff = fmodf(angle_diff, static_cast<float>(M_PI));
//         if (angle_diff < M_PI/-2) return angle_diff+M_PI;
//         else if (angle_diff > M_PI/2) return angle_diff-M_PI;
//         else return angle_diff;
// }
__device__ void sort(float *a, int n){

  float temp = 0;
  for(int i=0;i<n-1;i++)                                                             
  {                                                                              
    for(int j=0; j<n-i-1; j++)                                                       
    {                                                                            
      if(a[j]<a[j+1])                                                             
      {                                                                           
        temp = a[j];                                                               
        a[j] = a[j+1];                                                             
        a[j+1] = temp;                                                             
      }                                                                           
    }                                                                            
  }
}

__global__ void wnms_merge_kernel(
  const int boxes_num,
  const int keep_num,
  const float *boxes,
  const float *data2merge,
  const int data_dim,
  const long* keep_idx,
  long* count,
  const unsigned long long *mask,
  float* output_data
  ) {

  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= keep_num) return;
  long cur_box_idx = keep_idx[index];
  // long cur_box_idx = 0;
  if (cur_box_idx < 0) return;
  const int col_blocks = DIVUP(boxes_num, THREADS_PER_BLOCK_NMS);

  int col_start = cur_box_idx / THREADS_PER_BLOCK_NMS;
  const unsigned long long* cur_mask = mask + cur_box_idx * col_blocks;

  // find the approximate media yaw
  float median_yaw = boxes[cur_box_idx * 5 + 4];
  // float median_yaw = data2merge[cur_box_idx * data_dim + 8];
  float yaw_list[200];
  for (int i = 0; i<200; i++){ yaw_list[i] = -1000;}

  int median_counter = 0;
  for(int col=col_start; (col < col_blocks) && (median_counter < 200); col++){
    const unsigned long long cur_block_mask = cur_mask[col];
    if (cur_block_mask > 0) {
      for(int j=0; j<THREADS_PER_BLOCK_NMS; j++){
        if ( cur_block_mask & (1ULL << j) ){
          int target_data_idx = col * THREADS_PER_BLOCK_NMS + j;
          float target_yaw = boxes[target_data_idx * 5 + 4];
          // float target_yaw = data2merge[target_data_idx * data_dim + 8];
          if(median_counter<200){
            yaw_list[median_counter] = target_yaw;
            median_counter++;
          }
          else{
            break;
          }
        }
      }
    }
  }
  assert(median_counter <= 200);
  if (median_counter > 2){
    sort(yaw_list, median_counter);
    assert(yaw_list[0] >= yaw_list[1]);
    assert(yaw_list[1] >= yaw_list[2]);
    median_yaw = yaw_list[median_counter / 2];
  }
  assert(median_yaw != -1000);
  
  // merging

  float* cur_merged_data = output_data + index * data_dim;

  float cur_score = data2merge[cur_box_idx * data_dim + data_dim - 1];
  float score_sum = cur_score;
  // float cur_yaw = boxes[cur_box_idx * 5 + 4];


  count[index] = 1;
  for(int d=0; d < data_dim-1; d++) {
    cur_merged_data[d] = data2merge[cur_box_idx * data_dim + d] * cur_score;
  }
  for(int col=col_start; col < col_blocks; col++){
    const unsigned long long cur_block_mask = cur_mask[col];
    if (cur_block_mask > 0) {
      for(int j=0; j<THREADS_PER_BLOCK_NMS; j++){
        if ( cur_block_mask & (1ULL << j) ){
          int target_data_idx = col * THREADS_PER_BLOCK_NMS + j;
          float angle_diff = boxes[target_data_idx * 5 + 4] - median_yaw;
          angle_diff = get_old_angle_diff(angle_diff);
          if (fabsf(angle_diff) < 0.3) {
              float score = data2merge[target_data_idx * data_dim + data_dim - 1];
              for(int d = 0; d < data_dim-1; d++){
                cur_merged_data[d] += (data2merge[target_data_idx * data_dim + d] * score);
              }
              score_sum += score;
              count[index]++;
          }
        }
      }
    }
  }
  // update boxes value with merged boxes
  for(int d = 0; d < data_dim-1; d++){
    cur_merged_data[d] = cur_merged_data[d] / score_sum;
  }
  cur_merged_data[data_dim-1] = cur_score;
  assert(score_sum / static_cast<float>(count[index]) <= cur_score + 1e-3);
}

int fill_keep_data(const std::vector<unsigned long long> mask_cpu, long *keep_data, int boxes_num){

  const int col_blocks = DIVUP(boxes_num, THREADS_PER_BLOCK_NMS);
  unsigned long long remv_cpu[col_blocks];
  memset(remv_cpu, 0, col_blocks * sizeof(unsigned long long));

  int num_to_keep = 0;
  for (int i = 0; i < boxes_num; i++) {
    int nblock = i / THREADS_PER_BLOCK_NMS;
    int inblock = i % THREADS_PER_BLOCK_NMS;

    if (!(remv_cpu[nblock] & (1ULL << inblock))) {
      keep_data[num_to_keep++] = i;
      const unsigned long long *p = &mask_cpu[0] + i * col_blocks;
      for (int j = nblock; j < col_blocks; j++) {
        remv_cpu[j] |= p[j];
      }
    }
  }
  return num_to_keep;

}


int wnmsLauncher(
  const float *boxes,
  const float *data2merge,
  float *output_merged_data,
  int data_dim,
  long *keep_data,
  long *count,
  unsigned long long *mask,
  unsigned long long *merge_mask,
  int boxes_num,
  float nms_overlap_thresh,
  float merge_thresh
  ) {

  dim3 blocks(DIVUP(boxes_num, THREADS_PER_BLOCK_NMS),
              DIVUP(boxes_num, THREADS_PER_BLOCK_NMS));
  dim3 threads(THREADS_PER_BLOCK_NMS);

  wnms_kernel<<<blocks, threads>>>(boxes_num, nms_overlap_thresh, merge_thresh, boxes, mask, merge_mask);

  // get keep_data in CPU
  const int col_blocks = DIVUP(boxes_num, THREADS_PER_BLOCK_NMS);
  std::vector<unsigned long long> mask_cpu(boxes_num * col_blocks);
  CHECK_CALL(cudaMemcpy(&mask_cpu[0], mask,
                         boxes_num * col_blocks * sizeof(unsigned long long),
                         cudaMemcpyDeviceToHost));
  int num_to_keep = fill_keep_data(mask_cpu, keep_data, boxes_num);


  int num_all_thread = num_to_keep == 0 ? 1 : num_to_keep;

  dim3 merge_blocks(DIVUP(num_all_thread, 64));
  dim3 merge_threads(64);

  long *keep_data_gpu = NULL;
  CHECK_CALL(cudaMalloc((void **)&keep_data_gpu,
                         boxes_num * sizeof(long)));
  CHECK_CALL(cudaMemcpy(keep_data_gpu, keep_data,
                         boxes_num * sizeof(long),
                         cudaMemcpyHostToDevice));

  wnms_merge_kernel<<<merge_blocks, merge_threads>>>(boxes_num, num_to_keep, boxes, data2merge, data_dim, keep_data_gpu, count, merge_mask, output_merged_data);

  return num_to_keep;
  // const int boxes_num,
  // const int keep_num,
  // const float *boxes,
  // const float *data2merge,
  // const int data_dim,
  // const int* keep_idx,
  // const unsigned long long *mask,
  // float* output_data) {
}
