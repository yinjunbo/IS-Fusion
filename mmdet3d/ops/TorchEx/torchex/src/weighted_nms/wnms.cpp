// Modified from
// https://github.com/open-mmlab/OpenPCDet/blob/master/pcdet/ops/iou3d_nms/src/iou3d_nms.cpp

/*
3D IoU Calculation and Rotated NMS(modified from 2D NMS written by others)
Written by Shaoshuai Shi
All Rights Reserved 2019-2020.
*/

#include <cuda.h>
#include <cuda_runtime_api.h>
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

#define DIVUP(m, n) ((m) / (n) + ((m) % (n) > 0))

#define CHECK_ERROR(ans) \
  { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line,
                      bool abort = true) {
  if (code != cudaSuccess) {
    fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file,
            line);
    if (abort) exit(code);
  }
}

const int THREADS_PER_BLOCK_NMS = sizeof(unsigned long long) * 8;

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
  );

int wnms_gpu(
  at::Tensor boxes,
  at::Tensor data2merge,
  at::Tensor output,
  at::Tensor keep,
  at::Tensor count,
	float nms_overlap_thresh,
	float merge_thresh,
  int device_id) {
  // params boxes: (N, 5) [x1, y1, x2, y2, ry]
  // params keep: (N)

  CHECK_INPUT(boxes);
  CHECK_INPUT(data2merge);
  CHECK_INPUT(output);
  CHECK_INPUT(count);
  CHECK_CONTIGUOUS(keep);
  cudaSetDevice(device_id);
  assert(boxes.size(0) == data2merge.size(0));
  assert(boxes.size(0) == output.size(0));
  assert(data2merge.size(1) == output.size(1));
  long *count_ptr = count.data_ptr<long>();

  int boxes_num = boxes.size(0);
  const float *boxes_data = boxes.data_ptr<float>();
  long *keep_data = keep.data_ptr<long>();

  const int col_blocks = DIVUP(boxes_num, THREADS_PER_BLOCK_NMS);

  unsigned long long *mask_ptr = NULL;
  CHECK_ERROR(cudaMalloc((void **)&mask_ptr,
                         boxes_num * col_blocks * sizeof(unsigned long long)));

  unsigned long long *merge_mask_ptr = NULL;
  CHECK_ERROR(cudaMalloc((void **)&merge_mask_ptr,
                         boxes_num * col_blocks * sizeof(unsigned long long)));
  
  float *data2merge_ptr = data2merge.data_ptr<float>();
  float *output_ptr = output.data_ptr<float>();
  int output_dim = output.size(1);

  int num_to_keep = wnmsLauncher(boxes_data, data2merge_ptr, output_ptr, output_dim, keep_data, count_ptr, mask_ptr, merge_mask_ptr, boxes_num, nms_overlap_thresh, merge_thresh);

  cudaFree(mask_ptr);
  cudaFree(merge_mask_ptr);

  return num_to_keep;

}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("wnms_gpu", &wnms_gpu, "oriented nms gpu");
}
