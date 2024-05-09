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
#define DIVUP(m, n) ((m) / (n) + ((m) % (n) > 0))

// #define DEBUG
// #define ASSERTION

__global__ void ingroup_inds_kernel(
    const long *group_inds,
    long *out_inds,
    int *ingroup_counter,
    int N
) {

  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= N) return;
  long this_group_id = group_inds[idx];

  int cnt = atomicAdd(&ingroup_counter[this_group_id], 1);
  out_inds[idx] = cnt;
}


 void ingroup_inds_launcher(
  const long *group_inds,
  long *out_inds,
  int N,
  int max_group_id
  ) {

  int *ingroup_counter = NULL;
  CHECK_CALL(cudaMalloc(&ingroup_counter,   (max_group_id + 1) * sizeof(int)));
  CHECK_CALL(cudaMemset(ingroup_counter, 0, (max_group_id + 1) * sizeof(int)));

  dim3 blocks(DIVUP(N, THREADS_PER_BLOCK));
  dim3 threads(THREADS_PER_BLOCK);

  ingroup_inds_kernel<<<blocks, threads>>>(
      group_inds,
      out_inds,
      ingroup_counter,
      N
  );

  cudaFree(ingroup_counter);

  #ifdef DEBUG
  CHECK_CALL(cudaGetLastError());
  CHECK_CALL(cudaDeviceSynchronize());
  #endif

  return;

}
