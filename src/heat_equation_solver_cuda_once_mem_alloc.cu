#include "heat_equation_solver_impl.h"

#include "cuda_utils/cuda_utils.cuh"
#include "cuda_utils/cublas_utils.cuh"

#include <stdexcept>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <algorithm>
#include <memory>

namespace {
  constexpr std::size_t BLOCK_SIZE = 16;
}

#ifndef N_ERR_COMPUTING_IN_DEVICE
#define N_ERR_COMPUTING_IN_DEVICE 1500
#endif // N_ERR_COMPUTING_IN_DEVICE

template <typename T>
static void compute_error(const T *__restrict__ buff_grid, const std::size_t grid_pitch,
                       T *__restrict__ diff_buff, const std::size_t diff_pitch,
                       const size_t grid_size,
                       const cudaStream_t copy_stream, const cublasHandle_t &handle,
                       int &err_idx, T &err)
{
  const T a = 1.0;
  const T b = -1.0;

  CUBLAS_CHECK(cublasGeam(handle, CUBLAS_OP_T, CUBLAS_OP_T, grid_size, grid_size,
                          &a, buff_grid, grid_pitch,
                          &b, buff_grid + (grid_size * grid_pitch), grid_pitch,
                          diff_buff, diff_pitch));
  CUBLAS_CHECK(cublasIamax(handle, grid_size * diff_pitch, diff_buff, 1, &err_idx));
  CUDA_CHECK(cu_memcpy_async(&err, diff_buff + err_idx, 1, cudaMemcpyDefault, copy_stream));
}

__global__ void grid_recompute(const FLOAT_TYPE *__restrict__ curr_grid,
                               FLOAT_TYPE *__restrict__ next_grid,
                               const std::size_t grid_size,
                               const std::size_t pitch)
{
  unsigned int x_idx = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int y_idx = blockIdx.y * blockDim.y + threadIdx.y;

  if ((x_idx > 0 && x_idx < grid_size - 1) &&
      (y_idx > 0 && y_idx < grid_size - 1))
  {
    std::size_t grid_idx = y_idx * pitch + x_idx;
    next_grid[grid_idx] = (FLOAT_TYPE)0.25 * (curr_grid[grid_idx - pitch] +
                                              curr_grid[grid_idx + pitch] +
                                              curr_grid[grid_idx - 1] +
                                              curr_grid[grid_idx + 1]);
  }
}

int solve_heat_equation(FLOAT_TYPE *__restrict__ init_grid,
                        const size_t grid_size,
                        const size_t max_iter,
                        const FLOAT_TYPE etol,
                        size_t *last_iter,
                        FLOAT_TYPE *last_etol)
{
  const std::size_t grid_sqr = grid_size * grid_size;

  auto cublas_handle_unique = cublas_init_handle();
  auto cu_streams_unique = cu_create_streams(3);

  std::size_t grid_pitch;
  auto cu_buff_grid_unique = cu_make_pitched_memory_unique<FLOAT_TYPE>(2 * grid_size, grid_size, grid_pitch);
  std::size_t diff_pitch;
  auto cu_diff_buff_unique = cu_make_pitched_memory_unique<FLOAT_TYPE>(grid_size, grid_size, diff_pitch);

  auto cu_init_grid_pinned_unique = cu_make_pinned_memory_unique(init_grid, grid_sqr);

  const std::size_t half_buff_grid_size = grid_size * grid_pitch;
  CUDA_CHECK(cu_memcpy2D_async(cu_buff_grid_unique.get(), grid_pitch,
                               cu_init_grid_pinned_unique.get(), grid_size,
                               grid_size, grid_size,
                               cudaMemcpyDefault, cu_streams_unique[0]));
  CUDA_CHECK(cu_memcpy2D_async(cu_buff_grid_unique.get() + half_buff_grid_size, grid_pitch,
                               cu_init_grid_pinned_unique.get(), grid_size,
                               grid_size, grid_size,
                               cudaMemcpyDefault, cu_streams_unique[1]));
  CUDA_CHECK(cu_memset2D_async(cu_diff_buff_unique.get(), diff_pitch, 0, grid_size, grid_size, cu_streams_unique[2]));

  CUDA_CHECK(cudaDeviceSynchronize());

  CUBLAS_CHECK(cublasSetStream(*cublas_handle_unique.get(), cu_streams_unique[0]));

  auto cu_err_unique = cu_make_host_memory_unique<FLOAT_TYPE>();
  FLOAT_TYPE &err = *cu_err_unique.get();
  err = INFINITY;

  auto cu_err_idx_unique = cu_make_host_memory_unique<int>();

  size_t curr_iter;
  size_t n_err_iter;
  dim3 kernel_threads(BLOCK_SIZE, BLOCK_SIZE);
  dim3 kernel_blocks((grid_size / kernel_threads.x) + (grid_size % kernel_threads.x != 0),
                     (grid_size / kernel_threads.y) + (grid_size % kernel_threads.y != 0));
  for (curr_iter = 0u; curr_iter < max_iter && err > etol; curr_iter += n_err_iter)
  {

    for (n_err_iter = 0; n_err_iter < N_ERR_COMPUTING_IN_DEVICE; n_err_iter += 2)
    {

      grid_recompute<<<kernel_blocks, kernel_threads, 0, cu_streams_unique[0]>>>(
          cu_buff_grid_unique.get(), cu_buff_grid_unique.get() + half_buff_grid_size, grid_size, grid_pitch);

      grid_recompute<<<kernel_blocks, kernel_threads, 0, cu_streams_unique[0]>>>(
          cu_buff_grid_unique.get() + half_buff_grid_size, cu_buff_grid_unique.get(), grid_size, grid_pitch);
    }

    compute_error(cu_buff_grid_unique.get(), grid_pitch, cu_diff_buff_unique.get(), diff_pitch, grid_size, cu_streams_unique[0], *cublas_handle_unique.get(), *cu_err_idx_unique.get(), err);

    CUDA_CHECK(cudaStreamSynchronize(cu_streams_unique[0]));

    if (err < 0.0)
    {
      err = -err;
    }
  }

  CUDA_CHECK(cudaStreamSynchronize(cu_streams_unique[0]));

  CUDA_CHECK(cu_memcpy2D_async(cu_init_grid_pinned_unique.get(), grid_size,
                               cu_buff_grid_unique.get(), grid_pitch,
                               grid_size, grid_size, cudaMemcpyDefault));

  if (NULL != last_iter)
  {
    *last_iter = curr_iter;
  }
  if (NULL != last_etol)
  {
    *last_etol = err;
  }

  return 0;
}

const char *get_solver_version()
{
  return "CUDA once mem alloc";
}
