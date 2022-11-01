#include "heat_equation_solver_impl.h"

#include "cuda_utils/cuda_utils.cuh"

#include <cub/block/block_reduce.cuh>
#include <cub/device/device_reduce.cuh>

#include <stdexcept>
#include <cstdlib>
#include <cmath>

namespace
{
  constexpr std::size_t GRID_BLOCK_SIZE = 16;
  constexpr std::size_t ERROR_BLOCK_SIZE = 16;
}

#ifndef N_ERR_COMPUTING_IN_DEVICE
#define N_ERR_COMPUTING_IN_DEVICE 1500
#endif // N_ERR_COMPUTING_IN_DEVICE

template <int block_size>
__global__ void compute_partial_errors(const FLOAT_TYPE *__restrict__ curr_grid,
                                 const FLOAT_TYPE *__restrict__ next_grid,
                                 const std::size_t grid_size,
                                 const std::size_t pitch,
                                 FLOAT_TYPE *__restrict__ err_ptr)
{
  using BlockReduce = cub::BlockReduce<FLOAT_TYPE, block_size, cub::BLOCK_REDUCE_RAKING_COMMUTATIVE_ONLY, block_size>;

  __shared__ typename BlockReduce::TempStorage temp_storage;
  const auto thread_data_size = grid_size / block_size + ((grid_size % block_size) != 0);
  const auto i_y_begin = threadIdx.y * thread_data_size;
  const auto i_y_end = min(grid_size, i_y_begin + thread_data_size);
  const auto i_x_begin = threadIdx.x * thread_data_size;
  const auto i_x_end = min(grid_size, i_x_begin + thread_data_size);

  FLOAT_TYPE abs_diff = 0.0;

  #pragma unroll
  for (std::size_t i_y = i_y_begin; i_y < i_y_end; ++i_y)
  {
    for (std::size_t i_x = i_x_begin; i_x < i_x_end; ++i_x)
    {
      const auto grid_idx = i_y * pitch + i_x;
      abs_diff = fmax(abs_diff, fabs(curr_grid[grid_idx] - next_grid[grid_idx]));
    }
  }

  auto block_max_err = BlockReduce(temp_storage).Reduce(abs_diff, cub::Max{});

  if (threadIdx.x == 0)
  {
    *err_ptr = block_max_err;
  }
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
  auto cu_streams_unique = cu_create_streams(2);

  std::size_t grid_pitch;
  auto cu_buff_grid_unique = cu_make_pitched_memory_unique<FLOAT_TYPE>(2 * grid_size, grid_size, grid_pitch);

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

  CUDA_CHECK(cudaDeviceSynchronize());

  auto cu_err_unique = cu_make_host_memory_unique<FLOAT_TYPE>();
  FLOAT_TYPE &err = *cu_err_unique.get();
  err = INFINITY;

  size_t curr_iter;
  size_t n_err_iter;
  dim3 kernel_threads(GRID_BLOCK_SIZE, GRID_BLOCK_SIZE);
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

    compute_partial_errors<ERROR_BLOCK_SIZE><<<1, dim3(ERROR_BLOCK_SIZE, ERROR_BLOCK_SIZE), 0, cu_streams_unique[0]>>>(
        cu_buff_grid_unique.get(), cu_buff_grid_unique.get() + half_buff_grid_size, grid_size, grid_pitch, &err);

    CUDA_CHECK(cudaStreamSynchronize(cu_streams_unique[0]));
  }

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
  return "CUDA CUB one block";
}
