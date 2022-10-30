#include "heat_equation_solver_impl.h"

#include <stdlib.h>
#include <math.h>

double maxd(double v1, double v2)
{
  return v1 < v2 ? v2 : v1;
}

float maxf(float v1, float v2)
{
  return v1 < v2 ? v2 : v1;
}

long double maxld(long double v1, long double v2)
{
  return v1 < v2 ? v2 : v1;
}

#define fabs(x) _Generic((x),        \
                         long double \
                         : fabsl,    \
                           float     \
                         : fabsf,    \
                           default   \
                         : fabs)(x)

#define fmax(x, y) _Generic((x) + (y),  \
                            long double \
                            : maxld,    \
                              float     \
                            : maxf,     \
                              default   \
                            : maxd)(x, y)

#ifndef N_ERR_COMPUTING_IN_DEVICE
#define N_ERR_COMPUTING_IN_DEVICE 1500
#endif // N_ERR_COMPUTING_IN_DEVICE

int solve_heat_equation(FLOAT_TYPE * restrict init_grid, const size_t grid_size, const size_t max_iter, const FLOAT_TYPE etol, size_t *last_iter, FLOAT_TYPE *last_etol)
{
  const size_t grid_sqr = grid_size * grid_size;
  FLOAT_TYPE *restrict buff_grid = (FLOAT_TYPE *)malloc(sizeof(FLOAT_TYPE) * (grid_sqr * 2));
  if (NULL == buff_grid)
  {
    return 1;
  }

#pragma acc data copy(init_grid [0:grid_sqr]) create(buff_grid [0:grid_sqr * 2])
  {
    FLOAT_TYPE err = (FLOAT_TYPE)INFINITY;

    size_t curr_iter;

#pragma acc data present(init_grid [0:grid_sqr], buff_grid [0:grid_sqr * 2])
#pragma acc parallel
    {
#pragma acc loop independent collapse(2)
      for (size_t i = 0; i < grid_size; ++i)
      {
        for (size_t j = 0; j < grid_size; ++j)
        {
          size_t grid_idx = i * grid_size + j;
          buff_grid[grid_idx] = init_grid[grid_idx];
          buff_grid[grid_sqr + grid_idx] = init_grid[grid_idx];
        }
      }
    }

    size_t n_err_iter;
    for (curr_iter = 0u; curr_iter < max_iter && err > etol; curr_iter += n_err_iter)
    {

      for (n_err_iter = 0; n_err_iter + 2 < N_ERR_COMPUTING_IN_DEVICE; n_err_iter += 2)
      {
#pragma acc data present(buff_grid [0:grid_sqr * 2])
#pragma acc kernels async
        {
#pragma acc loop independent collapse(2)
          for (size_t i = 1; i < grid_size - 1; ++i)
          {
            for (size_t j = 1; j < grid_size - 1; ++j)
            {
              const size_t grid_idx = i * grid_size + j;
              buff_grid[grid_sqr + grid_idx] = (FLOAT_TYPE)0.25 * (buff_grid[grid_idx - grid_size] +
                                                                   buff_grid[grid_idx + grid_size] +
                                                                   buff_grid[grid_idx - 1] +
                                                                   buff_grid[grid_idx + 1]);
            }
          }

#pragma acc loop independent collapse(2)
          for (size_t i = 1; i < grid_size - 1; ++i)
          {
            for (size_t j = 1; j < grid_size - 1; ++j)
            {
              const size_t grid_idx = i * grid_size + j;
              const size_t next_grid_idx = grid_sqr + grid_idx;
              buff_grid[grid_idx] = (FLOAT_TYPE)0.25 * (buff_grid[next_grid_idx - grid_size] +
                                                        buff_grid[next_grid_idx + grid_size] +
                                                        buff_grid[next_grid_idx - 1] +
                                                        buff_grid[next_grid_idx + 1]);
            }
          }
        }
      }

      err = (FLOAT_TYPE)0.0;
#pragma acc wait
#pragma acc data present(buff_grid [0:grid_sqr * 2]) copy(err)
#pragma acc kernels
      {
#pragma acc loop independent collapse(2)
        for (size_t i = 1; i < grid_size - 1; ++i)
        {
          for (size_t j = 1; j < grid_size - 1; ++j)
          {
            const size_t grid_idx = i * grid_size + j;
            const size_t next_grid_idx = grid_sqr + grid_idx;
            buff_grid[next_grid_idx] = (FLOAT_TYPE)0.25 * (buff_grid[grid_idx - grid_size] +
                                                           buff_grid[grid_idx + grid_size] +
                                                           buff_grid[grid_idx - 1] +
                                                           buff_grid[grid_idx + 1]);
          }
        }

#pragma acc loop independent collapse(2) reduction(max \
                                                   : err)
        for (size_t i = 1; i < grid_size - 1; ++i)
        {
          for (size_t j = 1; j < grid_size - 1; ++j)
          {
            const size_t grid_idx = i * grid_size + j;
            const size_t next_grid_idx = grid_sqr + grid_idx;
            buff_grid[grid_idx] = (FLOAT_TYPE)0.25 * (buff_grid[next_grid_idx - grid_size] +
                                                      buff_grid[next_grid_idx + grid_size] +
                                                      buff_grid[next_grid_idx - 1] +
                                                      buff_grid[next_grid_idx + 1]);
            err = fmax(err, fabs(buff_grid[next_grid_idx] - buff_grid[grid_idx]));
          }
        }
      }

      n_err_iter += 2;
    }

#pragma acc data present(init_grid [0:grid_sqr], buff_grid [0:grid_sqr * 2])
#pragma acc parallel
    {
#pragma acc loop independent collapse(2)
      for (size_t i = 0; i < grid_size; ++i)
      {
        for (size_t j = 0; j < grid_size; ++j)
        {
          size_t grid_idx = i * grid_size + j;
          init_grid[grid_idx] = buff_grid[grid_idx];
        }
      }
    }

    if (NULL != last_iter)
    {
      *last_iter = curr_iter;
    }
    if (NULL != last_etol) {
      *last_etol = err;
    }
  }

  free(buff_grid);

  return 0;
}

const char *get_solver_version() {
  return "OpenACC";
}
