#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <omp.h>
#include <openacc.h>

#include "heat_equation_common.h"
#include "heat_equation_solver.h"

static FLOAT_TYPE
parse_etol(const char *etol_str);

static size_t
parse_grid_size(const char *grid_size_str);

static size_t
parse_max_iter(const char *max_iter_str);

static void
init_grid(FLOAT_TYPE left_upper_border, FLOAT_TYPE right_upper_border,
          FLOAT_TYPE left_lower_border, FLOAT_TYPE right_lower_border,
          FLOAT_TYPE *grid, size_t grid_size);

static void
linear_interpolate_col(FLOAT_TYPE y0, FLOAT_TYPE y1, FLOAT_TYPE *grid, size_t col_idx, size_t grid_size);
static void
linear_interpolate_row(FLOAT_TYPE y0, FLOAT_TYPE y1, FLOAT_TYPE *grid, size_t row_idx, size_t grid_size);

int main(int argc, const char *argv[])
{
  if (argc < 4)
  {
    fprintf(stderr, "usage: heat_equation_solver [error tolerance] [grid size] [max number of iterations]\n");
    return 0;
  }

  FLOAT_TYPE etol = parse_etol(argv[1]);
  size_t grid_size = parse_grid_size(argv[2]);
  size_t max_iter = parse_max_iter(argv[3]);

  size_t last_iter;
  FLOAT_TYPE last_etol;
  double t1, t2;
  double elapsed_time = INFINITY;
  size_t number_of_attempts = 1;
  for (size_t attempt = 0; attempt < number_of_attempts; ++attempt)
  {
    FLOAT_TYPE *grid = (FLOAT_TYPE *)calloc((grid_size * grid_size), sizeof(FLOAT_TYPE));
    if (NULL == grid)
    {
      perror("Cannot allocate memory for init grid\n");
      return 1;
    }

    init_grid(10.0, 20.0, 30.0, 20.0, grid, grid_size);

    t1 = omp_get_wtime();
    int err_code = solve_heat_equation(grid, grid_size, max_iter, etol, &last_iter, &last_etol);
    t2 = omp_get_wtime() - t1;
    if (t2 < elapsed_time)
    {
      elapsed_time = t2;
    }
    if (0 != err_code)
    {
      fprintf(stderr, "Error in heat equation solving\n");
    }
  }


  printf("%s;%s;%lu;%lu;%.4f;%.4f\n", get_target_device_type(), get_solver_version(), grid_size, last_iter, elapsed_time, last_etol);
}

static FLOAT_TYPE
parse_etol(const char *etol_str)
{
  char *etol_endptr = NULL;
  FLOAT_TYPE etol = strtod(etol_str, &etol_endptr);
  if (etol_endptr == etol_str)
  {
    fprintf(stderr, "Bat error tolerance parameter, must be a float number");
    exit(0);
  }

  return etol;
}

static size_t
parse_grid_size(const char *grid_size_str)
{
  char *grid_size_endptr = NULL;
  size_t grid_size = strtoull(grid_size_str, &grid_size_endptr, 10);
  if (grid_size_endptr == grid_size_str)
  {
    fprintf(stderr, "Bat grid size parameter, must be a positive integer number");
    exit(0);
  }

  return grid_size;
}

static size_t
parse_max_iter(const char *max_iter_str)
{
  char *max_iter_endptr = NULL;
  size_t max_iter = strtoull(max_iter_str, &max_iter_endptr, 10);
  if (max_iter_endptr == max_iter_str)
  {
    fprintf(stderr, "Bat max iter parameter, must be a positive integer number");
    exit(0);
  }

  return max_iter;
}

static void
init_grid(FLOAT_TYPE left_upper_border, FLOAT_TYPE right_upper_border,
          FLOAT_TYPE left_lower_border, FLOAT_TYPE right_lower_border,
          FLOAT_TYPE *grid, size_t grid_size)
{
  grid[0] = left_upper_border;
  grid[grid_size - 1] = right_lower_border;
  grid[(grid_size - 1) * grid_size] = left_lower_border;
  grid[(grid_size * grid_size) - 1] = right_lower_border;

  linear_interpolate_col(left_upper_border, left_lower_border, grid, 0, grid_size);
  linear_interpolate_col(right_upper_border, right_lower_border, grid, grid_size - 1, grid_size);
  linear_interpolate_row(left_upper_border, right_upper_border, grid, 0, grid_size);
  linear_interpolate_row(left_lower_border, right_lower_border, grid, grid_size - 1, grid_size);
}

static void
linear_interpolate_col(FLOAT_TYPE y0, FLOAT_TYPE y1, FLOAT_TYPE *grid, size_t col_idx, size_t grid_size)
{
  FLOAT_TYPE step = (y1 - y0) / (FLOAT_TYPE)(grid_size - 1);
  for (size_t row_idx = 0; row_idx < grid_size; ++row_idx)
  {
    grid[row_idx * (grid_size - 1) + col_idx] = y0 + ((FLOAT_TYPE)row_idx * step);
  }
}

static void
linear_interpolate_row(FLOAT_TYPE y0, FLOAT_TYPE y1, FLOAT_TYPE *grid, size_t row_idx, size_t grid_size)
{
  FLOAT_TYPE step = (y1 - y0) / (FLOAT_TYPE)(grid_size - 1);
  for (size_t col_idx = 0; col_idx < grid_size; ++col_idx)
  {
    grid[row_idx * (grid_size - 1) + col_idx] = y0 + ((FLOAT_TYPE)col_idx * step);
  }
}
