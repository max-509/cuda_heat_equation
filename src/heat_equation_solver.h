#ifndef HEAT_EQUATION_SOLVER_NAIVE_H
#define HEAT_EQUATION_SOLVER_NAIVE_H

#include "heat_equation_common.h"

#include <stddef.h>

#ifdef __cplusplus
extern "C" int solve_heat_equation(FLOAT_TYPE *init_grid, size_t gird_size, size_t max_iter, FLOAT_TYPE etol, size_t *last_iter, FLOAT_TYPE *last_etol);

extern "C" const char *get_target_device_type();

extern "C" const char *get_solver_version();
#else  // __cplusplus
int solve_heat_equation(FLOAT_TYPE *init_grid, size_t gird_size, size_t max_iter, FLOAT_TYPE etol, size_t *last_iter, FLOAT_TYPE *last_etol);

const char *get_target_device_type();

const char *get_solver_version();
#endif  // __cplusplus

#endif //  HEAT_EQUATION_SOLVER_NAIVE_H
