#ifndef HEAT_EQUATION_SOLVER_IMPL_H
#define HEAT_EQUATION_SOLVER_IMPL_H

#include "heat_equation_solver.h"
#include "heat_equation_utils.h"

#define GPU 1
#define CPU 2
#define DEFAULT CPU

#ifndef TARGET_DEVICE
#define TARGET_DEVICE DEFAULT
#endif  // TARGET_DEVICE

const char *get_target_device_type() {
  if (TARGET_DEVICE == GPU) {
    return "GPU";
  } else if (TARGET_DEVICE == CPU) {
    return "CPU";
  }

  return NULL;
}

#endif  // HEAT_EQUATION_SOLVER_IMPL_H
