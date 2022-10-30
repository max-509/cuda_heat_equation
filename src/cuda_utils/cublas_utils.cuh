#ifndef CUBLAS_UTILS_H
#define CUBLAS_UTILS_H

#include <cublas_v2.h>

#include <stdexcept>
#include <cstdio>
#include <type_traits>
#include <memory>

#define CUBLAS_CHECK(err)                                                           \
  do                                                                                \
  {                                                                                 \
    cublasStatus_t err_ = (err);                                                    \
    if (err_ != CUBLAS_STATUS_SUCCESS)                                              \
    {                                                                               \
      std::fprintf(stderr, "cublas error %d at %s:%d\n", err_, __FILE__, __LINE__); \
      throw std::runtime_error("cublas error");                                     \
    }                                                                               \
  } while (0)

struct cublas_handle_deleter
{
  void operator()(cublasHandle_t *handle_ptr)
  {
    CUBLAS_CHECK(cublasDestroy(*handle_ptr));
    delete handle_ptr;
  }
};

std::unique_ptr<cublasHandle_t, cublas_handle_deleter> cublas_init_handle();

cublasStatus_t
cublasCopy(cublasHandle_t handle,
           int n,
           const float *x,
           int incx,
           float *y,
           int incy);

cublasStatus_t cublasCopy(cublasHandle_t handle,
                          int n,
                          const double *x,
                          int incx,
                          double *y,
                          int incy);

cublasStatus_t cublasAxpy(cublasHandle_t handle,
                          int n,
                          const float *alpha, /* host or device pointer */
                          const float *x,
                          int incx,
                          float *y,
                          int incy);

cublasStatus_t cublasAxpy(cublasHandle_t handle,
                          int n,
                          const double *alpha, /* host or device pointer */
                          const double *x,
                          int incx,
                          double *y,
                          int incy);

cublasStatus_t cublasIamax(cublasHandle_t handle,
                           int n,
                           const float *x,
                           int incx,
                           int *result);

cublasStatus_t cublasIamax(cublasHandle_t handle,
                           int n,
                           const double *x,
                           int incx,
                           int *result);

cublasStatus_t cublasGeam(cublasHandle_t handle,
                          cublasOperation_t transa, cublasOperation_t transb,
                          int m, int n,
                          const float *alpha,
                          const float *A, int lda,
                          const float *beta,
                          const float *B, int ldb,
                          float *C, int ldc);

cublasStatus_t cublasGeam(cublasHandle_t handle,
                          cublasOperation_t transa, cublasOperation_t transb,
                          int m, int n,
                          const double *alpha,
                          const double *A, int lda,
                          const double *beta,
                          const double *B, int ldb,
                          double *C, int ldc);

#endif // CUBLAS_UTILS_H
