#include "cublas_utils.cuh"

std::unique_ptr<cublasHandle_t, cublas_handle_deleter> cublas_init_handle()
{
  std::unique_ptr<cublasHandle_t> handle_ptr{new cublasHandle_t};
  CUBLAS_CHECK(cublasCreate(handle_ptr.get()));

  return std::unique_ptr<cublasHandle_t, cublas_handle_deleter>{handle_ptr.release(), cublas_handle_deleter{}};
}

cublasStatus_t cublasCopy(cublasHandle_t handle,
                          int n,
                          const float *x,
                          int incx,
                          float *y,
                          int incy)
{
  return cublasScopy(handle, n, x, incx, y, incy);
}

cublasStatus_t cublasCopy(cublasHandle_t handle,
                          int n,
                          const double *x,
                          int incx,
                          double *y,
                          int incy)
{
  return cublasDcopy(handle, n, x, incx, y, incy);
}

cublasStatus_t cublasAxpy(cublasHandle_t handle,
                          int n,
                          const float *alpha, /* host or device pointer */
                          const float *x,
                          int incx,
                          float *y,
                          int incy)
{
  return cublasSaxpy(handle, n, alpha, x, incx, y, incy);
}

cublasStatus_t cublasAxpy(cublasHandle_t handle,
                          int n,
                          const double *alpha, /* host or device pointer */
                          const double *x,
                          int incx,
                          double *y,
                          int incy)
{
  return cublasDaxpy(handle, n, alpha, x, incx, y, incy);
}

cublasStatus_t cublasIamax(cublasHandle_t handle,
                           int n,
                           const float *x,
                           int incx,
                           int *result)

{
  return cublasIsamax(handle, n, x, incx, result);
}

cublasStatus_t cublasIamax(cublasHandle_t handle,
                           int n,
                           const double *x,
                           int incx,
                           int *result)
{
  return cublasIdamax(handle, n, x, incx, result);
}

cublasStatus_t cublasGeam(cublasHandle_t handle,
                          cublasOperation_t transa, cublasOperation_t transb,
                          int m, int n,
                          const float *alpha,
                          const float *A, int lda,
                          const float *beta,
                          const float *B, int ldb,
                          float *C, int ldc)
{
  return cublasSgeam(handle, transa, transb, m, n, alpha, A, lda, beta, B, ldb, C, ldc);
}

cublasStatus_t cublasGeam(cublasHandle_t handle,
                          cublasOperation_t transa, cublasOperation_t transb,
                          int m, int n,
                          const double *alpha,
                          const double *A, int lda,
                          const double *beta,
                          const double *B, int ldb,
                          double *C, int ldc)
{
  return cublasDgeam(handle, transa, transb, m, n, alpha, A, lda, beta, B, ldb, C, ldc);
}
