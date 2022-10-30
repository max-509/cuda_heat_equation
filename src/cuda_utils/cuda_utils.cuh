#ifndef CUDA_UTILS_H
#define CUDA_UTILS_H

#include <cuda_runtime.h>

#include <cstdio>
#include <stdexcept>
#include <vector>
#include <memory>

#define CUDA_CHECK(err)                                                                   \
  do                                                                                      \
  {                                                                                       \
    cudaError_t err_ = (err);                                                             \
    if (err_ != cudaSuccess)                                                              \
    {                                                                                     \
      std::fprintf(stderr, "cuda error %s at %s:%d with message: %s\n",                   \
                   cudaGetErrorName(err_), __FILE__, __LINE__, cudaGetErrorString(err_)); \
      throw std::runtime_error(cudaGetErrorString(err_));                                 \
    }                                                                                     \
  } while (0)

struct cu_memory_deleter
{
  void operator()(void *ptr) const
  {
    CUDA_CHECK(cudaFree(ptr));
  }
};

struct cu_stream_deleter
{
  void operator()(cudaStream_t *stream_ptr) const
  {
    CUDA_CHECK(cudaStreamDestroy(*stream_ptr));
    delete stream_ptr;
  }
};

struct cu_streams_deleter
{
  explicit cu_streams_deleter(std::size_t n_streams) : m_n_streams{n_streams} {}
  void operator()(cudaStream_t *streams_ptr)
  {
    for (std::size_t i = m_n_streams; i > 0; --i)
    {
      CUDA_CHECK(cudaStreamDestroy(streams_ptr[i - 1]));
    }

    delete[] streams_ptr;
  }

private:
  std::size_t m_n_streams;
};

struct cu_pinned_deleter
{
  void operator()(void *ptr) const
  {
    CUDA_CHECK(cudaHostUnregister(ptr));
  }
};

struct cu_host_deleter
{
  void operator()(void *ptr) const
  {
    CUDA_CHECK(cudaFreeHost(ptr));
  }
};

template <typename T>
std::unique_ptr<T, cu_memory_deleter> cu_make_memory_unique(std::size_t size)
{
  T *ptr = nullptr;
  CUDA_CHECK(cudaMalloc(&ptr, size * sizeof(T)));

  return std::unique_ptr<T, cu_memory_deleter>(ptr, cu_memory_deleter{});
}

template <typename T>
std::unique_ptr<T, cu_memory_deleter> cu_make_memory_unique()
{
  return cu_make_memory_unique<T>(1);
}

template <typename T>
std::unique_ptr<T, cu_memory_deleter> cu_make_pitched_memory_unique(std::size_t nrows,
                                                                    std::size_t ncols,
                                                                    std::size_t &pitch)
{
  T *ptr = nullptr;
  CUDA_CHECK(cudaMallocPitch(&ptr, &pitch, sizeof(T) * ncols, nrows));
  pitch /= sizeof(T);

  return std::unique_ptr<T, cu_memory_deleter>(ptr, cu_memory_deleter{});
}

template <typename T>
std::unique_ptr<T, cu_pinned_deleter> cu_make_pinned_memory_unique(T *ptr, const std::size_t size)
{
  CUDA_CHECK(cudaHostRegister(ptr, sizeof(T) * size, cudaHostRegisterDefault));

  return std::unique_ptr<T, cu_pinned_deleter>{ptr, cu_pinned_deleter{}};
}

template <typename T>
std::unique_ptr<T, cu_host_deleter> cu_make_host_memory_unique(const std::size_t size)
{
  T *ptr = nullptr;
  CUDA_CHECK(cudaHostAlloc(&ptr, sizeof(T) * size, cudaHostAllocDefault));

  return std::unique_ptr<T, cu_host_deleter>{ptr, cu_host_deleter{}};
}

template <typename T>
std::unique_ptr<T, cu_host_deleter> cu_make_host_memory_unique()
{
  return cu_make_host_memory_unique<T>(1);
}

template <typename T>
cudaError_t cu_memcpy_async(T *dst, const T *src, size_t count, enum cudaMemcpyKind kind, cudaStream_t stream = (cudaStream_t)0)
{
  return cudaMemcpyAsync(dst, src, sizeof(T) * count, kind, stream);
}

template <typename T>
cudaError_t cu_memcpy2D_async(T *dst, size_t dpitch, const T *src, size_t spitch, size_t width, size_t height, enum cudaMemcpyKind kind, cudaStream_t stream = (cudaStream_t)0)
{
  return cudaMemcpy2DAsync(dst, sizeof(T) * dpitch, src, sizeof(T) * spitch, sizeof(T) * width, height, kind, stream);
}

template <typename T>
cudaError_t cu_memset_async(T *devPtr, int value, size_t count, cudaStream_t stream = (cudaStream_t)0)
{
  return cudaMemsetAsync(devPtr, value, sizeof(T) * count, stream);
}

template <typename T>
cudaError_t cu_memset2D_async(T *devPtr, size_t pitch, int value, size_t width, size_t height, cudaStream_t stream = (cudaStream_t)0)
{
  return cudaMemset2DAsync(devPtr, sizeof(T) * pitch, value, sizeof(T) * width, height, stream);
}

std::unique_ptr<cudaStream_t[], cu_streams_deleter> cu_create_streams(std::size_t size);

std::unique_ptr<cudaStream_t, cu_stream_deleter> cu_create_stream();

#endif // CUDA_UTILS_H
