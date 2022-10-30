#include "cuda_utils.cuh"

std::unique_ptr<cudaStream_t[], cu_streams_deleter> cu_create_streams(std::size_t size)
{
  std::unique_ptr<cudaStream_t[]> streams_ptr{new cudaStream_t[size]};

  for (std::size_t i = 0; i < size; ++i)
  {
    CUDA_CHECK(cudaStreamCreate(&streams_ptr[i]));
  }

  return std::unique_ptr<cudaStream_t[], cu_streams_deleter>{streams_ptr.release(), cu_streams_deleter{size}};
}

std::unique_ptr<cudaStream_t, cu_stream_deleter> cu_create_stream()
{
  std::unique_ptr<cudaStream_t> stream_ptr{new cudaStream_t};
  CUDA_CHECK(cudaStreamCreate(stream_ptr.get()));
  return std::unique_ptr<cudaStream_t, cu_stream_deleter>{stream_ptr.release(), cu_stream_deleter{}};
}
