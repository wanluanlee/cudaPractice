#include <cuda_runtime.h>
#include <helper_cuda.h>

#include <iostream>
#include <memory>
#include <string>

#include <cuda.h>

cudaError_t checkCuda(cudaError_t result)
{
  if (result != cudaSuccess) {
    using namespace std::literals::string_literals;
    throw std::runtime_error("CUDA Runtime Error : "s + cudaGetErrorString(result));
    //assert(result == cudaSuccess);
  }
  return result;
}

int main(int argc, char **argv) 
{

    int deviceCount = 0;
    cudaError_t error_id = cudaGetDeviceCount(&deviceCount);

    if (error_id != cudaSuccess) 
    {
        printf("cudaGetDeviceCount returned %d\n-> %s\n",
           static_cast<int>(error_id), cudaGetErrorString(error_id));
        printf("Result = FAIL\n");
        exit(EXIT_FAILURE);
    }

        // This function call returns 0 if there are no CUDA capable devices.
    if (deviceCount == 0) 
    {
        printf("There are no available device(s) that support CUDA\n");
    } 
    else 
    {
        printf("Detected %d CUDA Capable device(s)\n", deviceCount);
    }

}