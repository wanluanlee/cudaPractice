
#include <cstdlib>
#include <iostream>
#include <random>
#include <time.h> 
#include <chrono>
#include <math.h>

cudaError_t checkCuda(cudaError_t result)
{
  if (result != cudaSuccess) {
    using namespace std::literals::string_literals;
    throw std::runtime_error("CUDA Runtime Error : "s + cudaGetErrorString(result));
    //assert(result == cudaSuccess);
  }
  return result;
}

__global__ void add_vectors(int *d_array1, int *d_array2, int *d_gpu_array)
{
  int idx = blockIdx.x * 512 + threadIdx.x;
  d_gpu_array[idx] = d_array1[idx] + d_array2[idx];
}

void cpu_vec_add(int* vec1, int* vec2, int* cpu_res, size_t N) 
{
  for(size_t i = 0; i < N; i++) 
  {
    cpu_res[i] = vec1[i] + vec2[i];
  }
}

// cudaMalloc to allocate space for the three vectors
// copy the data from CPU to GPU
// call kernel to compute vector addition
// copy the data from GPU to CPU
void gpu_vec_add(int* h_vec1, int* h_vec2, int* h_gpu_res, size_t N)
{   
  int *d_array1;
  int *d_array2;
  int *d_gpu_array;
  
  //cudaMalloc to allocate space for the three vectors
  
  checkCuda(cudaMalloc((void **)&d_array1, sizeof(int) * N));
  checkCuda(cudaMalloc((void **)&d_array2, sizeof(int) * N));
  checkCuda(cudaMalloc((void **)&d_gpu_array, sizeof(int) * N));
  
  //copy the data from CPU to GPU

  checkCuda(cudaMemcpy(d_array1, h_vec1, sizeof(int) * N, cudaMemcpyHostToDevice));
  checkCuda(cudaMemcpy(d_array2, h_vec2, sizeof(int) * N, cudaMemcpyHostToDevice));

  // call kernel to compute vector addition

  int number_block = ceil(N/512.0);
  add_vectors <<< number_block, 512 >>> (d_array1,d_array2,d_gpu_array);
  checkCuda(cudaDeviceSynchronize());

  //copy the data from GPU to CPU

  checkCuda(cudaMemcpy(h_gpu_res, d_gpu_array, sizeof(int) * N, cudaMemcpyDeviceToHost));
  checkCuda(cudaFree(d_array1));
  checkCuda(cudaFree(d_array2));
  checkCuda(cudaFree(d_gpu_array));
  
  
}

int main(int argc, char* argv[]) 
{
  if(argc != 2) 
  {
    std::cerr << "usage: ./a.out N" << "\n";
  }


  size_t length = atoi(argv[1]);

  std::vector<int> h_array1(length);
  std::vector<int> h_array2(length);
  std::vector<int> h_cpu_array(length);
  std::vector<int> h_gpu_array(length);

  // here create two vectors filled in random values
  srand(time(0));

  for(auto& element: h_array1) 
  {
	  element = (1+ (rand() % 10));

  }

  for(auto& element: h_array2) 
  {
	  element = (1+ (rand() % 10));

  }


  // run cpu_vec_add and measure the runtime using std::chrono library
  int *ptr1 = h_array1.data();
  int *ptr2 = h_array2.data();
  int *ptr_cpu = h_cpu_array.data();
  int *ptr_gpu = h_gpu_array.data();

  auto start_cpu = std::chrono::steady_clock::now();
  
  cpu_vec_add(ptr1, ptr2, ptr_cpu, length); 

  auto end_cpu = std::chrono::steady_clock::now();

  std::chrono::duration<double> elapsed_seconds_cpu = end_cpu - start_cpu;
  std::cout << "CPU elapsed time: " << elapsed_seconds_cpu.count() << "\n";

  // run_gpu_vec_add
  auto start_gpu = std::chrono::steady_clock::now();

  gpu_vec_add(ptr1, ptr2, ptr_gpu, length);

  auto end_gpu = std::chrono::steady_clock::now();

  std::chrono::duration<double> elapsed_seconds_gpu = end_gpu - start_gpu;
  std::cout << "GPU elapsed time: " << elapsed_seconds_gpu.count() << "\n";

  // compare the result
  //check if two array has the same size

  bool ifSame = true;

  for(size_t i = 0; i < length ;++i)
  {
      if(h_cpu_array.at(i) != h_gpu_array.at(i))
      {
        ifSame = false;
        break;
      }
  }

  if(!ifSame)
  {
    std::cerr << "The results are different \n";
  }

  else
  {
    std::cerr << "The results are the same \n";
  }

  // show the runtime

  std::cout << "CPU/GPU time is " << elapsed_seconds_cpu.count()/elapsed_seconds_gpu.count() << "\n";
}
