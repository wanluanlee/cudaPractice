
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
  }
  return result;
}

__global__ void matrix_multi(int *d_array1, int *d_array2, int *d_gpu_array, size_t nColumn)
{
  int row = blockIdx.y*blockDim.y+threadIdx.y;
  int column = blockIdx.x*blockDim.x+threadIdx.x;
  int val = 0;
  if (row < nColumn && column < nColumn) 
  {
    for(int i = 0; i < nColumn; ++i)
    {
      val += d_array1[row*nColumn+i] * d_array2[i * nColumn + column];
    }

    d_gpu_array[row*nColumn+column] = val;
  }
  
}

void cpu_matrx_multi(int* vec1, int* vec2, int* cpu_res, size_t nColumn) 
{
    for(int i = 0; i < nColumn; ++i)
    {
        for(int j = 0; j < nColumn; ++j)
        {
            for(int k = 0; k < nColumn; ++k)
            {
              cpu_res[i*nColumn+j] += vec1[i*nColumn+k] * vec2[k*nColumn+j];
            }
        }
    }
}

// cudaMalloc to allocate space for the three vectors
// copy the data from CPU to GPU
// call kernel to compute vector addition
// copy the data from GPU to CPU
void gpu_matrx_multi(int* h_vec1, int* h_vec2, int* h_gpu_res, size_t nColumn)
{   
  int *d_array1;
  int *d_array2;
  int *d_gpu_array;
  
  //cudaMalloc to allocate space for the three vectors

  checkCuda(cudaMalloc((void **)&d_array1, sizeof(int) * nColumn * nColumn));
  checkCuda(cudaMalloc((void **)&d_array2, sizeof(int) * nColumn * nColumn));
  checkCuda(cudaMalloc((void **)&d_gpu_array, sizeof(int) * nColumn * nColumn));
  
  //copy the data from CPU to GPU
  checkCuda(cudaMemcpy(d_array1, h_vec1, sizeof(int) * nColumn * nColumn, cudaMemcpyHostToDevice));
  checkCuda(cudaMemcpy(d_array2, h_vec2, sizeof(int) * nColumn * nColumn, cudaMemcpyHostToDevice));

  // call kernel to compute matrix multiplication
  // TODO:
  // integer division equivalent to ceil(A/B) = (A + B - 1) / B
  dim3 threadsPerBlock(nColumn, nColumn);
  dim3 blocksPerGrid(1, 1);
  if (nColumn*nColumn > 1024)
  {
    threadsPerBlock.x = 32;
    threadsPerBlock.y = 32;
    blocksPerGrid.x = (nColumn + 32 - 1) / 32;
    blocksPerGrid.y = (nColumn + 32 - 1) / 32;
  }
  matrix_multi <<< blocksPerGrid,threadsPerBlock >>> (d_array1,d_array2,d_gpu_array,nColumn);
  checkCuda(cudaDeviceSynchronize());

  //copy the data from GPU to CPU
  checkCuda(cudaMemcpy(h_gpu_res, d_gpu_array, sizeof(int) * nColumn * nColumn, cudaMemcpyDeviceToHost));
  checkCuda(cudaFree(d_array1));
  checkCuda(cudaFree(d_array2));
  checkCuda(cudaFree(d_gpu_array));
}

int main(int argc, char* argv[])  {

  if(argc != 2) 
  {
    std::cerr << "usage: ./a.out N" << "\n";
    return -1;
  }


  size_t length = atoi(argv[1]);

  //check if the input number is a prefect square
  // integer division equivalent to ceil(A/B) = (A + B - 1) / B
  if (!(ceil((double)sqrt(length)) == floor((double)sqrt(length))))
  {
    std::cerr << "not a perfect square" << "\n";
    return -1;
  }
  
  std::vector<int> h_array1(length);
  std::vector<int> h_array2(length);
  std::vector<int> h_cpu_array(length);
  std::vector<int> h_gpu_array(length);

  // std::vector<int> h_array1 = {10, 8, 10, 8};
  // std::vector<int> h_array2 = {2, 4, 2, 8};
  // std::vector<int> h_cpu_array(length);
  // std::vector<int> h_gpu_array(length);

  
  
  //here create two vectors filled in random values
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
  int numberRow = sqrt(length);

  auto start_cpu = std::chrono::steady_clock::now();
  
  cpu_matrx_multi(ptr1, ptr2, ptr_cpu, numberRow); 

  auto end_cpu = std::chrono::steady_clock::now();

  auto elapsed_seconds_cpu = std::chrono::duration_cast<std::chrono::nanoseconds>(end_cpu - start_cpu);
  std::cout << "CPU elapsed time: " << (double)elapsed_seconds_cpu.count() << "\n";

  // https://en.cppreference.com/w/cpp/chrono
  
  // run_gpu_vec_add

  auto start_gpu = std::chrono::steady_clock::now();

  gpu_matrx_multi(ptr1, ptr2, ptr_gpu, numberRow);

  auto end_gpu = std::chrono::steady_clock::now();

  auto elapsed_seconds_gpu = std::chrono::duration_cast<std::chrono::nanoseconds>(end_gpu - start_gpu);
  std::cout << "GPU elapsed time: " << (double) elapsed_seconds_gpu.count() << "\n";
  // nvcc -o test test.cu

  // compare the result
  //check if two array has the same size

  bool ifSame = true;

  for(size_t i = 0; i < length ;++i)
  {
      if(h_cpu_array.at(i) != h_gpu_array.at(i))
      {
        ifSame = false;
        std::cerr << "The results are the  different at " << i << "\n";
        std::cerr << "h_cpu_array.at(i) is " << h_cpu_array.at(i) << "\n";
        std::cerr << "h_gpu_array.at(i) is " << h_gpu_array.at(i) << "\n";
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

  // // show the runtime
  
  // // TODO:
  std::cout << "CPU/GPU time is " << (double)elapsed_seconds_cpu.count()/(double)elapsed_seconds_gpu.count() << "\n";
}
