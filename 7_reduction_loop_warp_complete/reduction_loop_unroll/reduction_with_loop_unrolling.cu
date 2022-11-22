#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "host_utils.h"
#include "cuda_utils.cuh"

#include <iostream>
#include <stdlib.h>

using namespace std;

/* //TODO
__global__ void ReduceSumGPU(int *input, int *sum, int size) {
	int tid = threadIdx.x;
	int gid = blockIdx.x * blockDim.x + tid;

	__syncthreads();
}
*/

__global__ void ReduceSumLoopUnroll(int *input, int *temp, int size) {
	int tid = threadIdx.x;
	int block_offset = blockIdx.x * blockDim.x * 2;
	int index = block_offset + tid;

	int *i_data = input + block_offset;

	if ((index + blockDim.x) < size) {
		input[index] += input[index + blockDim.x];
	}
	__syncthreads();

	for (int offset = blockDim.x / 2; offset > 0; offset = offset / 2) {
		if (tid < offset) {
			i_data[tid] += i_data[tid + offset];
		}
		__syncthreads();
	}

	if (tid == 0) {
		temp[blockIdx.x] = i_data[0];
	}
}

int main() {
	cout << "\n-----Reduce Sum with Loop Unroll-----\n" << endl;
	float gpu_total_milliseconds = 0;
	float kernel_milliseconds = 0;
	clock_t cpu_start, cpu_end;
	cudaEvent_t gpu_start, gpu_end, kernel_start, kernel_end;
	cudaEventCreate(&gpu_start);
	cudaEventCreate(&gpu_end);
	cudaEventCreate(&kernel_start);
	cudaEventCreate(&kernel_end);

	int size = 1 << 20; // 1MB
	int block_size = 128;
	int byte_size = size * sizeof(int);

	int *host_input, *host_ref_unroll;
	host_input = (int*)malloc(byte_size);

	InitializeData(host_input, size, INIT_ONE);

	// Perform reduce sum on CPU
	cpu_start = clock();
	int cpu_result = ReductionSumCPU(host_input, size);
	cpu_end = clock();

	dim3 block(block_size);
	dim3 grid(ceil((size / block_size) / 2));

	printf("Kernel launch parameters -> grid: (%d,%d,%d), block: (%d,%d,%d) \n\n",
            grid.x, grid.y, grid.z, block.x, block.y, block.z);

	int temp_array_byte_size = sizeof(int)* grid.x;
	host_ref_unroll = (int*)malloc(temp_array_byte_size);

	int *device_input, *device_temp;
	GPUErrorCheck(cudaEventRecord(gpu_start, 0));
	GPUErrorCheck(cudaMalloc((void**)&device_input, byte_size));
	GPUErrorCheck(cudaMemcpy(device_input, host_input, byte_size, cudaMemcpyHostToDevice));
	GPUErrorCheck(cudaMalloc((void**)&device_temp, temp_array_byte_size));
	GPUErrorCheck(cudaMemset(device_temp, 0, temp_array_byte_size));

	// Perform reduce sum with loop unroll on GPU
	GPUErrorCheck(cudaEventRecord(kernel_start, 0));
	ReduceSumLoopUnroll<<<grid, block>>>(device_input, device_temp, size);
	GPUErrorCheck(cudaEventRecord(kernel_end, 0));
	GPUErrorCheck(cudaEventSynchronize(kernel_end));

	GPUErrorCheck(cudaDeviceSynchronize());
	GPUErrorCheck(cudaMemcpy(host_ref_unroll, device_temp, temp_array_byte_size, cudaMemcpyDeviceToHost));

	int gpu_result = 0;
	for (int i = 0; i < grid.x; i++) {
		gpu_result += host_ref_unroll[i];
	}
	GPUErrorCheck(cudaEventRecord(gpu_end, 0));
	GPUErrorCheck(cudaEventSynchronize(gpu_end));
	GPUErrorCheck(cudaEventElapsedTime(&kernel_milliseconds, kernel_start, kernel_end));
	GPUErrorCheck(cudaEventElapsedTime(&gpu_total_milliseconds, gpu_start, gpu_end));

	CompareResults(gpu_result, cpu_result);

	printf("CPU execution time (Function Only): %4.6f milliseconds",
		(double)((double)(cpu_end - cpu_start) / CLOCKS_PER_SEC) * 1000.0);
	printf("\nGPU Execution Time (Kernel Only): %4.6f milliseconds", kernel_milliseconds);
	printf("\nTotal GPU Execution Time (Malloc, Memcpy, Kernel): %4.6f milliseconds\n", gpu_total_milliseconds);

	GPUErrorCheck(cudaFree(device_input));
	GPUErrorCheck(cudaFree(device_temp));
	free(host_input);
	free(host_ref_unroll);

	GPUErrorCheck(cudaDeviceReset());
	return 0;
}