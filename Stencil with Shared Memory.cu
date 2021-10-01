#include "stdio.h"
#include <cuda_runtime.h>

const int blockCount = 60;
const int threadsPerBlock = 256;
const int radius = 3;
const int arraySize = blockCount * threadsPerBlock;

__global__ void Stencil_1d(const int* in, int* out)
{
	__shared__ int temp[threadsPerBlock + 2 * radius];
	int gindex = threadIdx.x + blockIdx.x * blockDim.x;
	int lindex = threadIdx.x + radius;

	// Read input elements into shared memory
	temp[lindex] = in[gindex];
	if (threadIdx.x < radius)
	{
		temp[lindex - radius] = in[gindex - radius];
		temp[lindex + threadsPerBlock] = in[gindex + threadsPerBlock];
	}

	// Synchronize (ensure all the data is available)
	__syncthreads();
	
	// Apply the stencil
	int result = 0;
	for (int offset = -radius; offset <= radius; offset++)
		result += temp[lindex + offset];

	// Store the result
	out[gindex] = result;
}

int main()
{
	const int arraySizeWithHalos = arraySize + 2 * radius;
	int h_in[arraySizeWithHalos]; // add halos in the main input array too for simplicity in the kernel code
	int h_out[arraySize];
	for (int i = 0; i < arraySize; i++)
	{
		h_in[radius + i] = i + 1;
		h_out[i] = 0;
	}

	for (int i = 0; i < radius; i++)
		h_in[i] = h_in[arraySizeWithHalos - i - 1]= 0;

	int *d_in, *d_out;
	cudaMalloc(&d_in, arraySizeWithHalos * sizeof(int));
	cudaMalloc(&d_out, arraySize * sizeof(int));

	cudaMemcpy(d_in, h_in, arraySizeWithHalos * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_out, h_out, arraySize * sizeof(int), cudaMemcpyHostToDevice);

	Stencil_1d<<<blockCount, threadsPerBlock >>>(d_in + radius, d_out);
	
	cudaMemcpy(h_in, d_in, arraySizeWithHalos * sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_out, d_out, arraySize * sizeof(int), cudaMemcpyDeviceToHost);
	cudaFree(d_in);
	cudaFree(d_out);

	printf("Radius: %d\n", radius);
	printf("Input:\t\t|\t\tOutput:\n");
	for (int i = 0; i < arraySize; i++)
		printf("%d\t\t|\t\t%d\n", h_in[i + radius], h_out[i]);

	return 0;
}

