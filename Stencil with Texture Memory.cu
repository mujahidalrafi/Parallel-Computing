#include "stdio.h"
#include <cuda_runtime.h>

const int blockCount = 60;
const int threadsPerBlock = 256;
const int radius = 3;
const int arraySize = blockCount * threadsPerBlock;
const int arraySizeWithHalos = arraySize + 2 * radius;

texture<int, cudaTextureType1D, cudaReadModeElementType> texRef;

__global__ void Stencil_1d(int* out)
{
	__shared__ int temp[threadsPerBlock + 2 * radius];
	int gindex = threadIdx.x + blockIdx.x * blockDim.x + radius;
	int lindex = threadIdx.x + radius;

	temp[lindex] = tex1D(texRef, gindex);
	if (threadIdx.x < radius)
	{
		temp[lindex - radius] = tex1D(texRef, gindex - radius); 
		temp[lindex + threadsPerBlock] = tex1D(texRef, gindex + threadsPerBlock);
	}

	// Synchronize (ensure all the data is available)
	__syncthreads();

	// Apply the stencil
	int result = 0;
	for (int offset = -radius; offset <= radius; offset++)
		result += temp[lindex + offset];

	// Store the result
	out[gindex - radius] = result;
}

int main()
{
	int h_in[arraySizeWithHalos]; // add halos in the main input array too for simplicity in the kernel code
	int h_out[arraySize];
	for (int i = 0; i < arraySize; i++)
	{
		h_in[radius + i] = i + 1;
		h_out[i] = 0;
	}

	for (int i = 0; i < radius; i++)
	{
		h_in[i] = h_in[arraySizeWithHalos - i - 1] = 0;
	}

	int* d_out;
	cudaMalloc(&d_out, arraySize * sizeof(int));
	cudaMemcpy(d_out, h_out, arraySize * sizeof(int), cudaMemcpyHostToDevice);

	// 	Bind the device input array to the texture reference
	cudaArray_t d_in;
	cudaChannelFormatDesc channel = cudaCreateChannelDesc<int>();
	cudaMallocArray(&d_in, &channel, arraySizeWithHalos, 1, cudaArrayDefault);
	cudaMemcpyToArray(d_in, 0, 0, h_in, arraySizeWithHalos * sizeof(int), cudaMemcpyHostToDevice);
	cudaBindTextureToArray(texRef, d_in, channel);

	Stencil_1d <<<blockCount, threadsPerBlock >>> (d_out);

	cudaMemcpy(h_out, d_out, arraySize * sizeof(int), cudaMemcpyDeviceToHost);
	cudaFree(d_out);

	printf("Radius: %d\n", radius);
	printf("Input:\t\t|\t\tOutput:\n");
	for (int i = 0; i < arraySize; i++)
		printf("%d\t\t|\t\t%d\n", h_in[i + radius], h_out[i]);

	return 0;
}
