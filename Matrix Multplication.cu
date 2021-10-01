#include "stdio.h"
#include <stdlib.h>   
#include <cuda_runtime.h>
#include <math.h>

const int rowCountA = 10;
const int colCountA = 14;
const int colCountB = 15;
const int rowCountB = 12;

const int blockWidth = 4;

struct Matrix
{
	int *elements = nullptr;
	int rowCount = 0;
	int columnCount = 0;

	Matrix(int row, int col)
	{
		elements = nullptr;
		rowCount = row;
		columnCount = col;	
	}

	Matrix(){}

	void Host_Allocate(bool populateToo)
	{
		elements = (int*)malloc(rowCount * columnCount * sizeof(int));

		if (!populateToo)
			return;

		for (int row = 0; row < rowCount; row++)
		{
			for (int col = 0; col < columnCount; col++)
				elements[row*columnCount + col] = rand() % 10 + 1;
		}
	}
	
	void Print()
	{
		for (int row = 0; row < rowCount; row++)
		{
			for (int col = 0; col < columnCount; col++)
				printf("%d\t", elements[row * columnCount + col]);

			printf("\n");
		}
	}
};


__global__ void MatrixMultiply(struct Matrix a, struct Matrix b, struct Matrix out)
{
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	int product = 0;

	for (int i = 0; i < a.columnCount; i++)
		product += a.elements[row*a.columnCount + i] * b.elements[b.columnCount*i + col];
	
	out.elements[row * b.columnCount + col] = product;
}

int main()
{
	struct Matrix h_a(rowCountA, colCountA), h_b(rowCountB, colCountB), h_out(rowCountA, colCountB);
	h_a.Host_Allocate(true);
	h_b.Host_Allocate(true);
	h_out.Host_Allocate(false);

	struct Matrix d_a(rowCountA, colCountA), d_b(rowCountB, colCountB), d_out(rowCountA, colCountB);
	
	cudaMalloc(&d_a.elements, rowCountA * colCountA * sizeof(int));
	cudaMemcpy(d_a.elements, h_a.elements, rowCountA * colCountA * sizeof(int), cudaMemcpyHostToDevice);

	cudaMalloc(&d_b.elements, rowCountB * colCountB * sizeof(int));
	cudaMemcpy(d_b.elements, h_b.elements, rowCountB * colCountB * sizeof(int), cudaMemcpyHostToDevice);

	cudaMalloc(&d_out.elements, rowCountA * colCountB * sizeof(int));


	dim3 blockDim(blockWidth, blockWidth);
	dim3 gridDim(ceil(1.0 * colCountB / blockDim.x), ceil(1.0 * rowCountA / blockDim.y));
	MatrixMultiply<<<gridDim, blockDim >>> (d_a, d_b, d_out);

	cudaMemcpy(h_out.elements, d_out.elements, rowCountA * colCountB * sizeof(int), cudaMemcpyDeviceToHost);

	printf("\nA:\n");
	h_a.Print();
	printf("\nB:\n");
	h_b.Print();
	printf("\nOut:\n");
	h_out.Print();

	return 0;
}

