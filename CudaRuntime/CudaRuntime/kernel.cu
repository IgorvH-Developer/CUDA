#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <iostream>

using namespace std;

cudaError_t maxWithCuda(int* columns_mins, int* matr, unsigned int size);


__global__ void minsRows(int * rows_mins, const int *matrix)
{
    int thread_number = threadIdx.x;
    int matrix_size = blockDim.x;

    int min_in_column = matrix[thread_number * matrix_size];
    for (int i = 1; i < matrix_size; i++)
        if (matrix[thread_number * matrix_size + i] < min_in_column)
            min_in_column = matrix[thread_number * matrix_size + i];
   
    rows_mins[thread_number] = min_in_column;
}

int main()
{
    const int matrSize = 10;

    int *matrix = new int[matrSize * matrSize];
    int *rows_mins = new int[matrSize];
    for (int i = 0; i < matrSize; i++) 
    {
        rows_mins[i] = 0;
        for (int j = 0; j < matrSize; j++) 
        {
            matrix[i * matrSize + j] = i + j;
            printf("%i ", matrix[i * matrSize + j]);
        }
        printf("\n");
    }
    printf("\n");



    // Add vectors in parallel.
    cudaError_t cudaStatus = maxWithCuda(rows_mins, matrix, matrSize);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
        return 1;
    }

    //Print output
    printf("Mins of rows: ");
    int max_matrix = rows_mins[0];
    for (int i = 1; i < matrSize; i++) 
    {
        printf("%i ", rows_mins[i]);
        if (rows_mins[i] > max_matrix)
            max_matrix = rows_mins[i];
    }
    printf("\n");

    printf("Max of mins %i\n", max_matrix);




    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}

// Helper function for using CUDA
cudaError_t maxWithCuda(int *rows_mins, int *matr, unsigned int size)
{
    int* dev_rows_mins = 0;
    int *dev_matr = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers  
    cudaStatus = cudaMalloc((void**)&dev_matr, size * size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_rows_mins, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_matr, matr, size * size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_rows_mins, rows_mins, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }



    // Launch a kernel on the GPU with one thread for each column.
    minsRows <<<1, size>>>(dev_rows_mins, dev_matr);



    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }


    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(rows_mins, dev_rows_mins, size * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }



    Error:
    cudaFree(dev_rows_mins);
    cudaFree(dev_matr);
    
    return cudaStatus;
}
