#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <iostream>

using namespace std;

cudaError_t maxWithCuda(int* max_of_mins, int* matr, unsigned int size);


__global__ void minsRows(int* max_of_mins, const int *matrix, unsigned int matrix_size)
{
    //printf("%i %i\n", blockIdx.x, blockDim.x);
    int thread_number = threadIdx.x;
    int block_number = blockIdx.x;
    int block_size = blockDim.x;

    int raw_number = (block_number * block_size + thread_number);
    int min_in_column = matrix[raw_number * matrix_size];
    for (int i = 1; i < matrix_size; i++) 
        if (matrix[raw_number * matrix_size + i] < min_in_column)
            min_in_column = matrix[raw_number * matrix_size + i];
    
    if (block_number == 0 && thread_number == 0)
        *max_of_mins = min_in_column;
    else
        if (*max_of_mins < min_in_column)
            *max_of_mins = min_in_column;
}

int main()
{
    // Размер матрицы кратный 1024
    const int matrSize = 20480, outputSize = 10;

    printf("Matrix size %d\n", matrSize);

    int *matrix = new int[matrSize * matrSize];
    int *rows_mins = new int[matrSize];
    int max_of_mins = 0;
    for (int i = 0; i < matrSize; i++) 
    {
        rows_mins[i] = 0;
        for (int j = 0; j < matrSize; j++) 
        {
            matrix[i * matrSize + j] = i + j;
            if (i < outputSize && j < outputSize)
                printf("%i ", matrix[i * matrSize + j]);
        }
        if (i < outputSize)
            printf("\n");
    }
    printf("\n");
    


    // Add in parallel.
    cudaError_t cudaStatus = maxWithCuda(&max_of_mins, matrix, matrSize);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
        return 1;
    }

    printf("Max of mins %i\n", max_of_mins);


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
cudaError_t maxWithCuda(int* max_of_mins, int *matr, unsigned int size)
{
    int *dev_matr = 0;
    int *dev_max_of_mins = 0;
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
        fprintf(stderr, "cudaMalloc failed!1");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_max_of_mins, sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed! with max");
        goto Error;
    }


    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_matr, matr, size * size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }




    float elapsed = 0;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);


    // Launch a kernel on the GPU with one thread for each column.
    minsRows <<<size / 1024, 1024>>>(dev_max_of_mins, dev_matr, size);


    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    printf("The elapsed time %.2f ms\n", elapsed);



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

    cudaStatus = cudaMemcpy(max_of_mins, dev_max_of_mins, sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed! with max");
        goto Error;
    }



    Error:
    cudaFree(dev_max_of_mins);
    cudaFree(dev_matr);
    
    return cudaStatus;
}
