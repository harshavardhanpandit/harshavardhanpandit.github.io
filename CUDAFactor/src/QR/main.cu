/******************************************************************************
********************** CUDA Factor Project ************************************
********************* 15-618 Spring 2015 CMU **********************************
*******************************************************************************
*
* Authors: Harshavardhan Pandit (hpandit@andrew.cmu.edu)
*          Ravi Chandra Bandlamudi (rbandlam@andrew.cmu.edu)
* 
* main.cu - Top-level module to check the correctness and performance of the
*           CUDA Factor QR Factorization routine. Sets up the inputs to be passed
*           to the reference (CUBLAS library) and the CUDAFactor routines. 
*           Outputs from both the routines are compared to check for correctness
*           and the wall-clock time required by the functions is displayed as
*           a measure of the performance.
******************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <assert.h>
#include <math.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "qr.h"

int main()
{
     double *cudaFactorInput, *cublasInput, *cublasOutput, *tau;

     int tauDim = max(1, min(M,N));

     /* Set up the input arrays */
     cudaFactorInput = (double *)malloc(sizeof(double) * M * N);
     assert(cudaFactorInput);
     cublasInput = (double *)malloc(sizeof(double) * M * N);
     assert(cublasInput);
     cublasOutput = (double *)malloc(sizeof(double) * M * N);
     assert(cublasOutput);
     tau = (double *)malloc(sizeof(double) * tauDim);
     assert(tau);

     int min = 0, max = 10;

     /* Same random matrix passed as input to both routines */
     for (int i=0; i < M*N; i++)
     {
        cudaFactorInput[i] = ((double) rand() / (RAND_MAX)) * (max-min) + min; 
     }

     /* tau array required by CUBLAS reference routine */
    for (int i=0; i < tauDim; i++)
    {
        tau[i] = (double) 0.0f;
    }     

     /* Transpose the matrix since CUDA Factor works in
        row major order whereas CUBLAS works in column major order */     
     for (int i=0; i < M; i++)
     {
        for (int j=0; j< N; j++)
        {
            cublasInput[j*M+i] = cudaFactorInput[i*N+j];
        }
     }

     /* Reference QR routine */
    cublasReference(cublasInput, tau, cublasOutput);

    /* CUDAFactor QR routine */
    cudaFactorQR(cudaFactorInput);

    checkCorrectness(cudaFactorInput, cublasOutput); 

    free(cublasOutput); free(cublasInput); free(cudaFactorInput);
    return 0;
}


/******************************************************************************
* checkCorrectness : Takes two output matrices to be compared for 
*                     correctness as the inputs. If the difference between the
*                     elements of these matrices is below a particular
*                     threshold correctness is passed.
******************************************************************************/
void checkCorrectness(double *src, double *cublasOutput)
{
    double diff;

    double max = 0.0f;

    for (int i=0; i<M; i++)
    {
        for(int j=0; j<N; j++)
        {
            diff =  fabs(src[i*N+j] - cublasOutput[i*N+j]);
         
            if (diff > max)
                max = diff;
        }
    }

    printf("Maximum difference is %f\n", max);
    if (max < 0.005f)
        printf("************** Correctness Passed! ***********\n");

}

/******************************************************************************
* cublasReference : Sets up inputs for calling the reference CUBLAS QR routine
*                    on the given matrix. The result/output matrix from the
*                    reference routine is transposed before it is checked for
*                    correctness.
******************************************************************************/
void cublasReference(double *src, double *tauSrc, double *cublasOutput)
{

    double *srcDevice, *tauSrcDevice;

    /* Allocate device memory for the inputs */
    cudacall(cudaMalloc<double>(&srcDevice, M * N * sizeof(double)));
    cudacall(cudaMalloc<double>(&tauSrcDevice, M * sizeof(double)));

    /* Copy the inputs to device memory */
    cudacall(cudaMemcpy(srcDevice,src, M * N * sizeof(double),
                        cudaMemcpyHostToDevice));
    cudacall(cudaMemcpy(tauSrcDevice, tauSrc, 
                        max(1, min(M,N)) * sizeof(double),
                        cudaMemcpyHostToDevice));


    /* Wrapper for the CUBLAS QR routine */
    cublasQR(srcDevice, tauSrcDevice);

    /* Copy the result back to host memory */
    cudacall(cudaMemcpy(src,srcDevice, M * N * sizeof(double),
                        cudaMemcpyDeviceToHost));

    /* Transpose the result matrix since CUDA Factor works in
        row major order whereas CUBLAS works in column major order */ 
    for(int j=0; j<N; j++)        
    {
        for(int i=0; i<M; i++)
        {
            cublasOutput[i*N + j] = src[j*M + i];
        }
    }

    /* Free device memory */
    cudacall(cudaFree(srcDevice)); cudacall(cudaFree(tauSrcDevice));
}

/******************************************************************************
* cublasQR : Wrapper for the CUBLAS QR routine (cublasDgeqrfBatched)
*            Sets up device memory to call the reference routine and measures
*            the performance in terms of wall-clock time
******************************************************************************/
void cublasQR(double* srcDevice, double *tauSrcDevice)
{
    cublasHandle_t handle;
    cublascall(cublasCreate_v2(&handle));

    int batchSize = 1;

    int info;

    double *tau[] = { tauSrcDevice };

    double *src[] = { srcDevice };
    double** srcDeviceRef;
    double** tauDeviceRef;
    cudacall(cudaMalloc<double*>(&srcDeviceRef,sizeof(src)));
    cudacall(cudaMalloc<double*>(&tauDeviceRef,sizeof(tau)));

    cudacall(cudaMemcpy(srcDeviceRef,src,sizeof(src),cudaMemcpyHostToDevice));
    cudacall(cudaMemcpy(tauDeviceRef, tau,sizeof(tau),cudaMemcpyHostToDevice));

    clock_t tic = clock();
    cublascall(cublasDgeqrfBatched(handle, M, N, srcDeviceRef, M, tauDeviceRef,
                                   &info, batchSize));
    cudacall(cudaThreadSynchronize());
    cudacall(cudaDeviceSynchronize());
    clock_t toc = clock();

    printf("\nReference time: %f seconds\n",
          ((double)(toc - tic)) / CLOCKS_PER_SEC);

    if(info < 0)
    {
        fprintf(stderr, "Error in parameters to CUBLAS\n");
        cudaDeviceReset();
        exit(EXIT_FAILURE);
    } 

    cublasDestroy_v2(handle);
}




