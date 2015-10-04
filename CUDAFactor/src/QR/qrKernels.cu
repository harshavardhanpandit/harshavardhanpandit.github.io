/******************************************************************************
********************** CUDA Factor Project ************************************
********************* 15-618 Spring 2015 CMU **********************************
*******************************************************************************
*
* Authors: Harshavardhan Pandit (hpandit@andrew.cmu.edu)
*          Ravi Chandra Bandlamudi (rbandlam@andrew.cmu.edu)
* 
* qr_kernels.cu - Kernel calls used in qr.cu for QR Factorization
******************************************************************************/

#include <stdio.h>
#include <time.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>

#include "qr.h"

/******************************************************************************
* kernelLowerTriangular : 
******************************************************************************/
__global__ void kernelLowerTriangular(double *U, double *v, double *vprime, 
                                      int columnIndex)
{

    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= M-columnIndex)
        return;

    U[(index+columnIndex)*N + columnIndex] = v[index]/v[0];
    vprime[index] = v[index];

}

/******************************************************************************
* kernelPartialColumn : Choose a part of a single column within the input matrix
******************************************************************************/
__global__ void kernelPartialColumn(double *input, double *x, int columnIndex)
{

    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= M-columnIndex)
        return;

    x[index] = input[(columnIndex+index)*N + columnIndex];  

}

/******************************************************************************
* kernelScalarMultipliedInput : 
******************************************************************************/
__global__ void kernelScalarMultipliedInput(double *scalarMultipliedInput, 
                                            double *input, double *B, 
                                            int columnInTile, int columnIndex, 
                                            int tileStartColumn)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    if (row >= (M-columnIndex) || col >= (tileStartColumn+tileSize-columnIndex))
        return;

    scalarMultipliedInput[row*(tileStartColumn+tileSize-columnIndex) + col] = 
               B[columnInTile]*input[(row+columnIndex)*(N) + columnIndex + col]; 

}

/******************************************************************************
* kernelUpdateInput : 
******************************************************************************/
__global__ void kernelUpdateInput(double *input, double *productBetaVVprimeInput, 
                                 int columnIndex, int tileStartColumn)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    if (row >= (M-columnIndex) || col >= (tileStartColumn+tileSize-columnIndex))
        return;

    input[(row+columnIndex)*(N) + columnIndex + col] = 
                            input[(row+columnIndex)*(N) + columnIndex + col] - 
                            productBetaVVprimeInput[row *
                                (tileStartColumn+tileSize-columnIndex) + col];  

}

/******************************************************************************
* kernelConcatHouseholderVectors : Store vector v in tmp variable 
******************************************************************************/
__global__ void kernelConcatHouseholderVectors(double *v, double *V, 
                                               int columnInTile, 
                                               int tileStartColumn)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= M-tileStartColumn)
        return;

    if (index<columnInTile)
        V[index*tileSize + columnInTile] = 0.0f;
    else
        V[index*tileSize + columnInTile] = v[index-columnInTile];

}

/******************************************************************************
* kernelSharedMemMatMult : 
******************************************************************************/
__global__ void kernelSharedMemMatMult(double *A, int rowsA, int colsA, 
                                       double *B, int rowsB,  int colsB, 
                                       double *C)
{
    double tmp = 0.0;
    __shared__ double M_shared[BLOCK_SIZE][BLOCK_SIZE] ;
    __shared__ double N_shared[BLOCK_SIZE][BLOCK_SIZE] ;

    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    for(int m = 0; m < (BLOCK_SIZE + colsA - 1)/BLOCK_SIZE; m++) 
    {
      if(m * BLOCK_SIZE + threadIdx.x < colsA && row < rowsA) 
      {
        M_shared[threadIdx.y][threadIdx.x] =  
                                A[row * colsA + m * BLOCK_SIZE + threadIdx.x];
      }
      else 
      {
        M_shared[threadIdx.y][threadIdx.x] = 0.0;
      }

      if(m * BLOCK_SIZE + threadIdx.y < rowsB && col < colsB) 
      {
        N_shared[threadIdx.y][threadIdx.x] =  
                              B[(m * BLOCK_SIZE + threadIdx.y) * colsB + col];
      } 
      else 
      {
        N_shared[threadIdx.y][threadIdx.x] = 0.0;
      }
      __syncthreads();


      for(int tileIndex = 0; tileIndex < BLOCK_SIZE; tileIndex++) 
      {
        tmp += M_shared[threadIdx.y][tileIndex] * N_shared[tileIndex][threadIdx.x];
      }
      __syncthreads();
    }

    if(row < rowsA && col < colsB) 
    {
        C[((blockIdx.y * blockDim.y + threadIdx.y) * colsB) + 
                                (blockIdx.x * blockDim.x) + threadIdx.x] = tmp;
    }

}

/******************************************************************************
* kernelMatMult : 
******************************************************************************/
__global__ void kernelMatMult(double *a, int rowsA, int colsA,
                              double *b, int rowsB, int colsB, double *c)
{
  
  int col = blockIdx.y * blockDim.y + threadIdx.y;
  int row = blockIdx.x * blockDim.x + threadIdx.x;

  if (row >= rowsA || col >= colsB)
    return;

  float sum = 0.0f;
  for (int i = 0; i < colsA; i++)
  {
    sum += a[row * colsA + i] * b[i * colsB + col]; 
  }

  c[row * colsB + col] = sum;

}

/******************************************************************************
* kernelComputeYW : 
******************************************************************************/
__global__ void kernelComputeYW(double *Y, double *W, double *V, double *B,
                                int tileStartColumn)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= M-tileStartColumn)
        return;

    Y[index*tileSize+0] = V[index*tileSize+0];
    W[index*tileSize+0] = -B[0]*V[index*tileSize+0];

}

/******************************************************************************
* kernelCurrentWY : 
******************************************************************************/
__global__ void kernelCurrentWY(double *currentW, double *currentYPrime, 
                                double *Y, double *W, 
                                int columnInTile, int tileStartColumn)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    if (row >= (M-tileStartColumn) || col >= columnInTile )
        return;

    // Y has to be transposed
    currentYPrime[row+ col*(M-tileStartColumn)] =  Y[row*tileSize+col]; 
    currentW[row*columnInTile+col] = W[row*tileSize+col];

}

/******************************************************************************
* kernelExtractVnew : 
******************************************************************************/
__global__ void kernelExtractVnew(double *vNew, double *V, 
                                  int columnInTile, int tileStartColumn)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= M-tileStartColumn)
        return;

    vNew[index] = V[index*tileSize+columnInTile];

}

/******************************************************************************
* kernelComputezWY : 
******************************************************************************/
__global__ void kernelComputezWY(double *z, double *W, double *Y, double *vNew, 
                                 double *B, double *productWYprimeVnew, 
                                 int columnInTile, int tileStartColumn)
{

    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= M-tileStartColumn)
        return;

    z[row] = -B[columnInTile] * vNew[row] - B[columnInTile] * 
                                            productWYprimeVnew[row];
    W[row*tileSize+columnInTile] = z[row];
    Y[row*tileSize+columnInTile] = vNew[row];
}

/******************************************************************************
* kernelComputeWprime : 
******************************************************************************/
__global__ void kernelComputeWprime(double *W, double *Wprime,
                                    int tileStartColumn)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int columnInTile = blockIdx.y * blockDim.y + threadIdx.y;       

    if (row >= (M-tileStartColumn) || columnInTile>= tileSize)
        return;

    Wprime[columnInTile*(M-tileStartColumn) + row] = W[row*tileSize + 
                                                            columnInTile];

}


/******************************************************************************
* kernelPartialInput : 
******************************************************************************/
__global__ void kernelPartialInput(double *partialInput, double *input,
                                   int tileStartColumn)
{

    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int columnInTile = blockIdx.y * blockDim.y + threadIdx.y;       

    if (row >= (M-tileStartColumn) || 
        columnInTile>= ((N - tileStartColumn - tileSize)) )
        return;

    partialInput[row*(N-tileStartColumn-tileSize) + columnInTile] = 
                                        input[(row+tileStartColumn) * N + 
                                                columnInTile + tileStartColumn 
                                                + tileSize];
}


/******************************************************************************
* kernelFinalInputUpdate : 
******************************************************************************/
__global__ void kernelFinalInputUpdate(double *productYWprimePartialInput,
                                       double *input, int tileStartColumn)
{

    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int columnInTile = blockIdx.y * blockDim.y + threadIdx.y;       

    if (row >= (M-tileStartColumn) || 
        columnInTile>= ((N - tileStartColumn - tileSize)) )
        return;

    input[(row+tileStartColumn)*N + columnInTile + tileStartColumn + tileSize] += 
    productYWprimePartialInput[row*(N-tileStartColumn-tileSize) + columnInTile];
}

/******************************************************************************
* kernelMergeLowerUpper : Combine upper and lower triangular matrices to get 
                   final result matrix
******************************************************************************/
__global__ void kernelMergeLowerUpper(double *input, double *U)
{
    int columnInTile = blockIdx.y * blockDim.y + threadIdx.y;       
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= M || columnInTile>= N || row <= columnInTile)
        return;

    input[row*N + columnInTile] = U[row*N + columnInTile];

}
