#ifndef _QR_H_
#define _QR_H_

#define M 2048  /* Rows in the input matrix */
#define N 2048  /* Columns in the input matrix */
#define tileSize 32    /* Size of the tile */

#define BLOCK_SIZE (tileSize)

#define min(a,b) ((a) < (b) ? (a) : (b))
#define max(a,b) ((a) < (b) ? (a) : (b))

#define cudacall(call)                                                                                                          \
    do                                                                                                                          \
    {                                                                                                                           \
        cudaError_t err = (call);                                                                                               \
        if(cudaSuccess != err)                                                                                                  \
        {                                                                                                                       \
            fprintf(stderr,"CUDA Error:\nFile = %s\nLine = %d\nReason = %s\n", __FILE__, __LINE__, cudaGetErrorString(err));    \
            cudaDeviceReset();                                                                                                  \
            exit(EXIT_FAILURE);                                                                                                 \
        }                                                                                                                       \
    }                                                                                                                           \
    while (0)

#define cublascall(call)                                                                                        \
    do                                                                                                          \
    {                                                                                                           \
        cublasStatus_t status = (call);                                                                         \
        if(CUBLAS_STATUS_SUCCESS != status)                                                                     \
        {                                                                                                       \
            fprintf(stderr,"CUBLAS Error:\nFile = %s\nLine = %d\nCode = %d\n", __FILE__, __LINE__, status);     \
            cudaDeviceReset();                                                                                  \
            exit(EXIT_FAILURE);                                                                                 \
        }                                                                                                       \
                                                                                                                \
    }                                                                                                           \
    while(0)

/* CUDA Factor QR top-level functions */
void cudaFactorQR(double *input);
void cudaFactorQRRoutine(double *input);

/* Correctness and Performance reference functions */
void checkCorrectness(double *src, double *cublasOutput);
void cublasQR(double* srcDevice, double *tauSrcDevice);
void cublasReference(double *src, double *tauSrc, double *cublasOutput);

/* Helper host functions */
void householder(double *x, double *v, int lengthX, double *B, int columnInTile);
double computeNorm(double *x, int lengthX);

/* Helper device functions */
void matrixMultiplyDevice(double *a, int rowsA, int colsA, double *b, 
                           int rowsB, int colsB, double *c);

/* Helper kernels for QR factorization */
__global__ void kernelLowerTriangular(double *U, double *v, double *vprime, 
                                      int columnIndex);
__global__ void kernelPartialColumn(double *input, double *x, int columnIndex);
__global__ void kernelScalarMultipliedInput(double *scalarMultipliedInput, 
                                            double *input, double *B, 
                                            int columnInTile, int columnIndex, 
                                            int tileStartColumn);
__global__ void kernelUpdateInput(double *input, double *productBetaVVprimeInput, 
                                 int columnIndex, int tileStartColumn);
__global__ void kernelConcatHouseholderVectors(double *v, double *V, 
                                               int columnInTile, 
                                               int tileStartColumn);
__global__ void kernelSharedMemMatMult(double *A, int rowsA, int colsA, 
                                       double *B, int rowsB,  int colsB, 
                                       double *C);
__global__ void kernelMatMult(double *a, int rowsA, int colsA,
                              double *b, int rowsB, int colsB, double *c);
__global__ void kernelComputeYW(double *Y, double *W, double *V, double *B,
                                int tileStartColumn);
__global__ void kernelCurrentWY(double *currentW, double *currentYPrime, 
                                double *Y, double *W, 
                                int columnInTile, int tileStartColumn);
__global__ void kernelExtractVnew(double *vNew, double *V, 
                                  int columnInTile, int tileStartColumn);
__global__ void kernelComputezWY(double *z, double *W, double *Y, double *vNew, 
                                 double *B, double *productWYprimeVnew, 
                                 int columnInTile, int tileStartColumn);
__global__ void kernelComputeWprime(double *W, double *Wprime,
                                    int tileStartColumn);
__global__ void kernelPartialInput(double *partialInput, double *input,
                                   int tileStartColumn);
__global__ void kernelFinalInputUpdate(double *productYWprimePartialInput,
                                       double *input, int tileStartColumn);
__global__ void kernelMergeLowerUpper(double *input, double *U);




#endif
