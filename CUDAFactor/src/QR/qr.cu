/******************************************************************************
********************** CUDA Factor Project ************************************
********************* 15-618 Spring 2015 CMU **********************************
*******************************************************************************
*
* Authors: Harshavardhan Pandit (hpandit@andrew.cmu.edu)
*          Ravi Chandra Bandlamudi (rbandlam@andrew.cmu.edu)
* 
* qr.cu - Performs a QR factorization on the given input matrix using a blocked
*		  version of the Householder transformation algorithm
*
* Reference: 
************
* Kerr, Andrew, Dan Campbell, and Mark Richards. "QR decomposition on
* GPUs" Proceedings of 2nd Workshop on General Purpose Processing on Graphics
* Processing Units. ACM, 2009
******************************************************************************/

#include <stdio.h>
#include <time.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>

#include "qr.h"

/******************************************************************************
* cudaFactorQR : Sets up device memory to perform QR factorization and 
*				 measures performance in terms of wall-clock time
******************************************************************************/
void cudaFactorQR(double *input)
{

	/* Set up inputs */
	double *inputDevice;
    cudacall(cudaMalloc((void **)&inputDevice, sizeof(double) * M * N));
    cudacall(cudaMemcpy((void *)inputDevice, (const void *) input, 
    						(M * N) * sizeof(double), cudaMemcpyHostToDevice));

    clock_t tic = clock();
    /* Perform QR factorization */
    cudaFactorQRRoutine(inputDevice);
    cudacall(cudaThreadSynchronize());
    cudacall(cudaDeviceSynchronize());
    clock_t toc = clock();
    printf("\nCUDAFactor time: %f seconds\n\n", 
    							((double)(toc - tic)) / CLOCKS_PER_SEC);

    /* Copy output to host memory */
    cudacall(cudaMemcpy((void *)input, (const void *)inputDevice, 
    						(M * N) * sizeof(double), cudaMemcpyDeviceToHost));

}

/******************************************************************************
* cudaFactorQRRoutine : Uses blocked version of the Householder 
*					  	transformation algorithm to compute the QR 
*						factorization
******************************************************************************/
void cudaFactorQRRoutine(double *input)
{

	/* CUBLAS returns a single matrix as the QR Factorization 
	   R/input is the upper triangular matrix and U is the lower triangular
	   matrix */
	double *U; 
	cudacall(cudaMalloc((void **)&U, sizeof(double) * M * N));
	cudacall(cudaMemset(U, 0, sizeof(double) * M * N));

	#ifdef DEBUG
		clock_t tic, toc;
		double timeHouseholder = 0.0f;
		double timePartialColumn = 0.0f;
		double timeLowerTriangular = 0.0f;
		double timeScalarMultipliedInput = 0.0f;
		double timeAllMatrixMultiplications = 0.0f;
		double timeUpdateInput = 0.0f;
		double timeConcatHouseholderVectors = 0.0f;
		double timeComputeYW = 0.0f;
		double timeCurrentWY = 0.0f;
		double timeExtractVnew = 0.0f;
		double timeComputezWY = 0.0f;
		double timeComputeWprime = 0.0f;
		double timePartialInput = 0.0f;
		double timeFinalInputUpdate = 0.0f;
		double timeMergeLowerUpper = 0.0f;
		double timeCudaMalloc = 0.0f;
	#endif 

	/* N: Number of columns, tileSize: Number of columns in one block */
	for (int tileIndex=0; tileIndex < N/tileSize; tileIndex++)
	{
		int tileStartColumn = tileIndex*tileSize;
		double *V, *W, *Y, *B;
		double *BHost; /* B and BHost store scalars */
		BHost = (double *)malloc(sizeof(double) * tileSize);
		memset(BHost, 0, sizeof(double) * tileSize);

		#ifdef DEBUG
			tic = clock(); 
		#endif

		cudacall(cudaMalloc((void **)&V, sizeof(double) * (M-tileIndex*tileSize)
																   * tileSize));
		cudacall(cudaMalloc((void **)&W, sizeof(double) * (M-tileIndex*tileSize)
																   * tileSize));
		cudacall(cudaMalloc((void **)&Y, sizeof(double) * (M-tileIndex*tileSize)
																   * tileSize));
		cudacall(cudaMalloc((void **)&B, sizeof(double) * tileSize));

		#ifdef DEBUG
			toc = clock();
			timeCudaMalloc += ((double)(toc - tic)) / CLOCKS_PER_SEC;
		#endif

		/* Loop within the columns inside a block */
		for (int columnInTile=0; columnInTile<tileSize; columnInTile++)
		{
			int columnIndex = tileStartColumn + columnInTile;

			double *x, *v; 
			double *vVprime, *vprime;

			#ifdef DEBUG
				tic = clock();	
			#endif

			cudacall(cudaMalloc((void **)&x, sizeof(double) * (M-columnIndex)));
			cudacall(cudaMalloc((void **)&v, sizeof(double) * (M-columnIndex)));
			cudacall(cudaMalloc((void **)&vVprime, sizeof(double) * 
										     (M-columnIndex) * (M-columnIndex)));
			cudacall(cudaMalloc((void **)&vprime, sizeof(double) *
															  (M-columnIndex)));

			#ifdef DEBUG
				toc = clock();
				timeCudaMalloc += ((double)(toc - tic)) / CLOCKS_PER_SEC;
			#endif

			dim3 blockDim(BLOCK_SIZE, 1);
        	dim3 gridDim((M - columnIndex + blockDim.x - 1) / blockDim.x, 1);
			
			#ifdef DEBUG
        		tic = clock();
        	#endif
			
			/* Choose a part x of the current column within the input matrix */
			kernelPartialColumn<<<gridDim, blockDim>>>(input, x, columnIndex);
			cudacall(cudaThreadSynchronize());
        	cudacall(cudaDeviceSynchronize());
        	
        	#ifdef DEBUG
        		toc = clock();
        		timePartialColumn += ((double)(toc - tic)) / CLOCKS_PER_SEC;
        		tic = clock();
        	#endif

    		/* Perform a householder transformation on the part of the current
    		   column - this is done on the host side, can also be done on the
    		   device using parallel exclusive scan, the bottleneck however are
    		   the matrix multiplications */
    		/* v is a householder vector */   
			double *vHost = (double *)malloc(sizeof(double) * (M-columnIndex));
			double *xHost = (double *)malloc(sizeof(double) * (M-columnIndex));
			cudacall(cudaMemcpy((void *)xHost, (const void *)x, 
							    (M-columnIndex) * sizeof(double), 
							    cudaMemcpyDeviceToHost));
			householder(xHost, vHost, M-columnIndex, BHost, columnInTile);
			cudacall(cudaMemcpy((void *)v, (const void *)vHost, 
							    (M-columnIndex) * sizeof(double), 
							    cudaMemcpyHostToDevice));
			cudacall(cudaMemcpy((void *)B, (const void *)BHost, 
								 tileSize * sizeof(double),
								 cudaMemcpyHostToDevice));

			#ifdef DEBUG
				toc = clock();
        		timeHouseholder += ((double)(toc - tic)) / CLOCKS_PER_SEC;
        		tic = clock();
        	#endif

    		/* Use output of householder to fill in the lower triangular part
    		   of the result matrix */
			kernelLowerTriangular<<<gridDim, blockDim>>>(U, v, vprime,
														 columnIndex);
			cudacall(cudaThreadSynchronize());
        	cudacall(cudaDeviceSynchronize());
        	
        	#ifdef DEBUG
	        	toc = clock();
	        	timeLowerTriangular += ((double)(toc - tic)) / CLOCKS_PER_SEC;
	        	tic = clock();
        	#endif

	        /* Using output of householder to update input takes several 
	           intermediate steps */
			double *scalarMultipliedInput, *productBetaVVprimeInput; 
			cudacall(cudaMalloc((void **)&scalarMultipliedInput, sizeof(double)*
							    (M-columnIndex) * (tileStartColumn + 
							    				   tileSize - columnIndex)));
			cudacall(cudaMalloc((void **)&productBetaVVprimeInput, 
								sizeof(double) * (M-columnIndex) * 
								(tileStartColumn+tileSize-columnIndex)));

			dim3 blockDim2D(BLOCK_SIZE, BLOCK_SIZE);
        	dim3 gridDim2D((M - columnIndex + blockDim2D.x - 1) / blockDim2D.x, 
						   (tileStartColumn+tileSize-columnIndex + 
						   						  blockDim2D.y)/ blockDim2D.y);

        	/* Scalar multiplied input */
			kernelScalarMultipliedInput<<<gridDim2D, blockDim2D>>>
									   (scalarMultipliedInput, input, B, 
									    columnInTile, columnIndex, tileStartColumn);
			cudacall(cudaThreadSynchronize());
        	cudacall(cudaDeviceSynchronize());
        	
        	#ifdef DEBUG
        		toc = clock();
        		timeScalarMultipliedInput += ((double)(toc - tic)) 
        									 / CLOCKS_PER_SEC;
        		tic = clock();
        	#endif

        	/* Perform vv' */	
			matrixMultiplyDevice(v,M-columnIndex,1, vprime, 1, M-columnIndex,
								 vVprime);
			/* productBetaVVprimeInput = vv'scalarMultipliedInput  */
			matrixMultiplyDevice(vVprime, M-columnIndex, M-columnIndex, 
								 scalarMultipliedInput, M-columnIndex, 
								 tileStartColumn+tileSize-columnIndex,
								 productBetaVVprimeInput);

			#ifdef DEBUG
				toc = clock();
        		timeAllMatrixMultiplications += ((double)(toc - tic)) 
        										/ CLOCKS_PER_SEC;
        		tic = clock();
        	#endif

        	/* input = input - productBetaVVprimeInput */	
        	/* input = input - beta*vv'*input */	
			kernelUpdateInput<<<gridDim2D, blockDim2D>>>
							 (input, productBetaVVprimeInput, columnIndex, 
						 	  tileStartColumn);
			cudacall(cudaThreadSynchronize());
        	cudacall(cudaDeviceSynchronize());
        	
        	#ifdef DEBUG
	        	toc = clock();
	        	timeUpdateInput += ((double)(toc - tic)) / CLOCKS_PER_SEC;
        	#endif

			dim3 gridDimConcat((M - tileIndex*tileSize + blockDim.x - 1) 
								/ blockDim.x, 1);

			#ifdef DEBUG
				tic = clock();
			#endif
			
			/* Store v in accumulative V variable */
			kernelConcatHouseholderVectors<<<gridDimConcat, blockDim>>>
											(v, V, columnInTile, tileStartColumn);
			cudacall(cudaThreadSynchronize());
        	cudacall(cudaDeviceSynchronize());
        	
        	#ifdef DEBUG
	        	toc = clock();
	        	timeConcatHouseholderVectors += ((double)(toc - tic)) 
	        									 / CLOCKS_PER_SEC;
        	#endif

	        /* De-allocate memory on host */	
        	free(vHost); free(xHost); 

        	#ifdef DEBUG
        		tic = clock();
        	#endif

    		/* De-allocate memory on device */
			cudacall(cudaFree(x)); cudacall(cudaFree(v)); 
			cudacall(cudaFree(vprime)); cudacall(cudaFree(vVprime));
			cudacall(cudaFree(scalarMultipliedInput)); 
			cudacall(cudaFree(productBetaVVprimeInput));

			#ifdef DEBUG
				toc = clock();
				timeCudaMalloc += ((double)(toc - tic)) / CLOCKS_PER_SEC;
			#endif

		}
		free(BHost);

		dim3 blockDim(BLOCK_SIZE, 1);
		dim3 gridDim((M - tileIndex*tileSize + blockDim.x - 1) / blockDim.x, 1);
	
		#ifdef DEBUG
			tic = clock();
		#endif

		/* Y = V(:,0) W = -B(0)*V(:,0) */
		kernelComputeYW<<<gridDim, blockDim>>>(Y, W, V, B, tileStartColumn);
		cudacall(cudaThreadSynchronize());
    	cudacall(cudaDeviceSynchronize());
    	
    	#ifdef DEBUG
    		toc = clock();
    		timeComputeYW += ((double)(toc - tic)) / CLOCKS_PER_SEC;
    	#endif

		double *vNew, *z;
		#ifdef DEBUG
			tic = clock();
		#endif
		
		cudacall(cudaMalloc((void **)&vNew, sizeof(double) * 
											(M-tileIndex*tileSize)));
		cudacall(cudaMalloc((void **)&z, sizeof(double) * 
											(M-tileIndex*tileSize)));

		#ifdef DEBUG
			toc = clock();
			timeCudaMalloc += ((double)(toc - tic)) / CLOCKS_PER_SEC;
		#endif

		for (int columnInTile=1; columnInTile<tileSize; columnInTile++)
        {
    		#ifdef DEBUG
        		tic = clock();
        	#endif

			double *currentW, *currentYPrime;
			cudacall(cudaMalloc((void **)&currentW, sizeof(double) 
								   * (M-tileIndex*tileSize) * columnInTile));
			cudacall(cudaMalloc((void **)&currentYPrime, sizeof(double)
								   * (M-tileIndex*tileSize) * columnInTile));

			dim3 blockDim2D(BLOCK_SIZE, BLOCK_SIZE);
        	dim3 gridDim2D(((M-tileIndex*tileSize)+ blockDim2D.x - 1)/blockDim2D.x, 
						   (columnInTile + blockDim2D.y)/ blockDim2D.y);
        	
        	/* Store W and Y in currentW & currentYPrime */
        	kernelCurrentWY<<<gridDim2D, blockDim2D>>>
        					 (currentW, currentYPrime, Y, W, 
				 			  columnInTile, tileStartColumn);	
        	cudacall(cudaThreadSynchronize());
    		cudacall(cudaDeviceSynchronize());
    		
    		#ifdef DEBUG
    			toc = clock();
    			timeCurrentWY += ((double)(toc - tic)) / CLOCKS_PER_SEC;
    		#endif


        	dim3 blockDim(BLOCK_SIZE, 1);
			dim3 gridDim((M - tileIndex*tileSize + blockDim.x - 1) /
													  blockDim.x, 1);

			#ifdef DEBUG
				tic = clock();
			#endif
			
			/* vNew = V(:,columnInTile) */
			kernelExtractVnew<<<gridDim, blockDim>>>(vNew, V, 
												     columnInTile, tileStartColumn);
			cudacall(cudaThreadSynchronize());
    		cudacall(cudaDeviceSynchronize());
    		
    		#ifdef DEBUG
	    		toc = clock();
	    		timeExtractVnew += ((double)(toc - tic)) / CLOCKS_PER_SEC;
				tic = clock();
			#endif

			double *productWYprime, *productWYprimeVnew;
			cudacall(cudaMalloc((void **)&productWYprime, sizeof(double) *
							    (M-tileIndex*tileSize) * (M-tileIndex*tileSize)));
			cudacall(cudaMalloc((void **)&productWYprimeVnew, sizeof(double)
							  					* (M-tileIndex*tileSize) * 1 ));

			#ifdef DEBUG
				toc = clock();
				timeCudaMalloc += ((double)(toc - tic)) / CLOCKS_PER_SEC;
				tic = clock();
			#endif

			/* productWYprime = WY' */
			matrixMultiplyDevice(currentW, M-tileIndex*tileSize, columnInTile, 
								 currentYPrime, columnInTile, 
								 M-tileIndex*tileSize, productWYprime);
			/* productWYprimeVnew = WY'vNew*/
			matrixMultiplyDevice(productWYprime, (M-tileIndex*tileSize), 
								 (M-tileIndex*tileSize), vNew, 
								 (M-tileIndex*tileSize), 1, productWYprimeVnew);

			#ifdef DEBUG
				toc = clock();
	    		timeAllMatrixMultiplications += ((double)(toc - tic))
	    										 / CLOCKS_PER_SEC;
	    		tic = clock();
    		#endif

	    	/* z = -B(columnInTile)vNew - B(columnInTile)*WY'*vNew
    		   W = [W z]
	    	   Y = [Y vNew] */	
			kernelComputezWY<<<gridDim, blockDim>>>(z, W, Y, vNew, B, 
								productWYprimeVnew, columnInTile, tileStartColumn);
			cudacall(cudaThreadSynchronize());
    		cudacall(cudaDeviceSynchronize());
    		
    		#ifdef DEBUG
    			toc = clock();
    			timeComputezWY += ((double)(toc - tic)) / CLOCKS_PER_SEC;
    			tic = clock();
    		#endif
			
			cudacall(cudaFree(currentW)); cudacall(cudaFree(currentYPrime)); 
			cudacall(cudaFree(productWYprime)); 
			cudacall(cudaFree(productWYprimeVnew));

			#ifdef DEBUG
				toc = clock();
				timeCudaMalloc += ((double)(toc - tic)) / CLOCKS_PER_SEC;
			#endif
		}

		#ifdef DEBUG
			tic = clock();
		#endif

		cudacall(cudaFree(vNew)); cudacall(cudaFree(z));
		
		#ifdef DEBUG
			toc = clock();
			timeCudaMalloc += ((double)(toc - tic)) / CLOCKS_PER_SEC;
		#endif

		double *Wprime, *productYWprime;
		#ifdef DEBUG
			tic = clock();
		#endif
		cudacall(cudaMalloc((void **)&Wprime, sizeof(double) * 
									 		(M-tileIndex*tileSize) * tileSize));
		cudacall(cudaMalloc((void **)&productYWprime, sizeof(double) * 
							(M-tileIndex*tileSize) * (M-tileIndex*tileSize)));

		dim3 blockDim2D(BLOCK_SIZE, BLOCK_SIZE);
    	dim3 gridDim2D(( (M-tileIndex*tileSize)+ blockDim2D.x - 1) / blockDim2D.x, 
										  (tileSize + blockDim2D.y)/ blockDim2D.y);

    	kernelComputeWprime<<<gridDim2D, blockDim2D>>>(W, Wprime, tileStartColumn);
    	cudacall(cudaThreadSynchronize());
    	cudacall(cudaDeviceSynchronize());

    	#ifdef DEBUG
	    	toc = clock();
			timeComputeWprime += ((double)(toc - tic)) / CLOCKS_PER_SEC;
			tic = clock();
		#endif
    	
    	/* YW' */
    	matrixMultiplyDevice(Y, M-tileIndex*tileSize, tileSize, Wprime, 
    						 tileSize, M-tileIndex*tileSize, productYWprime);	
    	
    	#ifdef DEBUG
	    	toc = clock();
			timeAllMatrixMultiplications += ((double)(toc - tic))/CLOCKS_PER_SEC;	
			tic = clock();
		#endif

    	double *partialInput, *productYWprimePartialInput;
    	cudacall(cudaMalloc((void **)&partialInput, sizeof(double) * 
    						(M-tileIndex*tileSize) * 
    						(N - tileStartColumn - tileSize) ));
    	cudacall(cudaMalloc((void **)&productYWprimePartialInput, sizeof(double)
    						 * (M-tileIndex*tileSize) * 
    						 (N - tileStartColumn - tileSize) ));

    	dim3 blockDimInput(BLOCK_SIZE, BLOCK_SIZE);
    	dim3 gridDimInput(( (M-tileIndex*tileSize)+ blockDimInput.x - 1) 
    						/ blockDimInput.x, 
						   ((N - tileStartColumn - tileSize) + 
						   	blockDimInput.y)/ blockDimInput.y);

    	/* A part of the input be used to update input matrix */ 
    	kernelPartialInput<<<gridDimInput, blockDimInput>>>
    						(partialInput, input, tileStartColumn);		
    	cudacall(cudaThreadSynchronize());
    	cudacall(cudaDeviceSynchronize());
    	#ifdef DEBUG
	    	toc = clock();
			timePartialInput += ((double)(toc - tic)) / CLOCKS_PER_SEC;	
	    	tic = clock();
    	#endif

	    /* productYWprimePartialInput = YW'partialInput */	
    	matrixMultiplyDevice(productYWprime, (M-tileIndex*tileSize), 
    						(M-tileIndex*tileSize), partialInput, 
    						(M-tileIndex*tileSize), (N-tileStartColumn-tileSize),
    						 productYWprimePartialInput);

    	#ifdef DEBUG
	    	toc = clock();
			timeAllMatrixMultiplications += ((double)(toc - tic))/CLOCKS_PER_SEC;	
		#endif

		#ifdef DEBUG
			tic = clock();
		#endif

		/* Update input matrix input = input + productYWprimePartialInput */
		kernelFinalInputUpdate<<<gridDimInput, blockDimInput>>>
								(productYWprimePartialInput, input,
								 tileStartColumn);		    	
		cudacall(cudaThreadSynchronize());
    	cudacall(cudaDeviceSynchronize());
    	
    	#ifdef DEBUG
    		toc = clock();
			timeFinalInputUpdate += ((double)(toc - tic)) / CLOCKS_PER_SEC;	
			tic = clock();
		#endif

		cudacall(cudaFree(Wprime));  cudacall(cudaFree(productYWprime)); 
		cudacall(cudaFree(V)); cudacall(cudaFree(W)); 
		cudacall(cudaFree(Y)); cudacall(cudaFree(B));
		cudacall(cudaFree(productYWprimePartialInput));
		 cudacall(cudaFree(partialInput));

		#ifdef DEBUG
			toc = clock();
			timeCudaMalloc += ((double)(toc - tic)) / CLOCKS_PER_SEC;
		#endif
	}

	dim3 blockDimFinal(BLOCK_SIZE, BLOCK_SIZE);
	dim3 gridDimFinal(( M+ blockDimFinal.x - 1) / blockDimFinal.x, 
									(N + blockDimFinal.y)/ blockDimFinal.y);

	#ifdef DEBUG
		tic = clock();
	#endif

	/* Combine upper and lower triangular matrices to get final result matrix */
	kernelMergeLowerUpper<<<gridDimFinal, blockDimFinal>>>(input, U);
	cudacall(cudaThreadSynchronize());
	cudacall(cudaDeviceSynchronize());

	#ifdef DEBUG
		toc = clock();
		timeMergeLowerUpper += ((double)(toc - tic)) / CLOCKS_PER_SEC;	
	#endif

	cudacall(cudaFree(U));

	/* Print timing information if DEBUG on */
	#ifdef DEBUG
		printf("\n\n************** Timing Summary **************** \n\n");
		printf("Module\t\tTime taken\n");
		printf("House\t\t%lf\n", timeHouseholder);
		printf("compute x\t%lf\n", timePartialColumn);
		printf("compute U\t%lf\n", timeLowerTriangular);
		printf("computeRtmp\t%lf\n", timeScalarMultipliedInput);
		printf("modifyR\t\t%lf\n", timeUpdateInput);
		printf("productWYprime  %lf\n", timeConcatHouseholderVectors);
		printf("YW\t\t%lf\n", timeComputeYW);
		printf("curr\t\t%lf\n", timeCurrentWY);
		printf("vnew\t\t%lf\n", timeExtractVnew);
		printf("zWY\t\t%lf\n", timeComputezWY);
		printf("W prime\t\t%lf\n", timeComputeWprime);
		printf("Rtmp1\t\t%lf\n", timePartialInput);
		printf("Radd\t\t%lf\n", timeFinalInputUpdate);
		printf("Merge RU\t%lf\n", timeMergeLowerUpper);
		printf("Matrix mults\t%lf\n", timeAllMatrixMultiplications);
		printf("Memory ops\t%lf\n", timeCudaMalloc);
	#endif
}

/******************************************************************************
* computeNorm : computes the l2 norm of a vector
******************************************************************************/
double computeNorm(double *x, int lengthX)
{

    double norm = 0.0f;

    for (int i=0; i < lengthX; i++)
    {
        norm += x[i]*x[i];
    }

    return(sqrt(norm));

}

/******************************************************************************
* householder : performs the householder transformation on given input vector
******************************************************************************/
void householder(double *x, double *v, int lengthX, double *B, int columnInTile)
{
	double beta;

	if (lengthX == 1)
	{
		v[0] = 1;
		beta = 0.0f;
	}
	else
	{
	    double *e = (double *)malloc(sizeof(double) * lengthX);
	    memset(e, 0, sizeof(double) * lengthX);
	    e[0] = 1;

	    double norm = computeNorm(x, lengthX);

	    double sign = 0.0f;

	    if (x[0] > 0)
	        sign = 1.0f;

	    if (x[0] < 0)
	        sign = -1.0f;

	    for (int i=0; i < lengthX; i++)
	    {
	        v[i] = x[i] + e[i] * norm * sign;
	    }

	    double normV = computeNorm(v, lengthX);
	    double scalar = normV * normV;

	    beta = (double) 2/scalar;
	}

    B[columnInTile] = beta;

}

/******************************************************************************
* matrixMultiplyDevice : Wrapper for the matrix multiplication kernel
******************************************************************************/
void matrixMultiplyDevice(double *a, int rowsA, int colsA, double *b, 
											int rowsB, int colsB, double *c)
{

	dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
  	
  	dim3 gridDim((colsB + blockDim.x - 1) / blockDim.x, 
  										(rowsA + blockDim.y - 1) / blockDim.y);

  	kernelSharedMemMatMult<<<gridDim,blockDim>>>(a, rowsA, colsA,
  														 b, rowsB, colsB, c);
  	cudacall(cudaThreadSynchronize());
	cudacall(cudaDeviceSynchronize());

}