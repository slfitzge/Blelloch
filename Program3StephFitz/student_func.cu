/* Udacity Homework 3
   HDR Tone-mapping
  Background HDR
  ==============
  A High Definition Range (HDR) image contains a wider variation of intensity
  and color than is allowed by the RGB format with 1 byte per channel that we
  have used in the previous assignment.  
  To store this extra information we use single precision floating point for
  each channel.  This allows for an extremely wide range of intensity values.
  In the image for this assignment, the inside of church with light coming in
  through stained glass windows, the raw input floating point values for the
  channels range from 0 to 275.  But the mean is .41 and 98% of the values are
  less than 3!  This means that certain areas (the windows) are extremely bright
  compared to everywhere else.  If we linearly map this [0-275] range into the
  [0-255] range that we have been using then most values will be mapped to zero!
  The only thing we will be able to see are the very brightest areas - the
  windows - everything else will appear pitch black.
  The problem is that although we have cameras capable of recording the wide
  range of intensity that exists in the real world our monitors are not capable
  of displaying them.  Our eyes are also quite capable of observing a much wider
  range of intensities than our image formats / monitors are capable of
  displaying.
  Tone-mapping is a process that transforms the intensities in the image so that
  the brightest values aren't nearly so far away from the mean.  That way when
  we transform the values into [0-255] we can actually see the entire image.
  There are many ways to perform this process and it is as much an art as a
  science - there is no single "right" answer.  In this homework we will
  implement one possible technique.
  Background Chrominance-Luminance
  ================================
  The RGB space that we have been using to represent images can be thought of as
  one possible set of axes spanning a three dimensional space of color.  We
  sometimes choose other axes to represent this space because they make certain
  operations more convenient.
  Another possible way of representing a color image is to separate the color
  information (chromaticity) from the brightness information.  There are
  multiple different methods for doing this - a common one during the analog
  television days was known as Chrominance-Luminance or YUV.
  We choose to represent the image in this way so that we can remap only the
  intensity channel and then recombine the new intensity values with the color
  information to form the final image.
  Old TV signals used to be transmitted in this way so that black & white
  televisions could display the luminance channel while color televisions would
  display all three of the channels.
  
  Tone-mapping
  ============
  In this assignment we are going to transform the luminance channel (actually
  the log of the luminance, but this is unimportant for the parts of the
  algorithm that you will be implementing) by compressing its range to [0, 1].
  To do this we need the cumulative distribution of the luminance values.
  Example
  -------
  input : [2 4 3 3 1 7 4 5 7 0 9 4 3 2]
  min / max / range: 0 / 9 / 9
  histo with 3 bins: [4 7 3]
  cdf : [4 11 14]
  Your task is to calculate this cumulative distribution by following these
  steps.
*/


#include "reference_calc.cpp"
#include "utils.h"
#include "cuda_runtime.h"


#define BLOCKSIZEMAX 16	
#define BLOCKSIZEMAXHISTOGRAM 22				
#define BLOCKSIZEMAXSCAN 512	
				
//Modified simple histo from notes
__global__ void simple_histo(unsigned int *d_bins, const float *d_In, const unsigned int binCount, float _min, float _range, int numRows, int numCols)
{
    int myId = ((blockIdx.x + (blockIdx.y * gridDim.x)) * (blockDim.x * blockDim.y)) + (threadIdx.x + (threadIdx.y * blockDim.x));

	if ( myId < (numRows * numCols) )
	{
		float myItem = d_In[myId];
		unsigned int myBin = min(static_cast<unsigned int>(binCount - 1), static_cast<unsigned int>((myItem - _min) / _range * binCount));
		atomicAdd(&(d_bins[myBin]), 1);
	}
	else{
		return;
	}
}

//Found original in discussion forum...  modified for a .2 speed increase by reducing logic sequences
__global__ void globalMinMax(float *d_Out, const float *d_In, int numRows, int numCols, bool firstTime)
{
	__shared__ float sharedValue[2 * BLOCKSIZEMAX * BLOCKSIZEMAX];

     int myId = ((blockIdx.x + (blockIdx.y * gridDim.x)) * (blockDim.x * blockDim.y)) + (threadIdx.x + (threadIdx.y * blockDim.x));

	if ( myId < (numRows * numCols) )
	 {
		if (!firstTime)
		{
			sharedValue[(threadIdx.y * blockDim.x + threadIdx.x)] = d_In[myId];
			sharedValue[(threadIdx.y * blockDim.x + threadIdx.x) + (blockDim.x * blockDim.y)] = d_In[myId + (numRows * numCols)];
		}
		else
		{	
			sharedValue[(threadIdx.y * blockDim.x + threadIdx.x)] = d_In[myId];
			sharedValue[(threadIdx.y * blockDim.x + threadIdx.x) + (blockDim.x * blockDim.y)] = d_In[myId];
		}

		for (unsigned int s = (blockDim.x * blockDim.y) / 2; s > 0; s >>= 1)
		{
			if ((threadIdx.y * blockDim.x + threadIdx.x) < s && (myId + s) < (numRows * numCols))
			{
				sharedValue[(threadIdx.y * blockDim.x + threadIdx.x)] = min(sharedValue[(threadIdx.y * blockDim.x + threadIdx.x)], sharedValue[(threadIdx.y * blockDim.x + threadIdx.x) + s]);
				sharedValue[(threadIdx.y * blockDim.x + threadIdx.x) + (blockDim.x * blockDim.y)] = max(sharedValue[(threadIdx.y * blockDim.x + threadIdx.x) + (blockDim.x * blockDim.y)], sharedValue[(threadIdx.y * blockDim.x + threadIdx.x) + (blockDim.x * blockDim.y) + s]);
			}
			__syncthreads();      
		}
		if ((threadIdx.y * blockDim.x + threadIdx.x) == 0)
		{
			d_Out[myId / (blockDim.x * blockDim.y)] = d_In[myId];
			d_Out[myId / (blockDim.x * blockDim.y)] = sharedValue[(threadIdx.y * blockDim.x + threadIdx.x)];
			if (gridDim.x <= 1 || gridDim.y <= 1)
			{
				d_Out[(myId / (blockDim.x * blockDim.y)) + 1] = d_In[myId + (numRows * numCols)];
				d_Out[(myId / (blockDim.x * blockDim.y)) + 1] = sharedValue[(threadIdx.y * blockDim.x + threadIdx.x) + (blockDim.x * blockDim.y)];

			}
			else
			{
				d_Out[(myId / (blockDim.x * blockDim.y)) + (blockDim.x * blockDim.y)] = d_In[myId + (numRows * numCols)];
				d_Out[(myId / (blockDim.x * blockDim.y)) + (blockDim.x * blockDim.y)] = sharedValue[(threadIdx.y * blockDim.x + threadIdx.x) + (blockDim.x * blockDim.y)];

			}
		}
	}
	else{
		return;
	}
	
}

//https://github.com/steffenmartin/CUDA/blob/master/Scan/kernel.cu  found cool exclusive  Blelloch Scan
//was curious how would react after class discussion
//Suprisingly fast.... next step compare to inclusivew
// this really help me understadn Blelloch
//modified logic for .05 increase in speed.... 
__global__ void blellochScanExclusive(const unsigned int *d_In, unsigned int *d_Out, size_t size, size_t offset, bool isLastCall)
{
	__shared__ unsigned int CurrentBoundaryValue;
	__shared__ unsigned int finalSum;
	unsigned int remember;
	size_t remainingSteps = size;
	unsigned int neighbor = 1;
	unsigned int addTurn = 1;
	__shared__ unsigned int sharedValue[BLOCKSIZEMAXSCAN];

	if (threadIdx.x == 0)
	{
		CurrentBoundaryValue = 0;
		remember = d_In[offset + size - 1];

		if (offset > 0)
		{
			finalSum = d_Out[0] + d_Out[offset - 1];
		}
	}
	if (threadIdx.x < size)
	{
		// Initial data fetch
		sharedValue[threadIdx.x] = d_In[threadIdx.x + offset];

		__syncthreads();

		// Step 1: Adding neighbors
		while (remainingSteps)
		{
			if ((addTurn & threadIdx.x) == addTurn)
			{
				sharedValue[threadIdx.x] += sharedValue[threadIdx.x - neighbor];
			}

			remainingSteps >>= 1;
			neighbor <<= 1;
			addTurn <<= 1;
			addTurn++;

			__syncthreads();
		}

		// Step 2: Down-sweep and adding neighbors again
		
		addTurn--;
		addTurn >>= 1;
		neighbor >>= 1;
		remainingSteps = size;

		while (remainingSteps)
		{
			bool fillBoundary= true;

			if ((addTurn & threadIdx.x) == addTurn)
			{
				unsigned int tempValue = sharedValue[threadIdx.x];
				sharedValue[threadIdx.x] += sharedValue[threadIdx.x - neighbor];
				sharedValue[threadIdx.x - neighbor] = tempValue;
				fillBoundary= false;
			}

			__syncthreads();

			unsigned int crossSweep = addTurn >> 1;

			if (fillBoundary&&((addTurn & threadIdx.x) ^ crossSweep) == 0 && (threadIdx.x + neighbor) >= size)
			{
				sharedValue[threadIdx.x] = CurrentBoundaryValue;
				CurrentBoundaryValue = CurrentBoundaryValue + sharedValue[(threadIdx.x)];

			}
			addTurn--;
			addTurn >>= 1;
			neighbor >>= 1;
			remainingSteps >>= 1;

			__syncthreads();
		}

		if (offset > 0){sharedValue[threadIdx.x] += finalSum;}

		__syncthreads();

		d_Out[threadIdx.x + offset] = sharedValue[threadIdx.x];

		if (threadIdx.x == 0 && !isLastCall)
		{
			d_Out[0] = remember;
		}
		else
		{
			d_Out[0] = 0;
		}
		__syncthreads();
	}
}

void your_histogram_and_prefixsum(const float* const d_logLuminance,
                                  unsigned int* const d_cdf,
                                  float &min_logLum,
                                  float &max_logLum,
                                  const size_t numRows,
                                  const size_t numCols,
                                  const size_t numBins)
{
//TODO
  /*Here are the steps you need to implement
    1) find the minimum and maximum value in the input logLuminance channel
       store in min_logLum and max_logLum
    2) subtract them to find the range
    3) generate a histogram of all the values in the logLuminance channel using
       the formula: bin = (lum[i] - lumMin) / lumRange * numBins
    4) Perform an exclusive scan (prefix sum) on the histogram to get
       the cumulative distribution of luminance values (this should go in the
       incoming d_cdf pointer which already has been allocated for you)       */

	int gridSizeX = (numCols - 1) / BLOCKSIZEMAX + 1;
	int gridSizeY = (numRows - 1) / BLOCKSIZEMAX + 1;
	dim3 blockSize(BLOCKSIZEMAX, BLOCKSIZEMAX, 1);
	dim3 gridSize(gridSizeX, gridSizeY, 1);
	int numBinsLeft = numBins;
	float h_MinMaxOut[2];

	float *d_center;
	
	checkCudaErrors(cudaMalloc(&d_center, max((unsigned int)(2 * sizeof(float) * gridSizeX * gridSizeY), (unsigned int)(sizeof(unsigned int) * numBins))));
	checkCudaErrors(cudaMemset(d_center,0x0, 2 * sizeof(float) * gridSizeX * gridSizeY));

	globalMinMax<<<gridSize, blockSize>>>(d_center, d_logLuminance, numRows, numCols, true);
	globalMinMax<<<1, blockSize>>>(d_center, d_center, gridSizeX, gridSizeY, false);

	checkCudaErrors(cudaMemcpy(&h_MinMaxOut[0], d_center, 2 * sizeof(float),cudaMemcpyDeviceToHost));

	min_logLum = h_MinMaxOut[0];
	max_logLum = h_MinMaxOut[1];
	float lumRange = max_logLum - min_logLum;

	unsigned int *d_Bins = reinterpret_cast<unsigned int *>(d_center);
	checkCudaErrors(cudaMemset(d_Bins, 0x0, sizeof(unsigned int) * numBins));

	blockSize.x = BLOCKSIZEMAXHISTOGRAM;
	blockSize.y = BLOCKSIZEMAXHISTOGRAM;
	gridSize.x = (numCols - 1) / BLOCKSIZEMAXHISTOGRAM + 1;
	gridSize.y = (numRows - 1) / BLOCKSIZEMAXHISTOGRAM + 1;

	simple_histo<<<gridSize, blockSize>>>(d_Bins, d_logLuminance, numBins, h_MinMaxOut[0], lumRange, numRows, numCols);

	while (numBinsLeft)
	{
		blockSize.x = numBinsLeft > BLOCKSIZEMAXSCAN ? BLOCKSIZEMAXSCAN : numBinsLeft;
		blockSize.y = 1;
		
		gridSize.x = 1;
		gridSize.y = 1;

		blellochScanExclusive<<<gridSize, blockSize>>>(d_Bins, d_cdf, blockSize.x, numBins - numBinsLeft, (numBinsLeft - blockSize.x) <= 0);

		numBinsLeft -= blockSize.x;
	}

}