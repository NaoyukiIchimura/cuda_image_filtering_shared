////
//// cuda_image_filtering_constant.cu: an example program for image filtering using CUDA, shared memory version
////

///
/// The standard include files
///
#include <iostream>
#include <string>

#include <cmath>

///
/// The include files for CUDA
///
#include <cuda_runtime.h>
#include <helper_functions.h>
#include <helper_cuda.h>

///
/// The include files for image filtering
///
#include "path_handler.h"
#include "image_rw_cuda.h"
#include "padding.h"
#include "postprocessing.h"
#include "get_micro_second.h"

///
/// An inline function of the ceilling function for unsigned int variables
///
inline unsigned int iDivUp( const unsigned int &a, const unsigned int &b ) { return ( a%b != 0 ) ? (a/b+1):(a/b); }

///
/// The declaration of the constant memory space for filter coefficients
///
const unsigned int MAX_FILTER_SIZE = 90;
__device__ __constant__ float d_cFilterKernel[ MAX_FILTER_SIZE * MAX_FILTER_SIZE ];

///
/// The kernel function for image filtering using constant and shared memory
/// Note that passing references cannot be used.
///
template <typename T>
__global__ void imageFilteringKernel( const T *d_f, const unsigned int paddedW, const unsigned int paddedH,
				      const unsigned int blockW, const unsigned int blockH, const int S,
				      T *d_h, const unsigned int W, const unsigned int H )
{

    //
    // Note that blockDim.(x,y) cannot be used instead of blockW and blockH,
    // because the size of a thread block is not equal to the size of a data block
    // due to the apron and the use of subblocks.
    //
    
    //
    // Set the size of a tile
    //
    const unsigned int tileW = blockW + 2 * S;
    const unsigned int tileH = blockH + 2 * S;

    // 
    // Set the number of subblocks in a tile
    //
    const unsigned int noSubBlocks = static_cast<unsigned int>(ceil( static_cast<double>(tileH)/static_cast<double>(blockDim.y) ));

    //
    // Set the start position of the block, which is determined by blockIdx. 
    // Note that since padding is applied to the input image, the origin of the block is ( S, S )
    //
    const unsigned int blockStartCol = blockIdx.x * blockW + S;
    const unsigned int blockEndCol = blockStartCol + blockW;

    const unsigned int blockStartRow = blockIdx.y * blockH + S;
    const unsigned int blockEndRow = blockStartRow + blockH;

    //
    // Set the position of the tile which includes the data block and its apron
    //
    const unsigned int tileStartCol = blockStartCol - S;
    const unsigned int tileEndCol = blockEndCol + S;
    const unsigned int tileEndClampedCol = min( tileEndCol, paddedW );

    const unsigned int tileStartRow = blockStartRow - S;
    const unsigned int tileEndRow = blockEndRow + S;
    const unsigned int tileEndClampedRow = min( tileEndRow, paddedH );

    //
    // Set the size of the filter kernel
    //
    const unsigned int kernelSize = 2 * S + 1;

    //
    // Shared memory for the tile
    //
    extern __shared__ T sData[];

    //
    // Copy the tile into shared memory
    //
    unsigned int tilePixelPosCol = threadIdx.x;
    unsigned int iPixelPosCol = tileStartCol + tilePixelPosCol;
    for( unsigned int subBlockNo = 0; subBlockNo < noSubBlocks; subBlockNo++ ) {

	unsigned int tilePixelPosRow = threadIdx.y + subBlockNo * blockDim.y;
	unsigned int iPixelPosRow = tileStartRow + tilePixelPosRow;

	if( iPixelPosCol < tileEndClampedCol && iPixelPosRow < tileEndClampedRow ) { // Check if the pixel in the image
	    unsigned int iPixelPos = iPixelPosRow * paddedW + iPixelPosCol;
	    unsigned int tilePixelPos = tilePixelPosRow * tileW + tilePixelPosCol;
	    sData[tilePixelPos] = d_f[iPixelPos];
	}
	
    }

    //
    // Wait for all the threads for data loading
    //
    __syncthreads();

    //
    // Perform convolution
    //
    tilePixelPosCol = threadIdx.x;
    iPixelPosCol = tileStartCol + tilePixelPosCol;
    for( unsigned int subBlockNo = 0; subBlockNo < noSubBlocks; subBlockNo++ ) {

	unsigned int tilePixelPosRow = threadIdx.y + subBlockNo * blockDim.y;
	unsigned int iPixelPosRow = tileStartRow + tilePixelPosRow;

	// Check if the pixel in the tile and image.
	// Note that the apron of the tile is excluded.
	if( iPixelPosCol >= tileStartCol + S && iPixelPosCol < tileEndClampedCol - S &&
	    iPixelPosRow >= tileStartRow + S && iPixelPosRow < tileEndClampedRow - S ) {

	    // Compute the pixel position for the output image
	    unsigned int oPixelPosCol = iPixelPosCol - S; // removing the origin
	    unsigned int oPixelPosRow = iPixelPosRow - S;
	    unsigned int oPixelPos = oPixelPosRow * W + oPixelPosCol;

	    unsigned int tilePixelPos = tilePixelPosRow * tileW + tilePixelPosCol;

	    d_h[oPixelPos] = 0.0;
	    for( int i = -S; i <= S; i++ ) {
		for( int j = -S; j <= S; j++ ) {
		    int tilePixelPosOffset = i * tileW + j;
		    int coefPos = ( i + S ) * kernelSize + ( j + S );
		    d_h[oPixelPos] += sData[ tilePixelPos + tilePixelPosOffset ] * d_cFilterKernel[coefPos];
		}
	    }

	}
	
    }

}

///
/// The function for image filtering performed on a CPU
///
template <typename T>
int imageFiltering( const T *h_f, const unsigned int &paddedW, const unsigned int &paddedH,
                    const T *h_g, const int &S,
                    T *h_h, const unsigned int &W, const unsigned int &H )
{

    // Set the padding size and filter size
    unsigned int paddingSize = S;
    unsigned int filterSize = 2 * S + 1;

    // The loops for the pixel coordinates
    for( unsigned int i = paddingSize; i < paddedH - paddingSize; i++ ) {
        for( unsigned int j = paddingSize; j < paddedW - paddingSize; j++ ) {

            // The multiply-add operation for the pixel coordinate ( j, i )
            unsigned int oPixelPos = ( i - paddingSize ) * W + ( j - paddingSize );
            h_h[oPixelPos] = 0.0;
            for( int k = -S; k <=S; k++ ) {
                for( int l = -S; l <= S; l++ ) {
                    unsigned int iPixelPos = ( i + k ) * paddedW + ( j + l );
                    unsigned int coefPos = ( k + S ) * filterSize + ( l + S );
                    h_h[oPixelPos] += h_f[iPixelPos] * h_g[coefPos];
                }
            }

        }
    }

    return 0;

}       

///
/// Comopute the mean squared error between two images
///
template <typename T>
T calMSE( const T *image1, const T *image2, const unsigned int &iWidth, const unsigned int &iHeight )
{

    T mse = 0.0;
    for( unsigned int i = 0; i < iHeight; i++ ) {
	for( unsigned int j = 0; j < iWidth; j++ ) {
	    unsigned int pixelPos = i * iWidth + j;
	    mse += ( image1[pixelPos] - image2[pixelPos] ) * ( image1[pixelPos] - image2[pixelPos] );
	}
    }
    mse = sqrt( mse );

    return mse;

}

///
/// The main function
///
int main( int argc, char *argv[] )
{

    //----------------------------------------------------------------------

    //
    // Declare the variables for measuring elapsed time
    //
    double sTime;
    double eTime;

    //----------------------------------------------------------------------

    //
    // Input file paths
    //
    std::string inputImageFilePath;
    std::string filterDataFilePath;
    std::string outputImageFilePrefix;
    if( argc <= 1 ) {
	std::cerr << "Input image file path: ";
	std::cin >> inputImageFilePath;
	std::cerr << "Filter data file path: ";
	std::cin >> filterDataFilePath;
	std::cerr << "Output image file path: ";
	std::cin >> outputImageFilePrefix;
    } else if( argc <= 2 ) {
	inputImageFilePath = argv[1];
	std::cerr << "Filter data file path: ";
	std::cin >> filterDataFilePath;
	std::cerr << "Output image file path: ";
	std::cin >> outputImageFilePrefix;
    } else if( argc <= 3 ) {
	inputImageFilePath = argv[1];
	filterDataFilePath = argv[2];
	std::cerr << "Output image file path: ";
	std::cin >> outputImageFilePrefix;
    } else {
	inputImageFilePath = argv[1];
	filterDataFilePath = argv[2];
	outputImageFilePrefix = argv[3];
    }

    //----------------------------------------------------------------------

    //
    // Set the prefix and extension of the input image file
    //
    std::string imageFileDir;
    std::string imageFileName;
    getDirFileName( inputImageFilePath, &imageFileDir, &imageFileName );

    std::string imageFilePrefix;
    std::string imageFileExt;
    getPrefixExtension( imageFileName, &imageFilePrefix, &imageFileExt );

    //----------------------------------------------------------------------

    //
    // Read the intput image in pageable memory on a host
    // Page-locked memory (write-combining memory) is not used, because padding is performed on a host 
    //
    hsaImage<float> h_inputImage;
    if( imageFileExt == "tif" ) { // TIFF
      h_inputImage.tiffGetImageSize( inputImageFilePath );
      h_inputImage.allocImage( PAGEABLE_MEMORY );
      h_inputImage.tiffReadImage( inputImageFilePath );
    } else if( imageFileExt == "jpg" ) { // JPEG
      h_inputImage.jpegGetImageSize( inputImageFilePath );
      h_inputImage.allocImage( PAGEABLE_MEMORY );
      h_inputImage.jpegReadImage( inputImageFilePath );
    } else if( imageFileExt == "png" ) { // PNG
      h_inputImage.pngGetImageSize( inputImageFilePath );
      h_inputImage.allocImage( PAGEABLE_MEMORY );
      h_inputImage.pngReadImage( inputImageFilePath );
    }

    //
    // Show the size of the input image
    //
    std::cout << "The size of the input image: ("
	      << h_inputImage.getImageWidth()
	      << ", "
	      << h_inputImage.getImageHeight() 
	      << ")"
	      << std::endl;

    //----------------------------------------------------------------------

    //
    // Prepare an image for example
    //
    float *h_image;
    unsigned int iWidth = h_inputImage.getImageWidth();
    unsigned int iHeight = h_inputImage.getImageHeight();
    try {
	h_image = new float[ iWidth * iHeight ];
    } catch( std::bad_alloc & ) {
	std::cerr << "Could not allocate the memory space for h_image: "
		  << __FILE__ << " : " << __LINE__
		  << std::endl;
	exit(1);
    }

    // Compute Y component
    for( unsigned int i = 0; i < h_inputImage.getImageHeight(); i++ ) {
	for( unsigned int j = 0; j < h_inputImage.getImageWidth(); j++ ) {
	    unsigned int pixelPos = i * h_inputImage.getImageWidth() + j;
	    h_image[pixelPos] = 0.2126 * h_inputImage.getImagePtr( 0 )[pixelPos] +
		0.7152 * h_inputImage.getImagePtr( 1 )[pixelPos] +
		0.0722 * h_inputImage.getImagePtr( 2 )[pixelPos];
	}
    }

    //----------------------------------------------------------------------

    //
    // Read the filter data file
    //
    std::ifstream fin;
    fin.open( filterDataFilePath.c_str() );
    if( !fin ) {
	std::cerr << "Could not open the filter data file: "
		  << filterDataFilePath
		  << std::endl;
	exit(1);
    }

    // Read the size of the filter
    unsigned int filterSize;
    fin >> filterSize;

    // Check the filter size
    if( filterSize > MAX_FILTER_SIZE ) {
	std::cerr << "The filter size exceeds the maximum: "
		  << filterSize
		  << " > " << MAX_FILTER_SIZE
		  << std::endl;
	exit(1);
    }

    // Read the filter kernel
    float *h_filterKernel;
    try {
	h_filterKernel = new float[ filterSize * filterSize ];
    } catch( std::bad_alloc & ) {
	std::cerr << "Could not allocate the memory space for h_filterKernel: "
		  << __FILE__ << " : " << __LINE__
		  << std::endl;
	exit(1);
    }
    for( unsigned int i = 0; i < filterSize; i++ )
	for( unsigned int j = 0; j < filterSize; j++ )
	    fin >> h_filterKernel[ i * filterSize + j ];

    std::cout << "*** Filter coefficients ***" << std::endl;  
    for( unsigned int i = 0; i < filterSize; i++ ) {
	for( unsigned int j = 0; j < filterSize; j++ )
	    std::cout << h_filterKernel[ i * filterSize + j ] << " ";
	std::cout << std::endl;
    }

    //----------------------------------------------------------------------

    //
    // Check if the filter coefficients can be transfered to constant memory
    //
    if( filterSize > MAX_FILTER_SIZE ) {
	std::cerr << "The filter size is too large to transfer them into constant memory: "
		  << filterSize << std::endl;
	exit(1);
    }

    //----------------------------------------------------------------------

    //
    // Perform padding for the image
    //
    int hFilterSize = filterSize / 2;
    unsigned int paddedIWidth = iWidth + 2 * hFilterSize;
    unsigned int paddedIHeight = iHeight + 2 * hFilterSize;
    float *h_paddedImage;
    try {
	h_paddedImage = new float[ paddedIWidth * paddedIHeight ];
    } catch( std::bad_alloc & ) {
	std::cerr << "Could not allocate the memory space for h_paddedImage: "
		  << __FILE__ << " : " << __LINE__
		  << std::endl;
	exit(1);
    }

    replicationPadding( h_image, iWidth, iHeight,
			hFilterSize,
			h_paddedImage, paddedIWidth, paddedIHeight );
    
    //----------------------------------------------------------------------

    //
    // Perform image filtering by a GPU 
    //

    // Transfer the padded image to a device 
    float *d_paddedImage;
    unsigned int paddedImageSizeByte = paddedIWidth * paddedIHeight * sizeof(float);
    checkCudaErrors( cudaMalloc( reinterpret_cast<void **>(&d_paddedImage), paddedImageSizeByte ) );
    sTime = getMicroSecond();
    checkCudaErrors( cudaMemcpy( d_paddedImage, h_paddedImage, paddedImageSizeByte, cudaMemcpyHostToDevice ) );
    eTime = getMicroSecond();
    double dataTransferTime = eTime - sTime;

    // Transfer the filter to a device
    sTime = getMicroSecond();
    unsigned int filterKernelSizeByte = filterSize * filterSize * sizeof(float);
    checkCudaErrors( cudaMemcpyToSymbol( d_cFilterKernel, h_filterKernel, filterKernelSizeByte, 0, cudaMemcpyHostToDevice ) );
    eTime = getMicroSecond();
    dataTransferTime += ( eTime - sTime );

    // Set the execution configuration
    const unsigned int blockW = 32;
    const unsigned int blockH = 32;
    const unsigned int tileW = blockW + 2 * hFilterSize;
    const unsigned int tileH = blockH + 2 * hFilterSize;
    const unsigned int threadBlockH = 8;
    const dim3 grid( iDivUp( iWidth, blockW ), iDivUp( iHeight, blockH ) );
    const dim3 threadBlock( tileW, threadBlockH );
    
    // Set the size of shared memory
    const unsigned int sharedMemorySizeByte = tileW * tileH * sizeof(float);

    // call the kernel function for image filtering
    float *d_filteringResult;
    const unsigned int imageSizeByte = iWidth * iHeight * sizeof(float);
    checkCudaErrors( cudaMalloc( reinterpret_cast<void **>(&d_filteringResult), imageSizeByte ) );
    
    sTime = getMicroSecond();
    checkCudaErrors( cudaDeviceSynchronize() );
    imageFilteringKernel<<<grid,threadBlock,sharedMemorySizeByte>>>( d_paddedImage, paddedIWidth, paddedIHeight,
								     blockW, blockH, hFilterSize,
								     d_filteringResult, iWidth, iHeight );
    checkCudaErrors( cudaDeviceSynchronize() );
    eTime = getMicroSecond();
    double filteringTimeGPU = eTime - sTime;

    // Back-transfer the filtering result to a host
    float *h_filteringResultGPU;
    try {
	h_filteringResultGPU =new float[ iWidth * iHeight ];
    } catch( std::bad_alloc & ) {
	std::cerr << "Could not allocate the memory space for h_filteringResultGPU: "
		  << __FILE__ << " : " << __LINE__
		  << std::endl;
	exit(1);
    }

    sTime = getMicroSecond();
    checkCudaErrors( cudaMemcpy( h_filteringResultGPU, d_filteringResult, imageSizeByte, cudaMemcpyDeviceToHost ) );
    eTime = getMicroSecond();
    dataTransferTime += ( eTime - sTime );

    //----------------------------------------------------------------------

    //
    // Perform image filtering by a CPU
    //
    float *h_filteringResultCPU;
    try {
	h_filteringResultCPU =new float[ iWidth * iHeight ];
    } catch( std::bad_alloc & ) {
	std::cerr << "Could not allocate the memory space for h_filteringResultCPU: "
		  << __FILE__ << " : " << __LINE__
		  << std::endl;
	exit(1);
    }

    sTime = getMicroSecond();
    imageFiltering( h_paddedImage, paddedIWidth, paddedIHeight,
		    h_filterKernel, hFilterSize,
		    h_filteringResultCPU, iWidth, iHeight );
    eTime = getMicroSecond();
    double filteringTimeCPU = eTime - sTime;

    //----------------------------------------------------------------------

    //
    // Compare the filtering results by a GPU and a CPU
    //
    float mse = calMSE( h_filteringResultGPU, h_filteringResultCPU, iWidth, iHeight );

    std::cout << "MSE: " << mse << std::endl;

    //----------------------------------------------------------------------

    //
    // Show the compution time
    //
    std::cout << "The time for data transfer: " << dataTransferTime * 1e3 << "[ms]" <<std::endl;
    std::cout << "The time for filtering by GPU: " << filteringTimeGPU * 1e3 << "[ms]" << std::endl;
    std::cout << "The time for filtering by CPU: " << filteringTimeCPU * 1e3 << "[ms]" << std::endl;
    std::cout << "Filering: the GPU is " << filteringTimeCPU / filteringTimeGPU << "X faster than the CPU." << std::endl;
    std::cout << "The overall speed-up is " << filteringTimeCPU / ( dataTransferTime + filteringTimeGPU ) << "X." << std::endl;

    //----------------------------------------------------------------------

    //
    // Save the fitlering results
    //
    hsaImage<float> filteringResultImage;
    filteringResultImage.allocImage( iWidth, iHeight, PAGEABLE_MEMORY );

    // Set the number of channels
    const unsigned int RGB = 3;

    // The GPU result
    for( unsigned int i = 0; i < iHeight; i++ ) {
	for( unsigned int j = 0; j < iWidth; j++ ) {
	    unsigned int pixelPos = i * iWidth + j;
	    for( unsigned int k = 0; k < RGB; k++ )
		filteringResultImage.getImagePtr( k )[pixelPos] = h_filteringResultGPU[pixelPos];
	}
    }
    takeImageAbsoluteValueCPU( &filteringResultImage, RGB );
    normalizeImageCPU( &filteringResultImage, RGB );
    adjustImageLevelCPU( &filteringResultImage, RGB, static_cast<float>(255) );
    std::string filteringResultGPUFileName = outputImageFilePrefix + "_GPU.png";
    filteringResultImage.pngSaveImage( filteringResultGPUFileName, RGB_DATA );

    // The CPU result
    for( unsigned int i = 0; i < iHeight; i++ ) {
	for( unsigned int j = 0; j < iWidth; j++ ) {
	    unsigned int pixelPos = i * iWidth + j;
	    for( unsigned int k = 0; k < RGB; k++ )
		filteringResultImage.getImagePtr( k )[pixelPos] = h_filteringResultCPU[pixelPos];
	}
    }
    takeImageAbsoluteValueCPU( &filteringResultImage, RGB );
    normalizeImageCPU( &filteringResultImage, RGB );
    adjustImageLevelCPU( &filteringResultImage, RGB, static_cast<float>(255) );
    std::string filteringResultCPUFileName = outputImageFilePrefix + "_CPU.png";
    filteringResultImage.pngSaveImage( filteringResultCPUFileName, RGB_DATA );

    //----------------------------------------------------------------------

    //
    // Delete the memory spaces
    //
    filteringResultImage.freeImage();

    delete [] h_filteringResultCPU;
    h_filteringResultCPU = 0;

    delete [] h_filteringResultGPU;
    h_filteringResultGPU = 0;

    checkCudaErrors( cudaFree( d_filteringResult ) );
    d_filteringResult = 0;

    checkCudaErrors( cudaFree( d_paddedImage ) );
    d_paddedImage = 0;

    delete [] h_paddedImage;
    h_paddedImage = 0;

    delete [] h_image;
    h_image = 0;

    delete [] h_filterKernel;
    h_filterKernel = 0;

    h_inputImage.freeImage();

    //----------------------------------------------------------------------

    return 0;
    
}
