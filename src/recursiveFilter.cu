


#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cinttypes>

#include "misc/APRTimer.hpp"
#include "misc/CudaTools.cuh"
#include "recursiveFilter.h"


constexpr int blockWidth = 32;
constexpr int numOfThreads = 32;
extern __shared__ char sharedMemProcess[];
template<typename T>
__global__ void bsplineYdirProcess(T *image, const size_t x_num, const size_t y_num, const size_t z_num,
                                   const float b1, const float b2, const float norm_factor) {
    const int numOfWorkers = blockDim.x;
    const int currentWorkerId = threadIdx.x;
    const int xzOffset = blockIdx.x * blockDim.x;
    const int64_t maxXZoffset = x_num * z_num;
    const int64_t workersOffset = xzOffset * y_num;

    T (*cache)[blockWidth + 0] = (T (*)[blockWidth + 0]) &sharedMemProcess[0];

    float temp1 = 0, temp2 = 0;

    // ---------------- forward direction -------------------------------------------
    for (int yBlockBegin = 0; yBlockBegin < y_num - 2; yBlockBegin += blockWidth) {

        // Read from global mem to cache
        for (int i = currentWorkerId; i < blockWidth * numOfWorkers; i += numOfWorkers) {
            int offs = i % blockWidth;
            int work = i / blockWidth;
            if (offs + yBlockBegin < y_num && work + xzOffset < maxXZoffset) {
                cache[work][(offs + work)%blockWidth] = image[workersOffset + y_num * work + offs + yBlockBegin];
            }
        }
        __syncthreads();

        // Do operations
        if (xzOffset + currentWorkerId < maxXZoffset) {
            for (size_t k = 0; k < blockWidth && k + yBlockBegin < y_num; ++k) {
                float  temp = temp1*b2 + temp2*b1 + cache[currentWorkerId][(k + currentWorkerId)%blockWidth];
                cache[currentWorkerId][(k + currentWorkerId)%blockWidth] = temp;
                temp1 = temp2;
                temp2 = temp;
            }
        }
        __syncthreads();

        // Write from cache to global mem
        for (int i = currentWorkerId; i < blockWidth * numOfWorkers; i += numOfWorkers) {
            int offs = i % blockWidth;
            int work = i / blockWidth;
            if (offs + yBlockBegin < y_num && work + xzOffset < maxXZoffset) {
                image[workersOffset + y_num * work + offs + yBlockBegin] = cache[work][(offs + work)%blockWidth];
            }
        }
        __syncthreads();
    }

    // ---------------- backward direction -------------------------------------------
    temp1 = 0;
    temp2 = 0;
    for (int yBlockBegin = y_num - 1; yBlockBegin >= 0; yBlockBegin -= blockWidth) {

        // Read from global mem to cache
        for (int i = currentWorkerId; i < blockWidth * numOfWorkers; i += numOfWorkers) {
            int offs = i % blockWidth;
            int work = i / blockWidth;
            if (yBlockBegin - offs >= 0 && work + xzOffset < maxXZoffset) {
                cache[work][(offs + work)%blockWidth] = image[workersOffset + y_num * work - offs + yBlockBegin];
            }
        }
        __syncthreads();

        // Do operations
        if (xzOffset + currentWorkerId < maxXZoffset) {
            for (int64_t k = 0; k < blockWidth && yBlockBegin - k >= 0; ++k) {
                float  temp = temp2*b1 + temp1*b2 + cache[currentWorkerId][(k + currentWorkerId)%blockWidth];
                cache[currentWorkerId][(k + currentWorkerId)%blockWidth] = temp * norm_factor;
                temp1 = temp2;
                temp2 = temp;
            }
        }
        __syncthreads();

        // Write from cache to global mem
        for (int i = currentWorkerId; i < blockWidth * numOfWorkers; i += numOfWorkers) {
            int offs = i % blockWidth;
            int work = i / blockWidth;
            if (yBlockBegin - offs >= 0 && work + xzOffset < maxXZoffset) {
                image[workersOffset + y_num * work - offs + yBlockBegin] = cache[work][(offs + work)%blockWidth];
            }
        }
        __syncthreads();
    }
}

/**
 * Function for launching a kernel
 */
template <typename T>
void runBsplineYdir(T *cudaImage, size_t x_num, size_t y_num, size_t z_num,
                    float b1, float b2, float norm_factor,  cudaStream_t aStream) {
    dim3 threadsPerBlock(numOfThreads);
    dim3 numBlocks((x_num * z_num + threadsPerBlock.x - 1) / threadsPerBlock.x);
    size_t sharedMemSize = numOfThreads * blockWidth * sizeof(T);
    bsplineYdirProcess<T> <<< numBlocks, threadsPerBlock, sharedMemSize, aStream >>> (cudaImage, x_num, y_num, z_num, b1, b2, norm_factor);
}

/**
 * Runs bspline recursive filter in X direction. Each processed 2D patch consist of number of workes
 * (distributed in Y direction) and each of them is handling the whole row in X-dir.
 * Next patches are build on a top of first (like patch1 in example below) and they cover
 * whole y-dimension. Such a setup should be run for every plane in z-direction.
 *
 * Example block/threadblock calculation:
 *     constexpr int numOfWorkersY = 64;
 *     dim3 threadsPerBlock(1, numOfWorkersY, 1);
 *     dim3 numBlocks(1,
 *                    (input.y_num + threadsPerBlock.y - 1)/threadsPerBlock.y,
 *                    (input.z_num + threadsPerBlock.z - 1)/threadsPerBlock.z);
 *
 * Image memory setup is [z][x][y]
 *
 *     y_num
 *           XXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
 *           XX                            X
 *    y      X X                            X
 *           X  X                            X
 *    d      X   X                            X
 *    i      X    X                            X
 *    r ^    X     XXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
 *    e |    X     X                            X
 *    c |    X     X  ...                       X
 *    t      X     +----------------------------+                   X
 *    i      X     |                            |
 *    o      X     | ----->                     |
 *    n      X     | patch 1                    |
 *           X     |                            |
 *      z_num X    +----------------------------+
 *         ^   X   |                            |
 *          \   X  | ----->                     |
 *   z       \   X | patch 0                    |
 *   direction    X|                            |
 *                 +----------------------------+
 *                 0                              x_num
 *                          X direction ->
 *
 * @tparam T - input image type
 * @param image - device pointer to image
 * @param x_num - dimension len in x-direction
 * @param y_num - dimension len in y-direction
 * @param k0 - filter len
 * @param b1 - filter coefficient
 * @param b2 - filter coefficient
 * @param norm_factor - filter norm factor
 */
template<typename T>
__global__ void bsplineXdir(T *image, size_t x_num, size_t y_num,
                            float b1, float b2, float norm_factor) {

    const int yDirOffset = blockIdx.y * blockDim.y + threadIdx.y;
    const size_t zDirOffset = (blockIdx.z * blockDim.z + threadIdx.z) * x_num * y_num;
    const size_t nextElementXdirOffset = y_num;
    const size_t dirLen = x_num;

    if (yDirOffset < y_num) {
        float temp1 = 0;
        float temp2 = 0;
        float temp3 = 0;
        float temp4 = 0;

        // Causal Filter loop
        int64_t offset = zDirOffset + yDirOffset;
        int64_t offsetLimit = zDirOffset + dirLen * nextElementXdirOffset;
        while (offset < offsetLimit) {
            __syncthreads(); // only needed for speed imporovement (memory coalescing)
            const float temp = temp1 * b2 + temp2 * b1 + image[offset];
            image[offset] = temp;
            temp1 = temp2;
            temp2 = temp;

            offset += nextElementXdirOffset;
        }

        // Anti-Causal Filter loop
        offset = zDirOffset + (dirLen - 1) * nextElementXdirOffset + yDirOffset;
        offsetLimit = zDirOffset;
        while (offset >= offsetLimit) {
            __syncthreads(); // only needed for speed imporovement (memory coalescing)
            const float temp = temp3 * b1 + temp4 * b2 + image[offset];
            image[offset] = temp * norm_factor;
            temp4 = temp3;
            temp3 = temp;

            offset -= nextElementXdirOffset;
        }
    }
}

/**
 * Function for launching a kernel
 */
template<typename T>
void runBsplineXdir(T *cudaImage, size_t x_num, size_t y_num, size_t z_num,
                    float b1, float b2, float norm_factor, cudaStream_t aStream) {
    constexpr int numOfWorkersYdir = 128;
    dim3 threadsPerBlockX(1, numOfWorkersYdir, 1);
    dim3 numBlocksX(1,
                    (y_num + threadsPerBlockX.y - 1) / threadsPerBlockX.y,
                    (z_num + threadsPerBlockX.z - 1) / threadsPerBlockX.z);
    bsplineXdir<T> <<<numBlocksX, threadsPerBlockX, 0, aStream>>> (cudaImage, x_num, y_num, b1, b2, norm_factor);
}



// explicit instantiation of handled types
template void filterZeroBoundary(PixelData<float> &, TypeOfRecRecursiveFlags);
template <typename ImgType>
void filterZeroBoundary(PixelData<ImgType> &input, TypeOfRecRecursiveFlags flags) {
    cudaStream_t  aStream = 0;

    ScopedCudaMemHandler<PixelData<ImgType>, D2H | H2D> cudaInput(input);

    APRTimer timer(true);
    timer.start_timer("GpuDeviceTimeFull");
    if (flags & RECURSIVE_Y_DIR) {
        runBsplineYdir(cudaInput.get(), input.x_num, input.y_num, input.z_num, -0.5,0.5, 1, aStream);
    }
    if (flags & RECURSIVE_X_DIR) {
        runBsplineXdir(cudaInput.get(), input.x_num, input.y_num, input.z_num, -0.5,0.5, 1, aStream);
    }
    timer.stop_timer();
}