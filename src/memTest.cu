#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <chrono>
#include <memory>
#include <vector>

template<typename T>
__global__ void memCopy1dKernel(T *in, T *out, size_t len) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < len) {
        out[idx] = in[idx];
    }
}

template<typename T>
__global__ void memCopy2dA(const T *in, T *out, size_t xLen, size_t yLen) {
    size_t xi = blockIdx.x * blockDim.x + threadIdx.x;
    size_t yi = blockIdx.y * blockDim.y + threadIdx.y;
    if (xi < xLen && yi < yLen) {
        out[yi * xLen + xi] = in[yi * xLen + xi];
    }
}

template<typename T>
__global__ void memCopy2dB(const T *in, T *out, size_t xLen, size_t yLen) {
    size_t xi = blockIdx.x * blockDim.x + threadIdx.x;
    if (xi < xLen) {
        for (size_t idx = xi; idx < yLen * xLen; idx += xLen) {
            __syncthreads(); // don't need sychronization but it gives super speedup!
            out[idx] = in[idx];
        }
    }
}

template<typename T>
__global__ void memCopy2dBnotSynchronized(const T *in, T *out, size_t xLen, size_t yLen) {
    size_t xi = blockIdx.x * blockDim.x + threadIdx.x;
    if (xi < xLen) {
        for (size_t idx = xi; idx < yLen * xLen; idx += xLen) {
            out[idx] = in[idx];
        }
    }
}

static void waitForCuda() {
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) printf("Error: %s\n", cudaGetErrorString(err));
}

class ScopedBandwidth {
    const std::string name;
    const std::chrono::high_resolution_clock::time_point startTime;
    const size_t dataSize;
    const int numOfRepetitions;
    static constexpr double gigabyte = 1000 * 1000 * 1000;

public:
    ScopedBandwidth(const std::string &name, size_t dataSize, size_t numOfRepetitions) : name(name), startTime(std::chrono::high_resolution_clock::now()), dataSize(dataSize), numOfRepetitions(numOfRepetitions) {}
    ~ScopedBandwidth() {
        auto stopTime = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = stopTime - startTime;
        std::cout << name << " GB/s: " << (2 * dataSize * numOfRepetitions) / elapsed.count() / gigabyte << std::endl;
    }
};


template <typename T>
void warmUp(int numOfRepetitions, int numOfThreads, size_t xLen, size_t yLen, T *dInput, T *dOutput) {
    const size_t numOfElements = xLen * yLen;

    dim3 threadsPerBlock(numOfThreads);
    dim3 numBlocks((numOfElements + threadsPerBlock.x - 1) / threadsPerBlock.x);

    {
        for (int i = 0; i < numOfRepetitions; ++i) {
            memCopy1dKernel << < numBlocks, threadsPerBlock >> > (dInput, dOutput, numOfElements);
        }
        waitForCuda();
    }
}

template <typename T>
void test1D(int numOfRepetitions, int numOfThreads, size_t xLen, size_t yLen, T *dInput, T *dOutput) {
    const size_t numOfElements = xLen * yLen;
    const size_t dataSize = numOfElements * sizeof(T);

    dim3 threadsPerBlock(numOfThreads);
    dim3 numBlocks((numOfElements + threadsPerBlock.x - 1) / threadsPerBlock.x);

    {
        ScopedBandwidth sb("test1D", dataSize, numOfRepetitions);
        for (int i = 0; i < numOfRepetitions; ++i) {
            memCopy1dKernel << < numBlocks, threadsPerBlock >> > (dInput, dOutput, numOfElements);
        }
        waitForCuda();
    }
}

template <typename T>
void test2DA(int numOfRepetitions, int numOfThreads, size_t xLen, size_t yLen, T *dInput, T *dOutput) {
    const size_t numOfElements = xLen * yLen;
    const size_t dataSize = numOfElements * sizeof(T);

    dim3 threadsPerBlock(numOfThreads, 1);
    dim3 numBlocks((xLen + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (yLen + threadsPerBlock.y - 1) / threadsPerBlock.y);

    {
        ScopedBandwidth sb("test2DA", dataSize, numOfRepetitions);
        for (int i = 0; i < numOfRepetitions; ++i) {
            memCopy2dA <<< numBlocks, threadsPerBlock >>> (dInput, dOutput, xLen, yLen);
        }
        waitForCuda();
    }
}

template <typename T>
void test2DB(int numOfRepetitions, int numOfThreads, size_t xLen, size_t yLen, T *dInput, T *dOutput) {
    const size_t numOfElements = xLen * yLen;
    const size_t dataSize = numOfElements * sizeof(T);

    dim3 threadsPerBlock(numOfThreads);
    dim3 numBlocks((xLen + threadsPerBlock.x - 1) / threadsPerBlock.x);

    {
        ScopedBandwidth sb("test2DB", dataSize, numOfRepetitions);
        for (int i = 0; i < numOfRepetitions; ++i) {
            memCopy2dB <<< numBlocks, threadsPerBlock >>> (dInput, dOutput, xLen, yLen);
        }
        waitForCuda();
    }
}

template <typename T>
void test2DBnotSynchronized(int numOfRepetitions, int numOfThreads, size_t xLen, size_t yLen, T *dInput, T *dOutput) {
    const size_t numOfElements = xLen * yLen;
    const size_t dataSize = numOfElements * sizeof(T);

    dim3 threadsPerBlock(numOfThreads);
    dim3 numBlocks((xLen + threadsPerBlock.x - 1) / threadsPerBlock.x);

    {
        ScopedBandwidth sb("test2DBnotSynchronized", dataSize, numOfRepetitions);
        for (int i = 0; i < numOfRepetitions; ++i) {
            memCopy2dBnotSynchronized <<< numBlocks, threadsPerBlock >>> (dInput, dOutput, xLen, yLen);
        }
        waitForCuda();
    }
}

template <typename T>
void setMemory(T *dInput, T *dOutput, size_t dataSize) {
    cudaMemset(dInput, 0, dataSize);
    cudaMemset(dOutput, 11, dataSize);
}

template <typename T>
void checkMemory(T *dOutput, size_t numOfElements) {
    std::unique_ptr<T[]> hMemory(new T[numOfElements]);
    cudaMemcpy(hMemory.get(), dOutput, numOfElements * sizeof(T), cudaMemcpyDeviceToHost);
    size_t cnt = 0;
    for (size_t i = 0; i < numOfElements; ++i) {
        if (hMemory[i] != 0) { // check if memory was zeroed
            cnt++;
        }
    }

    if (cnt > 0) {
        std::cout << "Memory not copied properly! Found " << numOfElements << "/" << cnt << " errors." << std::endl;
        exit(1);
    }
}

#include "deviceQuery.cuh"


int main() {

    std::cout << getDeviceInfo() << std::endl;

    int deviceId = -1;
    cudaGetDevice(&deviceId);
    std::cout  << "Device ID = " << deviceId << std::endl;
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, deviceId);
    std::cout << "Shared memory per SM = " << deviceProp.sharedMemPerMultiprocessor << std::endl;
    const int smCount = deviceProp.multiProcessorCount;
    std::cout << "Name [" << deviceProp.name << "]\n";
    std::cout << "Number of SMs = " << smCount << std::endl;
    std::cout << "Size of global mem = " << static_cast<float>(deviceProp.totalGlobalMem / 1048576.0f) << "MB" << std::endl;

    using T=float;
    const size_t yLen = 1024;
    // calculate coefficient to get for both buffers about half of memory in GPU
    const size_t halfMemCoef = deviceProp.totalGlobalMem / (yLen * 32 * 64 * smCount * sizeof(T)) / 2 / 2;
    // max efficiency at #SM * 32 (max num of active blocks) * 64 (maximum num of active threads 2048/32 active blocks)
    const size_t xLen = smCount * 32 * 64 * halfMemCoef;
    const size_t numOfElements = xLen * yLen;
    const size_t dataSize = numOfElements * sizeof(T);
    std::cout << "Used DataSize = " << dataSize / 1024 / 1024 << "MB\n";

    T *dInput;
    cudaMalloc(&dInput, dataSize);
    T *dOutput;
    cudaMalloc(&dOutput, dataSize);

    const int numOfRepetitions = 50;

    warmUp(numOfRepetitions, 64, xLen, yLen, dInput, dOutput);

    std::vector<int> numThreadsPool = {32, 64, 128, 256};

    for (int numOfThreads : numThreadsPool) {
        std::cout << "-------- numOfThreads in block = " << numOfThreads << std::endl;

        setMemory(dInput, dOutput, dataSize);
        test1D(numOfRepetitions, numOfThreads, xLen, yLen, dInput, dOutput);
        checkMemory(dOutput, numOfElements);

        setMemory(dInput, dOutput, dataSize);
        test2DA(numOfRepetitions, numOfThreads, xLen, yLen, dInput, dOutput);
        checkMemory(dOutput, numOfElements);

        setMemory(dInput, dOutput, dataSize);
        test2DB(numOfRepetitions, numOfThreads, xLen, yLen, dInput, dOutput);
        checkMemory(dOutput, numOfElements);

        setMemory(dInput, dOutput, dataSize);
        test2DBnotSynchronized(numOfRepetitions, numOfThreads, xLen, yLen, dInput, dOutput);
        checkMemory(dOutput, numOfElements);
    }

    cudaFree(dInput);
    cudaFree(dOutput);

    return 0;
}
