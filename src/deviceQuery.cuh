#ifndef DEVICE_QUERY_CUH
#define DEVICE_QUERY_CUH


#include <sstream>
#include <cuda_runtime_api.h>


/**
 * @return string with very basic information about current GPU (useful for logging)
 */
static std::string getDeviceInfo() {
    std::ostringstream res;

    // Get current device count
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    res << "Number of GPUs: " << deviceCount << "\n";

    // current device ID
    int deviceId = -1;
    cudaGetDevice(&deviceId);
    res  << "Using device ID: " << deviceId << "\n";

    // Get some useful data
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, deviceId);

    res << "Name: [" << deviceProp.name << "]\n";
    res << "CUDA Capability Major.Minor version number: " << deviceProp.major << "." << deviceProp.minor << "\n";
    res << "GPU Max Clock rate: " << deviceProp.clockRate * 1e-3f << "MHz" << "\n";
    res << "Memory Clock rate: " << deviceProp.memoryClockRate * 1e-3f << "MHz" << "\n";
    res << "Memory Bus Width: " << deviceProp.memoryBusWidth << "-bit" << "\n";
    res << "L2 Cache Size: " << deviceProp.l2CacheSize << " bytes" << "\n";
    res << "Shared memory/SM: " << deviceProp.sharedMemPerMultiprocessor << " (" << deviceProp.sharedMemPerMultiprocessor / 1024 << "kB)" << "\n";
    res << "Number of SMs: " << deviceProp.multiProcessorCount << "\n";
    res << "Global memory size: " << deviceProp.totalGlobalMem << " (" << deviceProp.totalGlobalMem / 1048576.0f << "MB)" << "\n";
    res << "Device PCI DomainID/BusID/LocationID: " << deviceProp.pciDomainID << "/" << deviceProp.pciBusID << "/" << deviceProp.pciDeviceID << "\n";

    return res.str();
}

#endif
