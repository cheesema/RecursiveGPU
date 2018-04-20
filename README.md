# RecursiveGPU
A fast and efficient 3D recursive filter GPU implimentation


# How to build

```

git clone --recursive https://github.com/cheesema/RecursiveGPU.git

cd RecursiveGPU
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release  ..
cmake -DCMAKE_BUILD_TYPE=Release  ..
make -j 8

./benchmark | tee testOutput.txt

```
![LocalIntensityScale speedup plot](BenchmarkResults/localIntensityScaleCpuVsGpu.jpg?raw=true)
![Recursive Filter speedup plot](BenchmarkResults/recursiveCpuVsGpu.jpg?raw=true)

