//
// Created by gonciarz on 4/24/18.
//

#ifndef BENCHMARKTHINGS_MEMBENCHMARKS_H
#define BENCHMARKTHINGS_MEMBENCHMARKS_H

#include "data_structures/Mesh/MeshData.hpp"

template <typename T>
void memCopy1D(const MeshData<T> &in, MeshData<T> &out);
template <typename T>
void memCopy2D(const MeshData<T> &in, MeshData<T> &out);
template <typename T>
void conv2d(const MeshData<T> &in, MeshData<T> &out, const MeshData<T> &kernel);

template <typename T>
void conv3d(const MeshData<T> &in, MeshData<T> &out, const MeshData<T> &aKernel);

#endif //BENCHMARKTHINGS_MEMBENCHMARKS_H
