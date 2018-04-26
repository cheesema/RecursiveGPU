//
// Created by gonciarz on 4/15/18.
//

#ifndef BENCHMARKTHINGS_CONV_H
#define BENCHMARKTHINGS_CONV_H

#include "data_structures/Mesh/MeshData.hpp"

template <typename T>
void computeConv(const MeshData<T> &in, MeshData<T> &out, const MeshData<T> &kernel);

template <typename T>
void compute3rdPartyConv(const MeshData<T> &in, MeshData<T> &out, const MeshData<T> &kernel);

template <typename T>
void compute3rdParty2DConv(const MeshData<T> &in, MeshData<T> &out, const MeshData<T> &kernel);

int test();


#endif //BENCHMARKTHINGS_CONV_H
