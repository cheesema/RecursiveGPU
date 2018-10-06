//
// Created by gonciarz on 10/5/18.
//

#ifndef BENCHMARKTHINGS_RECURSIVEFILTER_H
#define BENCHMARKTHINGS_RECURSIVEFILTER_H

#include "data_structures/Mesh/PixelData.hpp"

using TypeOfRecRecursiveFlags = uint16_t;
constexpr TypeOfRecRecursiveFlags RECURSIVE_Y_DIR = 0x01;
constexpr TypeOfRecRecursiveFlags RECURSIVE_X_DIR = 0x02;
constexpr TypeOfRecRecursiveFlags RECURSIVE_ALL_DIR = RECURSIVE_Y_DIR | RECURSIVE_X_DIR;

template <typename ImgType>
void filterZeroBoundary(PixelData<ImgType> &input, TypeOfRecRecursiveFlags flags);

#endif //BENCHMARKTHINGS_RECURSIVEFILTER_H
