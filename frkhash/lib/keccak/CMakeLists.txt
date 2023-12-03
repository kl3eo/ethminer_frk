# frkhash: C/C++ implementation of Frankohash, the Expanse Proof of Work algorithm.
# Copyright 2019 Christopher Franko.
# SPDX-License-Identifier: Apache-2.0

add_library(keccak)
add_library(frkhash::keccak ALIAS keccak)
target_compile_features(keccak PUBLIC c_std_99)
target_include_directories(keccak PUBLIC ${PROJECT_SOURCE_DIR}/include)
target_sources(keccak PRIVATE
    ${PROJECT_SOURCE_DIR}/include/frkhash/keccak.h
    ${PROJECT_SOURCE_DIR}/include/frkhash/keccak.hpp
    keccak.c
    keccakf800.c
)
