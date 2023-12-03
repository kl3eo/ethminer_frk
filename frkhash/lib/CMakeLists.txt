# frkhash: C/C++ implementation of Frankohash, the Expanse Proof of Work algorithm.
# Copyright 2021 Christopher Franko.
# Licensed under the Apache License, Version 2.0.

add_subdirectory(keccak)
add_subdirectory(frkhash)

get_filename_component(FRKHASH_PRIVATE_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR} ABSOLUTE)
set(FRKHASH_PRIVATE_INCLUDE_DIR ${FRKHASH_PRIVATE_INCLUDE_DIR} PARENT_SCOPE)
