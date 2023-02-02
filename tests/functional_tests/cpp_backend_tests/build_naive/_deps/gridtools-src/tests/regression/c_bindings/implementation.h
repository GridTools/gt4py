// This file is generated!
#pragma once

#include <cpp_bindgen/array_descriptor.h>
#include <cpp_bindgen/handle.h>

#ifdef __cplusplus
extern "C" {
#endif

void copy_from_data_store(bindgen_handle*, double*);
void copy_to_data_store(bindgen_handle*, double*);
bindgen_handle* create_data_store(unsigned int, unsigned int, unsigned int);
void run_copy_stencil(bindgen_handle*, bindgen_handle*);

#ifdef __cplusplus
}
#endif
