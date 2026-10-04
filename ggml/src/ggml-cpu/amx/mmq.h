#pragma once
#include "common.h"

size_t ggml_backend_amx_desired_wsize(const struct ggml_tensor * dst);

size_t ggml_backend_amx_get_alloc_size(const struct ggml_tensor * tensor);

void ggml_backend_amx_convert_weight(struct ggml_tensor * tensor, const void * data, size_t offset, size_t size);

void ggml_backend_amx_mul_mat(const struct ggml_compute_params * params, struct ggml_tensor * dst);

// a gate/up product (dst) and the GEGLU node (glu) that reads only it, in one pass; false (nothing done) when the
// weights have no grouped copy or the shapes do not fit
bool ggml_backend_amx_mul_mat_geglu(const struct ggml_compute_params * params, struct ggml_tensor * dst, struct ggml_tensor * glu);
