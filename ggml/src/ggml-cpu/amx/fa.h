#pragma once

#include "ggml.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

struct ggml_compute_params;

// CPU flash attention with AMX-FP16 tiles. Returns false (and does nothing) when the build, the CPU or the operation
// is not supported; every thread of the operation gets the same answer.
bool ggml_fa_amx_compute(const struct ggml_compute_params * params, struct ggml_tensor * dst);

// work buffer bytes the AMX path needs for an operation with these head sizes
size_t ggml_fa_amx_work_size(int64_t DK, int64_t DV, int n_tasks);

#ifdef __cplusplus
}
#endif
