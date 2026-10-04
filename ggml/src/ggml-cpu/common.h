#pragma once

#include "ggml.h"
#include "traits.h"
#include "ggml-cpu-impl.h"
#include "ggml-impl.h"
#include "simd-mappings.h"

#define GGML_FA_TILE_Q  64
#define GGML_FA_TILE_KV 64

#ifdef __cplusplus

#include <utility>
#include <algorithm>
#include <cstdlib>

// convenience functions/macros for use in template calls
// note: these won't be required after the 'traits' lookup table is used.
static inline ggml_fp16_t f32_to_f16(float x) {
    return GGML_CPU_FP32_TO_FP16(x);
}

static inline float f16_to_f32(ggml_fp16_t x) {
    return GGML_CPU_FP16_TO_FP32(x);
}

static inline ggml_bf16_t f32_to_bf16(float x) {
    return GGML_FP32_TO_BF16(x);
}

static inline float bf16_to_f32(ggml_bf16_t x) {
    return GGML_BF16_TO_FP32(x);
}

static inline float i32_to_f32(int32_t x) {
    return x;
}

static inline int32_t f32_to_i32(float x) {
    return x;
}

static inline float f32_to_f32(float x) {
    return x;
}

// TODO - merge this into the traits table, after using row-based conversions
template <class T>
struct type_conversion_table;

template <>
struct type_conversion_table<ggml_fp16_t> {
    static constexpr float (*to_f32)(ggml_fp16_t) = f16_to_f32;
    static constexpr ggml_fp16_t (*from_f32)(float) = f32_to_f16;
};

template <>
struct type_conversion_table<float> {
    static constexpr float (*to_f32)(float) = f32_to_f32;
    static constexpr float (*from_f32)(float) = f32_to_f32;
};

template <>
struct type_conversion_table<ggml_bf16_t> {
    static constexpr float (*to_f32)(ggml_bf16_t) = bf16_to_f32;
    static constexpr ggml_bf16_t (*from_f32)(float) = f32_to_bf16;
};

template <>
struct type_conversion_table<int32_t> {
    static constexpr float (*to_f32)(int32_t) = i32_to_f32;
    static constexpr int32_t (*from_f32)(float) = f32_to_i32;
};

static std::pair<int64_t, int64_t> get_thread_range(const struct ggml_compute_params * params, const struct ggml_tensor * src0) {
    const int64_t ith = params->ith;
    const int64_t nth = params->nth;

    const int64_t nr  = ggml_nrows(src0);

    // rows per thread
    const int64_t dr = (nr + nth - 1)/nth;

    // row range for this thread
    const int64_t ir0 = dr*ith;
    const int64_t ir1 = MIN(ir0 + dr, nr);

    return {ir0, ir1};
}

// GGML_CPU_COL_SPLIT (default 1; 0: off): elementwise ops with fewer rows than threads (passes of 1-4 tokens) also split
// each row's columns, in slices of a multiple of 64 values, so every thread gets work instead of one per row. The ops
// compute each value on its own, so the results do not change.
static inline bool ggml_col_split_on() {
    static const bool v = [] {
        const char * e = getenv("GGML_CPU_COL_SPLIT");
        return e == nullptr || atoi(e) != 0;
    }();
    return v;
}

// column slices per row for nr rows (1: split rows only); with more than 1, thread ith takes item ith of nr * slices
static inline int64_t ggml_col_slices(int64_t nr, int nth) {
    return ggml_col_split_on() && nr < nth ? nth / nr : 1;
}

// columns [c0, c1) of slice s of n columns cut into cs slices
static inline void ggml_col_slice(int64_t n, int64_t cs, int64_t s, int64_t & c0, int64_t & c1) {
    const int64_t step = ((n + cs - 1) / cs + 63) / 64 * 64;
    c0 = std::min(n, s * step);
    c1 = std::min(n, c0 + step);
}

struct ggml_fa_tile_config {
    static constexpr size_t Q  = GGML_FA_TILE_Q;
    static constexpr size_t KV = GGML_FA_TILE_KV;
};

#endif
