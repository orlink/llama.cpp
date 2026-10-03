#include <mutex>
#include <string>
#include <unordered_map>
#include <cstdio>
#include <cstring>
#include "gguf.h"
#include <atomic>
#include <cmath>
#include <vector>
#if defined(__GNUC__)
#pragma GCC diagnostic ignored "-Wpedantic"
#pragma GCC diagnostic ignored "-Wunused-local-typedefs"
#endif

#include "amx.h"
#include "mmq.h"
#include "ggml-impl.h"
#include "ggml-cpu-impl.h"
#include "simd-mappings.h"
#include "quants.h"
#include "ggml-quants.h"
#include <algorithm>
#include <type_traits>

#if defined(__gnu_linux__)
#include <sys/syscall.h>
#include <unistd.h>
#endif

#if (defined(_WIN32) || defined(_WIN64))
#define RESTRICT __restrict
#else
#define RESTRICT __restrict__
#endif

#if (defined(_WIN32) || defined(_WIN64))
#define ALWAYS_INLINE __forceinline
#elif __has_attribute(always_inline) || defined(__GNUC__)
#define ALWAYS_INLINE __attribute__((__always_inline__)) inline
#else
#define ALWAYS_INLINE inline
#endif

#if defined(__AMX_INT8__) && defined(__AVX512VNNI__)

namespace {

// Forced unrolling
template <int n>
struct Unroll {
    template <typename Func, typename... Args>
    ALWAYS_INLINE void operator()(const Func& f, Args... args) const {
        Unroll<n - 1>{}(f, args...);
        f(std::integral_constant<int, n - 1>{}, args...);
    }
};

template <>
struct Unroll<1> {
    template <typename Func, typename... Args>
    ALWAYS_INLINE void operator()(const Func& f, Args... args) const {
        f(std::integral_constant<int, 0>{}, args...);
    }
};

// type traits
template <typename T> struct PackedTypes {};
template <> struct PackedTypes<block_q4_0> { using type = int8_t; };
template <> struct PackedTypes<block_q4_1> { using type = uint8_t; };
template <> struct PackedTypes<block_q8_0> { using type = int8_t; };
template <typename T> using packed_B_type = typename PackedTypes<T>::type;

template <typename T>
struct do_compensate : std::integral_constant<bool,
    std::is_same<T, block_q8_0>::value> {};

template <typename T>
struct do_unpack : std::integral_constant<bool,
    std::is_same<T, block_q4_0>::value ||
    std::is_same<T, block_q4_1>::value> {};

template <typename T>
struct is_type_qkk : std::integral_constant<bool,
    std::is_same<T, block_q4_K>::value ||
    std::is_same<T, block_q5_K>::value ||
    std::is_same<T, block_q6_K>::value ||
    std::is_same<T, block_iq4_xs>::value> {};

#define GGML_DISPATCH_FLOATING_TYPES(TYPE, ...)                                        \
    [&] {                                                                              \
        switch (TYPE) {                                                                \
            case GGML_TYPE_F16: {                                                      \
                using type = ggml_fp16_t;                                              \
                constexpr int blck_size = 16;                                          \
                return __VA_ARGS__();                                                  \
            }                                                                          \
            case GGML_TYPE_BF16: {                                                     \
                using type = ggml_bf16_t;                                              \
                constexpr int blck_size = 32;                                          \
                return __VA_ARGS__();                                                  \
            }                                                                          \
            default:                                                                   \
                fprintf(stderr, "Unsupported floating data type\n");                   \
        }                                                                              \
    }()

#define GGML_DISPATCH_QTYPES(QT, ...)                                                  \
    [&] {                                                                              \
        switch (QT) {                                                                  \
            case GGML_TYPE_Q4_0: {                                                     \
                using type = block_q4_0;                                               \
                using vec_dot_type = block_q8_0;                                       \
                constexpr int blck_size = QK4_0;                                       \
                return __VA_ARGS__();                                                  \
            }                                                                          \
            case GGML_TYPE_Q4_1: {                                                     \
                using type = block_q4_1;                                               \
                using vec_dot_type = block_q8_1;                                       \
                constexpr int blck_size = QK4_1;                                       \
                return __VA_ARGS__();                                                  \
            }                                                                          \
            case GGML_TYPE_Q8_0: {                                                     \
                using type = block_q8_0;                                               \
                using vec_dot_type = block_q8_0;                                       \
                constexpr int blck_size = QK8_0;                                       \
                return __VA_ARGS__();                                                  \
            }                                                                          \
            case GGML_TYPE_Q4_K: {                                                     \
                using type = block_q4_K;                                               \
                using vec_dot_type = block_q8_K;                                       \
                constexpr int blck_size = QK_K;                                        \
                return __VA_ARGS__();                                                  \
            }                                                                          \
            case GGML_TYPE_Q5_K: {                                                     \
                using type = block_q5_K;                                               \
                using vec_dot_type = block_q8_K;                                       \
                constexpr int blck_size = QK_K;                                        \
                return __VA_ARGS__();                                                  \
            }                                                                          \
            case GGML_TYPE_Q6_K: {                                                     \
                using type = block_q6_K;                                               \
                using vec_dot_type = block_q8_K;                                       \
                constexpr int blck_size = QK_K;                                        \
                return __VA_ARGS__();                                                  \
            }                                                                          \
            case GGML_TYPE_IQ4_XS: {                                                   \
                using type = block_iq4_xs;                                             \
                using vec_dot_type = block_q8_K;                                       \
                constexpr int blck_size = QK_K;                                        \
                return __VA_ARGS__();                                                  \
            }                                                                          \
            default:                                                                   \
                fprintf(stderr, "Unsupported quantized data type: %d\n", int(TYPE));   \
        }                                                                              \
    }()

#define GGML_DISPATCH_BOOL(BOOL_V, BOOL_NAME, ...)                                     \
    [&] {                                                                              \
        if (BOOL_V) {                                                                  \
            constexpr bool BOOL_NAME = true;                                           \
            return __VA_ARGS__();                                                      \
        } else {                                                                       \
            constexpr bool BOOL_NAME = false;                                          \
            return __VA_ARGS__();                                                      \
        }                                                                              \
    }()

// define amx tile config data structure
struct tile_config_t{
    uint8_t palette_id = 0;
    uint8_t start_row = 0;
    uint8_t reserved_0[14] = {0};
    uint16_t colsb[16] = {0};
    uint8_t rows[16] = {0};
};

// Notes: amx tile config
//
// Typically, TMUL calculates A and B of size 16 x 64 containing INT8 values,
// and accumulate the result to a 16 x 16 matrix C containing INT32 values,
//
// As many GGUF quantized types as `block_size` of 32, so a 16-16-32 config is used
// instead of the normally used 16-16-64 config.
//
//    Block A: {16, 32}, dtype = int8_t
//    Block B: {16, 32}, dtype = uint8_t/int8_t
//    Block C: {16, 16}, dtype = int32_t
//
// Block B needs to be prepacked to vnni format before feeding into  TMUL:
//    packed_B: from {n, k} to {k/vnni_blk, n, vnni_blck}, viewed in 2d, we get {8, 64}
//
// Therefore, we get tileconfig:
//             A    B    C
//    rows    16    8   16
//    colsb   32   64   16
//
// For tile distribution, follow a 2-2-4 pattern, e.g. A used TMM2-TMM3, B used TMM0-TMM1,
// C used TMM4-TMM7:
//            B TMM0  B TMM1
//    A TMM2  C TMM4  C TMM6
//    A TMM3  C TMM5  C TMM7
//
// Each `amx` kernel handles 4 blocks at a time: 2MB * 2NB, when m < 2 * BLOCK_M, unpack A
// will be needed.
//
// Here another commonly used pattern 1-3-3 is skipped, as it is mostly used when m <=16;
// and the single batch gemm (m=1) has a special fast path with `avx512-vnni`.
//
// ref: https://www.intel.com/content/www/us/en/developer/articles/code-sample/
//    advanced-matrix-extensions-intrinsics-functions.html
//

inline void ggml_tile_config_init(void) {
    static thread_local bool done = false;

    if (done) {
        return;
    }

    alignas(64) tile_config_t tc = {};
    tc.palette_id = 1;
    tc.start_row = 0;
    tc.rows[0] = 8;   tc.colsb[0] = 64;
    tc.rows[1] = 8;   tc.colsb[1] = 64;
    tc.rows[2] = 16;  tc.colsb[2] = 32;
    tc.rows[3] = 16;  tc.colsb[3] = 32;
    tc.rows[4] = 16;  tc.colsb[4] = 64;
    tc.rows[5] = 16;  tc.colsb[5] = 64;
    tc.rows[6] = 16;  tc.colsb[6] = 64;
    tc.rows[7] = 16;  tc.colsb[7] = 64;

    _tile_loadconfig(&tc);
    done = true;
}

// we need an extra 16 * 4B (TILE_N * int32_t) for each NB/KB block for compensation.
// See the notes `s8s8 igemm compensation in avx512-vnni` for detail.
template <typename TB>
int get_tile_size() {
    int tile_size = TILE_N * sizeof(TB);
    if (do_compensate<TB>::value) {
        tile_size += TILE_N * sizeof(int32_t);
    }
    if (std::is_same<TB, block_q4_K>::value ||
        std::is_same<TB, block_q5_K>::value) {
        tile_size += TILE_N * 4;
    }
    if (std::is_same<TB, block_iq4_xs>::value) {
        tile_size += TILE_N * 2;
    }
    return tile_size;
}

template <typename TB, int BLOCK_K>
int get_row_size(int K) {
    int KB = K / BLOCK_K;
    int row_size = KB * sizeof(TB);
    if (do_compensate<TB>::value) {
        row_size += KB * sizeof(int32_t);
    }
    if (std::is_same<TB, block_q4_K>::value ||
        std::is_same<TB, block_q5_K>::value) {
        row_size += KB * 4;
    }
    if (std::is_same<TB, block_iq4_xs>::value) {
        row_size += KB * 2;
    }
    return row_size;
}

// transpose utils
#define SHUFFLE_EPI32(a, b, mask) \
    _mm256_castps_si256(_mm256_shuffle_ps(_mm256_castsi256_ps(a), _mm256_castsi256_ps(b), mask))
inline void transpose_8x8_32bit(__m256i * v, __m256i * v1) {
    // unpacking and 32-bit elements
    v1[0] = _mm256_unpacklo_epi32(v[0], v[1]);
    v1[1] = _mm256_unpackhi_epi32(v[0], v[1]);
    v1[2] = _mm256_unpacklo_epi32(v[2], v[3]);
    v1[3] = _mm256_unpackhi_epi32(v[2], v[3]);
    v1[4] = _mm256_unpacklo_epi32(v[4], v[5]);
    v1[5] = _mm256_unpackhi_epi32(v[4], v[5]);
    v1[6] = _mm256_unpacklo_epi32(v[6], v[7]);
    v1[7] = _mm256_unpackhi_epi32(v[6], v[7]);

    // shuffling the 32-bit elements
    v[0] = SHUFFLE_EPI32(v1[0], v1[2], 0x44);
    v[1] = SHUFFLE_EPI32(v1[0], v1[2], 0xee);
    v[2] = SHUFFLE_EPI32(v1[4], v1[6], 0x44);
    v[3] = SHUFFLE_EPI32(v1[4], v1[6], 0xee);
    v[4] = SHUFFLE_EPI32(v1[1], v1[3], 0x44);
    v[5] = SHUFFLE_EPI32(v1[1], v1[3], 0xee);
    v[6] = SHUFFLE_EPI32(v1[5], v1[7], 0x44);
    v[7] = SHUFFLE_EPI32(v1[5], v1[7], 0xee);

    // shuffling 128-bit elements
    v1[0] = _mm256_permute2f128_si256(v[2], v[0], 0x02);
    v1[1] = _mm256_permute2f128_si256(v[3], v[1], 0x02);
    v1[2] = _mm256_permute2f128_si256(v[6], v[4], 0x02);
    v1[3] = _mm256_permute2f128_si256(v[7], v[5], 0x02);
    v1[4] = _mm256_permute2f128_si256(v[2], v[0], 0x13);
    v1[5] = _mm256_permute2f128_si256(v[3], v[1], 0x13);
    v1[6] = _mm256_permute2f128_si256(v[6], v[4], 0x13);
    v1[7] = _mm256_permute2f128_si256(v[7], v[5], 0x13);
}

inline void transpose_16x4_32bit(__m512i * r, __m512i * d) {

    static const __m512i index1 = _mm512_set_epi32(
        0x0f, 0x0b, 0x07, 0x03,
        0x0e, 0x0a, 0x06, 0x02,
        0x0d, 0x09, 0x05, 0x01,
        0x0c, 0x08, 0x04, 0x00);

    d[0] = _mm512_permutexvar_epi32(index1, r[0]);
    d[1] = _mm512_permutexvar_epi32(index1, r[1]);
    d[2] = _mm512_permutexvar_epi32(index1, r[2]);
    d[3] = _mm512_permutexvar_epi32(index1, r[3]);

    r[0] = _mm512_shuffle_i32x4(d[0], d[1], 0x44);
    r[1] = _mm512_shuffle_i32x4(d[0], d[1], 0xee);
    r[2] = _mm512_shuffle_i32x4(d[2], d[3], 0x44);
    r[3] = _mm512_shuffle_i32x4(d[2], d[3], 0xee);

    d[0] = _mm512_shuffle_i32x4(r[0], r[2], 0x88);
    d[1] = _mm512_shuffle_i32x4(r[0], r[2], 0xdd);
    d[2] = _mm512_shuffle_i32x4(r[1], r[3], 0x88);
    d[3] = _mm512_shuffle_i32x4(r[1], r[3], 0xdd);
}

inline void transpose_16x16_32bit(__m512i * v) {
    __m512i v1[16];
    v1[0] = _mm512_unpacklo_epi32(v[0], v[1]);
    v1[1] = _mm512_unpackhi_epi32(v[0], v[1]);
    v1[2] = _mm512_unpacklo_epi32(v[2], v[3]);
    v1[3] = _mm512_unpackhi_epi32(v[2], v[3]);
    v1[4] = _mm512_unpacklo_epi32(v[4], v[5]);
    v1[5] = _mm512_unpackhi_epi32(v[4], v[5]);
    v1[6] = _mm512_unpacklo_epi32(v[6], v[7]);
    v1[7] = _mm512_unpackhi_epi32(v[6], v[7]);
    v1[8] = _mm512_unpacklo_epi32(v[8], v[9]);
    v1[9] = _mm512_unpackhi_epi32(v[8], v[9]);
    v1[10] = _mm512_unpacklo_epi32(v[10], v[11]);
    v1[11] = _mm512_unpackhi_epi32(v[10], v[11]);
    v1[12] = _mm512_unpacklo_epi32(v[12], v[13]);
    v1[13] = _mm512_unpackhi_epi32(v[12], v[13]);
    v1[14] = _mm512_unpacklo_epi32(v[14], v[15]);
    v1[15] = _mm512_unpackhi_epi32(v[14], v[15]);

    v[0] = _mm512_unpacklo_epi64(v1[0], v1[2]);
    v[1] = _mm512_unpackhi_epi64(v1[0], v1[2]);
    v[2] = _mm512_unpacklo_epi64(v1[1], v1[3]);
    v[3] = _mm512_unpackhi_epi64(v1[1], v1[3]);
    v[4] = _mm512_unpacklo_epi64(v1[4], v1[6]);
    v[5] = _mm512_unpackhi_epi64(v1[4], v1[6]);
    v[6] = _mm512_unpacklo_epi64(v1[5], v1[7]);
    v[7] = _mm512_unpackhi_epi64(v1[5], v1[7]);
    v[8] = _mm512_unpacklo_epi64(v1[8], v1[10]);
    v[9] = _mm512_unpackhi_epi64(v1[8], v1[10]);
    v[10] = _mm512_unpacklo_epi64(v1[9], v1[11]);
    v[11] = _mm512_unpackhi_epi64(v1[9], v1[11]);
    v[12] = _mm512_unpacklo_epi64(v1[12], v1[14]);
    v[13] = _mm512_unpackhi_epi64(v1[12], v1[14]);
    v[14] = _mm512_unpacklo_epi64(v1[13], v1[15]);
    v[15] = _mm512_unpackhi_epi64(v1[13], v1[15]);

    v1[0] = _mm512_shuffle_i32x4(v[0], v[4], 0x88);
    v1[1] = _mm512_shuffle_i32x4(v[1], v[5], 0x88);
    v1[2] = _mm512_shuffle_i32x4(v[2], v[6], 0x88);
    v1[3] = _mm512_shuffle_i32x4(v[3], v[7], 0x88);
    v1[4] = _mm512_shuffle_i32x4(v[0], v[4], 0xdd);
    v1[5] = _mm512_shuffle_i32x4(v[1], v[5], 0xdd);
    v1[6] = _mm512_shuffle_i32x4(v[2], v[6], 0xdd);
    v1[7] = _mm512_shuffle_i32x4(v[3], v[7], 0xdd);
    v1[8] = _mm512_shuffle_i32x4(v[8], v[12], 0x88);
    v1[9] = _mm512_shuffle_i32x4(v[9], v[13], 0x88);
    v1[10] = _mm512_shuffle_i32x4(v[10], v[14], 0x88);
    v1[11] = _mm512_shuffle_i32x4(v[11], v[15], 0x88);
    v1[12] = _mm512_shuffle_i32x4(v[8], v[12], 0xdd);
    v1[13] = _mm512_shuffle_i32x4(v[9], v[13], 0xdd);
    v1[14] = _mm512_shuffle_i32x4(v[10], v[14], 0xdd);
    v1[15] = _mm512_shuffle_i32x4(v[11], v[15], 0xdd);

    v[0] = _mm512_shuffle_i32x4(v1[0], v1[8], 0x88);
    v[1] = _mm512_shuffle_i32x4(v1[1], v1[9], 0x88);
    v[2] = _mm512_shuffle_i32x4(v1[2], v1[10], 0x88);
    v[3] = _mm512_shuffle_i32x4(v1[3], v1[11], 0x88);
    v[4] = _mm512_shuffle_i32x4(v1[4], v1[12], 0x88);
    v[5] = _mm512_shuffle_i32x4(v1[5], v1[13], 0x88);
    v[6] = _mm512_shuffle_i32x4(v1[6], v1[14], 0x88);
    v[7] = _mm512_shuffle_i32x4(v1[7], v1[15], 0x88);
    v[8] = _mm512_shuffle_i32x4(v1[0], v1[8], 0xdd);
    v[9] = _mm512_shuffle_i32x4(v1[1], v1[9], 0xdd);
    v[10] = _mm512_shuffle_i32x4(v1[2], v1[10], 0xdd);
    v[11] = _mm512_shuffle_i32x4(v1[3], v1[11], 0xdd);
    v[12] = _mm512_shuffle_i32x4(v1[4], v1[12], 0xdd);
    v[13] = _mm512_shuffle_i32x4(v1[5], v1[13], 0xdd);
    v[14] = _mm512_shuffle_i32x4(v1[6], v1[14], 0xdd);
    v[15] = _mm512_shuffle_i32x4(v1[7], v1[15], 0xdd);
}

void quantize_row_q8_K_vnni(const float * RESTRICT x, void * RESTRICT vy, int64_t k) {
    assert(k % QK_K == 0);
    const int KB = k / QK_K;
    constexpr int kVecs = QK_K / 16;

    block_q8_K * y = reinterpret_cast<block_q8_K *>(vy);

    // hold 16 float vecs from x
    __m512  v[kVecs];

    // hold the quants vecs
    __m512i vq[kVecs / 4];

    // hold the packed quants vecs
    __m512i vq_packed[kVecs / 4];

    const __m512 signBit = _mm512_set1_ps(-0.f);

    for (int i = 0; i < KB; ++i) {
        // Compute max(abs(e)) for the block
        __m512 vamax = _mm512_set1_ps(0.f);
        for (int j = 0; j < kVecs; ++j) {
            v[j] = _mm512_loadu_ps(x); x += 16;
            vamax = _mm512_max_ps(vamax, _mm512_andnot_ps(signBit, v[j]));
        }
        const float amax = _mm512_reduce_max_ps(vamax);

        // Quantize these floats
        const float iscale = 127.f / amax;
        y[i].d = GGML_CPU_FP32_TO_FP16(1 / iscale);
        const float id = ( amax != 0.0f ) ? iscale : 0.f;
        const __m512 vscale = _mm512_set1_ps(id);

        // Apply multiplier and round to nearest integer
        for (int j = 0; j < kVecs; ++j) {
            v[j] = _mm512_mul_ps(v[j], vscale);
            v[j] = _mm512_roundscale_ps(v[j], (_MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
        }

        // Pack to epi8 vecs
        for (int j = 0; j < kVecs / 4; ++j) {
            __m128i q8_0 = _mm512_cvtepi32_epi8(_mm512_cvtps_epi32(v[j * 4 + 0]));
            __m128i q8_1 = _mm512_cvtepi32_epi8(_mm512_cvtps_epi32(v[j * 4 + 1]));
            __m128i q8_2 = _mm512_cvtepi32_epi8(_mm512_cvtps_epi32(v[j * 4 + 2]));
            __m128i q8_3 = _mm512_cvtepi32_epi8(_mm512_cvtps_epi32(v[j * 4 + 3]));

            __m256i q8_01 = _mm256_insertf128_si256(_mm256_castsi128_si256(q8_0), (q8_1), 1);
            __m256i q8_23 = _mm256_insertf128_si256(_mm256_castsi128_si256(q8_2), (q8_3), 1);

            vq[j] = _mm512_inserti32x8(_mm512_castsi256_si512(q8_01), q8_23, 1);
            _mm512_storeu_si512((__m512i *)(y[i].qs + j * 64), vq[j]);
        }

        // Compute the bsums with vnni
        transpose_16x4_32bit(vq, vq_packed);

        const __m512i one = _mm512_set1_epi8(1);
        __m512i sum = _mm512_setzero_si512();
        for (int k = 0; k < 4; ++k) {
            sum = _mm512_dpbusd_epi32(sum, one, vq_packed[k]);
        }
        _mm256_storeu_si256((__m256i *)(y[i].bsums), _mm512_cvtepi32_epi16(sum));
    }
}

// quantize A from float to `vec_dot_type`
template <typename T>
inline void from_float(const float * x, char * vy, int64_t k);

template <>
inline void from_float<block_q8_0>(const float * x, char * vy, int64_t k) {
    quantize_row_q8_0(x, (block_q8_0 *)vy, k);
}

template <>
inline void from_float<block_q8_1>(const float * x, char * vy, int64_t k) {
    quantize_row_q8_1(x, (block_q8_1 *)vy, k);
}

template <>
inline void from_float<block_q8_K>(const float * x, char * vy, int64_t k) {
#if 1
    // TODO: this is reference impl!
    quantize_row_q8_K_ref(x, (block_q8_K *)vy, k);
#else
    quantize_row_q8_K_vnni(x, vy, k);
#endif
}

// load A from memory to array when nrows can not fill in whole tile
void unpack_A(int8_t * RESTRICT tile, const block_q8_0 * RESTRICT A, int lda, int nr) {
    assert(nr != TILE_M);
    for (int m = 0; m < nr; ++m) {
        const __m256i v = _mm256_loadu_si256((const __m256i *)(A[m * lda].qs));
        _mm256_storeu_si256((__m256i *)(tile + m * TILE_K), v);
    }
}

void unpack_A(int8_t * RESTRICT tile, const block_q8_1 * RESTRICT A, int lda, int nr) {
    assert(nr != TILE_M);
    for (int m = 0; m < nr; ++m) {
        const __m256i v = _mm256_loadu_si256((const __m256i *)(A[m * lda].qs));
        _mm256_storeu_si256((__m256i *)(tile + m * TILE_K), v);
    }
}

template <typename TB>
void unpack_A(int8_t * RESTRICT tile, const block_q8_K * RESTRICT A, int lda, int k, int nr) {
    assert(nr <= TILE_M);
    for (int m = 0; m < nr; ++m) {
        const __m256i v = _mm256_loadu_si256((const __m256i *)(A[m * lda].qs + k * 32));
        _mm256_storeu_si256((__m256i *)(tile + m * TILE_K), v);
    }
}

template <>
void unpack_A<block_q6_K>(int8_t * RESTRICT tile, const block_q8_K * RESTRICT A, int lda, int k, int nr) {
    assert(nr <= TILE_M);
    // zero padding k from 16 to 32, so that we don't have to re-config amx
    const __m128i zero = _mm_setzero_si128();
    for (int m = 0; m < nr; ++m) {
        const __m128i v = _mm_loadu_si128((const __m128i *)(A[m * lda].qs + k * 16));
        const __m256i r = _mm256_insertf128_si256(_mm256_castsi128_si256(v), zero, 1);
        _mm256_storeu_si256((__m256i *)(tile + m * TILE_K), r);
    }
}

#define MM256_SET_M128I(a, b) _mm256_insertf128_si256(_mm256_castsi128_si256(b), (a), 1)
inline __m256i bytes_from_nibbles_32(const uint8_t * rsi) {
    const __m128i tmp = _mm_loadu_si128((const __m128i *)rsi);
    const __m256i bytes = MM256_SET_M128I(_mm_srli_epi16(tmp, 4), tmp);
    const __m256i lowMask = _mm256_set1_epi8(0xF);
    return _mm256_and_si256(lowMask, bytes);
}

// used for block_q4_K
inline __m512i bytes_from_nibbles_64(const uint8_t * rsi) {
    const __m256i tmp = _mm256_loadu_si256((const __m256i *)rsi);
    const __m256i lowMask = _mm256_set1_epi8(0xF);
    const __m256i q4l = _mm256_and_si256(tmp, lowMask);
    const __m256i q4h = _mm256_and_si256(_mm256_srli_epi16(tmp, 4), lowMask);
    return _mm512_inserti32x8(_mm512_castsi256_si512(q4l), q4h, 1);
}

// used for block_q5_K
inline __m512i bytes_from_nibbles_64(const uint8_t * qs, const uint8_t * qh, int k) {
    const __m256i lowMask = _mm256_set1_epi8(0xF);
    __m256i hmask = _mm256_set1_epi8(1);
    hmask = _mm256_slli_epi16(hmask, k);

    const __m256i q5bits = _mm256_loadu_si256((const __m256i *)qs);
    const __m256i hbits = _mm256_loadu_si256((const __m256i *)qh);

    const __m256i q5l_0 = _mm256_and_si256(q5bits, lowMask);
    const __m256i q5h_0 = _mm256_slli_epi16(_mm256_srli_epi16(_mm256_and_si256(hbits, hmask), k + 0), 4);
    const __m256i q5_0  = _mm256_add_epi8(q5l_0, q5h_0);
    hmask = _mm256_slli_epi16(hmask, 1);

    const __m256i q5l_1 = _mm256_and_si256(_mm256_srli_epi16(q5bits, 4), lowMask);
    const __m256i q5h_1 = _mm256_slli_epi16(_mm256_srli_epi16(_mm256_and_si256(hbits, hmask), k + 1), 4);
    const __m256i q5_1  = _mm256_add_epi8(q5l_1, q5h_1);

    return _mm512_inserti32x8(_mm512_castsi256_si512(q5_0), q5_1, 1);
}

// used for block_q6_K
inline void bytes_from_nibbles_128(__m512i& r0, __m512i& r1, const uint8_t * qs, const uint8_t * qh) {
    const __m256i m4 = _mm256_set1_epi8(0xF);
    const __m256i m2 = _mm256_set1_epi8(0x3);

    const __m256i q6bits1 = _mm256_loadu_si256((const __m256i *)qs);
    const __m256i q6bits2 = _mm256_loadu_si256((const __m256i *)(qs + 32));
    const __m256i q6bitsH = _mm256_loadu_si256((const __m256i *)qh);

    const __m256i q6h_0 = _mm256_slli_epi16(_mm256_and_si256(                  q6bitsH,     m2), 4);
    const __m256i q6h_1 = _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(q6bitsH, 2), m2), 4);
    const __m256i q6h_2 = _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(q6bitsH, 4), m2), 4);
    const __m256i q6h_3 = _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(q6bitsH, 6), m2), 4);

    const __m256i q6_0 = _mm256_or_si256(_mm256_and_si256(q6bits1, m4), q6h_0);
    const __m256i q6_1 = _mm256_or_si256(_mm256_and_si256(q6bits2, m4), q6h_1);
    const __m256i q6_2 = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(q6bits1, 4), m4), q6h_2);
    const __m256i q6_3 = _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(q6bits2, 4), m4), q6h_3);

    r0 = _mm512_inserti32x8(_mm512_castsi256_si512(q6_0), q6_1, 1);
    r1 = _mm512_inserti32x8(_mm512_castsi256_si512(q6_2), q6_3, 1);
}

inline __m512i packNibbles(__m512i r0, __m512i r1) {
    return _mm512_or_si512(r0, _mm512_slli_epi16(r1, 4));
}

template <typename TB>
inline void pack_qs(void * RESTRICT packed_B, const TB * RESTRICT B, int KB) {
    int8_t tmp[8 * 64];
    __m256i v[8], v2[8];
    for (int n = 0; n < 8; ++n) {
        v[n] = bytes_from_nibbles_32(B[n * KB].qs);
    }
    transpose_8x8_32bit(v, v2);
    for (int n = 0; n < 8; ++n) {
        _mm256_storeu_si256((__m256i *)(tmp + n * 64), v2[n]);
    }
    for (int n = 0; n < 8; ++n) {
        v[n] = bytes_from_nibbles_32(B[(n + 8) * KB].qs);
    }
    transpose_8x8_32bit(v, v2);
    for (int n = 0; n < 8; ++n) {
        _mm256_storeu_si256((__m256i *)(tmp + n * 64 + 32), v2[n]);
    }

    // pack again with 128 to fully utilize vector length
    for (int n = 0; n < 8; n += 2) {
        __m512i r0 = _mm512_loadu_si512((const __m512i *)(tmp + n * 64));
        __m512i r1 = _mm512_loadu_si512((const __m512i *)(tmp + n * 64 + 64));
        __m512i r1r0 = packNibbles(r0, r1);
        _mm512_storeu_si512((__m512i *)((char *)packed_B + n * 32), r1r0);
    }
}

template <>
inline void pack_qs<block_q8_0>(void * RESTRICT packed_B, const block_q8_0 * RESTRICT B, int KB) {
    __m256i v[8], v2[8];
    for (int n = 0; n < 8; ++n) {
        v[n] = _mm256_loadu_si256((const __m256i *)(B[n * KB].qs));
    }
    transpose_8x8_32bit(v, v2);
    for (int n = 0; n < 8; ++n) {
        _mm256_storeu_si256((__m256i *)((char *)packed_B + n * 64), v2[n]);
    }
    for (int n = 0; n < 8; ++n) {
        v[n] = _mm256_loadu_si256((const __m256i *)(B[(n + 8) * KB].qs));
    }
    transpose_8x8_32bit(v, v2);
    for (int n = 0; n < 8; ++n) {
        _mm256_storeu_si256((__m256i *)((char *)packed_B + n * 64 + 32), v2[n]);
    }
}

template <>
inline void pack_qs<block_q4_K>(void * RESTRICT packed_B, const block_q4_K * RESTRICT B, int KB) {
    __m512i v[16];
    // QK_K 256 with 8 groups, handle 2 groups at a time
    char * pb = (char *)packed_B;
    for (int k = 0; k < QK_K / 64; ++k) {
        // pack 2 groups { n, g,  k} to {g, k/4, 4n}
        //          e.g. {16, 2, 32} to {2,   8, 64}
        for (int n = 0; n < TILE_N; ++n) {
            v[n] = bytes_from_nibbles_64(B[n * KB].qs + k * 32);
        }

        transpose_16x16_32bit(v);

        // pack again with 128 to fully utilize vector length
        for (int n = 0; n < TILE_N; n += 2) {
            _mm512_storeu_si512((__m512i *)pb, packNibbles(v[n], v[n + 1]));
            pb += 64;
        }
    }
}

template <>
inline void pack_qs<block_q5_K>(void * RESTRICT packed_B, const block_q5_K * RESTRICT B, int KB) {
    __m512i v[16];
    const __m512i lowMask = _mm512_set1_epi8(0xF);
    // QK_K 256 with 8 groups, handle 2 groups at a time
    char * pb = (char *)packed_B;
    char * ph = (char *)packed_B + (QK_K / 2) * TILE_N;
    for (int k = 0; k < QK_K / 64; ++k) {
        // pack 2 groups { n, g,  k} to {g, k/4, 4n}
        //          e.g. {16, 2, 32} to {2,   8, 64}
        for (int n = 0; n < TILE_N; ++n) {
            v[n] = bytes_from_nibbles_64(B[n * KB].qs + k * 32, B[n * KB].qh, /* group */2 * k);
        }

        transpose_16x16_32bit(v);

        // 1. pack lower 4bits with 2 groups
        for (int n = 0; n < TILE_N; n += 2) {
            // get lower 4 bits
            const __m512i r0 = _mm512_and_si512(v[n], lowMask);
            const __m512i r1 = _mm512_and_si512(v[n + 1], lowMask);
            _mm512_storeu_si512((__m512i *)pb, packNibbles(r0, r1)); pb += 64;
        }

        // 2. pack higher 1bit with 2 groups
        const __m512i hmask = _mm512_set1_epi8(0x10);
        for (int g = 0; g < 2; ++g) {
            __m512i hbits = _mm512_setzero_si512();
            hbits = _mm512_add_epi8(hbits, _mm512_srli_epi16(_mm512_and_si512(v[g * 8 + 0], hmask), 4));
            hbits = _mm512_add_epi8(hbits, _mm512_srli_epi16(_mm512_and_si512(v[g * 8 + 1], hmask), 3));
            hbits = _mm512_add_epi8(hbits, _mm512_srli_epi16(_mm512_and_si512(v[g * 8 + 2], hmask), 2));
            hbits = _mm512_add_epi8(hbits, _mm512_srli_epi16(_mm512_and_si512(v[g * 8 + 3], hmask), 1));
            hbits = _mm512_add_epi8(hbits,                   _mm512_and_si512(v[g * 8 + 4], hmask)    );
            hbits = _mm512_add_epi8(hbits, _mm512_slli_epi16(_mm512_and_si512(v[g * 8 + 5], hmask), 1));
            hbits = _mm512_add_epi8(hbits, _mm512_slli_epi16(_mm512_and_si512(v[g * 8 + 6], hmask), 2));
            hbits = _mm512_add_epi8(hbits, _mm512_slli_epi16(_mm512_and_si512(v[g * 8 + 7], hmask), 3));
            _mm512_storeu_si512((__m512i *)ph, hbits); ph += 64;
        }
    }
}

template <>
inline void pack_qs<block_q6_K>(void * RESTRICT packed_B, const block_q6_K * RESTRICT B, int KB) {
    __m512i v[32];
    const __m512i lowMask = _mm512_set1_epi8(0xF);
    // QK_K 256 with 8 groups, handle 4 groups at a time
    char * pb = (char *)packed_B;
    char * ph = (char *)packed_B + (QK_K / 2) * TILE_N;
    for (int k = 0; k < QK_K / 128; ++k) {
        for (int n = 0; n < TILE_N; ++n) {
            bytes_from_nibbles_128(v[n], v[n + 16], B[n * KB].ql + k * 64, B[n * KB].qh + k * 32);
        }

        // top half: group 0,1 or 4,5; bottom half: group 2,3 or 6,7
        transpose_16x16_32bit(v);
        transpose_16x16_32bit(v + 16);

        // 1. pack lower 4bits with 4 groups
        for (int n = 0; n < 32; n += 2) {
            const __m512i r0 = _mm512_and_si512(v[n], lowMask);
            const __m512i r1 = _mm512_and_si512(v[n + 1], lowMask);
            _mm512_storeu_si512((__m512i *)pb, packNibbles(r0, r1)); pb += 64;
        }

        // 2. pack higher 2bit with 4 groups
        const __m512i hmask = _mm512_set1_epi8(0x30);
        for (int g = 0; g < 8; ++g) {
            __m512i hbits = _mm512_setzero_si512();
            hbits = _mm512_add_epi8(hbits, _mm512_srli_epi16(_mm512_and_si512(v[g * 4 + 0], hmask), 4));
            hbits = _mm512_add_epi8(hbits, _mm512_srli_epi16(_mm512_and_si512(v[g * 4 + 1], hmask), 2));
            hbits = _mm512_add_epi8(hbits,                   _mm512_and_si512(v[g * 4 + 2], hmask)    );
            hbits = _mm512_add_epi8(hbits, _mm512_slli_epi16(_mm512_and_si512(v[g * 4 + 3], hmask), 2));
            _mm512_storeu_si512((__m512i *)ph, hbits); ph += 64;
        }
    }
}

template <>
inline void pack_qs<block_iq4_xs>(void * RESTRICT packed_B, const block_iq4_xs * RESTRICT B, int KB) {
    __m512i v[16];
    char * pb = (char *)packed_B;
    for (int k = 0; k < QK_K / 64; ++k) {
        for (int n = 0; n < TILE_N; ++n) {
            __m256i r0 = bytes_from_nibbles_32(B[n * KB].qs + k * 32 +  0);
            __m256i r1 = bytes_from_nibbles_32(B[n * KB].qs + k * 32 + 16);
            v[n] = _mm512_inserti32x8(_mm512_castsi256_si512(r0), r1, 1);
        }

        transpose_16x16_32bit(v);

        // pack again with 128 to fully utilize vector length
        for (int n = 0; n < TILE_N; n += 2) {
            _mm512_storeu_si512((__m512i *)pb, packNibbles(v[n], v[n + 1]));
            pb += 64;
        }
    }
}

// pack B to vnni formats in 4bits or 8 bits
void pack_B(void * RESTRICT packed_B, const block_q4_0 * RESTRICT B, int KB) {
    pack_qs(packed_B, B, KB);
    ggml_half * d0 = reinterpret_cast<ggml_half *>((char *)packed_B + TILE_N * TILE_K / 2);
    for (int n = 0; n < TILE_N; ++n) {
        d0[n] = B[n * KB].d;
    }
}

void pack_B(void * RESTRICT packed_B, const block_q4_1 * RESTRICT B, int KB) {
    pack_qs(packed_B, B, KB);
    ggml_half * d0 = reinterpret_cast<ggml_half *>((char *)packed_B + TILE_N * TILE_K / 2);
    ggml_half * m0 = d0 + TILE_N;
    for (int n = 0; n < TILE_N; ++n) {
        d0[n] = B[n * KB].d;
        m0[n] = B[n * KB].m;
    }
}

inline void s8s8_compensation(void * RESTRICT packed_B) {
    // packed_B layout:
    //   quants {TILE_N, TILEK}  int8_t
    //   d0     {TILE_N}      ggml_half
    //   comp   {TILE_N}        int32_t
    const int offset = TILE_N * TILE_K + TILE_N * sizeof(ggml_half);
    __m512i vcomp = _mm512_setzero_si512();
    const __m512i off = _mm512_set1_epi8(static_cast<char>(0x80));
    for (int k = 0; k < 8; ++k) {
        __m512i vb = _mm512_loadu_si512((const __m512i *)((const char *)packed_B + k * 64));
        vcomp = _mm512_dpbusd_epi32(vcomp, off, vb);
    }
    _mm512_storeu_si512((__m512i *)((char *)(packed_B) + offset), vcomp);
}

void pack_B(void * RESTRICT packed_B, const block_q8_0 * RESTRICT B, int KB) {
    pack_qs(packed_B, B, KB);
    ggml_half * d0 = reinterpret_cast<ggml_half *>((char *)packed_B + TILE_N * TILE_K);
    for (int n = 0; n < TILE_N; ++n) {
        d0[n] = B[n * KB].d;
    }
    s8s8_compensation(packed_B);
}

// convert 8 * {min, scale} from int6 to int8
inline void unpack_mins_and_scales(const uint8_t * scales, uint32_t * utmp) {
    const uint32_t kmask1 = 0x3f3f3f3f;
    const uint32_t kmask2 = 0x0f0f0f0f;
    const uint32_t kmask3 = 0x03030303;

    memcpy(utmp, scales, 12);
    utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
    const uint32_t uaux = utmp[1] & kmask1;
    utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
    utmp[2] = uaux;
    utmp[0] &= kmask1;
}

// packed_B layout:
//   quants {8, TILE_N, 16}  uint8
//   scales {8, TILE_N}      uint8
//   mins   {8, TILE_N}      uint8
//   d      {TILE_N}     ggml_half
//   dmin   {TILE_N}     ggml_half
void pack_B(void * RESTRICT packed_B, const block_q4_K * RESTRICT B, int KB) {
    pack_qs(packed_B, B, KB);

    uint8_t * scales = reinterpret_cast<uint8_t *>((char *)packed_B + (QK_K / 2) * TILE_N);
    uint8_t * mins = scales + 8 * TILE_N;
    ggml_half * d = reinterpret_cast<ggml_half *>(mins + 8 * TILE_N);
    ggml_half * dmin = d + TILE_N;

    union {
        uint32_t u32[4];
        uint8_t  u8[16];
    } s;

    for (int n = 0; n < TILE_N; ++n) {
        unpack_mins_and_scales(B[n * KB].scales, s.u32);
        for (int k = 0; k < 8; ++k) {
            scales[k * TILE_N + n] = s.u8[k];
            mins[(k >> 1) * TILE_N * 2 + n * 2 + (k & 0x1)] = s.u8[k + 8];
        }
        d[n] = B[n * KB].d;
        dmin[n] = B[n * KB].dmin;
    }
}

// packed_B layout:
//   quants {8, TILE_N, 16}  uint8
//   qh     {8, TILE_N,  4}  uint8
//   scales {8, TILE_N}      uint8
//   mins   {8, TILE_N}      uint8
//   d      {TILE_N}     ggml_half
//   dmin   {TILE_N}     ggml_half
void pack_B(void * RESTRICT packed_B, const block_q5_K * RESTRICT B, int KB) {
    pack_qs(packed_B, B, KB);

    uint8_t * scales = reinterpret_cast<uint8_t *>((char *)packed_B + (QK_K / 2) * TILE_N + (QK_K / 8) * TILE_N);
    uint8_t * mins = scales + 8 * TILE_N;
    ggml_half * d = reinterpret_cast<ggml_half *>(mins + 8 * TILE_N);
    ggml_half * dmin = d + TILE_N;

    union {
        uint32_t u32[4];
        uint8_t  u8[16];
    } s;

    for (int n = 0; n < TILE_N; ++n) {
        unpack_mins_and_scales(B[n * KB].scales, s.u32);
        for (int k = 0; k < 8; ++k) {
            scales[k * TILE_N + n] = s.u8[k];
            mins[(k >> 1) * TILE_N * 2 + n * 2 + (k & 0x1)] = s.u8[k + 8];
        }
        d[n] = B[n * KB].d;
        dmin[n] = B[n * KB].dmin;
    }
}

// packed_B layout:
//   quants {16, TILE_N, 8}  uint8
//   qh     {16, TILE_N, 4}  uint8
//   scales {16, TILE_N}      uint8
//   d      {TILE_N}     ggml_half
void pack_B(void * RESTRICT packed_B, const block_q6_K * RESTRICT B, int KB) {
    pack_qs(packed_B, B, KB);

    uint8_t * scales = reinterpret_cast<uint8_t *>((char *)packed_B + (QK_K / 2) * TILE_N + (QK_K / 4) * TILE_N);
    ggml_half * d = reinterpret_cast<ggml_half *>(scales + 16 * TILE_N);
    for (int n = 0; n < TILE_N; ++n) {
        const int8_t * ps = B[n * KB].scales;
        for (int k = 0; k < 16; ++k) {
            scales[k * TILE_N + n] = ps[k];
        }
        d[n] = B[n * KB].d;
    }
}

// packed_B layout:
//   quants {8, TILE_N, 16}  uint8
//   scales {8, TILE_N}       int8
//   d      {TILE_N}     ggml_half
void pack_B(void * RESTRICT packed_B, const block_iq4_xs * RESTRICT B, int KB) {
    pack_qs(packed_B, B, KB);

    int8_t * scales = reinterpret_cast<int8_t *>((char *)packed_B + (QK_K / 2) * TILE_N);
    ggml_half * d = reinterpret_cast<ggml_half *>(scales + 8 * TILE_N);

    // pack the scales
    for (int n = 0; n < TILE_N; ++n) {
        uint16_t sh = B[n * KB].scales_h;
        for (int k = 0; k < 8; k += 2) {
            const int16_t ls1 = ((B[n * KB].scales_l[k / 2] & 0xf) | ((sh << 4) & 0x30)) - 32;
            const int16_t ls2 = ((B[n * KB].scales_l[k / 2] >>  4) | ((sh << 2) & 0x30)) - 32;
            scales[(k + 0) * TILE_N + n] = ls1;
            scales[(k + 1) * TILE_N + n] = ls2;
            sh >>= 4;
        }
        d[n] = B[n * KB].d;
    }
}

template<typename TB, typename packed_B_t = packed_B_type<TB>>
void unpack_B(packed_B_t * RESTRICT tile, const void * RESTRICT packed_B) {
    GGML_UNUSED(tile);
    GGML_UNUSED(packed_B);
}

template <>
void unpack_B<block_q4_0>(int8_t * RESTRICT tile, const void * RESTRICT packed_B) {
  const __m512i off = _mm512_set1_epi8(8);
  const __m512i lowMask = _mm512_set1_epi8(0xF);
  for (int n = 0; n < 8; n += 2) {
    __m512i bytes = _mm512_loadu_si512((const __m512i *)((const char *)packed_B + n * 32));
    const __m512i r0 = _mm512_sub_epi8(_mm512_and_si512(bytes, lowMask), off);
    const __m512i r1 = _mm512_sub_epi8(_mm512_and_si512(_mm512_srli_epi16(bytes, 4), lowMask), off);
    _mm512_storeu_si512((__m512i *)(tile + n * 64 +  0), r0);
    _mm512_storeu_si512((__m512i *)(tile + n * 64 + 64), r1);
  }
}

template <>
void unpack_B<block_q4_1>(uint8_t * RESTRICT tile, const void * RESTRICT packed_B) {
    const __m512i lowMask = _mm512_set1_epi8(0xF);
    for (int n = 0; n < 8; n += 2) {
        __m512i bytes = _mm512_loadu_si512((const __m512i *)((const char *)packed_B + n * 32));
        const __m512i r0 = _mm512_and_si512(bytes, lowMask);
        const __m512i r1 = _mm512_and_si512(_mm512_srli_epi16(bytes, 4), lowMask);
        _mm512_storeu_si512((__m512i *)(tile + n * 64 +  0), r0);
        _mm512_storeu_si512((__m512i *)(tile + n * 64 + 64), r1);
    }
}

// packed_B_t for QKK is int8_t
template <typename TB>
void unpack_B(int8_t * RESTRICT tile, const void * RESTRICT packed_B, int k) {
    const int packed_B_group_size = QK_K / 2 * TILE_N / 8;
    const char * packed_B_group = (const char *)packed_B + k * packed_B_group_size;
    const __m512i lowMask = _mm512_set1_epi8(0xF);
    for (int n = 0; n < 8; n += 2) {
        __m512i bytes = _mm512_loadu_si512(packed_B_group + n * 32);
        const __m512i r0 = _mm512_and_si512(bytes, lowMask);
        const __m512i r1 = _mm512_and_si512(_mm512_srli_epi16(bytes, 4), lowMask);
        _mm512_storeu_si512((__m512i *)(tile + n * 64 +  0), r0);
        _mm512_storeu_si512((__m512i *)(tile + n * 64 + 64), r1);
    }
}

template <>
void unpack_B<block_q5_K>(int8_t * RESTRICT tile, const void * RESTRICT packed_B, int k) {
    // lower 4bits, stride 256 bytes
    const int packed_l4_group_size = QK_K / 2 * TILE_N / 8;
    const char * pb = (const char *)packed_B + k * packed_l4_group_size;

    // higher 1bit, stride 64 bytes
    const int packed_h1_group_size = QK_K / 8 * TILE_N / 8;
    const char * ph = (const char *)packed_B + (QK_K / 2) * TILE_N + k * packed_h1_group_size;
    const __m512i hbits = _mm512_loadu_si512(ph);

    const __m512i lowMask = _mm512_set1_epi8(0xF);
    __m512i hmask0 = _mm512_set1_epi8(0x1);
    __m512i hmask1 = _mm512_set1_epi8(0x2);

    for (int n = 0; n < 8; n += 2) {
        __m512i bytes = _mm512_loadu_si512(pb + n * 32);
        __m512i r0 = _mm512_and_si512(bytes, lowMask);
        __m512i r1 = _mm512_and_si512(_mm512_srli_epi16(bytes, 4), lowMask);
        __m512i h0 = _mm512_slli_epi16(_mm512_srli_epi16(_mm512_and_si512(hbits, hmask0), n), 4);
        __m512i h1 = _mm512_slli_epi16(_mm512_srli_epi16(_mm512_and_si512(hbits, hmask1), n + 1), 4);

        hmask0 = _mm512_slli_epi16(hmask0, 2);
        hmask1 = _mm512_slli_epi16(hmask1, 2);
        r0 = _mm512_add_epi8(r0, h0);
        r1 = _mm512_add_epi8(r1, h1);
        _mm512_storeu_si512((__m512i *)(tile + n * 64 +  0), r0);
        _mm512_storeu_si512((__m512i *)(tile + n * 64 + 64), r1);
    }
}

template <>
void unpack_B<block_q6_K>(int8_t * RESTRICT tile, const void * RESTRICT packed_B, int k) {
    // lower 4bits, stride 128 bytes
    const int packed_l4_group_size = QK_K / 2 * TILE_N / 16;
    const char * pb = (const char *)packed_B + k * packed_l4_group_size;

    // higher 2bits, stride 64 bytes
    const int packed_h2_group_size = QK_K / 4 * TILE_N / 16;
    const char * ph = (const char *)packed_B + (QK_K / 2) * TILE_N + k * packed_h2_group_size;
    const __m512i hbits = _mm512_loadu_si512(ph);

    const __m512i off = _mm512_set1_epi8(32);
    const __m512i lowMask = _mm512_set1_epi8(0xF);
    __m512i hmask0 = _mm512_set1_epi8(0x3); // 0011
    __m512i hmask1 = _mm512_set1_epi8(0xC); // 1100

    // notes: skip zero padding from row4 to row7 as we have done so in `unpack_A`
    __m512i bytes = _mm512_loadu_si512(pb);
    __m512i r0 = _mm512_and_si512(bytes, lowMask);
    __m512i r1 = _mm512_and_si512(_mm512_srli_epi16(bytes, 4), lowMask);
    __m512i h0 = _mm512_slli_epi16(_mm512_and_si512(hbits, hmask0), 4);
    __m512i h1 = _mm512_slli_epi16(_mm512_and_si512(hbits, hmask1), 2);
    _mm512_storeu_si512((__m512i *)(tile +  0), _mm512_sub_epi8(_mm512_add_epi8(r0, h0), off));
    _mm512_storeu_si512((__m512i *)(tile + 64), _mm512_sub_epi8(_mm512_add_epi8(r1, h1), off));

    hmask0 = _mm512_slli_epi16(hmask0, 4);
    hmask1 = _mm512_slli_epi16(hmask1, 4);

    bytes = _mm512_loadu_si512(pb + 64);
    r0 = _mm512_and_si512(bytes, lowMask);
    r1 = _mm512_and_si512(_mm512_srli_epi16(bytes, 4), lowMask);
    h0 =                   _mm512_and_si512(hbits, hmask0);
    h1 = _mm512_srli_epi16(_mm512_and_si512(hbits, hmask1), 2);
    _mm512_storeu_si512((__m512i *)(tile + 128), _mm512_sub_epi8(_mm512_add_epi8(r0, h0), off));
    _mm512_storeu_si512((__m512i *)(tile + 192), _mm512_sub_epi8(_mm512_add_epi8(r1, h1), off));
}

template <>
void unpack_B<block_iq4_xs>(int8_t * RESTRICT tile, const void * RESTRICT packed_B, int k) {
    static const __m512i values128 = _mm512_set_epi8(
        113, 89, 69, 53, 38, 25, 13, 1, -10, -22, -35, -49, -65, -83, -104, -127,
        113, 89, 69, 53, 38, 25, 13, 1, -10, -22, -35, -49, -65, -83, -104, -127,
        113, 89, 69, 53, 38, 25, 13, 1, -10, -22, -35, -49, -65, -83, -104, -127,
        113, 89, 69, 53, 38, 25, 13, 1, -10, -22, -35, -49, -65, -83, -104, -127
    );

    const int packed_B_group_size = QK_K / 2 * TILE_N / 8;
    const char * pb = (const char *)packed_B + k * packed_B_group_size;
    const __m512i lowMask = _mm512_set1_epi8(0xF);

    for (int n = 0; n < 8; n += 2) {
        __m512i bytes = _mm512_loadu_si512(pb + n * 32);
        const __m512i r0 = _mm512_shuffle_epi8(values128, _mm512_and_si512(bytes, lowMask));
        const __m512i r1 = _mm512_shuffle_epi8(values128, _mm512_and_si512(_mm512_srli_epi16(bytes, 4), lowMask));
        _mm512_storeu_si512((__m512i *)(tile + n * 64 +  0), r0);
        _mm512_storeu_si512((__m512i *)(tile + n * 64 + 64), r1);
    }
}

template <typename TA, typename TB, bool is_acc>
struct acc_C {};

template <bool is_acc>
struct acc_C<block_q8_0, block_q4_0, is_acc> {
    static void apply(float * RESTRICT C, int ldc, const int32_t * RESTRICT tile, const block_q8_0 * A, int lda, const void * packed_B, int nr) {
        const int offset = TILE_N * TILE_K / 2;
        const __m512 vd0 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *)((const char *)packed_B + offset)));

        for (int m = 0; m < nr; ++m) {
            const __m512 vd1 = _mm512_set1_ps(GGML_CPU_FP16_TO_FP32(A[m * lda].d));
            const __m512 vtile = _mm512_cvtepi32_ps(_mm512_loadu_si512(tile + m * TILE_N));

            __m512 vsum;
            if (is_acc) {
                vsum = _mm512_loadu_ps(C + m * ldc);
            } else {
                vsum = _mm512_set1_ps(0.f);
            }
            vsum = _mm512_fmadd_ps(vtile, _mm512_mul_ps(vd0, vd1), vsum);
            _mm512_storeu_ps(C + m * ldc, vsum);
        }
    }
};

template <bool is_acc>
struct acc_C<block_q8_1, block_q4_1, is_acc> {
    static void apply(float * RESTRICT C, int ldc, const int32_t * RESTRICT tile, const block_q8_1 * A, int lda, const void * packed_B, int nr) {
        const int offset = TILE_N * TILE_K / 2;
        const __m512 vd0 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *)((const char *)packed_B + offset)));
        const __m512 vm0 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *)((const char *)packed_B + offset + TILE_N * sizeof(ggml_half))));

        for (int m = 0; m < nr; ++m) {
            const __m512 vd1 = _mm512_set1_ps(GGML_CPU_FP16_TO_FP32(A[m * lda].d));
            const __m512 vs1 = _mm512_set1_ps(GGML_CPU_FP16_TO_FP32(A[m * lda].s));
            const __m512 vtile = _mm512_cvtepi32_ps(_mm512_loadu_si512(tile + m * TILE_N));

            __m512 vsum;
            if (is_acc) {
                vsum = _mm512_loadu_ps(C + m * ldc);
            } else {
                vsum = _mm512_set1_ps(0.f);
            }
            vsum = _mm512_fmadd_ps(vtile, _mm512_mul_ps(vd0, vd1), vsum);
            vsum = _mm512_fmadd_ps(vm0, vs1, vsum);
            _mm512_storeu_ps(C + m * ldc, vsum);
        }
    }
};

template <bool is_acc>
struct acc_C<block_q8_0, block_q8_0, is_acc> {
    static void apply(float * RESTRICT C, int ldc, const int32_t * RESTRICT tile, const block_q8_0 * A, int lda, const void * packed_B, int nr) {
        const int offset = TILE_N * TILE_K;
        const __m512 vd0 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *)((const char *)packed_B + offset)));

        for (int m = 0; m < nr; ++m) {
            const __m512 vd1 = _mm512_set1_ps(GGML_CPU_FP16_TO_FP32(A[m * lda].d));
            const __m512 vtile = _mm512_cvtepi32_ps(_mm512_loadu_si512(tile + m * TILE_N));

            __m512 vsum;
            if (is_acc) {
                vsum = _mm512_loadu_ps(C + m * ldc);
            } else {
                vsum = _mm512_set1_ps(0.f);
            }
            vsum = _mm512_fmadd_ps(vtile, _mm512_mul_ps(vd0, vd1), vsum);
            _mm512_storeu_ps(C + m * ldc, vsum);
        }
    }
};

template <bool is_acc>
struct acc_C<block_q8_K, block_q4_K, is_acc> {
    static void apply(float * RESTRICT C, int ldc, const int32_t * RESTRICT tile, const block_q8_K * A, int lda, const void * packed_B, int nr) {
        const uint8_t * scales = reinterpret_cast<const uint8_t *>((const char *)packed_B + (QK_K / 2) * TILE_N);
        const uint8_t * mins = scales + 8 * TILE_N;
        const ggml_half * d0 = reinterpret_cast<const ggml_half *>(mins + 8 * TILE_N);
        const ggml_half * dmin = d0 + TILE_N;

        const __m512 vd0 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *)d0));
        const __m512 vdmin = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *)dmin));

        for (int m = 0; m < nr; ++m) {
            const float d1 = A[m * lda].d;
            const __m512 vd = _mm512_mul_ps(_mm512_set1_ps(d1), vd0);
            const __m512 vdm = _mm512_mul_ps(_mm512_set1_ps(-d1), vdmin);
            const __m512 vtile = _mm512_cvtepi32_ps(_mm512_loadu_si512(tile + m * TILE_N));

            __m512 vsum;
            if (is_acc) {
                vsum = _mm512_loadu_ps(C + m * ldc);
            } else {
                vsum = _mm512_set1_ps(0.f);
            }

            const __m256i q8sums = _mm256_loadu_si256((const __m256i *)A[m * lda].bsums);
            const __m128i q8s = _mm_hadd_epi16(_mm256_extracti128_si256(q8sums, 0), _mm256_extracti128_si256(q8sums, 1));

            __m512i acc_m = _mm512_setzero_si512();
            for (int k = 0; k < 4; ++k) {
                __m512i vmask = _mm512_set1_epi32(k);
                __m512i va = _mm512_permutexvar_epi32(vmask, _mm512_castsi128_si512(q8s));
                __m512i vb = _mm512_cvtepi8_epi16(_mm256_loadu_si256((const __m256i *)(mins + k * 32)));
                acc_m = _mm512_dpwssds_epi32(acc_m, va, vb);
            }

            vsum = _mm512_fmadd_ps(vtile, vd, vsum);
            vsum = _mm512_fmadd_ps(_mm512_cvtepi32_ps(acc_m), vdm, vsum);
            _mm512_storeu_ps(C + m * ldc, vsum);
        }
    }
};

template <bool is_acc>
struct acc_C<block_q8_K, block_q5_K, is_acc> {
    static void apply(float * RESTRICT C, int ldc, const int32_t * RESTRICT tile, const block_q8_K * A, int lda, const void * packed_B, int nr) {
        const uint8_t * scales = reinterpret_cast<const uint8_t *>((const char *)packed_B + (QK_K / 2) * TILE_N + (QK_K / 8) * TILE_N);
        const uint8_t * mins = scales + 8 * TILE_N;
        const ggml_half * d0 = reinterpret_cast<const ggml_half *>(mins + 8 * TILE_N);
        const ggml_half * dmin = d0 + TILE_N;

        const __m512 vd0 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *)d0));
        const __m512 vdmin = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *)dmin));

        for (int m = 0; m < nr; ++m) {
            const float d1 = A[m * lda].d;
            const __m512 vd = _mm512_mul_ps(_mm512_set1_ps(d1), vd0);
            const __m512 vdm = _mm512_mul_ps(_mm512_set1_ps(-d1), vdmin);
            const __m512 vtile = _mm512_cvtepi32_ps(_mm512_loadu_si512(tile + m * TILE_N));

            __m512 vsum;
            if (is_acc) {
                vsum = _mm512_loadu_ps(C + m * ldc);
            } else {
                vsum = _mm512_set1_ps(0.f);
            }

            const __m256i q8sums = _mm256_loadu_si256((const __m256i *)A[m * lda].bsums);
            const __m128i q8s = _mm_hadd_epi16(_mm256_extracti128_si256(q8sums, 0), _mm256_extracti128_si256(q8sums, 1));

            __m512i acc_m = _mm512_setzero_si512();
            for (int k = 0; k < 4; ++k) {
                __m512i vmask = _mm512_set1_epi32(k);
                __m512i va = _mm512_permutexvar_epi32(vmask, _mm512_castsi128_si512(q8s));
                __m512i vb = _mm512_cvtepi8_epi16(_mm256_loadu_si256((const __m256i *)(mins + k * 32)));
                acc_m = _mm512_dpwssds_epi32(acc_m, va, vb);
            }

            vsum = _mm512_fmadd_ps(vtile, vd, vsum);
            vsum = _mm512_fmadd_ps(_mm512_cvtepi32_ps(acc_m), vdm, vsum);
            _mm512_storeu_ps(C + m * ldc, vsum);
        }
    }
};

template <bool is_acc>
struct acc_C<block_q8_K, block_q6_K, is_acc> {
    static void apply(float * RESTRICT C, int ldc, const int32_t * RESTRICT tile, const block_q8_K * A, int lda, const void * packed_B, int nr) {
        const uint8_t * scales = reinterpret_cast<const uint8_t *>((const char *)packed_B + (QK_K / 2) * TILE_N + (QK_K / 4) * TILE_N);
        const ggml_half * d0 = reinterpret_cast<const ggml_half *>(scales + 16 * TILE_N);

        const __m512 vd0 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *)d0));

        for (int m = 0; m < nr; ++m) {
            const float d1 = A[m * lda].d;
            const __m512 vd = _mm512_mul_ps(_mm512_set1_ps(d1), vd0);
            const __m512 vtile = _mm512_cvtepi32_ps(_mm512_loadu_si512(tile + m * TILE_N));

            __m512 vsum;
            if (is_acc) {
                vsum = _mm512_loadu_ps(C + m * ldc);
            } else {
                vsum = _mm512_set1_ps(0.f);
            }

            vsum = _mm512_fmadd_ps(vtile, vd, vsum);
            _mm512_storeu_ps(C + m * ldc, vsum);
        }
    }
};

template <bool is_acc>
struct acc_C<block_q8_K, block_iq4_xs, is_acc> {
    static void apply(float * RESTRICT C, int ldc, const int32_t * RESTRICT tile, const block_q8_K * A, int lda, const void * packed_B, int nr) {
        const int8_t * scales = reinterpret_cast<const int8_t *>((const char *)packed_B + (QK_K / 2) * TILE_N);
        const ggml_half * d0 = reinterpret_cast<const ggml_half *>(scales + 8 * TILE_N);

        const __m512 vd0 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *)d0));

        for (int m = 0; m < nr; ++m) {
            const float d1 = A[m * lda].d;
            const __m512 vd = _mm512_mul_ps(_mm512_set1_ps(d1), vd0);
            const __m512 vtile = _mm512_cvtepi32_ps(_mm512_loadu_si512(tile + m * TILE_N));

            __m512 vsum;
            if (is_acc) {
                vsum = _mm512_loadu_ps(C + m * ldc);
            } else {
                vsum = _mm512_set1_ps(0.f);
            }

            vsum = _mm512_fmadd_ps(vtile, vd, vsum);
            _mm512_storeu_ps(C + m * ldc, vsum);
        }
    }
};

template <typename TB> constexpr int get_quants_size();
template <> constexpr int get_quants_size<block_q4_K>() { return (QK_K / 2) * TILE_N; }
template <> constexpr int get_quants_size<block_q5_K>() { return (QK_K / 2) * TILE_N + (QK_K / 8) * TILE_N; }
template <> constexpr int get_quants_size<block_q6_K>() { return (QK_K / 2) * TILE_N + (QK_K / 4) * TILE_N; }
template <> constexpr int get_quants_size<block_iq4_xs>() { return (QK_K / 2) * TILE_N; }

// used for QKK format
template <typename TB, bool is_acc,
          typename std::enable_if<is_type_qkk<TB>::value, int>::type = 0>
inline void scale_C(const int32_t * RESTRICT tile, int32_t * RESTRICT sumi, const void * packed_B, int k, int nr) {
    const uint8_t * scales = reinterpret_cast<const uint8_t *>((const char *)packed_B + get_quants_size<TB>());
    const __m512i vscale = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i *)(scales + k * TILE_N)));

    for (int m = 0; m < nr; ++m) {
        __m512i vsumi;
        if (is_acc) {
            vsumi = _mm512_loadu_si512(sumi + m * TILE_N);
        } else {
            vsumi = _mm512_setzero_si512();
        }
        __m512i vtile = _mm512_loadu_si512(tile + m * TILE_N);
        vsumi = _mm512_add_epi32(vsumi, _mm512_mullo_epi32(vtile, vscale));
        _mm512_storeu_si512((__m512i *)(sumi + m * TILE_N), vsumi);
    }
}

template <typename TA, typename TB, typename TC, int BLOCK_M, int BLOCK_N, int BLOCK_K>
struct tinygemm_kernel_avx {
    static void apply(int K, const TA * RESTRICT A, const TB * RESTRICT B, TC * RESTRICT C, int ldc) {
        GGML_UNUSED(K);
        GGML_UNUSED(A);
        GGML_UNUSED(B);
        GGML_UNUSED(C);
        GGML_UNUSED(ldc);
    }
};

template <int BLOCK_M, int BLOCK_N, int BLOCK_K>
struct tinygemm_kernel_avx<float, ggml_fp16_t, float, BLOCK_M, BLOCK_N, BLOCK_K> {
    static void apply(int K, const float * RESTRICT A, const ggml_fp16_t * RESTRICT B, float * RESTRICT C, int ldc) {
        constexpr int ROWS = BLOCK_M;
        constexpr int COLS = BLOCK_N;
        assert(BLOCK_K == 16);

        __m512 va;
        __m512 vb[COLS];
        __m512 vc[ROWS * COLS];

        auto loadc = [&](auto idx) {
            vc[idx] = _mm512_setzero_ps();
        };
        Unroll<ROWS * COLS>{}(loadc);

        auto compute = [&](auto idx, auto k) {
            constexpr int row = idx / COLS;
            constexpr int col = idx % COLS;

            if constexpr (col == 0) {
                va = _mm512_loadu_ps(A + row * K + k);
            }
            if constexpr (row == 0) {
                vb[col] =  _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *)(B + col * K + k)));
            }
            vc[idx] = _mm512_fmadd_ps(va, vb[col], vc[idx]);
        };

        for (int k = 0; k < K; k += 16) {
            Unroll<ROWS * COLS>{}(compute, k);
        }

        auto storec = [&](auto idx) {
            constexpr int row = idx / COLS;
            constexpr int col = idx % COLS;
            C[row * ldc + col] = _mm512_reduce_add_ps(vc[idx]);
        };
        Unroll<ROWS * COLS>{}(storec);
    }
};

#define LAUNCH_TINYGEMM_KERNEL_AVX(MB_SIZE, NB_SIZE)                                \
    tinygemm_kernel_avx<float, type, float, MB_SIZE, NB_SIZE, blck_size>::apply(    \
        K, (const float *)src1->data + src1_offset + mb_start * K,                  \
        (const type *)src0->data + src0_offset + nb_start * K,                      \
        (float *)dst->data + dst_offset + mb_start * ldc + nb_start, ldc)


// re-organize in the format {NB, KB, TILE_SIZE}:
#define PACKED_INDEX(n, k, KB, tile_size) (n * KB + k) * tile_size

template<typename TB, int BLOCK_K>
void convert_B_packed_format(void * RESTRICT packed_B, const TB * RESTRICT B, int N, int K) {
    const int NB = N / TILE_N;
    const int KB = K / BLOCK_K;
    const int TILE_SIZE = get_tile_size<TB>();

    // parallel on NB should be enough
    parallel_for(NB, [&](int begin, int end) {
        for (int n = begin; n < end; ++n) {
            for (int k = 0; k < KB; ++k) {
                int n0 = n * TILE_N;
                pack_B((char *)packed_B + PACKED_INDEX(n, k, KB, TILE_SIZE), &B[n0 * KB + k], KB);
            }
        }
    });
}

template <typename TA, typename TB, typename TC, int BLOCK_M, int BLOCK_N, int BLOCK_K>
struct tinygemm_kernel_vnni {};

template <int BLOCK_M, int BLOCK_N, int BLOCK_K>
struct tinygemm_kernel_vnni<block_q8_0, block_q4_0, float, BLOCK_M, BLOCK_N, BLOCK_K> {
    static void apply(int KB, const void * RESTRICT _A, const void * RESTRICT _B, float * RESTRICT C, int ldc) {

        constexpr int COLS = BLOCK_N / 16;
        const int TILE_SIZE = TILE_N * sizeof(block_q4_0);

        const block_q8_0 * RESTRICT A = static_cast<const block_q8_0 *>(_A);
        const char * RESTRICT B = static_cast<const char *>(_B);

        __m512i va[8];
        __m512 vc[COLS];
        __m512 vd1;

        // sum of offsets, shared across COLS
        //
        // avx512-vnni does not have `_mm512_dpbssd_epi32`,
        // need to transform ss to us:
        //   a * (b - 8) is equivalent to b * a - 8 * a
        //   s    u   u                   u   s   u   s
        //
        __m512i vcomp;

        const __m512i off = _mm512_set1_epi8(8);
        const __m512i lowMask = _mm512_set1_epi8(0xF);

        auto loadc = [&](auto col) {
            vc[col] = _mm512_setzero_ps();
        };
        Unroll<COLS>{}(loadc);

        auto compute = [&](auto col, auto i) {
            // load a and compute compensation
            if constexpr (col == 0) {
                const int32_t * a_ptr = reinterpret_cast<const int32_t *>(A[0 * KB + i].qs);
                vcomp = _mm512_setzero_si512();
                for (int k = 0; k < 8; ++k) {
                    va[k] = _mm512_set1_epi32(a_ptr[k]);
                    vcomp = _mm512_dpbusd_epi32(vcomp, off, va[k]);
                }
                vd1 = _mm512_set1_ps(GGML_CPU_FP16_TO_FP32(A[0 * KB + i].d));
            }

            // load b
            __m512i vsum = _mm512_setzero_si512();
            const char * b_ptr = B + PACKED_INDEX(col, i, KB, TILE_SIZE);
            for (int k = 0; k < 8; k += 2) {
                __m512i bytes = _mm512_loadu_si512((const __m512i *)(b_ptr + k * 32));
                __m512i vb0 = _mm512_and_si512(bytes, lowMask);
                vsum = _mm512_dpbusd_epi32(vsum, vb0, va[k + 0]);
                __m512i vb1 = _mm512_and_si512(_mm512_srli_epi16(bytes, 4), lowMask);
                vsum = _mm512_dpbusd_epi32(vsum, vb1, va[k + 1]);
            }
            const int offset = TILE_N * TILE_K / 2;
            const __m512 vd0 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *)(b_ptr + offset)));
            vsum = _mm512_sub_epi32(vsum, vcomp);

            vc[col] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(vsum), _mm512_mul_ps(vd0, vd1), vc[col]);
        };

        for (int i = 0; i < KB; ++i) {
            Unroll<COLS>{}(compute, i);
        }

        //store to C
        auto storec = [&](auto col) {
            _mm512_storeu_ps((__m512i*)(C + 0 * ldc + col * 16), vc[col]);
        };
        Unroll<COLS>{}(storec);
    }
};

template <int BLOCK_N, int BLOCK_K>
struct tinygemm_kernel_vnni<block_q8_1, block_q4_1, float, 1, BLOCK_N, BLOCK_K> {
    static void apply(int KB, const void * RESTRICT _A, const void * RESTRICT _B, float * RESTRICT C, int ldc) {

        constexpr int COLS = BLOCK_N / 16;
        const int TILE_SIZE = TILE_N * sizeof(block_q4_1);

        const block_q8_1 * RESTRICT A = static_cast<const block_q8_1 *>(_A);
        const char * RESTRICT B = static_cast<const char *>(_B);

        __m512i va[8];
        __m512i vb[8];
        __m512 vc[COLS];
        __m512 vd1, vs1;

        const __m512i lowMask = _mm512_set1_epi8(0xF);

        auto loadc = [&](auto col) {
            vc[col] = _mm512_setzero_ps();
        };
        Unroll<COLS>{}(loadc);

        auto compute = [&](auto col, auto i) {
            // load a
            if constexpr (col == 0) {
                const int32_t * a_ptr = reinterpret_cast<const int32_t *>(A[0 * KB + i].qs);
                for (int k = 0; k < 8; ++k) {
                    va[k] = _mm512_set1_epi32(a_ptr[k]);
                }
                vd1 = _mm512_set1_ps(GGML_CPU_FP16_TO_FP32(A[0 * KB + i].d));
                vs1 = _mm512_set1_ps(GGML_CPU_FP16_TO_FP32(A[0 * KB + i].s));
            }

            // load b
            const char * b_ptr = B + PACKED_INDEX(col, i, KB, TILE_SIZE);
            for (int k = 0; k < 8; k += 2) {
                __m512i bytes = _mm512_loadu_si512((const __m512i *)(b_ptr + k * 32));
                vb[k + 0] = _mm512_and_si512(bytes, lowMask);
                vb[k + 1] = _mm512_and_si512(_mm512_srli_epi16(bytes, 4), lowMask);
            }
            const int offset = TILE_N * TILE_K / 2;
            const __m512 vd0 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *)(b_ptr + offset)));
            const __m512 vm0 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *)(b_ptr + offset + TILE_N * sizeof(ggml_half))));

            __m512i vsum = _mm512_setzero_si512();
            for (int k = 0; k < 8; ++k) {
                vsum = _mm512_dpbusd_epi32(vsum, vb[k], va[k]);
            }

            vc[col] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(vsum), _mm512_mul_ps(vd0, vd1), vc[col]);
            vc[col] = _mm512_fmadd_ps(vm0, vs1, vc[col]);
        };

        for (int i = 0; i < KB; ++i) {
            Unroll<COLS>{}(compute, i);
        }

        //store to C
        auto storec = [&](auto col) {
            _mm512_storeu_ps((__m512i*)(C + 0 * ldc + col * 16), vc[col]);
        };
        Unroll<COLS>{}(storec);
    }
};

template <int BLOCK_M, int BLOCK_N, int BLOCK_K>
struct tinygemm_kernel_vnni<block_q8_0, block_q8_0, float, BLOCK_M, BLOCK_N, BLOCK_K> {
    static void apply(int KB, const void * RESTRICT _A, const void * RESTRICT _B, float * RESTRICT C, int ldc) {

        constexpr int COLS = BLOCK_N / 16;
        const int TILE_SIZE = TILE_N * sizeof(block_q8_0) + TILE_N * sizeof(int32_t);

        const block_q8_0 * RESTRICT A = static_cast<const block_q8_0 *>(_A);
        const char * RESTRICT B = static_cast<const char *>(_B);

        __m512i va[8];
        __m512i vb[8];
        __m512 vc[COLS];
        __m512 vd1;

        // Notes: s8s8 igemm compensation in avx512-vnni
        // change s8s8 to u8s8 with compensate
        //   a * b = (a + 128) * b - 128 * b
        //   s   s       u       s    u    s
        //
        // (128 * b is pre-computed when packing B to vnni formats)
        //
        const __m512i off = _mm512_set1_epi8(static_cast<char>(0x80));

        auto loadc = [&](auto col) {
            vc[col] = _mm512_setzero_ps();
        };
        Unroll<COLS>{}(loadc);

        auto compute = [&](auto col, auto i) {
            // load a and add offset 128
            if constexpr (col == 0) {
                const int32_t * a_ptr = reinterpret_cast<const int32_t *>(A[0 * KB + i].qs);
                for (int k = 0; k < 8; ++k) {
                    va[k] = _mm512_set1_epi32(a_ptr[k]);
                    va[k] = _mm512_add_epi8(va[k], off);
                }
                vd1 = _mm512_set1_ps(GGML_CPU_FP16_TO_FP32(A[0 * KB + i].d));
            }

            // load b
            const char * b_ptr = B + PACKED_INDEX(col, i, KB, TILE_SIZE);
            for (int k = 0; k < 8; ++k) {
                vb[k] = _mm512_loadu_si512((const __m512i *)(b_ptr + k * 64));
            }
            const int offset = TILE_N * TILE_K;
            const __m512 vd0 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *)(b_ptr + offset)));
            const int offset2 = TILE_N * TILE_K + TILE_N * sizeof(ggml_half);
            const __m512i vcomp = _mm512_loadu_si512((const __m512i *)(b_ptr + offset2));

            __m512i vsum = _mm512_setzero_si512();
            for (int k = 0; k < 8; ++k) {
                vsum = _mm512_dpbusd_epi32(vsum, va[k], vb[k]);
            }
            vsum = _mm512_sub_epi32(vsum, vcomp);

            vc[col] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(vsum), _mm512_mul_ps(vd0, vd1), vc[col]);
        };

        for (int i = 0; i < KB; ++i) {
            Unroll<COLS>{}(compute, i);
        }

        //store to C
        auto storec = [&](auto col) {
            _mm512_storeu_ps((__m512i*)(C + 0 * ldc + col * 16), vc[col]);
        };
        Unroll<COLS>{}(storec);
    }
};

template <int BLOCK_M, int BLOCK_N, int BLOCK_K>
struct tinygemm_kernel_vnni<block_q8_K, block_q4_K, float, BLOCK_M, BLOCK_N, BLOCK_K> {
    static void apply(int KB, const void * RESTRICT _A, const void * RESTRICT _B, float * RESTRICT C, int ldc) {

        constexpr int COLS = BLOCK_N / 16;
        const int TILE_SIZE = TILE_N * sizeof(block_q4_K) + TILE_N * 4;

        const block_q8_K * RESTRICT A = static_cast<const block_q8_K *>(_A);
        const char * RESTRICT B = static_cast<const char *>(_B);

        // a.qs:   8 groups, 32 bytes each group (m256i)
        __m512i va[8];
        // a.bsum: 8 groups,  2 bytes each group (m128i)
        __m512i va_bsum;
        __m512 vc[COLS];
        __m512 vd1;

        // packed_B:
        const int offset_scales = (QK_K / 2) * TILE_N;
        const int offset_mins   = (QK_K / 2) * TILE_N +  8 * TILE_N;
        const int offset_d0     = (QK_K / 2) * TILE_N + 16 * TILE_N;
        const int offset_dmin   = (QK_K / 2) * TILE_N + 16 * TILE_N + TILE_N * sizeof(ggml_half);

        const __m512i lowMask = _mm512_set1_epi8(0xF);

        auto loadc = [&](auto col) {
            vc[col] = _mm512_setzero_ps();
        };
        Unroll<COLS>{}(loadc);

        // Notes: vnni formats in QK_K
        //   a) quants vnni format
        //     int8  {k/4, n, 4}, viewed as 2d {k/4, 4n}, k = 32
        //     from {16, 32} to {8, 64}
        //
        //   b) min vnni format
        //     int16 {k/2, n, 2}, viewed as 2d {k/2, 2n}, k = 8
        //     from {16,  8} to {4, 32}
        //
        auto compute = [&](auto col, auto i) {
            // load a
            if constexpr (col == 0) {
                for (int k_group = 0; k_group < QK_K / 32; ++k_group) {
                    va[k_group] = _mm512_castsi256_si512(_mm256_loadu_si256((const __m256i *)(A[0 * KB + i].qs + k_group * 32)));
                }
                const __m256i q8sums = _mm256_loadu_si256((const __m256i *)A[0 * KB + i].bsums);
                const __m128i q8s = _mm_hadd_epi16(_mm256_extracti128_si256(q8sums, 0), _mm256_extracti128_si256(q8sums, 1));
                va_bsum = _mm512_castsi128_si512(q8s);
                vd1 = _mm512_set1_ps(A[0 * KB + i].d);
            }

            // step 1: accumultate the quants
            __m512i acc = _mm512_setzero_si512();
            const char * b_ptr = B + PACKED_INDEX(col, i, KB, TILE_SIZE);
            const char * b_qs  = b_ptr;
            for (int k_group = 0; k_group < QK_K / 32; ++k_group) {
                __m512i vsum = _mm512_setzero_si512();
                for (int k = 0; k < 8; k += 2) {
                    __m512i va0 = _mm512_permutexvar_epi32(_mm512_set1_epi32(k + 0), va[k_group]);
                    __m512i va1 = _mm512_permutexvar_epi32(_mm512_set1_epi32(k + 1), va[k_group]);

                    __m512i bytes = _mm512_loadu_si512((const __m512i *)b_qs);
                    __m512i vb0 = _mm512_and_si512(bytes, lowMask);
                    vsum = _mm512_dpbusd_epi32(vsum, vb0, va0);
                    __m512i vb1 = _mm512_and_si512(_mm512_srli_epi16(bytes, 4), lowMask);
                    vsum = _mm512_dpbusd_epi32(vsum, vb1, va1);

                    b_qs += 64;
                }
                // vacc += scale * (q8 @ q4)
                const __m512i vscale = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i *)(b_ptr + offset_scales + k_group * TILE_N)));
                acc = _mm512_add_epi32(acc, _mm512_mullo_epi32(vsum, vscale));
            }
            const __m512 vd0 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *)(b_ptr + offset_d0)));
            vc[col] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(acc), _mm512_mul_ps(vd0, vd1), vc[col]);

            // step 2: accumulate the mins
            __m512i acc_m = _mm512_setzero_si512();
            for (int k = 0; k < 4; ++k) {
                __m512i vmask = _mm512_set1_epi32(k);
                __m512i va = _mm512_permutexvar_epi32(vmask, va_bsum);
                __m512i vb = _mm512_cvtepi8_epi16(_mm256_loadu_si256((const __m256i *)(b_ptr + offset_mins + k * 32)));
                acc_m = _mm512_dpwssds_epi32(acc_m, va, vb);
            }
            const __m512 vdmin = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *)(b_ptr + offset_dmin)));
            vc[col] = _mm512_fnmadd_ps(_mm512_cvtepi32_ps(acc_m), _mm512_mul_ps(vdmin, vd1), vc[col]);
        };

        for (int i = 0; i < KB; ++i) {
            Unroll<COLS>{}(compute, i);
        }

        //store to C
        auto storec = [&](auto col) {
            _mm512_storeu_ps((__m512i*)(C + 0 * ldc + col * 16), vc[col]);
        };
        Unroll<COLS>{}(storec);
    }
};

template <int BLOCK_M, int BLOCK_N, int BLOCK_K>
struct tinygemm_kernel_vnni<block_q8_K, block_q5_K, float, BLOCK_M, BLOCK_N, BLOCK_K> {
    static void apply(int KB, const void * RESTRICT _A, const void * RESTRICT _B, float * RESTRICT C, int ldc) {

        constexpr int COLS = BLOCK_N / 16;
        const int TILE_SIZE = TILE_N * sizeof(block_q5_K) + TILE_N * 4;

        const block_q8_K * RESTRICT A = static_cast<const block_q8_K *>(_A);
        const char * RESTRICT B = static_cast<const char *>(_B);

        // a.qs:   8 groups, 32 bytes each group (m256i)
        __m512i va[8];
        // a.bsum: 8 groups,  2 bytes each group (m128i)
        __m512i va_bsum;
        __m512 vc[COLS];
        __m512 vd1;

        // packed_B:
        const int offset_qh     = (QK_K / 2) * TILE_N;
        const int offset_scales = (QK_K / 2) * TILE_N + (QK_K / 8) * TILE_N;
        const int offset_mins   = (QK_K / 2) * TILE_N + (QK_K / 8) * TILE_N +  8 * TILE_N;
        const int offset_d0     = (QK_K / 2) * TILE_N + (QK_K / 8) * TILE_N + 16 * TILE_N;
        const int offset_dmin   = (QK_K / 2) * TILE_N + (QK_K / 8) * TILE_N + 16 * TILE_N + TILE_N * sizeof(ggml_half);

        const __m512i lowMask = _mm512_set1_epi8(0xF);

        auto loadc = [&](auto col) {
            vc[col] = _mm512_setzero_ps();
        };
        Unroll<COLS>{}(loadc);

        // Q5_K and Q4_K shares the same vnni formats, refer to notes above.
        auto compute = [&](auto col, auto i) {
            // load a
            if constexpr (col == 0) {
                for (int k_group = 0; k_group < QK_K / 32; ++k_group) {
                    va[k_group] = _mm512_castsi256_si512(_mm256_loadu_si256((const __m256i *)(A[0 * KB + i].qs + k_group * 32)));
                }
                const __m256i q8sums = _mm256_loadu_si256((const __m256i *)A[0 * KB + i].bsums);
                const __m128i q8s = _mm_hadd_epi16(_mm256_extracti128_si256(q8sums, 0), _mm256_extracti128_si256(q8sums, 1));
                va_bsum = _mm512_castsi128_si512(q8s);
                vd1 = _mm512_set1_ps(A[0 * KB + i].d);
            }

            // step 1: accumultate the quants
            __m512i acc = _mm512_setzero_si512();
            const char * b_ptr = B + PACKED_INDEX(col, i, KB, TILE_SIZE);
            const char * b_qs  = b_ptr;
            const char * b_qh  = b_ptr + offset_qh;
            for (int k_group = 0; k_group < QK_K / 32; ++k_group) {
                __m512i vsum = _mm512_setzero_si512();
                __m512i hmask0 = _mm512_set1_epi8(0x1);
                __m512i hmask1 = _mm512_set1_epi8(0x2);
                __m512i hbits = _mm512_loadu_si512((const __m512i *)(b_qh + k_group * 64));
                for (int k = 0; k < 8; k += 2) {
                    __m512i va0 = _mm512_permutexvar_epi32(_mm512_set1_epi32(k + 0), va[k_group]);
                    __m512i va1 = _mm512_permutexvar_epi32(_mm512_set1_epi32(k + 1), va[k_group]);

                    __m512i bytes = _mm512_loadu_si512((const __m512i *)b_qs);
                    __m512i vb0 = _mm512_and_si512(bytes, lowMask);
                    __m512i vb1 = _mm512_and_si512(_mm512_srli_epi16(bytes, 4), lowMask);

                    __m512i vh0 = _mm512_slli_epi16(_mm512_srli_epi16(_mm512_and_si512(hbits, hmask0), k), 4);
                    __m512i vh1 = _mm512_slli_epi16(_mm512_srli_epi16(_mm512_and_si512(hbits, hmask1), k + 1), 4);

                    hmask0 = _mm512_slli_epi16(hmask0, 2);
                    hmask1 = _mm512_slli_epi16(hmask1, 2);
                    vb0 = _mm512_add_epi8(vb0, vh0);
                    vb1 = _mm512_add_epi8(vb1, vh1);

                    vsum = _mm512_dpbusd_epi32(vsum, vb0, va0);
                    vsum = _mm512_dpbusd_epi32(vsum, vb1, va1);

                    b_qs += 64;
                }
                // vacc += scale * (q8 @ q5)
                const __m512i vscale = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i *)(b_ptr + offset_scales + k_group * TILE_N)));
                acc = _mm512_add_epi32(acc, _mm512_mullo_epi32(vsum, vscale));
            }
            const __m512 vd0 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *)(b_ptr + offset_d0)));
            vc[col] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(acc), _mm512_mul_ps(vd0, vd1), vc[col]);

            // step 2: accumulate the mins
            __m512i acc_m = _mm512_setzero_si512();
            for (int k = 0; k < 4; ++k) {
                __m512i vmask = _mm512_set1_epi32(k);
                __m512i va = _mm512_permutexvar_epi32(vmask, va_bsum);
                __m512i vb = _mm512_cvtepi8_epi16(_mm256_loadu_si256((const __m256i *)(b_ptr + offset_mins + k * 32)));
                acc_m = _mm512_dpwssds_epi32(acc_m, va, vb);
            }
            const __m512 vdmin = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *)(b_ptr + offset_dmin)));
            vc[col] = _mm512_fnmadd_ps(_mm512_cvtepi32_ps(acc_m), _mm512_mul_ps(vdmin, vd1), vc[col]);
        };

        for (int i = 0; i < KB; ++i) {
            Unroll<COLS>{}(compute, i);
        }

        //store to C
        auto storec = [&](auto col) {
            _mm512_storeu_ps((__m512i*)(C + 0 * ldc + col * 16), vc[col]);
        };
        Unroll<COLS>{}(storec);
    }
};

template <int BLOCK_M, int BLOCK_N, int BLOCK_K>
struct tinygemm_kernel_vnni<block_q8_K, block_q6_K, float, BLOCK_M, BLOCK_N, BLOCK_K> {
    static void apply(int KB, const void * RESTRICT _A, const void * RESTRICT _B, float * RESTRICT C, int ldc) {

        constexpr int COLS = BLOCK_N / 16;
        const int TILE_SIZE = TILE_N * sizeof(block_q6_K);

        const block_q8_K * RESTRICT A = static_cast<const block_q8_K *>(_A);
        const char * RESTRICT B = static_cast<const char *>(_B);

        // load the 256 bytes from A to 4 avx512 vectors
        __m512i va[4];
        __m512 vc[COLS];
        __m512 vd1;

        // packed_B:
        const int offset_qh     = (QK_K / 2) * TILE_N;
        const int offset_scales = (QK_K / 2) * TILE_N + (QK_K / 4) * TILE_N;
        const int offset_d0     = (QK_K / 2) * TILE_N + (QK_K / 4) * TILE_N + 16 * TILE_N;

        // compensation
        __m512i vcomp;

        const __m512i m32s = _mm512_set1_epi32(32);
        const __m512i lowMask = _mm512_set1_epi8(0xF);

        auto loadc = [&](auto col) {
            vc[col] = _mm512_setzero_ps();
        };
        Unroll<COLS>{}(loadc);

        auto compute = [&](auto col, auto i) {
            if constexpr (col == 0) {
                // load a
                va[0] = _mm512_loadu_si512((const __m512i *)(A[0 * KB + i].qs +   0));
                va[1] = _mm512_loadu_si512((const __m512i *)(A[0 * KB + i].qs +  64));
                va[2] = _mm512_loadu_si512((const __m512i *)(A[0 * KB + i].qs + 128));
                va[3] = _mm512_loadu_si512((const __m512i *)(A[0 * KB + i].qs + 192));

                const __m256i q8sums = _mm256_loadu_si256((const __m256i *)A[0 * KB + i].bsums);
                vcomp = _mm512_mullo_epi32(_mm512_cvtepi16_epi32(q8sums), m32s);
                vd1 = _mm512_set1_ps(A[0 * KB + i].d);
            }

            // accmulate the quants
            __m512i acc = _mm512_setzero_si512();
            const char * b_ptr = B + PACKED_INDEX(col, i, KB, TILE_SIZE);
            const char * b_qs = b_ptr;
            const char * b_qh = b_ptr + offset_qh;
            int mask = 0;
            for (int k_group = 0; k_group < QK_K / 16; ++k_group) {
                int r = k_group >> 2;
                __m512i va0 = _mm512_permutexvar_epi32(_mm512_set1_epi32(mask++), va[r]);
                __m512i va1 = _mm512_permutexvar_epi32(_mm512_set1_epi32(mask++), va[r]);

                __m512i vsum = _mm512_setzero_si512();
                __m512i hmask = _mm512_set1_epi8(0x3);

                __m512i bytes = _mm512_loadu_si512(b_qs);
                __m512i hbits = _mm512_loadu_si512(b_qh);
                __m512i vb0 = _mm512_and_si512(bytes, lowMask);
                __m512i vb1 = _mm512_and_si512(_mm512_srli_epi16(bytes, 4), lowMask);
                __m512i vh0 = _mm512_slli_epi16(_mm512_and_si512(hbits, hmask), 4);
                __m512i vh1 = _mm512_slli_epi16(_mm512_and_si512(hbits, _mm512_slli_epi16(hmask, 2)), 2);

                vb0 = _mm512_add_epi8(vb0, vh0);
                vb1 = _mm512_add_epi8(vb1, vh1);
                vsum = _mm512_dpbusd_epi32(vsum, vb0, va0);
                vsum = _mm512_dpbusd_epi32(vsum, vb1, va1);
                b_qs += 64;

                va0 = _mm512_permutexvar_epi32(_mm512_set1_epi32(mask++), va[r]);
                va1 = _mm512_permutexvar_epi32(_mm512_set1_epi32(mask++), va[r]);

                bytes = _mm512_loadu_si512(b_qs);
                vb0 = _mm512_and_si512(bytes, lowMask);
                vb1 = _mm512_and_si512(_mm512_srli_epi16(bytes, 4), lowMask);
                vh0 =                   _mm512_and_si512(hbits, _mm512_slli_epi16(hmask, 4));
                vh1 = _mm512_srli_epi16(_mm512_and_si512(hbits, _mm512_slli_epi16(hmask, 6)), 2);
                vb0 = _mm512_add_epi8(vb0, vh0);
                vb1 = _mm512_add_epi8(vb1, vh1);
                vsum = _mm512_dpbusd_epi32(vsum, vb0, va0);
                vsum = _mm512_dpbusd_epi32(vsum, vb1, va1);
                b_qs += 64;
                b_qh += 64;

                // B * A - 32 * A
                __m512i vmask = _mm512_set1_epi32(k_group);
                vsum = _mm512_sub_epi32(vsum, _mm512_permutexvar_epi32(vmask, vcomp));

                // vacc += scale * (q8 @ q6)
                const __m512i vscale = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i *)(b_ptr + offset_scales + k_group * TILE_N)));
                acc = _mm512_add_epi32(acc, _mm512_mullo_epi32(vsum, vscale));
            }
            const __m512 vd0 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *)(b_ptr + offset_d0)));
            vc[col] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(acc), _mm512_mul_ps(vd0, vd1), vc[col]);
        };

        for (int i = 0; i < KB; ++i) {
            Unroll<COLS>{}(compute, i);
        }

        //store to C
        auto storec = [&](int col) {
            _mm512_storeu_ps((__m512i*)(C + 0 * ldc + col * 16), vc[col]);
        };
        Unroll<COLS>{}(storec);
    }
};

template <int BLOCK_M, int BLOCK_N, int BLOCK_K>
struct tinygemm_kernel_vnni<block_q8_K, block_iq4_xs, float, BLOCK_M, BLOCK_N, BLOCK_K> {
    static void apply(int KB, const void * RESTRICT _A, const void * RESTRICT _B, float * RESTRICT C, int ldc) {

        constexpr int COLS = BLOCK_N / 16;
        const int TILE_SIZE = TILE_N * sizeof(block_iq4_xs) + TILE_N * 2;

        const block_q8_K * RESTRICT A = static_cast<const block_q8_K *>(_A);
        const char * RESTRICT B = static_cast<const char *>(_B);

        // load the 256 bytes from A to 4 avx512 vectors
        __m512i va[4];
        __m512 vc[COLS];
        __m512 vd1;

        // packed_B:
        const int offset_scales = (QK_K / 2) * TILE_N ;
        const int offset_d0     = (QK_K / 2) * TILE_N + 8 * TILE_N;

        // compensation
        __m512i vcomp;

        const __m256i m128s = _mm256_set1_epi16(128);
        const __m512i lowMask = _mm512_set1_epi8(0xF);

        const __m512i values128 = _mm512_set_epi8(
            113, 89, 69, 53, 38, 25, 13, 1, -10, -22, -35, -49, -65, -83, -104, -127,
            113, 89, 69, 53, 38, 25, 13, 1, -10, -22, -35, -49, -65, -83, -104, -127,
            113, 89, 69, 53, 38, 25, 13, 1, -10, -22, -35, -49, -65, -83, -104, -127,
            113, 89, 69, 53, 38, 25, 13, 1, -10, -22, -35, -49, -65, -83, -104, -127
        );
        const __m512i off = _mm512_set1_epi8(static_cast<char>(0x80));
        const __m512i values256 = _mm512_add_epi8(values128, off);

        auto loadc = [&](auto col) {
            vc[col] = _mm512_setzero_ps();
        };
        Unroll<COLS>{}(loadc);

        auto compute = [&](auto col, auto i) {
            if constexpr (col == 0) {
                // load a
                va[0] = _mm512_loadu_si512((const __m512i *)(A[0 * KB + i].qs +   0));
                va[1] = _mm512_loadu_si512((const __m512i *)(A[0 * KB + i].qs +  64));
                va[2] = _mm512_loadu_si512((const __m512i *)(A[0 * KB + i].qs + 128));
                va[3] = _mm512_loadu_si512((const __m512i *)(A[0 * KB + i].qs + 192));

                // compensation: 128 * A
                const __m256i q8sums = _mm256_loadu_si256((const __m256i *)A[0 * KB + i].bsums);
                vcomp = _mm512_castsi256_si512(_mm256_madd_epi16(q8sums, m128s));
                vd1 = _mm512_set1_ps(A[0 * KB + i].d);
            }

            // accmulate the quants
            __m512i acc = _mm512_setzero_si512();
            const char * b_ptr = B + PACKED_INDEX(col, i, KB, TILE_SIZE);
            const char * b_qs = b_ptr;
            int mask = 0;
            for (int k_group = 0; k_group < QK_K / 32; ++k_group) {
                int r = k_group >> 1;
                __m512i vmask = _mm512_set1_epi32(k_group);
                __m512i vsum = _mm512_setzero_si512();
                for (int k = 0; k < 8; k += 2) {
                    __m512i va0 = _mm512_permutexvar_epi32(_mm512_set1_epi32(mask++), va[r]);
                    __m512i va1 = _mm512_permutexvar_epi32(_mm512_set1_epi32(mask++), va[r]);

                    __m512i bytes = _mm512_loadu_si512(b_qs);
                    __m512i vb0 = _mm512_shuffle_epi8(values256, _mm512_and_si512(bytes, lowMask));
                    __m512i vb1 = _mm512_shuffle_epi8(values256, _mm512_and_si512(_mm512_srli_epi16(bytes, 4), lowMask));

                    vsum = _mm512_dpbusd_epi32(vsum, vb0, va0);
                    vsum = _mm512_dpbusd_epi32(vsum, vb1, va1);
                    b_qs += 64;
                }
                // (B + 128) * A - 128 * A
                vsum = _mm512_sub_epi32(vsum, _mm512_permutexvar_epi32(vmask, vcomp));

                // vacc += scale * (q8 @ q4)
                const __m512i vscale = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i *)(b_ptr + offset_scales + k_group * TILE_N)));
                acc = _mm512_add_epi32(acc, _mm512_mullo_epi32(vsum, vscale));
            }
            const __m512 vd0 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *)(b_ptr + offset_d0)));
            vc[col] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(acc), _mm512_mul_ps(vd0, vd1), vc[col]);
        };

        for (int i = 0; i < KB; ++i) {
            Unroll<COLS>{}(compute, i);
        }

        //store to C
        auto storec = [&](auto col) {
            _mm512_storeu_ps((__m512i*)(C + 0 * ldc + col * 16), vc[col]);
        };
        Unroll<COLS>{}(storec);
    }
};

// Multi-row AVX512-VNNI kernels for small batches (inserted into ggml-cpu/amx/mmq.cpp by apply_patch.py).
//
// Upstream sends M == 1 to a VNNI kernel and every M >= 2 to the AMX tile kernel, which costs a fixed overhead
// (per 32-value block: tile multiply, store, rescale; Q4_0 also unpacks B per block) and is slow for 2-16 rows.
// These kernels keep the inner loops of the M == 1 kernels above (same packed VNNI layouts) and multiply each
// weight block read with up to 8 activation rows: 4 rows x 64 columns or 8 rows x 32 columns per call (16
// accumulator registers).
//
// Probe history (2026-10-02, 4 Granite Rapids cores, ms per pass): 4 rows x 64 columns per weight read was fastest for 2-4 rows
// and lost from 6 rows (weights re-read per 4 rows); 16 rows x 16 columns with the activation sums precomputed was
// slower at every size; 8 rows x 32 columns was best for 5-8 rows. Above 8 rows the AMX kernel is as fast.

constexpr int VNNI_ROWS_MAX = 8;

// Q4_0 weights: unsigned nibbles (u8) x signed activations (s8), minus 8 * sum(a) per block (acomp, precomputed).
template <int ROWS, int BLOCK_N>
static void vnni_rows_q4_0(int KB, const block_q8_0 * RESTRICT A, const int32_t * RESTRICT acomp,
                           const char * RESTRICT B, float * RESTRICT C, int ldc) {
    constexpr int COLS = BLOCK_N / 16;
    const int TILE_SIZE = TILE_N * sizeof(block_q4_0);
    const __m512i lowMask = _mm512_set1_epi8(0xF);

    __m512 vc[ROWS][COLS];
#pragma GCC unroll 8
    for (int r = 0; r < ROWS; ++r)
#pragma GCC unroll 4
        for (int c = 0; c < COLS; ++c) vc[r][c] = _mm512_setzero_ps();

    for (int i = 0; i < KB; ++i) {
#pragma GCC unroll 4
        for (int c = 0; c < COLS; ++c) {
            const char * b_ptr = B + PACKED_INDEX(c, i, KB, TILE_SIZE);
            __m512i vb[8];
#pragma GCC unroll 4
            for (int k = 0; k < 8; k += 2) {
                const __m512i bytes = _mm512_loadu_si512((const __m512i *)(b_ptr + k * 32));
                vb[k + 0] = _mm512_and_si512(bytes, lowMask);
                vb[k + 1] = _mm512_and_si512(_mm512_srli_epi16(bytes, 4), lowMask);
            }
            const __m512 vd0 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *)(b_ptr + TILE_N * TILE_K / 2)));
#pragma GCC unroll 8
            for (int r = 0; r < ROWS; ++r) {
                const block_q8_0 & a = A[r * KB + i];
                const int32_t * a_ptr = reinterpret_cast<const int32_t *>(a.qs);
                __m512i vsum = _mm512_setzero_si512();
#pragma GCC unroll 8
                for (int k = 0; k < 8; ++k) {
                    vsum = _mm512_dpbusd_epi32(vsum, vb[k], _mm512_set1_epi32(a_ptr[k]));
                }
                vsum = _mm512_sub_epi32(vsum, _mm512_set1_epi32(acomp[r * KB + i]));
                vc[r][c] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(vsum),
                                           _mm512_mul_ps(vd0, _mm512_set1_ps(GGML_CPU_FP16_TO_FP32(a.d))), vc[r][c]);
            }
        }
    }
#pragma GCC unroll 8
    for (int r = 0; r < ROWS; ++r)
#pragma GCC unroll 4
        for (int c = 0; c < COLS; ++c) _mm512_storeu_ps(C + r * ldc + c * 16, vc[r][c]);
}

// Q8_0 weights: (a + 128) as u8 x weights s8, minus 128 * sum(b) (packed with B), as the M == 1 kernel.
template <int ROWS, int BLOCK_N>
static void vnni_rows_q8_0(int KB, const block_q8_0 * RESTRICT A, const int32_t * RESTRICT /*acomp*/,
                           const char * RESTRICT B, float * RESTRICT C, int ldc) {
    constexpr int COLS = BLOCK_N / 16;
    const int TILE_SIZE = TILE_N * sizeof(block_q8_0) + TILE_N * sizeof(int32_t);
    const __m512i off = _mm512_set1_epi8(static_cast<char>(0x80));

    __m512 vc[ROWS][COLS];
#pragma GCC unroll 8
    for (int r = 0; r < ROWS; ++r)
#pragma GCC unroll 4
        for (int c = 0; c < COLS; ++c) vc[r][c] = _mm512_setzero_ps();

    for (int i = 0; i < KB; ++i) {
#pragma GCC unroll 8
        for (int r = 0; r < ROWS; ++r) {
            const block_q8_0 & a = A[r * KB + i];
            const int32_t * a_ptr = reinterpret_cast<const int32_t *>(a.qs);
            __m512i va[8];
#pragma GCC unroll 8
            for (int k = 0; k < 8; ++k) va[k] = _mm512_add_epi8(_mm512_set1_epi32(a_ptr[k]), off);
            const __m512 vd1 = _mm512_set1_ps(GGML_CPU_FP16_TO_FP32(a.d));
#pragma GCC unroll 4
            for (int c = 0; c < COLS; ++c) {
                const char * b_ptr = B + PACKED_INDEX(c, i, KB, TILE_SIZE);
                __m512i vsum = _mm512_setzero_si512();
#pragma GCC unroll 8
                for (int k = 0; k < 8; ++k) {
                    vsum = _mm512_dpbusd_epi32(vsum, va[k], _mm512_loadu_si512((const __m512i *)(b_ptr + k * 64)));
                }
                const __m512 vd0 = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *)(b_ptr + TILE_N * TILE_K)));
                const __m512i vcomp = _mm512_loadu_si512((const __m512i *)(b_ptr + TILE_N * TILE_K + TILE_N * sizeof(ggml_half)));
                vsum = _mm512_sub_epi32(vsum, vcomp);
                vc[r][c] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(vsum), _mm512_mul_ps(vd0, vd1), vc[r][c]);
            }
        }
    }
#pragma GCC unroll 8
    for (int r = 0; r < ROWS; ++r)
#pragma GCC unroll 4
        for (int c = 0; c < COLS; ++c) _mm512_storeu_ps(C + r * ldc + c * 16, vc[r][c]);
}

// 8 * sum(a) of every 32-value block of the quantized activation rows (for Q4_0)
static void vnni_rows_prepare(const block_q8_0 * A, int n, int32_t * acomp) {
    const __m256i flip = _mm256_set1_epi8(static_cast<char>(0x80));
    for (int j = 0; j < n; ++j) {
        const __m256i qu = _mm256_xor_si256(_mm256_loadu_si256((const __m256i *)A[j].qs), flip);
        const __m256i s = _mm256_sad_epu8(qu, _mm256_setzero_si256());
        acomp[j] = 8 * ((int32_t)(_mm256_extract_epi64(s, 0) + _mm256_extract_epi64(s, 1) +
                                  _mm256_extract_epi64(s, 2) + _mm256_extract_epi64(s, 3)) - 32 * 128);
    }
}

// rows (1..8) x cols (32, or 64 for up to 4 rows) per call
template <typename TB>
static void vnni_rows(int rows, int cols, int KB, const block_q8_0 * A, const int32_t * acomp, const char * B, float * C,
                      int ldc) {
#define VNNI_ROWS_CALL(R, NC) \
        if constexpr (std::is_same<TB, block_q4_0>::value) { vnni_rows_q4_0<R, NC>(KB, A, acomp, B, C, ldc); } \
        else { vnni_rows_q8_0<R, NC>(KB, A, acomp, B, C, ldc); }
    switch (rows * 100 + cols) {
        case 164: VNNI_ROWS_CALL(1, 64) break;
        case 264: VNNI_ROWS_CALL(2, 64) break;
        case 364: VNNI_ROWS_CALL(3, 64) break;
        case 464: VNNI_ROWS_CALL(4, 64) break;
        case 132: VNNI_ROWS_CALL(1, 32) break;
        case 232: VNNI_ROWS_CALL(2, 32) break;
        case 332: VNNI_ROWS_CALL(3, 32) break;
        case 432: VNNI_ROWS_CALL(4, 32) break;
        case 532: VNNI_ROWS_CALL(5, 32) break;
        case 632: VNNI_ROWS_CALL(6, 32) break;
        case 732: VNNI_ROWS_CALL(7, 32) break;
        case 832: VNNI_ROWS_CALL(8, 32) break;
        default: fprintf(stderr, "Unexpected vnni_rows block %d x %d!\n", rows, cols);
    }
#undef VNNI_ROWS_CALL
}

// GGML_AMX_VNNI_CHECK=1 recomputes every row with the M == 1 kernel and reports differences (slow; for testing).
static bool vnni_rows_check() {
    static const bool v = getenv("GGML_AMX_VNNI_CHECK") != nullptr;
    return v;
}

static void vnni_rows_compare(const float * ref, const float * got, int n, int M, int N, int K) {
    static std::atomic<long> checked{0}, bad{0};
    float worst = 0;
    for (int j = 0; j < n; ++j) {
        const float d = std::fabs(ref[j] - got[j]) / (std::fabs(ref[j]) + 1e-3f);
        worst = std::max(worst, d);
    }
    const long c = ++checked;
    if (worst > 1e-4f && bad++ < 20) {
        fprintf(stderr, "vnni_rows MISMATCH M=%d N=%d K=%d rel=%g ref=%g got=%g\n", M, N, K, worst, ref[0], got[0]);
    }
    if ((c & (c - 1)) == 0 && c >= 1024) {
        fprintf(stderr, "vnni_rows checked %ld rows x 32, mismatches %ld\n", c, bad.load());
    }
}

// Largest M sent to the multi-row kernels (GGML_AMX_VNNI_MAX_M; 0 or 1 disables them). Rows go 8 at a time.
static int vnni_rows_max_m() {
    static const int v = [] {
        const char * s = getenv("GGML_AMX_VNNI_MAX_M");
        return s ? atoi(s) : 8;
    }();
    return v;
}

// AMX-FP16 / AMX-BF16 kernels for Q4_0 / Q8_0 weights and larger batches (inserted into ggml-cpu/amx/mmq.cpp by
// apply_patch.py).
//
// ggml's AMX int8 kernel multiplies one 32-value block at a time (TILE_K = 32, the quantization block), stores the
// int32 tile and rescales it with AVX512 by both blocks' scales before the next block: the AVX512 work, not the tile
// multiply, sets its speed (about 1 TOPS on 4 Granite Rapids cores). Here each 32-column slice of the weights is
// dequantized once per call into 16-bit float tiles (scale applied, kept in L2 for all row blocks), the activations
// are converted once, and the tile multiply accumulates in FP32 tiles over the whole K: no per-block store.
//
// FP16 (AMX-FP16 + AVX512-FP16, Granite Rapids): int8 -> FP16 and the scale in about 4 instructions per 32 weights;
// activation rows are scaled by a power of two into FP16 range (exact) and the outputs scaled back.
// BF16 (AMX-BF16 + AVX512-BF16, Sapphire Rapids and later): the fallback, about 10 instructions per 32 weights.
// Numerics: activations in FP16 / BF16 instead of Q8_0, weights q * d rounded to FP16 / BF16; FP32 accumulation.

#if defined(__AMX_BF16__) && defined(__AVX512BF16__)
#define GGML_AMX_HALF_KERNELS 1
// ggml's PACKED_INDEX does not parenthesize its arguments (PACKED_INDEX(h + 1, ...) silently means h + KB + ...)
#define AMX_HALF_PACKED_INDEX(n, k, KB, tile_size) ((size_t)((n) * (KB) + (k)) * (tile_size))
// GCC's _tile_loadd / _tile_stored are inline asm without a memory clobber: the compiler does not know they read or
// write memory, so it may sink or drop the AVX512 stores of a buffer a tile then loads (seen with a stack buffer:
// garbage results), or read a tile's stored output too early. A compiler barrier between the two fixes the order.
#define AMX_HALF_MEMORY_BARRIER() __asm__ __volatile__("" ::: "memory")
#if defined(__AMX_FP16__) && defined(__AVX512FP16__)
#define GGML_AMX_HALF_FP16 1
#endif

// GGML_AMX_HALF_MIN_M: smallest M sent here (default 9, above the multi-row VNNI kernels; 0 disables)
static int amx_half_min_m() {
    static const int v = [] {
        const char * s = getenv("GGML_AMX_HALF_MIN_M");
        return s ? atoi(s) : 9;
    }();
    return v;
}

// GGML_AMX_HALF_BF16=1 uses BF16 even where FP16 is available (for comparisons)
static bool amx_half_use_fp16() {
#if defined(GGML_AMX_HALF_FP16)
    static const bool v = getenv("GGML_AMX_HALF_BF16") == nullptr;
    return v;
#else
    return false;
#endif
}

// tiles 0-1: B (16 rows of 16 columns x 2 values), 2-3: A (16 rows x 32 values), 4-7: C (16 x 16 FP32)
static void amx_half_tile_config(void) {
    alignas(64) tile_config_t tc = {};
    tc.palette_id = 1;
    for (int t = 0; t < 8; ++t) {
        tc.rows[t] = 16;
        tc.colsb[t] = 64;
    }
    _tile_loadconfig(&tc);
}

// ggml's int8 configuration (ggml_tile_config_init loads it only once per thread)
static void amx_int8_tile_config_restore(void) {
    alignas(64) tile_config_t tc = {};
    tc.palette_id = 1;
    tc.rows[0] = 8;   tc.colsb[0] = 64;
    tc.rows[1] = 8;   tc.colsb[1] = 64;
    tc.rows[2] = 16;  tc.colsb[2] = 32;
    tc.rows[3] = 16;  tc.colsb[3] = 32;
    for (int t = 4; t < 8; ++t) {
        tc.rows[t] = 16;
        tc.colsb[t] = 64;
    }
    _tile_loadconfig(&tc);
}

// byte order within a 64-byte int8 VNNI row: column n, values 4n..4n+3. This picks values 0,1 of every column into
// bytes 0-31 and values 2,3 into bytes 32-63: the two 16-bit tile rows (16 columns x 2 values) made from one int8 row.
static inline __m512i amx_half_pick() {
    return _mm512_set_epi8(
        63, 62, 59, 58, 55, 54, 51, 50, 47, 46, 43, 42, 39, 38, 35, 34,
        31, 30, 27, 26, 23, 22, 19, 18, 15, 14, 11, 10,  7,  6,  3,  2,
        61, 60, 57, 56, 53, 52, 49, 48, 45, 44, 41, 40, 37, 36, 33, 32,
        29, 28, 25, 24, 21, 20, 17, 16, 13, 12,  9,  8,  5,  4,  1,  0);
}

// One packed 16-column x 32-value block: int8 rows q[8] (VNNI layout) and the columns' FP16 scales d[16]
template <typename TB>
static inline void amx_half_load_block(const char * RESTRICT b_ptr, __m512i (&q)[8], const ggml_half *& d) {
    if constexpr (std::is_same<TB, block_q4_0>::value) {
        const __m512i lowMask = _mm512_set1_epi8(0xF);
        const __m512i off = _mm512_set1_epi8(8);
        for (int k = 0; k < 8; k += 2) {
            const __m512i bytes = _mm512_loadu_si512((const __m512i *)(b_ptr + k * 32));
            q[k + 0] = _mm512_sub_epi8(_mm512_and_si512(bytes, lowMask), off);
            q[k + 1] = _mm512_sub_epi8(_mm512_and_si512(_mm512_srli_epi16(bytes, 4), lowMask), off);
        }
        d = (const ggml_half *)(b_ptr + TILE_N * TILE_K / 2);
    } else {
        for (int k = 0; k < 8; ++k) {
            q[k] = _mm512_loadu_si512((const __m512i *)(b_ptr + k * 64));
        }
        d = (const ggml_half *)(b_ptr + TILE_N * TILE_K);
    }
}

// ... to a BF16 tile (16 rows x 64 bytes)
template <typename TB>
static inline void amx_half_dequant_bf16(const char * RESTRICT b_ptr, uint16_t * RESTRICT out) {
    __m512i q[8];
    const ggml_half * dh;
    amx_half_load_block<TB>(b_ptr, q, dh);
    const __m512 d = _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *) dh));
    const __m512i pick = amx_half_pick();
    const __m512 d_lo = _mm512_permutexvar_ps(_mm512_set_epi32(7, 7, 6, 6, 5, 5, 4, 4, 3, 3, 2, 2, 1, 1, 0, 0), d);
    const __m512 d_hi = _mm512_permutexvar_ps(_mm512_set_epi32(15, 15, 14, 14, 13, 13, 12, 12, 11, 11, 10, 10, 9, 9, 8, 8), d);
    for (int r = 0; r < 8; ++r) {
        const __m512i p = _mm512_permutexvar_epi8(pick, q[r]);
        for (int h = 0; h < 2; ++h) {
            const __m512 f0 = _mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(_mm512_extracti32x4_epi32(p, 2 * h))), d_lo);
            const __m512 f1 = _mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(_mm512_extracti32x4_epi32(p, 2 * h + 1))), d_hi);
            _mm512_storeu_si512((__m512i *)(out + (2 * r + h) * 32), (__m512i)_mm512_cvtne2ps_pbh(f1, f0));
        }
    }
}

#if defined(GGML_AMX_HALF_FP16)
// ... to an FP16 tile: int8 -> int16 -> FP16, times the column's scale, both in FP16
template <typename TB>
static inline void amx_half_dequant_fp16(const char * RESTRICT b_ptr, uint16_t * RESTRICT out) {
    __m512i q[8];
    const ggml_half * dh;
    amx_half_load_block<TB>(b_ptr, q, dh);
    const __m512i dup = _mm512_set_epi16(15, 15, 14, 14, 13, 13, 12, 12, 11, 11, 10, 10, 9, 9, 8, 8,
                                         7, 7, 6, 6, 5, 5, 4, 4, 3, 3, 2, 2, 1, 1, 0, 0);
    const __m512h d = _mm512_castsi512_ph(_mm512_permutexvar_epi16(dup, _mm512_castsi256_si512(_mm256_loadu_si256((const __m256i *) dh))));
    const __m512i pick = amx_half_pick();
    for (int r = 0; r < 8; ++r) {
        const __m512i p = _mm512_permutexvar_epi8(pick, q[r]);
        const __m512h h0 = _mm512_mul_ph(_mm512_cvtepi16_ph(_mm512_cvtepi8_epi16(_mm512_castsi512_si256(p))), d);
        const __m512h h1 = _mm512_mul_ph(_mm512_cvtepi16_ph(_mm512_cvtepi8_epi16(_mm512_extracti64x4_epi64(p, 1))), d);
        _mm512_storeu_si512((__m512i *)(out + (2 * r + 0) * 32), _mm512_castph_si512(h0));
        _mm512_storeu_si512((__m512i *)(out + (2 * r + 1) * 32), _mm512_castph_si512(h1));
    }
}
#endif

template <typename TB>
static inline void amx_half_dequant(bool fp16, const char * RESTRICT b_ptr, uint16_t * RESTRICT out) {
#if defined(GGML_AMX_HALF_FP16)
    if (fp16) {
        amx_half_dequant_fp16<TB>(b_ptr, out);
        return;
    }
#endif
    GGML_UNUSED(fp16);
    amx_half_dequant_bf16<TB>(b_ptr, out);
}

// One float row to 16-bit floats, written in tile order: value k of the row goes to y[(k / 32) * 512 + k % 32]
// (y points at the row's place in its 16-row tile; one tile = 16 rows x 32 values, KB tiles per 16-row block, so
// every tile load reads 1 KB in a row). x == nullptr writes a zero row. K is a multiple of 32.
// FP16: scaled by a power of two so the largest value is about 2^14 (FP16 tops out at 65504; Gemma's activations can
// exceed it), *inv = the factor that undoes it. BF16: unscaled, *inv = 1.
static void amx_half_convert_A(bool fp16, const float * RESTRICT x, uint16_t * RESTRICT y, int K, float * inv) {
    *inv = 1.0f;
    if (x == nullptr) {
        for (int k = 0; k < K; k += 32) {
            _mm512_storeu_si512((__m512i *)(y + (k / 32) * 512), _mm512_setzero_si512());
        }
        return;
    }
#if defined(GGML_AMX_HALF_FP16)
    if (fp16) {
        __m512 vmax = _mm512_setzero_ps();
        for (int k = 0; k < K; k += 16) {
            vmax = _mm512_max_ps(vmax, _mm512_abs_ps(_mm512_loadu_ps(x + k)));
        }
        const float amax = _mm512_reduce_max_ps(vmax);
        int e = 0;
        if (amax > 0 && std::isfinite(amax)) {
            std::frexp(amax, &e);  // amax = f * 2^e, 0.5 <= f < 1
        }
        const __m512 vs = _mm512_set1_ps(std::ldexp(1.0f, 14 - e));  // amax * s < 2^14
        *inv = std::ldexp(1.0f, e - 14);
        for (int k = 0; k < K; k += 32) {
            const __m256i lo = _mm512_cvtps_ph(_mm512_mul_ps(_mm512_loadu_ps(x + k), vs), _MM_FROUND_TO_NEAREST_INT);
            const __m256i hi = _mm512_cvtps_ph(_mm512_mul_ps(_mm512_loadu_ps(x + k + 16), vs), _MM_FROUND_TO_NEAREST_INT);
            _mm512_storeu_si512((__m512i *)(y + (k / 32) * 512), _mm512_inserti64x4(_mm512_castsi256_si512(lo), hi, 1));
        }
        return;
    }
#endif
    GGML_UNUSED(fp16);
    for (int k = 0; k < K; k += 32) {
        const __m512bh v = _mm512_cvtne2ps_pbh(_mm512_loadu_ps(x + k + 16), _mm512_loadu_ps(x + k));
        _mm512_storeu_si512((__m512i *)(y + (k / 32) * 512), (__m512i)v);
    }
}

// GGML_AMX_HALF_BUFFER_M: from this M on a slice's weights are converted once into a buffer and reused by every row
// block; otherwise (the default, 0) each 32-value weight block is converted right before its tile multiplies (it
// stays in L1), once per 32-row block. Converting again per row block was faster at every size measured (up to 500
// rows): a converted tile written by AVX512 and loaded by AMX from L1 costs ~12 ns more than one already in a tile
// register, and from a buffer in L2 more still.
static int amx_half_buffer_m() {
    static const int v = [] {
        const char * s = getenv("GGML_AMX_HALF_BUFFER_M");
        return s ? atoi(s) : 0;
    }();
    return v;
}

#if defined(GGML_AMX_HALF_FP16)
#define AMX_HALF_DP(c, a, b) do { if (fp16) { _tile_dpfp16ps(c, a, b); } else { _tile_dpbf16ps(c, a, b); } } while (0)
#else
#define AMX_HALF_DP(c, a, b) _tile_dpbf16ps(c, a, b)
#endif

// Store four 16 x 16 FP32 tiles (4-7) as rows x (16 * nt) values, undoing each row's scaling.
// layout 0: one 16-row block, tiles 4-7 = columns 0-15, 16-31, 32-47, 48-63;
// layout 1: two 16-row blocks x 32 columns, tiles 4/6 = rows 0-15, 5/7 = rows 16-31, 4/5 = columns 0-15, 6/7 = 16-31.
static inline void amx_half_store_C(int layout, int nt, int rows, const float * RESTRICT inv, float * RESTRICT C, int ldc) {
    alignas(64) float Ct[32 * 64];
    if (layout == 0) {
        _tile_stored(4, Ct, 256); _tile_stored(5, Ct + 16, 256);
        if (nt == 4) { _tile_stored(6, Ct + 32, 256); _tile_stored(7, Ct + 48, 256); }
    } else {
        _tile_stored(4, Ct, 256); _tile_stored(6, Ct + 16, 256);
        _tile_stored(5, Ct + 16 * 64, 256); _tile_stored(7, Ct + 16 * 64 + 16, 256);
    }
    AMX_HALF_MEMORY_BARRIER();
    for (int r = 0; r < rows; ++r) {
        const __m512 s = _mm512_set1_ps(inv[r]);
        for (int t = 0; t < nt; ++t) {
            _mm512_storeu_ps(C + (size_t)r * ldc + t * 16, _mm512_mul_ps(_mm512_load_ps(Ct + r * 64 + t * 16), s));
        }
    }
}

// C[M x ncols] for one slice of ncols = 16 * nt columns (nt = 4, or 2 at the edge of N). A: tiles in 16-row blocks
// ([Mpad / 16][KB][16 x 32 values], zero rows past M; Mpad a multiple of 32 when M > 16); inv[m]: the factor undoing
// row m's scaling. Tiles: 0-2 B, 3 (and 2) A, 4-7 C.
//   M <= 16: per 32-value block, 4 weight tiles converted, one A load, four multiplies (1 x 4 output tiles).
//   M > 16: per 32-row block and 32 columns, 2 weight tiles converted, two A loads, four multiplies (2 x 2 output
//   tiles); with GGML_AMX_HALF_BUFFER_M set and reached, the slice is converted once into Bh ([KB][nt][16 x 32]).
template <typename TB>
static void amx_half_slice(bool fp16, int M, int K, const uint16_t * RESTRICT A, const float * RESTRICT inv,
                           const char * RESTRICT Bq, int nt, uint16_t * RESTRICT Bh, float * RESTRICT C, int ldc) {
    const int KB = K / 32;
    const int TILE_SIZE = get_tile_size<TB>();
    alignas(64) uint16_t Bl[4 * 512];
    if (M <= 16) {
        _tile_zero(4); _tile_zero(5); _tile_zero(6); _tile_zero(7);
        for (int i = 0; i < KB; ++i) {
            AMX_HALF_MEMORY_BARRIER();  // the previous block's tile loads of Bl come first
            for (int t = 0; t < nt; ++t) {
                amx_half_dequant<TB>(fp16, Bq + AMX_HALF_PACKED_INDEX(t, i, KB, TILE_SIZE), Bl + t * 512);
            }
            AMX_HALF_MEMORY_BARRIER();
            _tile_loadd(3, A + (size_t)i * 512, 64);
            _tile_loadd(0, Bl, 64);        AMX_HALF_DP(4, 3, 0);
            _tile_loadd(1, Bl + 512, 64);  AMX_HALF_DP(5, 3, 1);
            if (nt == 4) {
                _tile_loadd(2, Bl + 1024, 64); AMX_HALF_DP(6, 3, 2);
                _tile_loadd(0, Bl + 1536, 64); AMX_HALF_DP(7, 3, 0);
            }
        }
        amx_half_store_C(0, nt, M, inv, C, ldc);
        return;
    }
    const bool buffered = amx_half_buffer_m() > 0 && M >= amx_half_buffer_m();
    if (buffered) {
        for (int i = 0; i < KB; ++i) {
            for (int t = 0; t < nt; ++t) {
                amx_half_dequant<TB>(fp16, Bq + AMX_HALF_PACKED_INDEX(t, i, KB, TILE_SIZE), Bh + (size_t)(i * nt + t) * 512);
            }
        }
        AMX_HALF_MEMORY_BARRIER();
    }
    for (int m0 = 0; m0 < M; m0 += 32) {
        const uint16_t * A0 = A + (size_t)(m0 / 16) * KB * 512;
        const uint16_t * A1 = A0 + (size_t)KB * 512;
        for (int h = 0; h < nt; h += 2) {  // 32 columns at a time
            _tile_zero(4); _tile_zero(5); _tile_zero(6); _tile_zero(7);
            for (int i = 0; i < KB; ++i) {
                const uint16_t * Bi;
                if (buffered) {
                    Bi = Bh + (size_t)(i * nt + h) * 512;
                } else {
                    AMX_HALF_MEMORY_BARRIER();  // the previous block's tile loads of Bl come first
                    amx_half_dequant<TB>(fp16, Bq + AMX_HALF_PACKED_INDEX(h, i, KB, TILE_SIZE), Bl);
                    amx_half_dequant<TB>(fp16, Bq + AMX_HALF_PACKED_INDEX(h + 1, i, KB, TILE_SIZE), Bl + 512);
                    AMX_HALF_MEMORY_BARRIER();
                    Bi = Bl;
                }
                _tile_loadd(0, Bi, 64);
                _tile_loadd(3, A0 + (size_t)i * 512, 64);
                AMX_HALF_DP(4, 3, 0);
                _tile_loadd(1, Bi + 512, 64);
                AMX_HALF_DP(6, 3, 1);
                _tile_loadd(2, A1 + (size_t)i * 512, 64);
                AMX_HALF_DP(5, 2, 0);
                AMX_HALF_DP(7, 2, 1);
            }
            amx_half_store_C(1, 2, std::min(32, M - m0), inv + m0, C + (size_t)m0 * ldc + h * 16, ldc);
        }
    }
}

// GGML_AMX_HALF_CHECK=1: compare sampled outputs with an FP64 reference from the packed weights and the FP32
// activations, and report the int8 path's error (Q8_0 activations) for the same outputs (slow; for testing)
static bool amx_half_check() {
    static const bool v = getenv("GGML_AMX_HALF_CHECK") != nullptr;
    return v;
}

template <typename TB>
static double amx_half_ref_dot(const char * B_slice, int KB, int col, const float * a, bool quantize_a,
                               double * abs_sum = nullptr) {
    const int TILE_SIZE = get_tile_size<TB>();
    const int t = col / 16, n = col % 16;
    double s = 0;
    for (int i = 0; i < KB; ++i) {
        const char * b_ptr = B_slice + AMX_HALF_PACKED_INDEX(t, i, KB, TILE_SIZE);
        float d;
        int8_t q[32];
        if constexpr (std::is_same<TB, block_q4_0>::value) {
            d = GGML_CPU_FP16_TO_FP32(((const ggml_half *)(b_ptr + TILE_N * TILE_K / 2))[n]);
            for (int r = 0; r < 8; r += 2) {
                for (int j = 0; j < 4; ++j) {
                    const uint8_t byte = (uint8_t) b_ptr[r * 32 + n * 4 + j];
                    q[r * 4 + j] = (int8_t)((byte & 0xF) - 8);
                    q[(r + 1) * 4 + j] = (int8_t)((byte >> 4) - 8);
                }
            }
        } else {
            d = GGML_CPU_FP16_TO_FP32(((const ggml_half *)(b_ptr + TILE_N * TILE_K))[n]);
            for (int r = 0; r < 8; ++r) {
                for (int j = 0; j < 4; ++j) {
                    q[r * 4 + j] = (int8_t) b_ptr[r * 64 + n * 4 + j];
                }
            }
        }
        const float * x = a + i * 32;
        if (quantize_a) {  // as the int8 path: Q8_0 activations
            float amax = 0;
            for (int j = 0; j < 32; ++j) amax = std::max(amax, std::fabs(x[j]));
            const float da = amax / 127.f;
            const float ida = da ? 1.f / da : 0.f;
            double bs = 0;
            for (int j = 0; j < 32; ++j) bs += (double) q[j] * (double) roundf(x[j] * ida);
            s += bs * (double) d * (double) GGML_CPU_FP16_TO_FP32(GGML_CPU_FP32_TO_FP16(da));
        } else {
            for (int j = 0; j < 32; ++j) {
                const double v = (double) q[j] * (double) d * (double) x[j];
                s += v;
                if (abs_sum) *abs_sum += std::fabs(v);
            }
        }
    }
    return s;
}

static void amx_half_report(double err, double err_int8, double scale) {
    static std::atomic<long> n{0};
    static std::atomic<double> sum_h{0}, sum_i{0}, max_h{0};
    const double rh = err / scale, ri = err_int8 / scale;
    double cur = sum_h.load(); while (!sum_h.compare_exchange_weak(cur, cur + rh)) {}
    cur = sum_i.load(); while (!sum_i.compare_exchange_weak(cur, cur + ri)) {}
    cur = max_h.load(); while (rh > cur && !max_h.compare_exchange_weak(cur, rh)) {}
    const long c = ++n;
    if ((c & (c - 1)) == 0 && c >= 256) {
        fprintf(stderr, "amx_half (%s) checked %ld outputs: mean rel err %.3g (int8 path %.3g), max %.3g\n",
                amx_half_use_fp16() ? "fp16" : "bf16", c, sum_h.load() / c, sum_i.load() / c, max_h.load());
    }
}

#endif // __AMX_BF16__ && __AVX512BF16__

// Prototype: AMX int8 kernel with large scale groups from a second copy of the weights (inserted into
// ggml-cpu/amx/mmq.cpp by apply_patch.py, after tools/llama_amx_half).
//
// ggml's int8 AMX kernel rescales after every 32 values (the Q4_0 / Q8_0 block). With one scale per G values
// (G = 128 / 256 ...) for both weights and activations, G / 64 full-width int8 tile multiplies (TDPBSSD, 64 values
// each) accumulate in an int32 tile before one rescale. The weights for it are a second copy made at load time, so
// passes of 1-8 tokens keep reading the smaller original weights (VNNI kernels).
//
// GGML_AMX_I8G=G                  group size (0 / unset: off)
// GGML_AMX_I8G_FILE=model.gguf    the copy's values: a GGUF whose matrix tensors are Q8_0 blocks sharing one scale per
//                                 G values (tools/llama_quant_groups/requant_groups.py, also 4-bit values -8..7)
// GGML_AMX_I8G_BITS4=a,b          tensors (name contains a or b) to store in fewer bits when their values allow:
//                                 -8..7 as 4-bit nibbles (half the bytes), -16..15 / -32..31 as nibbles plus one / two
//                                 planes of high bits (5/8, 6/8 of the bytes); unpacked to int8 in L1 before tile loads
// GGML_AMX_I8G_AHEAD=L            unpack the low-bit weight tiles L 64-value steps ahead into a ring of L1 buffers
//                                 (default 0). Measured 2026-10-03: 1 and 2 made no difference (within 1 ms); the
//                                 store-to-tile-load wait is already hidden by out-of-order execution / the sibling thread
// GGML_AMX_I8G_MIN_M=9            smallest pass (rows) sent here
// GGML_AMX_I8G_CHECK=1            compare sampled outputs with an FP64 sum of the same int8 products (testing)
// GGML_AMX_I8G_PREFETCH=D         software-prefetch each weight tile D 64-value steps ahead (default 0: off). Measured
//                                 2026-10-03: 4, 8 and 16 (L2) were all 5-10% slower; the hardware prefetcher keeps up
// GGML_AMX_I8G_PREFETCH_L2=1      prefetch into L2 (_MM_HINT_T1) instead of L1

#if defined(__AMX_INT8__) && defined(__AVX512VNNI__)
// (<mutex>, <string>, <unordered_map>, <cstdio>, <cstring> and gguf.h are included at the top of the file by apply_patch.py)

#define AMX_I8G_PACKED_INDEX(n, k, KB, tile_size) ((size_t)((n) * (KB) + (k)) * (tile_size))
#define AMX_I8G_MEMORY_BARRIER() __asm__ __volatile__("" ::: "memory")

struct amx_i8g_copy {
    int N = 0, K = 0, G = 0;
    int bits = 8;             // 8, 6, 5 or 4
    std::vector<int8_t> q;    // int8: tiles [N / 16][K / 64] of 16 rows (4 values each) x 16 columns x 4 = 1 KB
    std::vector<uint8_t> q4;  // 4 / 5-bit: the same tiles, 512 bytes each: byte i = u[i] & 15 | (u[i + 512] & 15) << 4,
                              // u = t + 2^(bits - 1)
    std::vector<uint64_t> q5; // 5 / 6-bit: per tile and high bit b (bits - 4 planes) 16 words: bit k of word w is
                              // bit 4 + b of u[64 w + k], u = t + 2^(bits - 1)
    std::vector<float> s;     // scales [N / 16][K / G][16]
};

static int amx_i8g_env_int(const char * name, int dflt) {
    const char * s = getenv(name);
    return s ? atoi(s) : dflt;
}
static int amx_i8g_group() { static const int v = amx_i8g_env_int("GGML_AMX_I8G", 0); return v; }
static int amx_i8g_min_m() { static const int v = amx_i8g_env_int("GGML_AMX_I8G_MIN_M", 9); return v; }
static bool amx_i8g_check() { static const bool v = getenv("GGML_AMX_I8G_CHECK") != nullptr; return v; }
static int amx_i8g_prefetch() { static const int v = amx_i8g_env_int("GGML_AMX_I8G_PREFETCH", 0); return v; }
static bool amx_i8g_prefetch_l2() { static const bool v = amx_i8g_env_int("GGML_AMX_I8G_PREFETCH_L2", 0) != 0; return v; }

// prefetch one weight tile (column tile nt, step ks) of the copy
static inline void amx_i8g_prefetch_tile(const amx_i8g_copy & cp, int nt, int ks, int KS) {
    const char * p;
    int lines;
    if (cp.bits < 8) { p = (const char *)cp.q4.data() + AMX_I8G_PACKED_INDEX(nt, ks, KS, 512); lines = 8; }
    else { p = (const char *)cp.q.data() + AMX_I8G_PACKED_INDEX(nt, ks, KS, 1024); lines = 16; }
    if (amx_i8g_prefetch_l2()) { for (int j = 0; j < lines; ++j) _mm_prefetch(p + 64 * j, _MM_HINT_T1); }
    else { for (int j = 0; j < lines; ++j) _mm_prefetch(p + 64 * j, _MM_HINT_T0); }
}

static std::mutex amx_i8g_mutex;
static std::unordered_map<const void *, amx_i8g_copy *> amx_i8g_copies;

static const amx_i8g_copy * amx_i8g_find(const void * packed) {
    if (amx_i8g_group() <= 0) return nullptr;
    std::lock_guard<std::mutex> lock(amx_i8g_mutex);
    auto it = amx_i8g_copies.find(packed);
    return it == amx_i8g_copies.end() ? nullptr : it->second;
}

// Called after a weight is repacked into the AMX buffer: builds the copy from GGML_AMX_I8G_FILE's tensor of the same
// name (Q8_0 blocks, one shared scale per G values).
static void amx_i8g_make_copy(const struct ggml_tensor * tensor) {
    const int G = amx_i8g_group();
    const char * path = getenv("GGML_AMX_I8G_FILE");
    if (G <= 0 || G % 64 != 0 || path == nullptr) return;
    const int K = tensor->ne[0], N = tensor->ne[1];
    if (K % G != 0 || N % 32 != 0 || ggml_nrows(tensor) != N) return;

    static std::mutex file_mutex;
    static gguf_context * ctx = nullptr;
    static FILE * fp = nullptr;
    std::lock_guard<std::mutex> lock(file_mutex);
    if (ctx == nullptr) {
        gguf_init_params params = { /* .no_alloc = */ true, /* .ctx = */ nullptr };
        ctx = gguf_init_from_file(path, params);
        fp = fopen(path, "rb");
        if (ctx == nullptr || fp == nullptr) {
            fprintf(stderr, "amx_i8g: cannot read %s\n", path);
            return;
        }
    }
    const int64_t idx = gguf_find_tensor(ctx, tensor->name);
    if (idx < 0 || gguf_get_tensor_type(ctx, idx) != GGML_TYPE_Q8_0) return;
    const size_t nblocks = (size_t) N * K / 32;
    std::vector<block_q8_0> src(nblocks);
    if (gguf_get_tensor_size(ctx, idx) != nblocks * sizeof(block_q8_0)) return;
    if (fseeko(fp, (off_t)(gguf_get_data_offset(ctx) + gguf_get_tensor_offset(ctx, idx)), SEEK_SET) != 0 ||
        fread(src.data(), sizeof(block_q8_0), nblocks, fp) != nblocks) {
        fprintf(stderr, "amx_i8g: read failed for %s\n", tensor->name);
        return;
    }

    bool lowbit = false;
    if (const char * b = getenv("GGML_AMX_I8G_BITS4")) {
        std::string list = b;
        size_t p = 0;
        while (p <= list.size()) {
            const size_t e = std::min(list.find(',', p), list.size());
            const std::string part = list.substr(p, e - p);
            if (!part.empty() && strstr(tensor->name, part.c_str())) lowbit = true;
            p = e + 1;
        }
    }

    // the copy holds one scale per group: only tensors whose blocks share it within each group (token_embd, for one,
    // is kept per 32 by requant_groups.py); anything else would compute wrong results
    for (size_t b = 0; b < nblocks; ++b) {
        const size_t first = b - b % (size_t)(G / 32);
        if (src[b].d != src[first].d) {
            static int skipped = 0;
            if (skipped++ < 3) fprintf(stderr, "amx_i8g: no copy of %s (scales not shared per %d values)\n", tensor->name, G);
            return;
        }
    }
    auto * cp = new amx_i8g_copy();
    cp->N = N; cp->K = K; cp->G = G;
    const int KS = K / 64, NG = K / G, KB = K / 32;
    cp->s.resize((size_t) N / 16 * NG * 16);
    std::vector<int8_t> tiles((size_t) N / 16 * KS * 1024);
    for (int n = 0; n < N; ++n) {
        const block_q8_0 * row = src.data() + (size_t) n * KB;
        for (int g = 0; g < NG; ++g) {
            cp->s[((size_t)(n / 16) * NG + g) * 16 + n % 16] = GGML_CPU_FP16_TO_FP32(row[g * (G / 32)].d);
        }
        for (int k = 0; k < K; ++k) {
            const int8_t v = row[k / 32].qs[k % 32];
            tiles[((size_t)(n / 16) * KS + k / 64) * 1024 + ((k % 64) / 4) * 64 + (n % 16) * 4 + k % 4] = v;
        }
    }
    int bits = 8;
    if (lowbit) {  // the fewest bits every value fits
        int8_t lo = 0, hi = 0;
        for (int8_t v : tiles) { lo = std::min(lo, v); hi = std::max(hi, v); }
        bits = (lo >= -8 && hi <= 7) ? 4 : (lo >= -16 && hi <= 15) ? 5 : (lo >= -32 && hi <= 31) ? 6 : 8;
    }
    cp->bits = bits;
    if (bits < 8) {
        const int bias = 1 << (bits - 1), planes = bits - 4;
        const size_t ntiles = tiles.size() / 1024;
        cp->q4.resize(ntiles * 512);
        if (planes > 0) cp->q5.assign(ntiles * 16 * planes, 0);
        for (size_t t = 0; t < ntiles; ++t) {
            for (int i = 0; i < 1024; ++i) {
                const int u = tiles[t * 1024 + i] + bias;
                if (i < 512) cp->q4[t * 512 + i] = (uint8_t)(u & 15);
                else cp->q4[t * 512 + i - 512] |= (uint8_t)((u & 15) << 4);
                for (int b = 0; b < planes; ++b) {
                    if ((u >> (4 + b)) & 1) cp->q5[(t * planes + b) * 16 + i / 64] |= 1ull << (i % 64);
                }
            }
        }
    } else {
        cp->q = std::move(tiles);
    }
    std::lock_guard<std::mutex> lock2(amx_i8g_mutex);
    amx_i8g_copies[tensor->data] = cp;
    static int reported = 0;
    if (reported++ < 3) {
        fprintf(stderr, "amx_i8g: copy of %s (%d x %d, groups of %d, %d-bit)\n", tensor->name, N, K, G, cp->bits);
    }
}

// Unpack one 4 / 5-bit weight tile (column tile nt, step ks) to int8 at dst (1 KB, in L1)
static inline void amx_i8g_unpack(const amx_i8g_copy & cp, int nt, int ks, int KS, int8_t * RESTRICT dst) {
    const __m512i lowMask = _mm512_set1_epi8(0xF);
    const size_t t = (size_t) nt * KS + ks;
    const uint8_t * pk = cp.q4.data() + t * 512;
    if (cp.bits == 4) {
        const __m512i off = _mm512_set1_epi8(8);
        for (int j = 0; j < 8; ++j) {
            const __m512i v = _mm512_loadu_si512((const __m512i *)(pk + 64 * j));
            _mm512_store_si512((__m512i *)(dst + 64 * j), _mm512_sub_epi8(_mm512_and_si512(v, lowMask), off));
            _mm512_store_si512((__m512i *)(dst + 512 + 64 * j), _mm512_sub_epi8(_mm512_and_si512(_mm512_srli_epi16(v, 4), lowMask), off));
        }
    } else {
        const int planes = cp.bits - 4;
        const __m512i off = _mm512_set1_epi8((char)(1 << (cp.bits - 1)));
        const __m512i b16 = _mm512_set1_epi8(16), b32 = _mm512_set1_epi8(32);
        const uint64_t * hb = cp.q5.data() + t * 16 * planes;
        for (int j = 0; j < 8; ++j) {
            const __m512i v = _mm512_loadu_si512((const __m512i *)(pk + 64 * j));
            __m512i lo = _mm512_and_si512(v, lowMask);
            __m512i hi = _mm512_and_si512(_mm512_srli_epi16(v, 4), lowMask);
            lo = _mm512_mask_add_epi8(lo, _cvtu64_mask64(hb[j]), lo, b16);       // bit 4
            hi = _mm512_mask_add_epi8(hi, _cvtu64_mask64(hb[j + 8]), hi, b16);
            if (planes == 2) {                                                  // bit 5
                lo = _mm512_mask_add_epi8(lo, _cvtu64_mask64(hb[16 + j]), lo, b32);
                hi = _mm512_mask_add_epi8(hi, _cvtu64_mask64(hb[16 + j + 8]), hi, b32);
            }
            _mm512_store_si512((__m512i *)(dst + 64 * j), _mm512_sub_epi8(lo, off));
            _mm512_store_si512((__m512i *)(dst + 512 + 64 * j), _mm512_sub_epi8(hi, off));
        }
    }
}

static int amx_i8g_ahead() { static const int v = std::max(0, std::min(2, amx_i8g_env_int("GGML_AMX_I8G_AHEAD", 0))); return v; }

// One activation row to int8, one scale per G values (FP16-rounded max / 127, as Q8_0), in tile order:
// value k goes to y[(k / 64) * 1024 + k % 64] (y = the row's place in its 16-row tile block). x == nullptr: zeros.
static void amx_i8g_quantize_row(const float * x, int8_t * y, float * scales, int K, int G) {
    for (int g0 = 0; g0 < K; g0 += G) {
        float d = 0.0f;
        if (x != nullptr) {
            __m512 vmax = _mm512_setzero_ps();
            for (int k = g0; k < g0 + G; k += 16) vmax = _mm512_max_ps(vmax, _mm512_abs_ps(_mm512_loadu_ps(x + k)));
            d = GGML_CPU_FP16_TO_FP32(GGML_CPU_FP32_TO_FP16(_mm512_reduce_max_ps(vmax) / 127.0f));
        }
        scales[g0 / G] = d;
        const __m512 id = _mm512_set1_ps(d ? 1.0f / d : 0.0f);
        for (int k = g0; k < g0 + G; k += 16) {
            __m128i q = _mm_setzero_si128();
            if (x != nullptr) {
                const __m512i v = _mm512_cvtps_epi32(_mm512_roundscale_ps(_mm512_mul_ps(_mm512_loadu_ps(x + k), id), _MM_FROUND_TO_NEAREST_INT));
                q = _mm512_cvtsepi32_epi8(v);
            }
            _mm_storeu_si128((__m128i *)(y + (k / 64) * 1024 + k % 64), q);
        }
    }
}

// C[M x 32 columns] for 32-column block nb32 of the copy. A: tiles [Mpad / 16][K / 64][16 x 64] int8; as: scales
// [Mpad][K / G]. Per 32-row block: 2 x 2 output tiles, G / 64 multiplies per group, one rescale per group.
static void amx_i8g_slice(const amx_i8g_copy & cp, int nb32, int M, const int8_t * RESTRICT A, const float * RESTRICT as,
                          float * RESTRICT C, int ldc) {
    const int K = cp.K, G = cp.G, KS = K / 64, NG = K / G, GS = G / 64;
    const int nt0 = nb32 * 2, nt1 = nt0 + 1;
    const int PD = amx_i8g_prefetch();
    alignas(64) float acc[32 * 32];
    alignas(64) int32_t Ci[32 * 32];
    alignas(64) int8_t Bl[3 * 2 * 1024];  // ring of 3 steps x 2 tiles (GGML_AMX_I8G_AHEAD)
    const int AH = cp.bits < 8 ? amx_i8g_ahead() : 0;
    for (int m0 = 0; m0 < M; m0 += 32) {
        const int8_t * A0 = A + (size_t)(m0 / 16) * KS * 1024;
        const int8_t * A1 = A0 + (size_t)KS * 1024;
        memset(acc, 0, sizeof(acc));
        if (cp.bits < 8) {
            for (int k = 0; k < AH && k < KS; ++k) {
                amx_i8g_unpack(cp, nt0, k, KS, Bl + (k % 3) * 2048);
                amx_i8g_unpack(cp, nt1, k, KS, Bl + (k % 3) * 2048 + 1024);
            }
        }
        for (int g = 0; g < NG; ++g) {
            _tile_zero(4); _tile_zero(5); _tile_zero(6); _tile_zero(7);
            for (int s = 0; s < GS; ++s) {
                const int ks = g * GS + s;
                if (PD > 0 && ks + PD < KS) {
                    amx_i8g_prefetch_tile(cp, nt0, ks + PD, KS);
                    amx_i8g_prefetch_tile(cp, nt1, ks + PD, KS);
                }
                const int8_t * B0;
                const int8_t * B1;
                if (cp.bits < 8) {
                    AMX_I8G_MEMORY_BARRIER();
                    const int ku = ks + AH;
                    if (ku < KS) {
                        amx_i8g_unpack(cp, nt0, ku, KS, Bl + (ku % 3) * 2048);
                        amx_i8g_unpack(cp, nt1, ku, KS, Bl + (ku % 3) * 2048 + 1024);
                    }
                    AMX_I8G_MEMORY_BARRIER();
                    B0 = Bl + (ks % 3) * 2048; B1 = B0 + 1024;
                } else {
                    B0 = cp.q.data() + AMX_I8G_PACKED_INDEX(nt0, ks, KS, 1024);
                    B1 = cp.q.data() + AMX_I8G_PACKED_INDEX(nt1, ks, KS, 1024);
                }
                _tile_loadd(0, B0, 64);
                _tile_loadd(2, A0 + (size_t)ks * 1024, 64);
                _tile_dpbssd(4, 2, 0);
                _tile_loadd(1, B1, 64);
                _tile_dpbssd(6, 2, 1);
                _tile_loadd(3, A1 + (size_t)ks * 1024, 64);
                _tile_dpbssd(5, 3, 0);
                _tile_dpbssd(7, 3, 1);
            }
            _tile_stored(4, Ci, 128);
            _tile_stored(6, Ci + 16, 128);
            _tile_stored(5, Ci + 16 * 32, 128);
            _tile_stored(7, Ci + 16 * 32 + 16, 128);
            AMX_I8G_MEMORY_BARRIER();
            const __m512 ws0 = _mm512_loadu_ps(cp.s.data() + ((size_t)nt0 * NG + g) * 16);
            const __m512 ws1 = _mm512_loadu_ps(cp.s.data() + ((size_t)nt1 * NG + g) * 16);
            const int rows = std::min(32, M - m0);
            for (int r = 0; r < rows; ++r) {
                const __m512 sa = _mm512_set1_ps(as[(size_t)(m0 + r) * NG + g]);
                _mm512_store_ps(acc + r * 32, _mm512_fmadd_ps(_mm512_cvtepi32_ps(_mm512_load_si512(Ci + r * 32)), _mm512_mul_ps(ws0, sa), _mm512_load_ps(acc + r * 32)));
                _mm512_store_ps(acc + r * 32 + 16, _mm512_fmadd_ps(_mm512_cvtepi32_ps(_mm512_load_si512(Ci + r * 32 + 16)), _mm512_mul_ps(ws1, sa), _mm512_load_ps(acc + r * 32 + 16)));
            }
        }
        const int rows = std::min(32, M - m0);
        for (int r = 0; r < rows; ++r) {
            _mm512_storeu_ps(C + (size_t)(m0 + r) * ldc + nb32 * 32, _mm512_load_ps(acc + r * 32));
            _mm512_storeu_ps(C + (size_t)(m0 + r) * ldc + nb32 * 32 + 16, _mm512_load_ps(acc + r * 32 + 16));
        }
    }
}

// Up to 16 rows: one activation tile against nt (4, or 2 at the edge of N) weight tiles, 64 columns per call, so no
// tile multiply works on padding rows (the 2 x 2 layout above wastes half of them below 17 rows). Tiles: 3 A,
// 0-2 B (rotating), 4-7 C.
static void amx_i8g_slice16(const amx_i8g_copy & cp, int nt0, int nt, int M, const int8_t * RESTRICT A,
                            const float * RESTRICT as, float * RESTRICT C, int ldc) {
    const int K = cp.K, G = cp.G, KS = K / 64, NG = K / G, GS = G / 64;
    const int PD = amx_i8g_prefetch();
    alignas(64) float acc[16 * 64];
    alignas(64) int32_t Ci[16 * 64];
    alignas(64) int8_t Bl[3 * 4 * 1024];  // ring of 3 steps x up to 4 tiles (GGML_AMX_I8G_AHEAD)
    const int AH = cp.bits < 8 ? amx_i8g_ahead() : 0;
    if (cp.bits < 8) {
        for (int k = 0; k < AH && k < KS; ++k) {
            for (int t = 0; t < nt; ++t) amx_i8g_unpack(cp, nt0 + t, k, KS, Bl + (k % 3) * 4096 + t * 1024);
        }
    }
    memset(acc, 0, sizeof(acc));
    for (int g = 0; g < NG; ++g) {
        _tile_zero(4); _tile_zero(5); _tile_zero(6); _tile_zero(7);
        for (int s = 0; s < GS; ++s) {
            const int ks = g * GS + s;
            if (PD > 0 && ks + PD < KS) {
                for (int t = 0; t < nt; ++t) amx_i8g_prefetch_tile(cp, nt0 + t, ks + PD, KS);
            }
            const int8_t * Bt[4];
            if (cp.bits < 8) {
                AMX_I8G_MEMORY_BARRIER();
                const int ku = ks + AH;  // the step unpacked now; step ks was unpacked AH steps ago
                if (ku < KS) {
                    for (int t = 0; t < nt; ++t) amx_i8g_unpack(cp, nt0 + t, ku, KS, Bl + (ku % 3) * 4096 + t * 1024);
                }
                for (int t = 0; t < nt; ++t) Bt[t] = Bl + (ks % 3) * 4096 + t * 1024;
                AMX_I8G_MEMORY_BARRIER();
            } else {
                for (int t = 0; t < nt; ++t) Bt[t] = cp.q.data() + AMX_I8G_PACKED_INDEX(nt0 + t, ks, KS, 1024);
            }
            _tile_loadd(3, A + (size_t)ks * 1024, 64);
            _tile_loadd(0, Bt[0], 64); _tile_dpbssd(4, 3, 0);
            _tile_loadd(1, Bt[1], 64); _tile_dpbssd(5, 3, 1);
            if (nt == 4) {
                _tile_loadd(2, Bt[2], 64); _tile_dpbssd(6, 3, 2);
                _tile_loadd(0, Bt[3], 64); _tile_dpbssd(7, 3, 0);
            }
        }
        _tile_stored(4, Ci, 256);
        _tile_stored(5, Ci + 16, 256);
        if (nt == 4) {
            _tile_stored(6, Ci + 32, 256);
            _tile_stored(7, Ci + 48, 256);
        }
        AMX_I8G_MEMORY_BARRIER();
        for (int r = 0; r < M; ++r) {
            const __m512 sa = _mm512_set1_ps(as[(size_t)r * NG + g]);
            for (int t = 0; t < nt; ++t) {
                const __m512 ws = _mm512_loadu_ps(cp.s.data() + ((size_t)(nt0 + t) * NG + g) * 16);
                _mm512_store_ps(acc + r * 64 + t * 16, _mm512_fmadd_ps(_mm512_cvtepi32_ps(_mm512_load_si512(Ci + r * 64 + t * 16)),
                                                                       _mm512_mul_ps(ws, sa), _mm512_load_ps(acc + r * 64 + t * 16)));
            }
        }
    }
    for (int r = 0; r < M; ++r) {
        for (int t = 0; t < nt; ++t) {
            _mm512_storeu_ps(C + (size_t)r * ldc + (nt0 + t) * 16, _mm512_load_ps(acc + r * 64 + t * 16));
        }
    }
}

// The same output from the copy's integers in FP64 (GGML_AMX_I8G_CHECK)
static double amx_i8g_ref(const amx_i8g_copy & cp, int n, const int8_t * A, const float * as, int m, double * abs_sum) {
    const int K = cp.K, G = cp.G, KS = K / 64, NG = K / G;
    double sum = 0;
    for (int g = 0; g < NG; ++g) {
        int64_t isum = 0;
        for (int k = g * G; k < (g + 1) * G; ++k) {
            const size_t ti = ((size_t)(n / 16) * KS + k / 64);
            const int within = ((k % 64) / 4) * 64 + (n % 16) * 4 + k % 4;
            int w;
            if (cp.bits < 8) {
                const uint8_t b = cp.q4[ti * 512 + within % 512];
                int u = within < 512 ? (b & 0xF) : (b >> 4);
                for (int b = 0; b < cp.bits - 4; ++b) {
                    u |= (int)((cp.q5[(ti * (cp.bits - 4) + b) * 16 + within / 64] >> (within % 64)) & 1) << (4 + b);
                }
                w = u - (1 << (cp.bits - 1));
            } else {
                w = cp.q[ti * 1024 + within];
            }
            const int a = A[((size_t)(m / 16) * KS + k / 64) * 1024 + (m % 16) * 64 + k % 64];
            isum += (int64_t) w * a;
        }
        const double term = (double) isum * as[(size_t)m * NG + g] * cp.s[((size_t)(n / 16) * NG + g) * 16 + n % 16];
        sum += term;
        *abs_sum += std::fabs(term);
    }
    return sum;
}

// difference relative to the sum of the groups' magnitudes (a relative difference to the result itself is unbounded
// when the groups cancel out)
static void amx_i8g_report(double rel) {
    static std::atomic<long> n{0};
    static std::atomic<double> mx{0}, sum{0};
    double cur = mx.load(); while (rel > cur && !mx.compare_exchange_weak(cur, rel)) {}
    cur = sum.load(); while (!sum.compare_exchange_weak(cur, cur + rel)) {}
    const long c = ++n;
    if ((c & (c - 1)) == 0 && c >= 256) fprintf(stderr, "amx_i8g checked %ld outputs, diff relative to the terms: mean %.3g, max %.3g\n", c, sum.load() / c, mx.load());
}
#endif

#define LAUNCH_TINYGEMM_KERNEL_VNNI(NB_SIZE)                                                   \
    tinygemm_kernel_vnni<vec_dot_type, type, float, 1, NB_SIZE, blck_size>::apply(             \
        KB, wdata_batch,                                                                       \
        (const char *)src0->data + src0_offset + PACKED_INDEX(nb * kTilesN, 0, KB, TILE_SIZE), \
        (float *) dst->data + dst_offset + nb_start, ldc)

template <typename TA, typename TB, typename TC, int BLOCK_K,
          typename std::enable_if<!is_type_qkk<TB>::value, int>::type = 0>
void tinygemm_kernel_amx(int M, int N, int KB, const void * RESTRICT _A, const void * RESTRICT _B, TC * RESTRICT C, int ldc) {
    using packed_B_t = packed_B_type<TB>;
    const int TILE_SIZE = get_tile_size<TB>();
    const bool need_unpack = do_unpack<TB>::value;

    GGML_ASSERT(M <= 2 * TILE_M && N == 2 * TILE_N);
    const TA * RESTRICT A = static_cast<const TA *>(_A);
    const char * RESTRICT B = static_cast<const char *>(_B);

    const int m0 = std::min(M, TILE_M);
    const int m1 = std::max(M - TILE_M, 0);
    const int lda = KB * sizeof(TA);
    //const int ldb = KB * sizeof(TB);

    alignas(64) static thread_local packed_B_t Tile0[TILE_N * TILE_K];
    alignas(64) static thread_local packed_B_t Tile1[TILE_N * TILE_K];
    alignas(64) static thread_local int8_t Tile23[TILE_M * TILE_K];

    alignas(64) static thread_local int32_t TileC0[TILE_M * TILE_N * 4];
    alignas(64) static thread_local int32_t TileC1[TILE_M * TILE_N * 4];

    // double buffering C to interleave avx512 and amx
    int32_t * C_cur = TileC0;
    int32_t * C_pre = TileC1;

    auto Tile4 = [&](int32_t * base) { return base; };
    auto Tile5 = [&](int32_t * base) { return base + TILE_M * TILE_N; };
    auto Tile6 = [&](int32_t * base) { return base + 2 * TILE_M * TILE_N; };
    auto Tile7 = [&](int32_t * base) { return base + 3 * TILE_M * TILE_N; };

    if (M == 2 * TILE_M) {
        // i = 0
        const char * B_blk0 = B + PACKED_INDEX(0, 0, KB, TILE_SIZE);
        const char * B_blk1 = B + PACKED_INDEX(1, 0, KB, TILE_SIZE);
        if (need_unpack) {
            unpack_B<TB>(Tile0, B_blk0);
            _tile_loadd(TMM0, Tile0, TILE_N * VNNI_BLK);
        } else {
            _tile_loadd(TMM0, B_blk0, TILE_N * VNNI_BLK);
        }

        _tile_zero(TMM4);
        _tile_loadd(TMM2, A[0].qs, lda);
        _tile_dpbssd(TMM4, TMM2, TMM0);
        _tile_stored(TMM4, Tile4(C_pre), TILE_N * sizeof(int32_t));

        _tile_zero(TMM5);
        _tile_loadd(TMM3, A[TILE_M * KB + 0].qs, lda);
        _tile_dpbssd(TMM5, TMM3, TMM0);
        _tile_stored(TMM5, Tile5(C_pre), TILE_N * sizeof(int32_t));

        if (need_unpack) {
            unpack_B<TB>(Tile1, B_blk1);
            _tile_loadd(TMM1, Tile1, TILE_N * VNNI_BLK);
        } else {
            _tile_loadd(TMM1, B_blk1, TILE_N * VNNI_BLK);
        }

        _tile_zero(TMM6);
        _tile_dpbssd(TMM6, TMM2, TMM1);
        _tile_stored(TMM6, Tile6(C_pre), TILE_N * sizeof(int32_t));

        _tile_zero(TMM7);
        _tile_dpbssd(TMM7, TMM3, TMM1);
        _tile_stored(TMM7, Tile7(C_pre), TILE_N * sizeof(int32_t));

        for (int i = 1; i < KB; ++i) {
            // index of previous iter
            const int ii = i - 1;
            const char * B_blk0 = B + PACKED_INDEX(0, i, KB, TILE_SIZE);
            const char * B_blk1 = B + PACKED_INDEX(1, i, KB, TILE_SIZE);
            GGML_DISPATCH_BOOL(ii > 0, is_acc, [&] {
                if (need_unpack) {
                    unpack_B<TB>(Tile0, B_blk0);
                    _tile_loadd(TMM0, Tile0, TILE_N * VNNI_BLK);
                } else {
                    _tile_loadd(TMM0, B_blk0, TILE_N * VNNI_BLK);
                }
                _tile_zero(TMM4);
                _tile_loadd(TMM2, A[i].qs, lda);
                acc_C<TA, TB, is_acc>::apply(C, ldc, Tile4(C_pre), &A[ii], KB, B + PACKED_INDEX(0, ii, KB, TILE_SIZE), TILE_M);

                _tile_dpbssd(TMM4, TMM2, TMM0);
                _tile_stored(TMM4, Tile4(C_cur), TILE_N * sizeof(int32_t));

                _tile_zero(TMM5);
                _tile_loadd(TMM3, A[TILE_M * KB + i].qs, lda);
                acc_C<TA, TB, is_acc>::apply(C + TILE_M * ldc, ldc, Tile5(C_pre), &A[TILE_M * KB + ii], KB, B + PACKED_INDEX(0, ii, KB, TILE_SIZE), TILE_M);

                _tile_dpbssd(TMM5, TMM3, TMM0);
                _tile_stored(TMM5, Tile5(C_cur), TILE_N * sizeof(int32_t));

                if (need_unpack) {
                    unpack_B<TB>(Tile1, B_blk1);
                    _tile_loadd(TMM1, Tile1, TILE_N * VNNI_BLK);
                } else {
                    _tile_loadd(TMM1, B_blk1, TILE_N * VNNI_BLK);
                }
                _tile_zero(TMM6);
                acc_C<TA, TB, is_acc>::apply(C + TILE_N, ldc, Tile6(C_pre), &A[ii], KB, B + PACKED_INDEX(1, ii, KB, TILE_SIZE), TILE_M);

                _tile_dpbssd(TMM6, TMM2, TMM1);
                _tile_stored(TMM6, Tile6(C_cur), TILE_N * sizeof(int32_t));

                _tile_zero(TMM7);
                acc_C<TA, TB, is_acc>::apply(C + TILE_M * ldc + TILE_N, ldc, Tile7(C_pre), &A[TILE_M * KB + ii], KB, B + PACKED_INDEX(1, ii, KB, TILE_SIZE), TILE_M);

                _tile_dpbssd(TMM7, TMM3, TMM1);
                _tile_stored(TMM7, Tile7(C_cur), TILE_N * sizeof(int32_t));

                std::swap(C_cur, C_pre);
            });
        }
        // final accumulation
        {
            int ii = KB - 1;
            acc_C<TA, TB, true>::apply(C, ldc, Tile4(C_pre), &A[ii], KB, B + PACKED_INDEX(0, ii, KB, TILE_SIZE), TILE_M);
            acc_C<TA, TB, true>::apply(C + TILE_M * ldc, ldc, Tile5(C_pre), &A[TILE_M * KB + ii], KB, B + PACKED_INDEX(0, ii, KB, TILE_SIZE), TILE_M);
            acc_C<TA, TB, true>::apply(C + TILE_N, ldc, Tile6(C_pre), &A[ii], KB, B + PACKED_INDEX(1, ii, KB, TILE_SIZE), TILE_M);
            acc_C<TA, TB, true>::apply(C + TILE_M * ldc + TILE_N, ldc, Tile7(C_pre), &A[TILE_M * KB + ii], KB, B + PACKED_INDEX(1, ii, KB, TILE_SIZE), TILE_M);
        }
    } else {
        for (int i = 0; i < KB; ++i) {
            _tile_zero(TMM4);
            _tile_zero(TMM6);
            if (m1 != 0) {
                _tile_zero(TMM5);
                _tile_zero(TMM7);
            }

            const char * B_blk0 = B + PACKED_INDEX(0, i, KB, TILE_SIZE);
            const char * B_blk1 = B + PACKED_INDEX(1, i, KB, TILE_SIZE);
            if (need_unpack) {
                unpack_B<TB>(Tile0, B_blk0);
                _tile_loadd(TMM0, Tile0, TILE_N * VNNI_BLK);
            } else {
                _tile_loadd(TMM0, B_blk0, TILE_N * VNNI_BLK);
            }

            if (need_unpack) {
                unpack_B<TB>(Tile1, B_blk1);
                _tile_loadd(TMM1, Tile1, TILE_N * VNNI_BLK);
            } else {
                _tile_loadd(TMM1, B_blk1, TILE_N * VNNI_BLK);
            }

            if (m0 == TILE_M) {
                _tile_loadd(TMM2, A[i].qs, lda);
            } else {
                unpack_A(Tile23, &A[i], KB, m0);
                _tile_loadd(TMM2, Tile23, TILE_K);
            }

            _tile_dpbssd(TMM4, TMM2, TMM0);
            _tile_dpbssd(TMM6, TMM2, TMM1);

            _tile_stored(TMM4, Tile4(C_cur), TILE_N * sizeof(int32_t));
            _tile_stored(TMM6, Tile6(C_cur), TILE_N * sizeof(int32_t));

            GGML_DISPATCH_BOOL(i > 0, is_acc, [&] {
                acc_C<TA, TB, is_acc>::apply(C,          ldc, Tile4(C_cur), &A[i], KB, B + PACKED_INDEX(0, i, KB, TILE_SIZE), m0);
                acc_C<TA, TB, is_acc>::apply(C + TILE_N, ldc, Tile6(C_cur), &A[i], KB, B + PACKED_INDEX(1, i, KB, TILE_SIZE), m0);
            });

            if (m1 != 0) {
                unpack_A(Tile23, &A[TILE_M * KB + i], KB, m1);
                _tile_loadd(TMM3, Tile23, TILE_K);

                _tile_dpbssd(TMM5, TMM3, TMM0);
                _tile_dpbssd(TMM7, TMM3, TMM1);
                _tile_stored(TMM5, Tile5(C_cur), TILE_N * sizeof(int32_t));
                _tile_stored(TMM7, Tile7(C_cur), TILE_N * sizeof(int32_t));
                GGML_DISPATCH_BOOL(i > 0, is_acc, [&] {
                    acc_C<TA, TB, is_acc>::apply(C + TILE_M * ldc,          ldc, Tile5(C_cur), &A[TILE_M * KB + i], KB, B + PACKED_INDEX(0, i, KB, TILE_SIZE), m1);
                    acc_C<TA, TB, is_acc>::apply(C + TILE_M * ldc + TILE_N, ldc, Tile7(C_cur), &A[TILE_M * KB + i], KB, B + PACKED_INDEX(1, i, KB, TILE_SIZE), m1);
                });
            }
        }
    }
    return;
}

template <typename TA, typename TB, typename TC, int BLOCK_K,
          typename std::enable_if<is_type_qkk<TB>::value, int>::type = 0>
void tinygemm_kernel_amx(int M, int N, int KB, const void * RESTRICT _A, const void * RESTRICT _B, float * RESTRICT C, int ldc) {
    static_assert(std::is_same<TA, block_q8_K>::value);
    const int TILE_SIZE = get_tile_size<TB>();

    GGML_ASSERT(M <= 2 * TILE_M && N == 2 * TILE_N);
    const TA * RESTRICT A = static_cast<const TA *>(_A);
    const char * RESTRICT B = static_cast<const char *>(_B);

    const int m0 = std::min(M, TILE_M);
    const int m1 = std::max(M - TILE_M, 0);
    //const int lda = KB * sizeof(TA);

    alignas(64) static thread_local int8_t Tile0[TILE_N * TILE_K];
    alignas(64) static thread_local int8_t Tile1[TILE_N * TILE_K];
    alignas(64) static thread_local int8_t Tile23[TILE_M * TILE_K];

    // mat mul result for each group
    alignas(64) static thread_local int32_t Tile4[TILE_M * TILE_N];
    alignas(64) static thread_local int32_t Tile5[TILE_M * TILE_N];
    alignas(64) static thread_local int32_t Tile6[TILE_M * TILE_N];
    alignas(64) static thread_local int32_t Tile7[TILE_M * TILE_N];

    // sum of each QK_K block, contains 8 groups, int32
    alignas(64) static thread_local int32_t Sumi4[TILE_M * TILE_N];
    alignas(64) static thread_local int32_t Sumi5[TILE_M * TILE_N];
    alignas(64) static thread_local int32_t Sumi6[TILE_M * TILE_N];
    alignas(64) static thread_local int32_t Sumi7[TILE_M * TILE_N];

    const int k_group_size = std::is_same<TB, block_q6_K>::value ? 16 : 32;
    for (int i = 0; i < KB; ++i) {
        // step 1: accumulate the quants across 8 groups, each group with 32
        for (int k = 0; k < QK_K / k_group_size; ++k) {
            GGML_DISPATCH_BOOL(k > 0, is_acc, [&] {
                _tile_zero(TMM4);
                _tile_zero(TMM6);

                unpack_B<TB>(Tile0, B + PACKED_INDEX(0, i, KB, TILE_SIZE), k);
                _tile_loadd(TMM0, Tile0, TILE_N * VNNI_BLK);

                unpack_B<TB>(Tile1, B + PACKED_INDEX(1, i, KB, TILE_SIZE), k);
                _tile_loadd(TMM1, Tile1, TILE_N * VNNI_BLK);

                unpack_A<TB>(Tile23, &A[i], KB, k, m0);
                _tile_loadd(TMM2, Tile23, TILE_K);

                _tile_dpbssd(TMM4, TMM2, TMM0);
                _tile_dpbssd(TMM6, TMM2, TMM1);

                _tile_stored(TMM4, Tile4, TILE_N * sizeof(int32_t));
                _tile_stored(TMM6, Tile6, TILE_N * sizeof(int32_t));

                scale_C<TB, is_acc>(Tile4, Sumi4, B + PACKED_INDEX(0, i, KB, TILE_SIZE), k, m0);
                scale_C<TB, is_acc>(Tile6, Sumi6, B + PACKED_INDEX(1, i, KB, TILE_SIZE), k, m0);

                if (m1 != 0) {
                    _tile_zero(TMM5);
                    _tile_zero(TMM7);

                    unpack_A<TB>(Tile23, &A[TILE_M * KB + i], KB, k, m1);
                    _tile_loadd(TMM3, Tile23, TILE_K);

                    _tile_dpbssd(TMM5, TMM3, TMM0);
                    _tile_dpbssd(TMM7, TMM3, TMM1);

                    _tile_stored(TMM5, Tile5, TILE_N * sizeof(int32_t));
                    _tile_stored(TMM7, Tile7, TILE_N * sizeof(int32_t));

                    scale_C<TB, is_acc>(Tile5, Sumi5, B + PACKED_INDEX(0, i, KB, TILE_SIZE), k, m1);
                    scale_C<TB, is_acc>(Tile7, Sumi7, B + PACKED_INDEX(1, i, KB, TILE_SIZE), k, m1);
                }
            });
        }

        // step 2: accmulate the mins
        GGML_DISPATCH_BOOL(i > 0, is_acc, [&] {
            acc_C<TA, TB, is_acc>::apply(C,          ldc, Sumi4, &A[i], KB, B + PACKED_INDEX(0, i, KB, TILE_SIZE), m0);
            acc_C<TA, TB, is_acc>::apply(C + TILE_N, ldc, Sumi6, &A[i], KB, B + PACKED_INDEX(1, i, KB, TILE_SIZE), m0);
            if (m1 != 0) {
                acc_C<TA, TB, is_acc>::apply(C + TILE_M * ldc,          ldc, Sumi5, &A[TILE_M * KB + i], KB, B + PACKED_INDEX(0, i, KB, TILE_SIZE), m1);
                acc_C<TA, TB, is_acc>::apply(C + TILE_M * ldc + TILE_N, ldc, Sumi7, &A[TILE_M * KB + i], KB, B + PACKED_INDEX(1, i, KB, TILE_SIZE), m1);
            }
        });
    }
    return;
}

} // anonymous namespace

// get the packed tensor size for quantized weights
size_t ggml_backend_amx_get_alloc_size(const struct ggml_tensor * tensor) {
    const enum ggml_type TYPE = tensor->type;

    const int K = tensor->ne[0]; // ne0: in_features
    const int N = tensor->ne[1]; // ne1: out_features

    auto get_tensor_size = [&] {
        size_t row_size_B{0};
        GGML_DISPATCH_QTYPES(TYPE, [&] {
            row_size_B = get_row_size<type, blck_size>(K);
        });
        return N * row_size_B;
    };

    if (qtype_has_amx_kernels(TYPE)) {
        return get_tensor_size();
    } else {
        // for f16, bf16 we don't do packing
        return ggml_nbytes(tensor);
    }
}

// pack weight to vnni format
void ggml_backend_amx_convert_weight(struct ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    GGML_ASSERT(offset == 0 && size == ggml_nbytes(tensor)); // only full tensor conversion is supported for now

    const enum ggml_type TYPE = tensor->type;

    const int K = tensor->ne[0]; // ne0: in_features
    const int N = tensor->ne[1]; // ne1: out_features

    GGML_DISPATCH_QTYPES(TYPE, [&] {
        convert_B_packed_format<type, blck_size>((void *)((char *)tensor->data + offset), (const type *)data, N, K);
    });
#if defined(__AMX_INT8__) && defined(__AVX512VNNI__)
    amx_i8g_make_copy(tensor);
#endif
}

// ne2 is passed explicitly to help compiler optimize repeated calls
inline int64_t ggml_batch_offset(const ggml_tensor * t, int64_t batch_idx, int64_t ne2) {
    const int64_t i2 = batch_idx % ne2;
    const int64_t i3 = batch_idx / ne2;
    return i3 * t->nb[3] + i2 * t->nb[2];
}

size_t ggml_backend_amx_desired_wsize(const struct ggml_tensor * dst) {
    struct ggml_tensor * src0 = dst->src[0];

    const enum ggml_type TYPE = src0->type;

    const bool is_floating_type = TYPE == GGML_TYPE_F16;
    if (is_floating_type) {
        return 0;
    }

    const int M = dst->ne[1];
    const int K = src0->ne[0];
    const int64_t n_batch = dst->ne[2] * dst->ne[3];

    size_t desired_wsize = 0;

    GGML_DISPATCH_QTYPES(TYPE, [&] {
        const size_t row_size_A = K / blck_size * sizeof(vec_dot_type);
        desired_wsize = n_batch * M * row_size_A;
    });
    if (TYPE == GGML_TYPE_Q4_0 || TYPE == GGML_TYPE_Q8_0) {
        const size_t Mpad = (M + 31) / 32 * 32;  // FP16 / BF16 activations and the rows' scale factors
        desired_wsize = std::max(desired_wsize, (size_t) n_batch * Mpad * (K * sizeof(uint16_t) + sizeof(float)));
        // the grouped int8 copy's activations (int8, rows padded to 32) and scales (at most K / 64 per row)
        desired_wsize = std::max(desired_wsize, (size_t) n_batch * Mpad * (K + (K / 64) * sizeof(float)));
    }

    return desired_wsize;
}

// NB: mixed dtype gemm with Advanced Matrix Extensions (Intel AMX)
//
// src0: weight in shape of {N, K}, quantized
// src1: input  in shape of {M, K}, float32
// dst:  output in shape of {M, N}, float32
//
// the function performs: dst = src1 @ src0.T for each batch
//
void ggml_backend_amx_mul_mat(const ggml_compute_params * params, struct ggml_tensor * dst) {
    struct ggml_tensor * src0 = dst->src[0];
    struct ggml_tensor * src1 = dst->src[1];

    const enum ggml_type TYPE = src0->type;

    // f16 only has avx512 kernels for now,
    // amx kernels will be added once 6th gen xeon is released.
    const bool is_floating_type = TYPE == GGML_TYPE_F16;

    const int M = dst->ne[1];
    const int N = dst->ne[0];
    const int K = src0->ne[0];
    const int ldc = dst->nb[1] / dst->nb[0];

    const int64_t ne2 = dst->ne[2];
    const int64_t n_batch = ne2 * dst->ne[3];

    if (is_floating_type) {
        constexpr int BLOCK_M = 4;
        constexpr int BLOCK_N = 6;
        const int MB = div_up(M, BLOCK_M);
        const int NB = div_up(N, BLOCK_N);

        parallel_for_ggml(params, n_batch * MB * NB, [&](int begin, int end) {
            GGML_DISPATCH_FLOATING_TYPES(TYPE, [&] {
                for (int i = begin; i < end; ++i) {
                    int batch_idx = i / (MB * NB);
                    int remaining = i % (MB * NB);
                    int mb = remaining / NB;
                    int nb = remaining % NB;

                    int64_t src0_offset = ggml_batch_offset(src0, batch_idx, ne2);
                    int64_t src1_offset = ggml_batch_offset(src1, batch_idx, ne2);
                    int64_t dst_offset  = ggml_batch_offset(dst,  batch_idx, ne2);

                    int mb_start = mb * BLOCK_M;
                    int mb_size = std::min(BLOCK_M, M - mb_start);
                    int nb_start = nb * BLOCK_N;
                    int nb_size = std::min(BLOCK_N, N - nb_start);

                    switch (mb_size << 4 | nb_size) {
                        case 0x12: LAUNCH_TINYGEMM_KERNEL_AVX(1, 2); break;
                        case 0x14: LAUNCH_TINYGEMM_KERNEL_AVX(1, 4); break;
                        case 0x16: LAUNCH_TINYGEMM_KERNEL_AVX(1, 6); break;
                        case 0x22: LAUNCH_TINYGEMM_KERNEL_AVX(2, 2); break;
                        case 0x24: LAUNCH_TINYGEMM_KERNEL_AVX(2, 4); break;
                        case 0x26: LAUNCH_TINYGEMM_KERNEL_AVX(2, 6); break;
                        case 0x32: LAUNCH_TINYGEMM_KERNEL_AVX(3, 2); break;
                        case 0x34: LAUNCH_TINYGEMM_KERNEL_AVX(3, 4); break;
                        case 0x36: LAUNCH_TINYGEMM_KERNEL_AVX(3, 6); break;
                        case 0x42: LAUNCH_TINYGEMM_KERNEL_AVX(4, 2); break;
                        case 0x44: LAUNCH_TINYGEMM_KERNEL_AVX(4, 4); break;
                        case 0x46: LAUNCH_TINYGEMM_KERNEL_AVX(4, 6); break;
                        default: fprintf(stderr, "Unexpected block size!\n");
                    }
                }
            });
        });
        return;
    }

    // pointer to work space, used convert A from float to quantized type
    void * wdata = params->wdata;

#if defined(__AMX_INT8__) && defined(__AVX512VNNI__)
    // passes of GGML_AMX_I8G_MIN_M or more rows with a grouped int8 copy of the weights (amx_i8g.inc)
    if (M >= amx_i8g_min_m()) {
        if (const amx_i8g_copy * cp = amx_i8g_find(src0->data); cp != nullptr && cp->K == K && cp->N == N) {
            const int G = cp->G, NG = K / G, KS = K / 64;
            const int Mpad = div_up(M, 32) * 32;
            int8_t * Aq = (int8_t *) wdata;                                // [n_batch][Mpad * K], tile order
            float * as = (float *)(Aq + (size_t) n_batch * Mpad * K);       // [n_batch][Mpad][NG]
            parallel_for_ggml(params, n_batch * Mpad, [&](int begin, int end) {
                for (int idx = begin; idx < end; ++idx) {
                    const int batch_idx = idx / Mpad;
                    const int m = idx % Mpad;
                    int8_t * y = Aq + (size_t) batch_idx * Mpad * K + (size_t) (m / 16) * KS * 1024 + (m % 16) * 64;
                    const float * x = m < M ? (const float *)((const char *) src1->data + ggml_batch_offset(src1, batch_idx, ne2)) + (size_t) m * K : nullptr;
                    amx_i8g_quantize_row(x, y, as + (size_t) idx * NG, K, G);
                }
            });
            ggml_barrier(params->threadpool);
            const bool rows16 = M <= 16;              // one activation tile, 64 columns per task
            const int NB = rows16 ? div_up(N, 64) : N / 32;
            parallel_for_ggml(params, n_batch * NB, [&](int begin, int end) {
                if (begin >= end) {
                    return;
                }
                amx_half_tile_config();
                for (int i = begin; i < end; ++i) {
                    const int batch_idx = i / NB;
                    const int nb = i % NB;
                    const int64_t dst_offset = ggml_batch_offset(dst, batch_idx, ne2);
                    const int8_t * A = Aq + (size_t) batch_idx * Mpad * K;
                    const float * asb = as + (size_t) batch_idx * Mpad * NG;
                    float * C = (float *) dst->data + dst_offset;
                    int ncols;
                    if (rows16) {
                        const int nt = std::min(4, N / 16 - nb * 4);
                        amx_i8g_slice16(*cp, nb * 4, nt, M, A, asb, C, ldc);
                        ncols = nt * 16;
                    } else {
                        amx_i8g_slice(*cp, nb, M, A, asb, C, ldc);
                        ncols = 32;
                    }
                    if (amx_i8g_check()) {
                        for (int s = 0; s < 2; ++s) {
                            const int m = (nb * 7 + s * 13) % M;
                            const int n = nb * (rows16 ? 64 : 32) + (nb * 5 + s * 17) % ncols;
                            double abs_sum = 0;
                            const double ref = amx_i8g_ref(*cp, n, A, asb, m, &abs_sum);
                            const double got = C[(size_t) m * ldc + n];
                            if (abs_sum > 0) {
                                amx_i8g_report(std::fabs(got - ref) / abs_sum);
                            }
                        }
                    }
                }
                amx_int8_tile_config_restore();
            });
            return;
        }
    }
#endif

#if defined(GGML_AMX_HALF_KERNELS)
    // Q4_0 / Q8_0 with GGML_AMX_HALF_MIN_M or more rows: FP16 / BF16 tiles (amx_half.inc) instead of the int8 kernel
    if ((TYPE == GGML_TYPE_Q4_0 || TYPE == GGML_TYPE_Q8_0) && amx_half_min_m() > 0 && M >= amx_half_min_m()) {
        const bool fp16 = amx_half_use_fp16();
        const int Mpad = M <= 16 ? 16 : div_up(M, 32) * 32;
        uint16_t * Ah = (uint16_t *) wdata;                                // [n_batch][Mpad * K], tile order
        float * inv = (float *)(Ah + (size_t) n_batch * Mpad * K);         // [n_batch][Mpad]
        // activations to FP16 / BF16, rows split over the threads; padding rows zero
        parallel_for_ggml(params, n_batch * Mpad, [&](int begin, int end) {
            for (int idx = begin; idx < end; ++idx) {
                const int batch_idx = idx / Mpad;
                const int m = idx % Mpad;
                // tile order: 16-row blocks of KB tiles of 16 rows x 32 values
                uint16_t * y = Ah + (size_t) batch_idx * Mpad * K + (size_t) (m / 16) * K * 16 + (m % 16) * 32;
                const float * x = m < M ? (const float *)((const char *) src1->data + ggml_batch_offset(src1, batch_idx, ne2)) + (size_t) m * K : nullptr;
                amx_half_convert_A(fp16, x, y, K, inv + idx);
            }
        });
        ggml_barrier(params->threadpool);

        constexpr int BLOCK_N = 4 * TILE_N;  // 64 columns per slice (32 at the edge of N)
        const int NB = div_up(N, BLOCK_N);
        parallel_for_ggml(params, n_batch * NB, [&](int begin, int end) {
            if (begin >= end) {
                return;
            }
            GGML_DISPATCH_QTYPES(TYPE, [&] {
                if constexpr (std::is_same<type, block_q4_0>::value || std::is_same<type, block_q8_0>::value) {
                    const int KB = K / blck_size;
                    const int TILE_SIZE = get_tile_size<type>();
                    static thread_local std::vector<uint16_t> Bh;
                    if (amx_half_buffer_m() > 0 && M >= amx_half_buffer_m()) {
                        Bh.resize((size_t) KB * 4 * 512);
                    }
                    amx_half_tile_config();
                    for (int i = begin; i < end; ++i) {
                        const int batch_idx = i / NB;
                        const int nb = i % NB;
                        const int64_t src0_offset = ggml_batch_offset(src0, batch_idx, ne2);
                        const int64_t dst_offset  = ggml_batch_offset(dst,  batch_idx, ne2);
                        const int nt = std::min(BLOCK_N, N - nb * BLOCK_N) / TILE_N;
                        // the macro does not parenthesize n
                        const char * B = (const char *) src0->data + src0_offset + PACKED_INDEX((nb * 4), 0, KB, TILE_SIZE);
                        const uint16_t * A = Ah + (size_t) batch_idx * Mpad * K;
                        float * C = (float *) dst->data + dst_offset + nb * BLOCK_N;
                        amx_half_slice<type>(fp16, M, K, A, inv + (size_t) batch_idx * Mpad, B, nt, Bh.data(), C, ldc);
                        if (amx_half_check()) {  // GGML_AMX_HALF_CHECK=1: two sampled outputs per slice
                            for (int s = 0; s < 2; ++s) {
                                const int m = (nb * 7 + s * 13) % M;
                                const int col = (nb * 5 + s * 17) % (nt * TILE_N);
                                const float * x = (const float *)((const char *) src1->data + ggml_batch_offset(src1, batch_idx, ne2)) + (size_t) m * K;
                                double abs_sum = 0;
                                const double ref = amx_half_ref_dot<type>(B, KB, col, x, false, &abs_sum);
                                const double ref_int8 = amx_half_ref_dot<type>(B, KB, col, x, true);
                                if (abs_sum > 0) {
                                    amx_half_report(std::fabs(C[(size_t) m * ldc + col] - ref), std::fabs(ref_int8 - ref), abs_sum);
                                }
                            }
                        }
                    }
                    amx_int8_tile_config_restore();
                }
            });
        });
        return;
    }
#endif

    //TODO: performance improvement: merge quant A
 // if (params->ith == 0) {
        GGML_DISPATCH_QTYPES(TYPE, [&] {
            const size_t row_size_A = K / blck_size * sizeof(vec_dot_type);
            const size_t desired_wsize = n_batch * M * row_size_A;
            if (params->wsize < desired_wsize) {
                GGML_ABORT("insufficient work space size");
            }

            // Q4_0, Q4_1, Q8_0 handles 1 TILE_K per blck_size
            // Q4_K, Q5_K, Q6_K, IQ4_XS handles 8 TILE_K per blck_size
            GGML_ASSERT(TILE_K == blck_size || TILE_K * 8 == blck_size);

            parallel_for_ggml(params, n_batch * M, [&](int begin, int end) {
                for (int idx = begin; idx < end; ++idx) {
                    int batch_idx = idx / M;
                    int m         = idx % M;
                    int64_t src1_offset = ggml_batch_offset(src1, batch_idx, ne2);
                    const float * A_data = (const float *)((const char *)src1->data + src1_offset);
                    char * wdata_batch = (char *)wdata + batch_idx * M * row_size_A;
                    from_float<vec_dot_type>(A_data + m * K, wdata_batch + m * row_size_A, K);
                }
            });
        });
 // }

    ggml_barrier(params->threadpool);

    // small batches of Q4_0 / Q8_0: multi-row VNNI kernels (vnni_rows.inc) instead of AMX tiles
    if (M > 1 && M <= vnni_rows_max_m() && (TYPE == GGML_TYPE_Q4_0 || TYPE == GGML_TYPE_Q8_0)) {
        constexpr int BLOCK_N = TILE_N * 4;
        const int NB = div_up(N, BLOCK_N);

        parallel_for_ggml(params, n_batch * NB, [&](int begin, int end) {
            GGML_DISPATCH_QTYPES(TYPE, [&] {
                if constexpr (std::is_same<type, block_q4_0>::value || std::is_same<type, block_q8_0>::value) {
                    const int KB = K / blck_size;
                    const int TILE_SIZE = get_tile_size<type>();
                    const int row_size_A = KB * sizeof(vec_dot_type);
                    static thread_local std::vector<int32_t> acomp;
                    int64_t prepared = -1;
                    for (int i = begin; i < end; ++i) {
                        const int batch_idx = i / NB;
                        const int nb = i % NB;
                        const int64_t src0_offset = ggml_batch_offset(src0, batch_idx, ne2);
                        const int64_t dst_offset  = ggml_batch_offset(dst,  batch_idx, ne2);
                        const char * wdata_batch = (const char *)wdata + batch_idx * M * row_size_A;
                        if (std::is_same<type, block_q4_0>::value && prepared != batch_idx) {
                            acomp.resize((size_t)M * KB);
                            vnni_rows_prepare((const block_q8_0 *)wdata_batch, M * KB, acomp.data());
                            prepared = batch_idx;
                        }
                        const int nb_start = nb * BLOCK_N;
                        const int tiles = std::min(BLOCK_N, N - nb_start) / TILE_N;
                        const int cols = M <= 4 && tiles == 4 ? 64 : 32;  // 4 rows x 64 or 8 rows x 32 columns per call
                        const int rows_per_call = cols == 64 ? 4 : VNNI_ROWS_MAX;
                        for (int t = 0; t < tiles; t += cols / TILE_N) {
                            const char * B = (const char *)src0->data + src0_offset + PACKED_INDEX((nb * 4 + t), 0, KB, TILE_SIZE);  // the macro does not parenthesize n
                            for (int m0 = 0; m0 < M; m0 += rows_per_call) {
                                const int rows = std::min(rows_per_call, M - m0);
                                const block_q8_0 * A = (const block_q8_0 *)(wdata_batch + m0 * row_size_A);
                                float * C = (float *)dst->data + dst_offset + m0 * ldc + nb_start + t * TILE_N;
                                vnni_rows<type>(rows, cols, KB, A, acomp.data() + m0 * KB, B, C, ldc);
                                if (vnni_rows_check()) {  // GGML_AMX_VNNI_CHECK=1: compare with the M == 1 kernel
                                    for (int r = 0; r < rows; ++r) {
                                        for (int h = 0; h < cols; h += 32) {
                                            alignas(64) float ref[32];
                                            tinygemm_kernel_vnni<vec_dot_type, type, float, 1, 32, blck_size>::apply(
                                                KB, A + r * KB, B + PACKED_INDEX((h / TILE_N), 0, KB, TILE_SIZE), ref, 32);
                                            vnni_rows_compare(ref, C + r * ldc + h, 32, M, N, K);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            });
        });
        return;
    }

    if (M == 1) {
        // MB = 1 and handle 8 tiles in each block
        constexpr int kTilesN = 4;
        constexpr int BLOCK_N = TILE_N * kTilesN;
        const int NB = div_up(N, BLOCK_N);

        parallel_for_ggml(params, n_batch * NB, [&](int begin, int end) {
            GGML_DISPATCH_QTYPES(TYPE, [&] {
                const int KB = K / blck_size;
                const int TILE_SIZE = get_tile_size<type>();
                const int row_size_A = KB * sizeof(vec_dot_type);
                for (int i = begin; i < end; ++i) {
                    int batch_idx = i / NB;
                    int nb = i % NB;

                    int64_t src0_offset = ggml_batch_offset(src0, batch_idx, ne2);
                    int64_t dst_offset  = ggml_batch_offset(dst,  batch_idx, ne2);
                    const char * wdata_batch = (const char *)wdata + batch_idx * row_size_A;

                    int nb_start = nb * BLOCK_N;
                    int nb_size = std::min(BLOCK_N, N - nb_start); // 32, 64, 96

                    switch (nb_size) {
                        //case 160: LAUNCH_TINYGEMM_KERNEL_VNNI(160); break;
                        case 128: LAUNCH_TINYGEMM_KERNEL_VNNI(128); break;
                        case 96: LAUNCH_TINYGEMM_KERNEL_VNNI(96); break;
                        case 64: LAUNCH_TINYGEMM_KERNEL_VNNI(64); break;
                        case 32: LAUNCH_TINYGEMM_KERNEL_VNNI(32); break;
                        default: fprintf(stderr, "Unexpected n block size!\n");
                    }
                }
            });
        });
        return;
    }

    // handle 4 tiles at a tile
    constexpr int BLOCK_M = TILE_M * 2;
    constexpr int BLOCK_N = TILE_N * 2;
    const int MB = div_up(M, BLOCK_M);
    const int NB = div_up(N, BLOCK_N);

    parallel_for_ggml(params, n_batch * MB * NB, [&](int begin, int end) {
        // init tile config for each thread
        ggml_tile_config_init();

        GGML_DISPATCH_QTYPES(TYPE, [&] {
            const int KB = K / blck_size;
            const int TILE_SIZE = get_tile_size<type>();
            const int row_size_A = KB * sizeof(vec_dot_type);

            for (int i = begin; i < end; ++i) {
                int batch_idx = i / (MB * NB);
                int remaining = i % (MB * NB);
                int mb = remaining / NB;
                int nb = remaining % NB;

                int64_t src0_offset = ggml_batch_offset(src0, batch_idx, ne2);
                int64_t dst_offset  = ggml_batch_offset(dst,  batch_idx, ne2);
                const char * wdata_batch = (const char *)wdata + batch_idx * M * row_size_A;

                int mb_start = mb * BLOCK_M;
                int mb_size = std::min(BLOCK_M, M - mb_start);
                int nb_start = nb * BLOCK_N;
                int nb_size = BLOCK_N;

                tinygemm_kernel_amx<vec_dot_type, type, float, blck_size>(
                    mb_size, nb_size, KB,
                    wdata_batch + mb_start * row_size_A,
                    (const char *)src0->data + src0_offset + PACKED_INDEX(nb * 2, 0, KB, TILE_SIZE),
                    (float *) dst->data + dst_offset + mb_start * N + nb_start, ldc);
            }
        });
    });
}

#endif // if defined(__AMX_INT8__) && defined(__AVX512VNNI__)
