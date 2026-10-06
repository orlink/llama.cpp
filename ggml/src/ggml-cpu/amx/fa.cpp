// CPU flash attention on AMX-FP16 tiles (TDPFP16PS: Granite Rapids and later).
//
// The query heads that share one KV head are packed into the same tiles, query-major: tile row r is query r / gqa,
// head r % gqa of the group. A 1-token pass of a model with 8 query heads per KV head fills half of one tile instead
// of 1/16 of eight tiles, and each tile reads the cached keys and values once for all of its heads.
//
// Per tile of 16 rows, over blocks of 32 cached keys:
//   scores   S^T (keys x rows) = K . Q^T: K rows straight from the F16 cache, the queries rounded to F16 and repacked
//            into AMX's pair layout once per tile; F32 sums
//   softmax  online, in F32 with AVX-512, one lane per row
//   values   O (rows x dims) += P . V: the weights P rounded to F16, V repacked into pairs of keys per block; O in F32
// Key blocks masked for every row of a tile are skipped (causal and sliding-window masks).
// When there are fewer tiles than threads (short passes), the keys are split among threads as well and the parts are
// merged at the end.
//
// Numerics: queries and softmax weights rounded to F16 (ggml's vector path also rounds the queries, and sums the values
// in F16); scores, softmax and value sums in F32.
// Supported: F32 queries, F16 keys and values, head sizes divisible by 32 (keys) and 16 (values), cached keys in blocks
// of 32, no ALiBi, logit soft-capping or sinks. Everything else, and single queries over 256 or fewer keys, goes to
// ggml's own paths. GGML_FA_AMX=0 turns it off.

#include "fa.h"

#include "ggml-cpu-impl.h"
#include "ggml-impl.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>

#if defined(__AMX_TILE__) && defined(__AMX_FP16__) && defined(__AVX512F__) && defined(__AVX512BW__) && defined(__linux__)
#define GGML_FA_AMX_KERNEL 1
#include <immintrin.h>
#include <sys/syscall.h>
#include <unistd.h>
#endif

// per-thread scratch: Q rows in F16 (16 x DK), Q pairs (16 x DK), V pairs (32 x DV), O (16 x DV F32),
// scores (32 x 16 F32), weights (16 x 32 F16), mask (32 x 16 F32), m and l (2 x 16 F32)
static size_t fa_amx_thread_bytes(int64_t DK, int64_t DV) {
    return (size_t) (64 * DK + 128 * DV + 2048 + 1024 + 2048 + 128);
}

size_t ggml_fa_amx_work_size(int64_t DK, int64_t DV, int n_tasks) {
    // plus the partial results of fewer than 2 x n_tasks tiles when the keys are split among threads
    return fa_amx_thread_bytes(DK, DV) * n_tasks + (size_t) 2 * n_tasks * 16 * (DV + 2) * sizeof(float) + 64;
}

#if defined(GGML_FA_AMX_KERNEL)

namespace {

struct fa_tile_cfg {
    uint8_t  palette_id;
    uint8_t  start_row;
    uint8_t  reserved[14];
    uint16_t colsb[16];
    uint8_t  rows[16];
};

// all tiles 16 rows x 64 bytes: 0-1 scores, 2 keys, 3 query pairs, 4 output, 5 weights, 6 value pairs
void fa_tile_config() {
    alignas(64) fa_tile_cfg c = {};
    c.palette_id = 1;
    for (int t = 0; t < 8; ++t) {
        c.rows[t]  = 16;
        c.colsb[t] = 64;
    }
    _tile_loadconfig(&c);
}

// ggml's int8 configuration (ggml_tile_config_init in mmq.cpp loads it only once per thread)
void fa_tile_config_restore() {
    alignas(64) fa_tile_cfg c = {};
    c.palette_id = 1;
    c.rows[0] = 8;   c.colsb[0] = 64;
    c.rows[1] = 8;   c.colsb[1] = 64;
    c.rows[2] = 16;  c.colsb[2] = 32;
    c.rows[3] = 16;  c.colsb[3] = 32;
    for (int t = 4; t < 8; ++t) {
        c.rows[t]  = 16;
        c.colsb[t] = 64;
    }
    _tile_loadconfig(&c);
}

// GCC's _tile_loadd / _tile_stored have no memory clobber: keep the AVX-512 stores and tile loads in order
#define FA_AMX_MEMORY_BARRIER() __asm__ __volatile__("" ::: "memory")

bool fa_amx_permission() {
    // ARCH_REQ_XCOMP_PERM (0x1023) for XFEATURE_XTILEDATA (18)
    static const bool ok = syscall(SYS_arch_prctl, 0x1023, 18) == 0;
    return ok;
}

bool fa_amx_enabled() {
    static const bool v = [] {
        const char * s = getenv("GGML_FA_AMX");
        return !(s && atoi(s) == 0);
    }();
    return v;
}

// GGML_FA_AMX_MIN_SPLIT: fewest blocks of 32 keys per thread when the keys are split (short passes)
int64_t fa_amx_min_split_blocks() {
    static const int64_t v = [] {
        const char * s = getenv("GGML_FA_AMX_MIN_SPLIT");
        return s ? std::max(1, atoi(s)) : 2;
    }();
    return v;
}

// GGML_FA_AMX_MIN_Q: fewest queries per pass sent to AMX (default 1; 64 = only where ggml's tiled F32 path runs)
int64_t fa_amx_min_queries() {
    static const int64_t v = [] {
        const char * s = getenv("GGML_FA_AMX_MIN_Q");
        return s ? std::max(1, atoi(s)) : 1;
    }();
    return v;
}

bool fa_amx_supported(const ggml_tensor * dst) {
    const ggml_tensor * q     = dst->src[0];
    const ggml_tensor * k     = dst->src[1];
    const ggml_tensor * v     = dst->src[2];
    const ggml_tensor * mask  = dst->src[3];
    const ggml_tensor * sinks = dst->src[4];

    if (q->type != GGML_TYPE_F32 || k->type != GGML_TYPE_F16 || v->type != GGML_TYPE_F16 || sinks) {
        return false;
    }
    if (mask && mask->type != GGML_TYPE_F16) {
        return false;
    }
    float max_bias, logit_softcap;
    memcpy(&max_bias,      (const float *) dst->op_params + 1, sizeof(float));
    memcpy(&logit_softcap, (const float *) dst->op_params + 2, sizeof(float));
    if (max_bias != 0.0f || logit_softcap != 0.0f) {
        return false;
    }
    if (q->ne[1] < fa_amx_min_queries()) {
        return false;
    }
    // one query over a short cache: ggml's vector path has less fixed cost
    if (q->ne[1] == 1 && k->ne[1] <= 256) {
        return false;
    }
    const int64_t DK = k->ne[0], DV = v->ne[0];
    if (DK % 32 != 0 || DV % 16 != 0 || k->ne[1] == 0 || k->ne[1] % 32 != 0 || v->ne[1] != k->ne[1]) {
        return false;
    }
    if (k->ne[2] != v->ne[2] || k->ne[3] != v->ne[3] || q->ne[2] % k->ne[2] != 0 || q->ne[3] % k->ne[3] != 0) {
        return false;
    }
    return q->nb[0] == sizeof(float) && k->nb[0] == sizeof(ggml_fp16_t) && v->nb[0] == sizeof(ggml_fp16_t) &&
           dst->nb[0] == sizeof(float);
}

// exp for 16 floats: 2^k * p(r), r = x - k ln2, |r| <= ln2/2, degree-6 polynomial (relative error ~2e-7)
inline __m512 fa_exp16(__m512 x) {
    x = _mm512_max_ps(x, _mm512_set1_ps(-100.0f));
    x = _mm512_min_ps(x, _mm512_set1_ps(88.0f));
    const __m512 k = _mm512_roundscale_ps(_mm512_mul_ps(x, _mm512_set1_ps(1.44269504f)), _MM_FROUND_TO_NEAREST_INT);
    __m512 r = _mm512_fnmadd_ps(k, _mm512_set1_ps(0.693145751953125f), x);
    r = _mm512_fnmadd_ps(k, _mm512_set1_ps(1.428606765330187e-06f), r);
    __m512 p = _mm512_set1_ps(1.0f / 720);
    p = _mm512_fmadd_ps(p, r, _mm512_set1_ps(1.0f / 120));
    p = _mm512_fmadd_ps(p, r, _mm512_set1_ps(1.0f / 24));
    p = _mm512_fmadd_ps(p, r, _mm512_set1_ps(1.0f / 6));
    p = _mm512_fmadd_ps(p, r, _mm512_set1_ps(0.5f));
    p = _mm512_fmadd_ps(p, r, _mm512_set1_ps(1.0f));
    p = _mm512_fmadd_ps(p, r, _mm512_set1_ps(1.0f));
    return _mm512_scalef_ps(p, k);
}

struct fa_shape {
    const ggml_tensor * q, * k, * v, * mask, * dst;
    int64_t DK, DV, N, gqa, Hkv, rk3, rows, tiles, splits, blocks_per_split, nblocks;
    float scale;
};

struct fa_scratch {
    uint16_t * qrows;  // 16 x DK, F16
    uint32_t * qp;     // [DK/32][16 pair rows][16 rows], pairs of F16
    uint16_t * vp;     // [DV/16][16 pair rows][16 dims x 2], F16
    float    * o;      // 16 x DV
    float    * s;      // 32 x 16 (keys x rows)
    uint16_t * p;      // 16 x 32 (rows x keys), F16
    float    * mt;     // 32 x 16 (keys x rows)
    float    * m;      // 16
    float    * l;      // 16
};

// destination row of query iq1, head hq
inline float * fa_dst_row(const ggml_tensor * dst, int64_t iq1, int64_t hq, int64_t iq3) {
    return (float *) ((char *) dst->data + (iq3 * dst->ne[2] * dst->ne[1] + hq + iq1 * dst->ne[1]) * dst->nb[1]);
}

// one tile of 16 rows of one KV head group over one split of the keys
void fa_tile(const fa_shape & sh, const fa_scratch & w, int64_t item, float * partials) {
    const ggml_tensor * q = sh.q, * k = sh.k, * v = sh.v, * mask = sh.mask;
    const int64_t DK = sh.DK, DV = sh.DV;

    const int64_t split = item % sh.splits;
    const int64_t bt    = item / sh.splits;
    const int64_t t     = bt % sh.tiles;
    const int64_t grp   = bt / sh.tiles;
    const int64_t ik2   = grp % sh.Hkv;
    const int64_t iq3   = grp / sh.Hkv;
    const int64_t ik3   = iq3 / sh.rk3;
    const int64_t r0    = t * 16;
    const int     nr    = (int) std::min<int64_t>(16, sh.rows - r0);

    const uint16_t * mrow[16] = {};
    for (int n = 0; n < 16; ++n) {
        uint16_t * dq = w.qrows + n * DK;
        if (n >= nr) {
            memset(dq, 0, DK * sizeof(uint16_t));
            continue;
        }
        const int64_t r   = r0 + n;
        const int64_t iq1 = r / sh.gqa;
        const int64_t hq  = ik2 * sh.gqa + r % sh.gqa;
        const float * qf  = (const float *) ((const char *) q->data + iq1 * q->nb[1] + hq * q->nb[2] + iq3 * q->nb[3]);
        for (int64_t d = 0; d < DK; d += 16) {
            _mm256_storeu_si256((__m256i *) (dq + d), _mm512_cvtps_ph(_mm512_loadu_ps(qf + d), _MM_FROUND_TO_NEAREST_INT));
        }
        if (mask) {
            mrow[n] = (const uint16_t *) ((const char *) mask->data + iq1 * mask->nb[1] +
                                          (hq % mask->ne[2]) * mask->nb[2] + (iq3 % mask->ne[3]) * mask->nb[3]);
        }
    }
    // pairs of F16 values: qp[c][pr][n] = (q[n][32c + 2pr], q[n][32c + 2pr + 1])
    for (int n = 0; n < 16; ++n) {
        const uint32_t * src = (const uint32_t *) (w.qrows + n * DK);
        for (int64_t c = 0; c < DK / 32; ++c) {
            for (int pr = 0; pr < 16; ++pr) {
                w.qp[(c * 16 + pr) * 16 + n] = src[c * 16 + pr];
            }
        }
    }

    memset(w.o, 0, 16 * DV * sizeof(float));
    for (int n = 0; n < 16; ++n) {
        w.m[n] = -INFINITY;
        w.l[n] = 0.0f;
    }

    const char * kbase = (const char *) k->data + ik2 * k->nb[2] + ik3 * k->nb[3];
    const char * vbase = (const char *) v->data + ik2 * v->nb[2] + ik3 * v->nb[3];
    const int64_t b0 = split * sh.blocks_per_split;
    const int64_t b1 = std::min(sh.nblocks, b0 + sh.blocks_per_split);
    const __m512i neg_inf16 = _mm512_set1_epi16((short) 0xFC00);
    const __m512i abs16     = _mm512_set1_epi16(0x7FFF);

    for (int64_t b = b0; b < b1; ++b) {
        const int64_t n0 = b * 32;

        // mask: skip the block when it is masked for every row; no addition when it is all zeros
        bool any = mask == nullptr, zeros = true;
        if (mask) {
            const uint16_t * prev = nullptr;
            for (int n = 0; n < nr; ++n) {
                if (mrow[n] == prev) {
                    continue;
                }
                prev = mrow[n];
                const __m512i x = _mm512_loadu_si512((const void *) (mrow[n] + n0));
                any   |= _mm512_cmpeq_epi16_mask(x, neg_inf16) != (__mmask32) 0xFFFFFFFF;
                zeros &= _mm512_test_epi16_mask(x, abs16) == 0;
            }
            if (!any) {
                continue;
            }
            if (!zeros) {
                for (int n = 0; n < 16; ++n) {
                    alignas(64) float row[32];
                    if (n < nr) {
                        _mm512_store_ps(row,      _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *) (mrow[n] + n0))));
                        _mm512_store_ps(row + 16, _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *) (mrow[n] + n0 + 16))));
                    } else {
                        std::fill(row, row + 32, -INFINITY);
                    }
                    for (int kk = 0; kk < 32; ++kk) {
                        w.mt[kk * 16 + n] = row[kk];
                    }
                }
            }
        }

        // scores, keys x rows
        const char * kb = kbase + n0 * k->nb[1];
        FA_AMX_MEMORY_BARRIER();
        _tile_zero(0);
        _tile_zero(1);
        for (int64_t c = 0; c < DK / 32; ++c) {
            _tile_loadd(3, w.qp + c * 16 * 16, 64);
            _tile_loadd(2, kb + 64 * c, k->nb[1]);
            _tile_dpfp16ps(0, 2, 3);
            _tile_loadd(2, kb + 16 * k->nb[1] + 64 * c, k->nb[1]);
            _tile_dpfp16ps(1, 2, 3);
        }
        _tile_stored(0, w.s, 64);
        _tile_stored(1, w.s + 16 * 16, 64);
        FA_AMX_MEMORY_BARRIER();

        // online softmax, one lane per row
        const __m512 old_m = _mm512_load_ps(w.m);
        __m512 mx = old_m;
        __m512 sv[32];
        for (int kk = 0; kk < 32; ++kk) {
            sv[kk] = _mm512_mul_ps(_mm512_load_ps(w.s + kk * 16), _mm512_set1_ps(sh.scale));
            if (!zeros) {
                sv[kk] = _mm512_add_ps(sv[kk], _mm512_load_ps(w.mt + kk * 16));
            }
            mx = _mm512_max_ps(mx, sv[kk]);
        }
        const __mmask16 live = _mm512_cmp_ps_mask(mx, _mm512_set1_ps(-INFINITY), _CMP_NEQ_OQ);
        const __m512 corr = _mm512_mask_blend_ps(live, _mm512_set1_ps(1.0f), fa_exp16(_mm512_sub_ps(old_m, mx)));
        __m512 sum = _mm512_mul_ps(_mm512_load_ps(w.l), corr);
        for (int kk = 0; kk < 32; ++kk) {
            const __m512 e = _mm512_maskz_mov_ps(live, fa_exp16(_mm512_sub_ps(sv[kk], mx)));
            sum = _mm512_add_ps(sum, e);
            alignas(32) uint16_t eh[16];
            _mm256_store_si256((__m256i *) eh, _mm512_cvtps_ph(e, _MM_FROUND_TO_NEAREST_INT));
            for (int n = 0; n < 16; ++n) {
                w.p[n * 32 + kk] = eh[n];
            }
        }
        _mm512_store_ps(w.l, sum);
        _mm512_store_ps(w.m, _mm512_mask_blend_ps(live, old_m, mx));
        alignas(64) float cf[16];
        _mm512_store_ps(cf, corr);
        for (int n = 0; n < nr; ++n) {
            if (cf[n] != 1.0f) {
                const __m512 f = _mm512_set1_ps(cf[n]);
                for (int64_t d = 0; d < DV; d += 16) {
                    _mm512_storeu_ps(w.o + n * DV + d, _mm512_mul_ps(_mm512_loadu_ps(w.o + n * DV + d), f));
                }
            }
        }

        // values -> pairs of keys: vp[c][pr][2j + i] = v[n0 + 2pr + i][16c + j]
        const char * vb = vbase + n0 * v->nb[1];
        for (int64_t c = 0; c < DV / 16; ++c) {
            for (int pr = 0; pr < 16; ++pr) {
                const __m256i a  = _mm256_loadu_si256((const __m256i *) (vb + (2 * pr)     * v->nb[1] + 32 * c));
                const __m256i bb = _mm256_loadu_si256((const __m256i *) (vb + (2 * pr + 1) * v->nb[1] + 32 * c));
                const __m256i lo = _mm256_unpacklo_epi16(a, bb), hi = _mm256_unpackhi_epi16(a, bb);
                uint16_t * dst = w.vp + (c * 16 + pr) * 32;
                _mm256_storeu_si256((__m256i *) dst,        _mm256_permute2x128_si256(lo, hi, 0x20));
                _mm256_storeu_si256((__m256i *) (dst + 16), _mm256_permute2x128_si256(lo, hi, 0x31));
            }
        }

        // O += P . V
        FA_AMX_MEMORY_BARRIER();
        _tile_loadd(5, w.p, 64);
        for (int64_t c = 0; c < DV / 16; ++c) {
            _tile_loadd(4, w.o + 16 * c, DV * sizeof(float));
            _tile_loadd(6, w.vp + c * 16 * 32, 64);
            _tile_dpfp16ps(4, 5, 6);
            _tile_stored(4, w.o + 16 * c, DV * sizeof(float));
        }
        FA_AMX_MEMORY_BARRIER();
    }

    if (sh.splits > 1) {
        for (int n = 0; n < nr; ++n) {
            float * pp = partials + (item * 16 + n) * (DV + 2);
            pp[0] = w.m[n];
            pp[1] = w.l[n];
            memcpy(pp + 2, w.o + n * DV, DV * sizeof(float));
        }
        return;
    }
    for (int n = 0; n < nr; ++n) {
        const int64_t r   = r0 + n;
        float * out = fa_dst_row(sh.dst, r / sh.gqa, ik2 * sh.gqa + r % sh.gqa, iq3);
        const __m512 inv = _mm512_set1_ps(w.l[n] > 0.0f ? 1.0f / w.l[n] : 0.0f);
        for (int64_t d = 0; d < DV; d += 16) {
            _mm512_storeu_ps(out + d, _mm512_mul_ps(_mm512_loadu_ps(w.o + n * DV + d), inv));
        }
    }
}

// merge the key splits of one row
void fa_merge_row(const fa_shape & sh, const float * partials, int64_t bt, int n, float * acc) {
    const int64_t DV = sh.DV;
    const int64_t t = bt % sh.tiles, grp = bt / sh.tiles;
    const int64_t r = t * 16 + n;
    if (r >= sh.rows) {
        return;
    }
    float M = -INFINITY, S = 0.0f;
    memset(acc, 0, DV * sizeof(float));
    for (int64_t sp = 0; sp < sh.splits; ++sp) {
        const float * pp = partials + ((bt * sh.splits + sp) * 16 + n) * (DV + 2);
        if (pp[1] == 0.0f) {
            continue;
        }
        const float Mn = std::max(M, pp[0]);
        const __m512 a = _mm512_set1_ps(expf(M - Mn)), b = _mm512_set1_ps(expf(pp[0] - Mn));
        for (int64_t d = 0; d < DV; d += 16) {
            _mm512_storeu_ps(acc + d, _mm512_fmadd_ps(_mm512_loadu_ps(acc + d), a,
                                                      _mm512_mul_ps(_mm512_loadu_ps(pp + 2 + d), b)));
        }
        S = S * expf(M - Mn) + pp[1] * expf(pp[0] - Mn);
        M = Mn;
    }
    const int64_t ik2 = grp % sh.Hkv, iq3 = grp / sh.Hkv;
    float * out = fa_dst_row(sh.dst, r / sh.gqa, ik2 * sh.gqa + r % sh.gqa, iq3);
    const __m512 inv = _mm512_set1_ps(S > 0.0f ? 1.0f / S : 0.0f);
    for (int64_t d = 0; d < DV; d += 16) {
        _mm512_storeu_ps(out + d, _mm512_mul_ps(_mm512_loadu_ps(acc + d), inv));
    }
}

} // namespace

#endif // GGML_FA_AMX_KERNEL

bool ggml_fa_amx_compute(const struct ggml_compute_params * params, struct ggml_tensor * dst) {
#if defined(GGML_FA_AMX_KERNEL)
    if (params->use_ref || !fa_amx_enabled() || !fa_amx_supported(dst) || !fa_amx_permission()) {
        return false;
    }
    const int ith = params->ith;
    const int nth = params->nth;

    fa_shape sh;
    sh.q    = dst->src[0];
    sh.k    = dst->src[1];
    sh.v    = dst->src[2];
    sh.mask = dst->src[3];
    sh.dst  = dst;
    sh.DK   = sh.k->ne[0];
    sh.DV   = sh.v->ne[0];
    sh.N    = sh.q->ne[1];
    sh.Hkv  = sh.k->ne[2];
    sh.gqa  = sh.q->ne[2] / sh.Hkv;
    sh.rk3  = sh.q->ne[3] / sh.k->ne[3];
    memcpy(&sh.scale, (const float *) dst->op_params + 0, sizeof(float));
    sh.rows    = sh.gqa * sh.N;
    sh.tiles   = (sh.rows + 15) / 16;
    sh.nblocks = sh.k->ne[1] / 32;

    // split the keys among threads when there are fewer tiles than threads, at least fa_amx_min_split_blocks() blocks
    // of 32 keys per split
    const int64_t base = sh.Hkv * sh.q->ne[3] * sh.tiles;
    sh.splits = 1;
    if (base < nth) {
        sh.splits = std::max<int64_t>(1, std::min<int64_t>((nth + base - 1) / base, sh.nblocks / fa_amx_min_split_blocks()));
    }
    sh.blocks_per_split = (sh.nblocks + sh.splits - 1) / sh.splits;
    const int64_t items = base * sh.splits;

    char * wbase = (char *) (((uintptr_t) params->wdata + 63) & ~(uintptr_t) 63);
    const size_t tb = fa_amx_thread_bytes(sh.DK, sh.DV);
    GGML_ASSERT(params->wsize >= ggml_fa_amx_work_size(sh.DK, sh.DV, nth));
    char * ws = wbase + ith * tb;
    fa_scratch w;
    w.qrows = (uint16_t *) ws;
    w.qp    = (uint32_t *) (ws + 32 * sh.DK);
    w.vp    = (uint16_t *) (ws + 64 * sh.DK);
    w.o     = (float *)    (ws + 64 * sh.DK + 64 * sh.DV);
    w.s     = (float *)    (ws + 64 * sh.DK + 128 * sh.DV);
    w.p     = (uint16_t *) (ws + 64 * sh.DK + 128 * sh.DV + 2048);
    w.mt    = (float *)    (ws + 64 * sh.DK + 128 * sh.DV + 3072);
    w.m     = (float *)    (ws + 64 * sh.DK + 128 * sh.DV + 5120);
    w.l     = w.m + 16;
    float * partials = (float *) (wbase + nth * tb);

    fa_tile_config();
    if (ith == 0) {
        ggml_threadpool_chunk_set(params->threadpool, nth);
    }
    ggml_barrier(params->threadpool);
    for (int64_t item = ith; item < items; item = ggml_threadpool_chunk_add(params->threadpool, 1)) {
        fa_tile(sh, w, item, partials);
    }
    fa_tile_config_restore();

    if (sh.splits > 1) {
        ggml_barrier(params->threadpool);
        for (int64_t idx = ith; idx < base * 16; idx += nth) {
            fa_merge_row(sh, partials, idx / 16, (int) (idx % 16), w.o);
        }
    }
    return true;
#else
    GGML_UNUSED(params);
    GGML_UNUSED(dst);
    return false;
#endif
}
