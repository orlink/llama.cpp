// Generated kernels from the corrected Granite Rapids weight-reuse experiment.
// Keep arithmetic order identical to the grouped ct16fa kernels.
static void amx_i8g_tile_config(int rows, bool small) {
    alignas(64) tile_config_t tc = {};
    tc.palette_id = 1;
    for (int t = 0; t < 8; ++t) { tc.rows[t] = 16; tc.colsb[t] = 64; }
    if (small) {
        tc.rows[3] = rows;
        for (int t = 4; t < 8; ++t) tc.rows[t] = rows;
    } else {
        tc.rows[2] = tc.rows[4] = tc.rows[6] = std::min(16, rows);
        tc.rows[3] = tc.rows[5] = tc.rows[7] = std::max(1, rows - 16);
    }
    _tile_loadconfig(&tc);
}

static void amx_reuse_panel(const amx_i8g_copy & cp, int nb32, int M, const int8_t * RESTRICT A, const float * RESTRICT as,
                          float * RESTRICT C, int ldc, bool tight, int8_t * panel) {
    const int K = cp.K, G = cp.G, KS = K / 64, NG = K / G, GS = G / 64;
    const int nt0 = nb32 * 2, nt1 = nt0 + 1;
    const int PD = amx_i8g_prefetch();
    alignas(64) float acc[32 * 32];
    alignas(64) int32_t Ci[32 * 32];
    if (cp.bits < 8) {
        for (int k = 0; k < KS; ++k) {
            amx_i8g_unpack(cp, nt0, k, KS, panel + k * 2048);
            amx_i8g_unpack(cp, nt1, k, KS, panel + k * 2048 + 1024);
        }
        AMX_I8G_MEMORY_BARRIER();
    }
    for (int m0 = 0; m0 < M; m0 += 32) {
        const int rows = std::min(32, M - m0);
        const bool second = !tight || rows > 16;
        if (tight && M > 32 && rows < 32) amx_i8g_tile_config(rows, false);
        const int8_t * A0 = A + (size_t)(m0 / 16) * KS * 1024;
        const int8_t * A1 = A0 + (size_t)KS * 1024;
        memset(acc, 0, sizeof(acc));
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
                    B0 = panel + ks * 2048; B1 = B0 + 1024;
                } else {
                    B0 = cp.q.data() + AMX_I8G_PACKED_INDEX(nt0, ks, KS, 1024);
                    B1 = cp.q.data() + AMX_I8G_PACKED_INDEX(nt1, ks, KS, 1024);
                }
                _tile_loadd(0, B0, 64);
                _tile_loadd(2, A0 + (size_t)ks * 1024, 64);
                _tile_dpbssd(4, 2, 0);
                _tile_loadd(1, B1, 64);
                _tile_dpbssd(6, 2, 1);
                if (second) {
                    _tile_loadd(3, A1 + (size_t)ks * 1024, 64);
                    _tile_dpbssd(5, 3, 0);
                    _tile_dpbssd(7, 3, 1);
                }
            }
            _tile_stored(4, Ci, 128);
            _tile_stored(6, Ci + 16, 128);
            if (second) {
                _tile_stored(5, Ci + 16 * 32, 128);
                _tile_stored(7, Ci + 16 * 32 + 16, 128);
            }
            AMX_I8G_MEMORY_BARRIER();
            const __m512 ws0 = _mm512_loadu_ps(cp.s.data() + ((size_t)nt0 * NG + g) * 16);
            const __m512 ws1 = _mm512_loadu_ps(cp.s.data() + ((size_t)nt1 * NG + g) * 16);
            for (int r = 0; r < rows; ++r) {
                const __m512 sa = _mm512_set1_ps(as[(size_t)(m0 + r) * NG + g]);
                _mm512_store_ps(acc + r * 32, _mm512_fmadd_ps(_mm512_cvtepi32_ps(_mm512_load_si512(Ci + r * 32)), _mm512_mul_ps(ws0, sa), _mm512_load_ps(acc + r * 32)));
                _mm512_store_ps(acc + r * 32 + 16, _mm512_fmadd_ps(_mm512_cvtepi32_ps(_mm512_load_si512(Ci + r * 32 + 16)), _mm512_mul_ps(ws1, sa), _mm512_load_ps(acc + r * 32 + 16)));
            }
        }
        for (int r = 0; r < rows; ++r) {
            _mm512_storeu_ps(C + (size_t)(m0 + r) * ldc + nb32 * 32, _mm512_load_ps(acc + r * 32));
            _mm512_storeu_ps(C + (size_t)(m0 + r) * ldc + nb32 * 32 + 16, _mm512_load_ps(acc + r * 32 + 16));
        }
    }
    if (tight && M > 32 && M % 32) amx_i8g_tile_config(32, false);
}

template<int CHUNK> static void amx_reuse_group(const amx_i8g_copy & cp, int nb32, int M, const int8_t * RESTRICT A, const float * RESTRICT as,
                          float * RESTRICT C, int ldc, bool tight) {
    const int K = cp.K, G = cp.G, KS = K / 64, NG = K / G, GS = G / 64;
    const int nt0 = nb32 * 2, nt1 = nt0 + 1;
    const int PD = amx_i8g_prefetch();
    alignas(64) float accum[CHUNK * 32];
    alignas(64) int32_t Ci[32 * 32];
    alignas(64) int8_t panel[4 * 2048];
    for (int chunk = 0; chunk < M; chunk += CHUNK) {
        memset(accum, 0, sizeof(accum));
        const int stop = std::min(M, chunk + CHUNK);
        for (int g = 0; g < NG; ++g) {
            if (cp.bits < 8) {
                for (int s = 0; s < GS; ++s) {
                    amx_i8g_unpack(cp, nt0, g * GS + s, KS, panel + s * 2048);
                    amx_i8g_unpack(cp, nt1, g * GS + s, KS, panel + s * 2048 + 1024);
                }
                AMX_I8G_MEMORY_BARRIER();
            }
            for (int m0 = chunk; m0 < stop; m0 += 32) {
                float * acc = accum + (m0 - chunk) * 32;
        const int rows = std::min(32, M - m0);
        const bool second = !tight || rows > 16;
        if (tight && stop == M && M > 32 && M % 32 && (m0 == chunk || rows < 32)) amx_i8g_tile_config(rows, false);
        const int8_t * A0 = A + (size_t)(m0 / 16) * KS * 1024;
        const int8_t * A1 = A0 + (size_t)KS * 1024;
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
                    B0 = panel + s * 2048; B1 = B0 + 1024;
                } else {
                    B0 = cp.q.data() + AMX_I8G_PACKED_INDEX(nt0, ks, KS, 1024);
                    B1 = cp.q.data() + AMX_I8G_PACKED_INDEX(nt1, ks, KS, 1024);
                }
                _tile_loadd(0, B0, 64);
                _tile_loadd(2, A0 + (size_t)ks * 1024, 64);
                _tile_dpbssd(4, 2, 0);
                _tile_loadd(1, B1, 64);
                _tile_dpbssd(6, 2, 1);
                if (second) {
                    _tile_loadd(3, A1 + (size_t)ks * 1024, 64);
                    _tile_dpbssd(5, 3, 0);
                    _tile_dpbssd(7, 3, 1);
                }
            }
            _tile_stored(4, Ci, 128);
            _tile_stored(6, Ci + 16, 128);
            if (second) {
                _tile_stored(5, Ci + 16 * 32, 128);
                _tile_stored(7, Ci + 16 * 32 + 16, 128);
            }
            AMX_I8G_MEMORY_BARRIER();
            const __m512 ws0 = _mm512_loadu_ps(cp.s.data() + ((size_t)nt0 * NG + g) * 16);
            const __m512 ws1 = _mm512_loadu_ps(cp.s.data() + ((size_t)nt1 * NG + g) * 16);
            for (int r = 0; r < rows; ++r) {
                const __m512 sa = _mm512_set1_ps(as[(size_t)(m0 + r) * NG + g]);
                _mm512_store_ps(acc + r * 32, _mm512_fmadd_ps(_mm512_cvtepi32_ps(_mm512_load_si512(Ci + r * 32)), _mm512_mul_ps(ws0, sa), _mm512_load_ps(acc + r * 32)));
                _mm512_store_ps(acc + r * 32 + 16, _mm512_fmadd_ps(_mm512_cvtepi32_ps(_mm512_load_si512(Ci + r * 32 + 16)), _mm512_mul_ps(ws1, sa), _mm512_load_ps(acc + r * 32 + 16)));
            }
        }
        }
        for (int m0 = chunk; m0 < stop; m0 += 32) {
            const int rows = std::min(32, stop - m0);
            const float * acc = accum + (m0 - chunk) * 32;
        for (int r = 0; r < rows; ++r) {
            _mm512_storeu_ps(C + (size_t)(m0 + r) * ldc + nb32 * 32, _mm512_load_ps(acc + r * 32));
            _mm512_storeu_ps(C + (size_t)(m0 + r) * ldc + nb32 * 32 + 16, _mm512_load_ps(acc + r * 32 + 16));
        }
    }
    }
    if (tight && M > 32 && M % 32) amx_i8g_tile_config(32, false);
}


static void amx_reuse_register_config(int rows) {
    alignas(64) tile_config_t cfg = {};
    cfg.palette_id = 1;
    cfg.rows[0] = 16; cfg.colsb[0] = 64;
    cfg.rows[2] = cfg.rows[4] = rows;
    cfg.colsb[2] = cfg.colsb[4] = 64;
    _tile_loadconfig(&cfg);
}

template<bool PANEL> static void amx_reuse_register(const amx_i8g_copy & cp, int nt, int M,
        const int8_t * RESTRICT A, const float * RESTRICT as, float * RESTRICT C, int ldc, int8_t * panel,
        bool configure = true) {
    const int KS = cp.K / 64, NG = cp.K / 256;
    alignas(64) int8_t buf[1024];
    alignas(64) int32_t Ci[16 * 16];
    if (PANEL && cp.bits < 8) {
        for (int k = 0; k < KS; ++k) amx_i8g_unpack(cp, nt, k, KS, panel + k * 1024);
        AMX_I8G_MEMORY_BARRIER();
    }
    for (int m0 = 0; m0 < M; m0 += 16) {
        const int rows = std::min(16, M - m0);
        if (configure) amx_reuse_register_config(rows);
        __m512 v0 = _mm512_setzero_ps();
        __m512 v1 = _mm512_setzero_ps();
        __m512 v2 = _mm512_setzero_ps();
        __m512 v3 = _mm512_setzero_ps();
        __m512 v4 = _mm512_setzero_ps();
        __m512 v5 = _mm512_setzero_ps();
        __m512 v6 = _mm512_setzero_ps();
        __m512 v7 = _mm512_setzero_ps();
        __m512 v8 = _mm512_setzero_ps();
        __m512 v9 = _mm512_setzero_ps();
        __m512 v10 = _mm512_setzero_ps();
        __m512 v11 = _mm512_setzero_ps();
        __m512 v12 = _mm512_setzero_ps();
        __m512 v13 = _mm512_setzero_ps();
        __m512 v14 = _mm512_setzero_ps();
        __m512 v15 = _mm512_setzero_ps();
        for (int g = 0; g < NG; ++g) {
            _tile_zero(4);
            for (int s = 0; s < 4; ++s) {
                const int ks = g * 4 + s;
                const int8_t * b;
                if (cp.bits == 8) b = cp.q.data() + AMX_I8G_PACKED_INDEX(nt, ks, KS, 1024);
                else if (PANEL) b = panel + ks * 1024;
                else {
                    amx_i8g_unpack(cp, nt, ks, KS, buf);
                    AMX_I8G_MEMORY_BARRIER();
                    b = buf;
                }
                _tile_loadd(0, b, 64);
                _tile_loadd(2, A + ((size_t)(m0 / 16) * KS + ks) * 1024, 64);
                _tile_dpbssd(4, 2, 0);
            }
            _tile_stored(4, Ci, 64);
            AMX_I8G_MEMORY_BARRIER();
            const __m512 ws = _mm512_loadu_ps(cp.s.data() + ((size_t)nt * NG + g) * 16);
            if (rows > 0) v0 = _mm512_fmadd_ps(_mm512_cvtepi32_ps(_mm512_load_si512(Ci + 0 * 16)), _mm512_mul_ps(ws, _mm512_set1_ps(as[(size_t)(m0 + 0) * NG + g])), v0);
            if (rows > 1) v1 = _mm512_fmadd_ps(_mm512_cvtepi32_ps(_mm512_load_si512(Ci + 1 * 16)), _mm512_mul_ps(ws, _mm512_set1_ps(as[(size_t)(m0 + 1) * NG + g])), v1);
            if (rows > 2) v2 = _mm512_fmadd_ps(_mm512_cvtepi32_ps(_mm512_load_si512(Ci + 2 * 16)), _mm512_mul_ps(ws, _mm512_set1_ps(as[(size_t)(m0 + 2) * NG + g])), v2);
            if (rows > 3) v3 = _mm512_fmadd_ps(_mm512_cvtepi32_ps(_mm512_load_si512(Ci + 3 * 16)), _mm512_mul_ps(ws, _mm512_set1_ps(as[(size_t)(m0 + 3) * NG + g])), v3);
            if (rows > 4) v4 = _mm512_fmadd_ps(_mm512_cvtepi32_ps(_mm512_load_si512(Ci + 4 * 16)), _mm512_mul_ps(ws, _mm512_set1_ps(as[(size_t)(m0 + 4) * NG + g])), v4);
            if (rows > 5) v5 = _mm512_fmadd_ps(_mm512_cvtepi32_ps(_mm512_load_si512(Ci + 5 * 16)), _mm512_mul_ps(ws, _mm512_set1_ps(as[(size_t)(m0 + 5) * NG + g])), v5);
            if (rows > 6) v6 = _mm512_fmadd_ps(_mm512_cvtepi32_ps(_mm512_load_si512(Ci + 6 * 16)), _mm512_mul_ps(ws, _mm512_set1_ps(as[(size_t)(m0 + 6) * NG + g])), v6);
            if (rows > 7) v7 = _mm512_fmadd_ps(_mm512_cvtepi32_ps(_mm512_load_si512(Ci + 7 * 16)), _mm512_mul_ps(ws, _mm512_set1_ps(as[(size_t)(m0 + 7) * NG + g])), v7);
            if (rows > 8) v8 = _mm512_fmadd_ps(_mm512_cvtepi32_ps(_mm512_load_si512(Ci + 8 * 16)), _mm512_mul_ps(ws, _mm512_set1_ps(as[(size_t)(m0 + 8) * NG + g])), v8);
            if (rows > 9) v9 = _mm512_fmadd_ps(_mm512_cvtepi32_ps(_mm512_load_si512(Ci + 9 * 16)), _mm512_mul_ps(ws, _mm512_set1_ps(as[(size_t)(m0 + 9) * NG + g])), v9);
            if (rows > 10) v10 = _mm512_fmadd_ps(_mm512_cvtepi32_ps(_mm512_load_si512(Ci + 10 * 16)), _mm512_mul_ps(ws, _mm512_set1_ps(as[(size_t)(m0 + 10) * NG + g])), v10);
            if (rows > 11) v11 = _mm512_fmadd_ps(_mm512_cvtepi32_ps(_mm512_load_si512(Ci + 11 * 16)), _mm512_mul_ps(ws, _mm512_set1_ps(as[(size_t)(m0 + 11) * NG + g])), v11);
            if (rows > 12) v12 = _mm512_fmadd_ps(_mm512_cvtepi32_ps(_mm512_load_si512(Ci + 12 * 16)), _mm512_mul_ps(ws, _mm512_set1_ps(as[(size_t)(m0 + 12) * NG + g])), v12);
            if (rows > 13) v13 = _mm512_fmadd_ps(_mm512_cvtepi32_ps(_mm512_load_si512(Ci + 13 * 16)), _mm512_mul_ps(ws, _mm512_set1_ps(as[(size_t)(m0 + 13) * NG + g])), v13);
            if (rows > 14) v14 = _mm512_fmadd_ps(_mm512_cvtepi32_ps(_mm512_load_si512(Ci + 14 * 16)), _mm512_mul_ps(ws, _mm512_set1_ps(as[(size_t)(m0 + 14) * NG + g])), v14);
            if (rows > 15) v15 = _mm512_fmadd_ps(_mm512_cvtepi32_ps(_mm512_load_si512(Ci + 15 * 16)), _mm512_mul_ps(ws, _mm512_set1_ps(as[(size_t)(m0 + 15) * NG + g])), v15);
        }
        if (rows > 0) _mm512_storeu_ps(C + (size_t)(m0 + 0) * ldc + nt * 16, v0);
        if (rows > 1) _mm512_storeu_ps(C + (size_t)(m0 + 1) * ldc + nt * 16, v1);
        if (rows > 2) _mm512_storeu_ps(C + (size_t)(m0 + 2) * ldc + nt * 16, v2);
        if (rows > 3) _mm512_storeu_ps(C + (size_t)(m0 + 3) * ldc + nt * 16, v3);
        if (rows > 4) _mm512_storeu_ps(C + (size_t)(m0 + 4) * ldc + nt * 16, v4);
        if (rows > 5) _mm512_storeu_ps(C + (size_t)(m0 + 5) * ldc + nt * 16, v5);
        if (rows > 6) _mm512_storeu_ps(C + (size_t)(m0 + 6) * ldc + nt * 16, v6);
        if (rows > 7) _mm512_storeu_ps(C + (size_t)(m0 + 7) * ldc + nt * 16, v7);
        if (rows > 8) _mm512_storeu_ps(C + (size_t)(m0 + 8) * ldc + nt * 16, v8);
        if (rows > 9) _mm512_storeu_ps(C + (size_t)(m0 + 9) * ldc + nt * 16, v9);
        if (rows > 10) _mm512_storeu_ps(C + (size_t)(m0 + 10) * ldc + nt * 16, v10);
        if (rows > 11) _mm512_storeu_ps(C + (size_t)(m0 + 11) * ldc + nt * 16, v11);
        if (rows > 12) _mm512_storeu_ps(C + (size_t)(m0 + 12) * ldc + nt * 16, v12);
        if (rows > 13) _mm512_storeu_ps(C + (size_t)(m0 + 13) * ldc + nt * 16, v13);
        if (rows > 14) _mm512_storeu_ps(C + (size_t)(m0 + 14) * ldc + nt * 16, v14);
        if (rows > 15) _mm512_storeu_ps(C + (size_t)(m0 + 15) * ldc + nt * 16, v15);
    }
}
