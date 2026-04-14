/**
 * astralx_similarity.cu
 * =====================
 * GPU similarity-matrix computation via Euler tour + O(1) sparse-table RMQ.
 *
 * DESIGN OVERVIEW
 * ───────────────
 * For k gene trees and n taxa we compute:
 *
 *   numSum[a][b] += Σ_{t: a,b present}  (S_t[u] − C2(subLC[c_a]) − C2(subLC[c_b]))
 *   denSum[a][b] += Σ_{t: a,b present}  C2(k_t − 2)
 *
 * where u = LCA_t(a,b), c_a/c_b are the children of u toward a and b.
 *
 * Per pair per tree the computation uses:
 *   1. O(1) LCA depth via standard Euler-tour RMQ (sparseMin).
 *   2. subLC[c_a]  = prevChildSubLC at LEFTMOST  argmin in [l,r]  → sparseSubLCLeft
 *   3. subLC[c_b]  = nextChildSubLC at RIGHTMOST argmin in [l,r]  → sparseSubLCRight
 *   4. S[u]        = eulerS         at LEFTMOST  argmin in [l,r]  → sparseSLeft
 *
 * where l = min(firstOcc[a], firstOcc[b]) and r = max(...).
 *
 * For binary trees (current implementation):
 *   num = S[u] − C2(subLC[c_a]) − C2(subLC[c_b])  =  C2(k_t − subLC[u])
 * For polytomies the full formula is used automatically via the payload tables.
 *
 * GPU memory budget (same architecture as distance matrix):
 *   Δ-tree batching : tree data  O(Δ n log n)
 *   B-pair tiling   : output tile O(B²)
 *   B ≈ sqrt(n·k)  so tile VRAM ≈ n·k entries  =  O(n·k)
 *
 * No atomics: thread (da,db) owns pair (a0+da, b0+db), writes only its cell.
 */

#include <cuda_runtime.h>
#include <jni.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <unistd.h>

// ── Utility: timing ──────────────────────────────────────────────────────────

static double sim_now_sec() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static void sim_fmt_duration(double sec, char* buf, int bufsz) {
    int s = (int)sec, m = s / 60;
    s %= 60;
    if (m >= 60) snprintf(buf, bufsz, "%dh%02dm%02ds", m/60, m%60, s);
    else         snprintf(buf, bufsz, "%02d:%02d",     m, s);
}

static int sim_use_color() {
    if (getenv("NO_COLOR"))    return 0;
    if (getenv("FORCE_COLOR")) return 1;
    return isatty(STDERR_FILENO);
}

static void sim_fmt_rate(double rate, char* buf, int bufsz) {
    if (rate <= 0)       snprintf(buf, bufsz, "?it/s");
    else if (rate >= 1)  snprintf(buf, bufsz, "%.1fit/s", rate);
    else                 snprintf(buf, bufsz, "%.2fs/it", 1.0 / rate);
}

static void sim_print_progress(int work_done, int total_work, double elapsed,
                                int color, int is_last) {
    double pct  = (total_work > 0) ? 100.0 * work_done / total_work : 100.0;
    double rate = (work_done > 0 && elapsed > 0) ? work_done / elapsed : 0.0;
    double eta  = (rate > 0 && work_done < total_work) ? (total_work - work_done) / rate : 0.0;

    char elapsed_buf[32], eta_buf[32], rate_buf[32];
    sim_fmt_duration(elapsed, elapsed_buf, sizeof(elapsed_buf));
    sim_fmt_duration(eta,     eta_buf,     sizeof(eta_buf));
    sim_fmt_rate(rate, rate_buf, sizeof(rate_buf));

    const int BAR_W = 28;
    int filled = (int)(BAR_W * pct / 100.0);
    const char* FULL  = "\xE2\x96\x88";
    const char* EMPTY = "\xE2\x96\x91";
    char bar[4 + BAR_W * 3 + 4];
    int pos = 0;
    bar[pos++] = '[';
    for (int i = 0; i < BAR_W; i++) {
        const char* ch = (i < filled) ? FULL : EMPTY;
        bar[pos++] = ch[0]; bar[pos++] = ch[1]; bar[pos++] = ch[2];
    }
    bar[pos++] = ']'; bar[pos] = '\0';

    if (color)
        fprintf(stderr,
            "     \033[2m▸  \033[0mSimilarity matrix (GPU)  "
            "\033[32m%s\033[0m  %d/%d (%d%%)"
            "  \033[2m[%s<%s, \033[0m\033[33m%s\033[0m\033[2m]\033[0m\r",
            bar, work_done, total_work, (int)pct,
            elapsed_buf, work_done > 0 ? eta_buf : "?", rate_buf);
    else
        fprintf(stderr,
            "     ▸  Similarity matrix (GPU)  %s  %d/%d (%d%%)  [%s<%s, %s]\r",
            bar, work_done, total_work, (int)pct,
            elapsed_buf, work_done > 0 ? eta_buf : "?", rate_buf);
    fflush(stderr);
    if (is_last) fprintf(stderr, "\n");
}

// ── Utility: CUDA error check ─────────────────────────────────────────────────

#define SIM_CUDA_CHECK(call) \
    do { \
        cudaError_t _e = (call); \
        if (_e != cudaSuccess) { \
            fprintf(stderr, "[ASTRAL-X sim] CUDA error %s at %s:%d: %s\n", \
                    #call, __FILE__, __LINE__, cudaGetErrorString(_e)); \
            return; \
        } \
    } while(0)

// ── C2 helper (device) ────────────────────────────────────────────────────────

__device__ __forceinline__ long long c2_dev(int x) {
    return (x < 2) ? 0LL : (long long)x * (x - 1) / 2;
}

// ── GPU Kernel ────────────────────────────────────────────────────────────────

/**
 * sim_tile_kernel
 * ───────────────
 * One thread per (da, db) = position within the B×B pair tile.
 * Global taxa: a = a0+da, b = b0+db.
 *
 * For each of the delta trees in the current batch:
 *   1. Presence check via leafDepth.
 *   2. RMQ to find lca_depth, subLC_ca, subLC_cb, S_u.
 *   3. Accumulate numTile and denTile.
 *
 * Payload RMQ overlap query (all four payloads in one pass):
 *   k_lvl = floor(log2(r-l+1))
 *   l2    = r - (1 << k_lvl) + 1
 *
 *   d_l = sparseMin[t][k_lvl][l]   d_r = sparseMin[t][k_lvl][l2]
 *
 *   Left-biased (leftmost argmin):  d_l <= d_r → take [l] payload
 *   Right-biased (rightmost argmin): d_r <= d_l → take [l2] payload
 *
 *   subLC_leftleaf  = (d_l <= d_r) ? sparseSubLCLeft[k_lvl][l]  : sparseSubLCLeft[k_lvl][l2]
 *   subLC_rightleaf = (d_r <= d_l) ? sparseSubLCRight[k_lvl][l2]: sparseSubLCRight[k_lvl][l]
 *   S_u             = (d_l <= d_r) ? sparseSLeft[k_lvl][l]      : sparseSLeft[k_lvl][l2]
 *
 *   if (fa <= fb):  subLC_ca = subLC_leftleaf,  subLC_cb = subLC_rightleaf
 *   else:           subLC_cb = subLC_leftleaf,  subLC_ca = subLC_rightleaf
 *
 * Layout: all sparse tables stored as [delta][LOG][E_max] on device.
 */
__global__ void sim_tile_kernel(
    const short* __restrict__ euler_depths,      // [delta * E_max]
    const short* __restrict__ euler_prev_sublc,  // [delta * E_max]
    const short* __restrict__ euler_next_sublc,  // [delta * E_max]
    const int*   __restrict__ euler_s,           // [delta * E_max]
    const short* __restrict__ sparse_min,        // [delta * LOG * E_max]
    const short* __restrict__ sparse_sublc_left, // [delta * LOG * E_max]
    const short* __restrict__ sparse_sublc_right,// [delta * LOG * E_max]
    const int*   __restrict__ sparse_s_left,     // [delta * LOG * E_max]
    const short* __restrict__ leaf_depth,        // [delta * n]   (-1 absent)
    const int*   __restrict__ first_occ,         // [delta * n]   (-1 absent)
    const int*   __restrict__ leaf_count,        // [delta]        k_t per tree
    int delta, int n, int E_max, int LOG,
    int a0, int b0, int bA, int bB,
    double* __restrict__ tile_num,               // [bA * bB] — zeroed before kernel
    double* __restrict__ tile_den                // [bA * bB] — zeroed before kernel
) {
    int da = blockIdx.x * blockDim.x + threadIdx.x;
    int db = blockIdx.y * blockDim.y + threadIdx.y;
    if (da >= bA || db >= bB) return;

    int a = a0 + da;
    int b = b0 + db;
    if (a >= n || b >= n || a >= b) return;   // upper triangle only; diagonal set by CPU

    double local_num = 0.0;
    double local_den = 0.0;

    for (int t = 0; t < delta; t++) {
        // ── Presence check ──────────────────────────────────────────────────
        short da_d = leaf_depth[(long)t * n + a];
        short db_d = leaf_depth[(long)t * n + b];
        if (da_d < 0 || db_d < 0) continue;

        int kt = leaf_count[t];
        long den_val = (long)(kt - 2) * (kt - 3) / 2;   // C2(kt-2)
        if (den_val <= 0) continue;

        // ── firstOcc for both leaves ─────────────────────────────────────────
        int fa = first_occ[(long)t * n + a];
        int fb = first_occ[(long)t * n + b];
        int l  = (fa < fb) ? fa : fb;
        int r  = (fa < fb) ? fb : fa;

        // ── RMQ overlap query — 4 payloads in one pass ───────────────────────
        int len   = r - l + 1;
        int k_lvl = 31 - __clz(len);
        int l2    = r - (1 << k_lvl) + 1;

        const short* sp_min  = sparse_min        + (long)t * LOG * E_max;
        const short* sp_left = sparse_sublc_left + (long)t * LOG * E_max;
        const short* sp_rght = sparse_sublc_right+ (long)t * LOG * E_max;
        const int*   sp_s    = sparse_s_left     + (long)t * LOG * E_max;

        long offset_l  = (long)k_lvl * E_max + l;
        long offset_l2 = (long)k_lvl * E_max + l2;

        short d_l = sp_min[offset_l];
        short d_r = sp_min[offset_l2];

        // Left-biased: leftmost argmin → subLC of child containing the LEFT leaf (min firstOcc)
        short subLC_lf = (d_l <= d_r) ? sp_left[offset_l]  : sp_left[offset_l2];

        // Right-biased: rightmost argmin → subLC of child containing the RIGHT leaf (max firstOcc)
        short subLC_rf = (d_r <= d_l) ? sp_rght[offset_l2] : sp_rght[offset_l];

        // S[u] at leftmost argmin
        int S_u = (d_l <= d_r) ? sp_s[offset_l] : sp_s[offset_l2];

        // Assign to c_a / c_b based on which leaf is "left" in the tour
        int subLC_ca = (fa <= fb) ? (int)subLC_lf : (int)subLC_rf;
        int subLC_cb = (fa <= fb) ? (int)subLC_rf : (int)subLC_lf;

        // ── Compute contribution ─────────────────────────────────────────────
        long num_val = (long)S_u - c2_dev(subLC_ca) - c2_dev(subLC_cb);
        if (num_val <= 0) {
            local_den += (double)den_val;
            continue;
        }

        local_num += (double)num_val;
        local_den += (double)den_val;
    }

    // Write to tile (unique location per thread, no atomics)
    long tidx = (long)da * bB + db;
    tile_num[tidx] += local_num;
    tile_den[tidx] += local_den;
}

// ── JNI entry point ───────────────────────────────────────────────────────────

extern "C" JNIEXPORT void JNICALL
Java_astralx_gpu_GPUSimilarityMatrix_computeSimilarityGPU(
    JNIEnv*  env,
    jclass   cls,
    jshortArray j_euler_depths,
    jshortArray j_euler_prev_sublc,
    jshortArray j_euler_next_sublc,
    jintArray   j_euler_s,
    jshortArray j_sparse_min,
    jshortArray j_sparse_sublc_left,
    jshortArray j_sparse_sublc_right,
    jintArray   j_sparse_s_left,
    jintArray   j_first_occ,
    jshortArray j_leaf_depth,
    jintArray   j_euler_len,
    jintArray   j_leaf_count,
    jint     numTrees,
    jint     n,
    jint     E_max,
    jint     LOG,
    jint     tileSizeB,
    jdouble  progressInterval,
    jint     progressMaxSteps,
    jdoubleArray j_num_sum_out,
    jdoubleArray j_den_sum_out
) {
    // ── Pin Java arrays ───────────────────────────────────────────────────────
    jboolean isCopy;
    jshort* h_euler     = env->GetShortArrayElements(j_euler_depths,       &isCopy);
    jshort* h_eprev     = env->GetShortArrayElements(j_euler_prev_sublc,   &isCopy);
    jshort* h_enext     = env->GetShortArrayElements(j_euler_next_sublc,   &isCopy);
    jint*   h_es        = env->GetIntArrayElements  (j_euler_s,            &isCopy);
    jshort* h_spmin     = env->GetShortArrayElements(j_sparse_min,         &isCopy);
    jshort* h_spleft    = env->GetShortArrayElements(j_sparse_sublc_left,  &isCopy);
    jshort* h_spright   = env->GetShortArrayElements(j_sparse_sublc_right, &isCopy);
    jint*   h_spsleft   = env->GetIntArrayElements  (j_sparse_s_left,      &isCopy);
    jint*   h_focc      = env->GetIntArrayElements  (j_first_occ,          &isCopy);
    jshort* h_lddepth   = env->GetShortArrayElements(j_leaf_depth,         &isCopy);
    jint*   h_elen      = env->GetIntArrayElements  (j_euler_len,          &isCopy);
    jint*   h_lcount    = env->GetIntArrayElements  (j_leaf_count,         &isCopy);
    jdouble* h_num      = env->GetDoubleArrayElements(j_num_sum_out,        &isCopy);
    jdouble* h_den      = env->GetDoubleArrayElements(j_den_sum_out,        &isCopy);

    // ── Query free VRAM ───────────────────────────────────────────────────────
    size_t free_vram = 0, total_vram = 0;
    cudaMemGetInfo(&free_vram, &total_vram);

    // ── Determine tile size B ─────────────────────────────────────────────────
    int B = tileSizeB;
    if (B <= 0) {
        B = (int)ceil(sqrt((double)n * numTrees));
        if (B > n) B = n;
    }
    // Cap: 2·B²·8 ≤ 40% free VRAM  (numTile + denTile, double)
    while (B > 1 && 2LL * B * B * sizeof(double) > (long long)(free_vram * 0.40)) B /= 2;
    if (B < 1) B = 1;

    // ── Determine tree batch size Δ ───────────────────────────────────────────
    size_t tile_vram = 2ULL * B * B * sizeof(double);
    size_t remaining = (free_vram > tile_vram + 32*1024*1024ULL)
                     ? (size_t)((free_vram - tile_vram) * 0.50)
                     : 16*1024*1024ULL;

    // Per-tree bytes on GPU (euler arrays + sparse tables + leaf maps)
    size_t per_tree = (size_t)E_max * (sizeof(short)*3 + sizeof(int))        // euler
                    + (size_t)LOG * E_max * (sizeof(short)*3 + sizeof(int))   // sparse
                    + (size_t)n * (sizeof(int) + sizeof(short))               // firstOcc+leafDepth
                    + sizeof(int);                                             // leafCount
    int delta = (per_tree > 0) ? (int)(remaining / per_tree) : numTrees;
    if (delta < 1)        delta = 1;
    if (delta > numTrees) delta = numTrees;

    int num_batches    = (numTrees + delta - 1) / delta;
    int num_tiles_side = (n + B - 1) / B;
    int num_tiles      = num_tiles_side * (num_tiles_side + 1) / 2;

    fprintf(stderr,
        "\n[ASTRAL-X sim] GPU similarity matrix: n=%d  k=%d  "
        "tile B=%d  tree-batch Δ=%d  (%d batches × %d tiles)\n",
        n, numTrees, B, delta, num_batches, num_tiles);
    fprintf(stderr,
        "[ASTRAL-X sim] GPU VRAM budget: tile %.1f MB  tree-data %.1f MB  "
        "(free %.0f MB total %.0f MB)\n",
        tile_vram / 1e6,
        (double)delta * per_tree / 1e6,
        free_vram / 1e6, total_vram / 1e6);

    // ── Allocate GPU tree-data buffers (one batch) ────────────────────────────
    short *d_euler   = nullptr, *d_eprev  = nullptr, *d_enext  = nullptr;
    int   *d_es      = nullptr;
    short *d_spmin   = nullptr, *d_spleft = nullptr, *d_spright= nullptr;
    int   *d_spsleft = nullptr;
    short *d_lddepth = nullptr;
    int   *d_focc    = nullptr, *d_lcount = nullptr;

    SIM_CUDA_CHECK(cudaMalloc(&d_euler,   (long long)delta * E_max * sizeof(short)));
    SIM_CUDA_CHECK(cudaMalloc(&d_eprev,   (long long)delta * E_max * sizeof(short)));
    SIM_CUDA_CHECK(cudaMalloc(&d_enext,   (long long)delta * E_max * sizeof(short)));
    SIM_CUDA_CHECK(cudaMalloc(&d_es,      (long long)delta * E_max * sizeof(int)));
    SIM_CUDA_CHECK(cudaMalloc(&d_spmin,   (long long)delta * LOG * E_max * sizeof(short)));
    SIM_CUDA_CHECK(cudaMalloc(&d_spleft,  (long long)delta * LOG * E_max * sizeof(short)));
    SIM_CUDA_CHECK(cudaMalloc(&d_spright, (long long)delta * LOG * E_max * sizeof(short)));
    SIM_CUDA_CHECK(cudaMalloc(&d_spsleft, (long long)delta * LOG * E_max * sizeof(int)));
    SIM_CUDA_CHECK(cudaMalloc(&d_lddepth, (long long)delta * n * sizeof(short)));
    SIM_CUDA_CHECK(cudaMalloc(&d_focc,    (long long)delta * n * sizeof(int)));
    SIM_CUDA_CHECK(cudaMalloc(&d_lcount,  delta * sizeof(int)));

    // ── Allocate GPU tile buffers ─────────────────────────────────────────────
    double* d_tile_num = nullptr;
    double* d_tile_den = nullptr;
    SIM_CUDA_CHECK(cudaMalloc(&d_tile_num, (long long)B * B * sizeof(double)));
    SIM_CUDA_CHECK(cudaMalloc(&d_tile_den, (long long)B * B * sizeof(double)));

    // Pinned host tile buffers for fast download
    double* h_tile_num = nullptr;
    double* h_tile_den = nullptr;
    SIM_CUDA_CHECK(cudaMallocHost(&h_tile_num, (long long)B * B * sizeof(double)));
    SIM_CUDA_CHECK(cudaMallocHost(&h_tile_den, (long long)B * B * sizeof(double)));

    // ── Progress tracking ─────────────────────────────────────────────────────
    int    total_work   = num_batches * num_tiles;
    int    work_done    = 0;
    double t_start      = sim_now_sec();
    double t_last_print = t_start - progressInterval;
    double last_pct_printed = -1.0;
    const bool step_mode = (progressMaxSteps > 0);
    int    use_color    = sim_use_color();

    // ── Outer loop: tree batches ──────────────────────────────────────────────
    for (int t0 = 0; t0 < numTrees; t0 += delta) {
        int dt = (t0 + delta > numTrees) ? numTrees - t0 : delta;

        // Upload tree batch: euler, sparse, leaf maps
        SIM_CUDA_CHECK(cudaMemcpy(d_euler,
            h_euler   + (long long)t0 * E_max,
            (long long)dt * E_max * sizeof(short), cudaMemcpyHostToDevice));
        SIM_CUDA_CHECK(cudaMemcpy(d_eprev,
            h_eprev   + (long long)t0 * E_max,
            (long long)dt * E_max * sizeof(short), cudaMemcpyHostToDevice));
        SIM_CUDA_CHECK(cudaMemcpy(d_enext,
            h_enext   + (long long)t0 * E_max,
            (long long)dt * E_max * sizeof(short), cudaMemcpyHostToDevice));
        SIM_CUDA_CHECK(cudaMemcpy(d_es,
            h_es      + (long long)t0 * E_max,
            (long long)dt * E_max * sizeof(int),   cudaMemcpyHostToDevice));
        SIM_CUDA_CHECK(cudaMemcpy(d_spmin,
            h_spmin   + (long long)t0 * LOG * E_max,
            (long long)dt * LOG * E_max * sizeof(short), cudaMemcpyHostToDevice));
        SIM_CUDA_CHECK(cudaMemcpy(d_spleft,
            h_spleft  + (long long)t0 * LOG * E_max,
            (long long)dt * LOG * E_max * sizeof(short), cudaMemcpyHostToDevice));
        SIM_CUDA_CHECK(cudaMemcpy(d_spright,
            h_spright + (long long)t0 * LOG * E_max,
            (long long)dt * LOG * E_max * sizeof(short), cudaMemcpyHostToDevice));
        SIM_CUDA_CHECK(cudaMemcpy(d_spsleft,
            h_spsleft + (long long)t0 * LOG * E_max,
            (long long)dt * LOG * E_max * sizeof(int),   cudaMemcpyHostToDevice));
        SIM_CUDA_CHECK(cudaMemcpy(d_lddepth,
            h_lddepth + (long long)t0 * n,
            (long long)dt * n * sizeof(short), cudaMemcpyHostToDevice));
        SIM_CUDA_CHECK(cudaMemcpy(d_focc,
            h_focc    + (long long)t0 * n,
            (long long)dt * n * sizeof(int),   cudaMemcpyHostToDevice));
        SIM_CUDA_CHECK(cudaMemcpy(d_lcount,
            h_lcount  + t0,
            dt * sizeof(int), cudaMemcpyHostToDevice));

        // ── Inner loop: B×B pair tiles (upper triangle) ───────────────────────
        for (int a0 = 0; a0 < n; a0 += B) {
            int bA = (a0 + B > n) ? n - a0 : B;
            for (int b0 = a0; b0 < n; b0 += B) {
                int bB_tile = (b0 + B > n) ? n - b0 : B;

                // Zero tile accumulators
                SIM_CUDA_CHECK(cudaMemset(d_tile_num, 0,
                    (long long)bA * bB_tile * sizeof(double)));
                SIM_CUDA_CHECK(cudaMemset(d_tile_den, 0,
                    (long long)bA * bB_tile * sizeof(double)));

                // Launch kernel: 32×32 thread blocks
                dim3 block(32, 32);
                dim3 grid((bA + 31) / 32, (bB_tile + 31) / 32);
                sim_tile_kernel<<<grid, block>>>(
                    d_euler, d_eprev, d_enext, d_es,
                    d_spmin, d_spleft, d_spright, d_spsleft,
                    d_lddepth, d_focc, d_lcount,
                    dt, n, E_max, LOG,
                    a0, b0, bA, bB_tile,
                    d_tile_num, d_tile_den
                );
                SIM_CUDA_CHECK(cudaDeviceSynchronize());

                // Download tile
                SIM_CUDA_CHECK(cudaMemcpy(h_tile_num, d_tile_num,
                    (long long)bA * bB_tile * sizeof(double), cudaMemcpyDeviceToHost));
                SIM_CUDA_CHECK(cudaMemcpy(h_tile_den, d_tile_den,
                    (long long)bA * bB_tile * sizeof(double), cudaMemcpyDeviceToHost));

                // CPU: merge tile into full n×n arrays (symmetric)
                for (int da = 0; da < bA; da++) {
                    int a = a0 + da;
                    for (int db = 0; db < bB_tile; db++) {
                        int b = b0 + db;
                        if (a >= b) continue;
                        long tidx = (long)da * bB_tile + db;
                        double nv = h_tile_num[tidx];
                        double dv = h_tile_den[tidx];
                        if (dv == 0.0) continue;
                        h_num[a * n + b] += nv;  h_num[b * n + a] += nv;
                        h_den[a * n + b] += dv;  h_den[b * n + a] += dv;
                    }
                }

                // Progress
                work_done++;
                double now = sim_now_sec();
                double pct = (total_work > 0) ? 100.0 * work_done / total_work : 100.0;
                bool is_last = (work_done == total_work);
                bool should_print = step_mode
                    ? (is_last || pct - last_pct_printed >= 100.0 / progressMaxSteps)
                    : (is_last || now - t_last_print >= progressInterval);
                if (should_print) {
                    sim_print_progress(work_done, total_work, now - t_start, use_color, is_last);
                    t_last_print      = now;
                    last_pct_printed  = pct;
                }
            }
        }
    }

    // ── Release GPU buffers ───────────────────────────────────────────────────
    cudaFree(d_euler);   cudaFree(d_eprev);  cudaFree(d_enext);  cudaFree(d_es);
    cudaFree(d_spmin);   cudaFree(d_spleft); cudaFree(d_spright);cudaFree(d_spsleft);
    cudaFree(d_lddepth); cudaFree(d_focc);   cudaFree(d_lcount);
    cudaFree(d_tile_num); cudaFree(d_tile_den);
    cudaFreeHost(h_tile_num); cudaFreeHost(h_tile_den);

    // ── Release Java array references ─────────────────────────────────────────
    env->ReleaseShortArrayElements(j_euler_depths,       h_euler,   JNI_ABORT);
    env->ReleaseShortArrayElements(j_euler_prev_sublc,   h_eprev,   JNI_ABORT);
    env->ReleaseShortArrayElements(j_euler_next_sublc,   h_enext,   JNI_ABORT);
    env->ReleaseIntArrayElements  (j_euler_s,            h_es,      JNI_ABORT);
    env->ReleaseShortArrayElements(j_sparse_min,         h_spmin,   JNI_ABORT);
    env->ReleaseShortArrayElements(j_sparse_sublc_left,  h_spleft,  JNI_ABORT);
    env->ReleaseShortArrayElements(j_sparse_sublc_right, h_spright, JNI_ABORT);
    env->ReleaseIntArrayElements  (j_sparse_s_left,      h_spsleft, JNI_ABORT);
    env->ReleaseIntArrayElements  (j_first_occ,          h_focc,    JNI_ABORT);
    env->ReleaseShortArrayElements(j_leaf_depth,         h_lddepth, JNI_ABORT);
    env->ReleaseIntArrayElements  (j_euler_len,          h_elen,    JNI_ABORT);
    env->ReleaseIntArrayElements  (j_leaf_count,         h_lcount,  JNI_ABORT);
    // Commit output arrays
    env->ReleaseDoubleArrayElements(j_num_sum_out, h_num, 0);
    env->ReleaseDoubleArrayElements(j_den_sum_out, h_den, 0);
}
