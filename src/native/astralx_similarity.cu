/**
 * astralx_similarity.cu
 * =====================
 * GPU similarity-matrix computation via Euler tour + sparse-table RMQ.
 *
 * FORMULA
 * ───────
 * For gene tree T (kt leaves) and pair (a,b) both present in T:
 *
 *   num_T(a,b) = C2(kt − sub[LCA_T(a,b)])
 *   den_T(a,b) = C2(kt − 2)
 *
 * where sub[v] = number of leaves in subtree(v), and C2(x) = x*(x-1)/2.
 * Pairs with num_T = 0 (i.e., LCA spans ≥ kt−1 leaves) contribute nothing.
 *
 * O(1) LCA QUERY via Euler tour RMQ
 * ───────────────────────────────────
 * Given first occurrences fa = firstOcc[a], fb = firstOcc[b]:
 *   l = min(fa, fb),  r = max(fa, fb)
 *   k_lvl = floor(log2(r − l + 1)),  l2 = r − 2^k_lvl + 1
 *   dL = sparseMin[k_lvl][l],  dR = sparseMin[k_lvl][l2]
 *   Left-biased argmin (prefer left on tie):
 *     sub_lca = (dL <= dR) ? sparseSubLC[k_lvl][l] : sparseSubLC[k_lvl][l2]
 *   num = C2(kt - sub_lca)
 *
 * ARCHITECTURE (same as distance matrix)
 * ────────────────────────────────────────
 *   Δ-tree batching : tree data on GPU  O(Δ · n · log n)
 *   B×B pair tiling : output tile       O(B²) doubles
 *   No atomics: each thread owns a unique (a,b) cell.
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
 * For each tree t in the current batch:
 *   1. Presence check: firstOcc[a] >= 0 && firstOcc[b] >= 0
 *   2. Skip if C2(kt-2) = 0 (kt < 4)
 *   3. O(1) RMQ: find sub_lca via sparseMin/sparseSubLC
 *   4. num = C2(kt - sub_lca);  if num <= 0: skip
 *   5. Accumulate num and den = C2(kt-2)
 *
 * Flat array layout:
 *   sparseMin / sparseSubLC : [delta][LOG][E_max]
 *   eulerDepths / eulerSubLC: [delta][E_max]
 *   firstOcc                : [delta][n]   (-1 absent)
 *   leafCount               : [delta]
 */
__global__ void sim_tile_kernel(
    const short* __restrict__ euler_depths,   // [delta * E_max]
    const short* __restrict__ euler_sublc,    // [delta * E_max]
    const short* __restrict__ sparse_min,     // [delta * LOG * E_max]
    const short* __restrict__ sparse_sublc,   // [delta * LOG * E_max]
    const int*   __restrict__ first_occ,      // [delta * n]   (-1 absent)
    const int*   __restrict__ leaf_count,     // [delta]        kt per tree
    int delta, int n, int E_max, int LOG,
    int a0, int b0, int bA, int bB,
    double* __restrict__ tile_num,            // [bA * bB] — zeroed before kernel
    double* __restrict__ tile_den             // [bA * bB] — zeroed before kernel
) {
    int da = blockIdx.x * blockDim.x + threadIdx.x;
    int db = blockIdx.y * blockDim.y + threadIdx.y;
    if (da >= bA || db >= bB) return;

    int a = a0 + da;
    int b = b0 + db;
    if (a >= n || b >= n || a >= b) return;   // upper triangle only

    double local_num = 0.0;
    double local_den = 0.0;

    for (int t = 0; t < delta; t++) {
        // ── Presence check ───────────────────────────────────────────────────
        long focc_off = (long)t * n;
        int fa = first_occ[focc_off + a];
        int fb = first_occ[focc_off + b];
        if (fa < 0 || fb < 0) continue;

        int kt = leaf_count[t];
        long long den_val = (long long)(kt - 2) * (kt - 3) / 2;   // C2(kt-2)
        if (den_val <= 0) continue;

        // ── O(1) RMQ: find sub[LCA(a,b)] ────────────────────────────────────
        int l   = (fa < fb) ? fa : fb;
        int r   = (fa < fb) ? fb : fa;
        int len = r - l + 1;
        int k_lvl = 31 - __clz(len);
        int l2    = r - (1 << k_lvl) + 1;

        long sp_off = (long)t * LOG * E_max;
        long ol     = sp_off + (long)k_lvl * E_max + l;
        long ol2    = sp_off + (long)k_lvl * E_max + l2;

        short dL = sparse_min[ol];
        short dR = sparse_min[ol2];

        // Left-biased argmin: prefer left half on tie
        int sub_lca = (dL <= dR) ? (int)sparse_sublc[ol] : (int)sparse_sublc[ol2];

        // ── Compute contribution ─────────────────────────────────────────────
        long long num_val = c2_dev(kt - sub_lca);
        if (num_val <= 0) continue;

        local_num += (double)num_val;
        local_den += (double)den_val;
    }

    // Write to tile (unique cell per thread — no atomics)
    long tidx = (long)da * bB + db;
    tile_num[tidx] += local_num;
    tile_den[tidx] += local_den;
}

// ── JNI entry point ───────────────────────────────────────────────────────────

extern "C" JNIEXPORT void JNICALL
Java_astralx_gpu_GPUSimilarityMatrix_computeSimilarityGPU(
    JNIEnv*  env,
    jclass   cls,
    jshortArray  j_euler_depths,
    jshortArray  j_euler_sublc,
    jshortArray  j_sparse_min,
    jshortArray  j_sparse_sublc,
    jintArray    j_first_occ,
    jintArray    j_euler_len,
    jintArray    j_leaf_count,
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
    jshort*  h_euler  = env->GetShortArrayElements(j_euler_depths, &isCopy);
    jshort*  h_sublc  = env->GetShortArrayElements(j_euler_sublc,  &isCopy);
    jshort*  h_spmin  = env->GetShortArrayElements(j_sparse_min,   &isCopy);
    jshort*  h_spsub  = env->GetShortArrayElements(j_sparse_sublc, &isCopy);
    jint*    h_focc   = env->GetIntArrayElements  (j_first_occ,    &isCopy);
    jint*    h_elen   = env->GetIntArrayElements  (j_euler_len,    &isCopy);
    jint*    h_lcount = env->GetIntArrayElements  (j_leaf_count,   &isCopy);
    jdouble* h_num    = env->GetDoubleArrayElements(j_num_sum_out, &isCopy);
    jdouble* h_den    = env->GetDoubleArrayElements(j_den_sum_out, &isCopy);

    // ── Query free VRAM ───────────────────────────────────────────────────────
    size_t free_vram = 0, total_vram = 0;
    cudaMemGetInfo(&free_vram, &total_vram);

    // ── Determine tile size B ─────────────────────────────────────────────────
    int B = tileSizeB;
    if (B <= 0) {
        B = (int)ceil(sqrt((double)n * numTrees));
        if (B > n) B = n;
    }
    // Cap: 2·B²·8 ≤ 40% free VRAM
    while (B > 1 && 2LL * B * B * sizeof(double) > (long long)(free_vram * 0.40)) B /= 2;
    if (B < 1) B = 1;

    // ── Determine tree batch size Δ ───────────────────────────────────────────
    size_t tile_vram = 2ULL * B * B * sizeof(double);
    size_t remaining = (free_vram > tile_vram + 32*1024*1024ULL)
                     ? (size_t)((free_vram - tile_vram) * 0.50)
                     : 16*1024*1024ULL;

    // Per-tree bytes:
    //   euler:  E_max × (2+2) = E_max × 4
    //   sparse: LOG × E_max × (2+2) = LOG × E_max × 4
    //   leaf:   n × 4
    //   misc:   sizeof(int)
    size_t per_tree = (size_t)E_max * 4
                    + (size_t)LOG * E_max * 4
                    + (size_t)n * 4
                    + sizeof(int);
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
        "[ASTRAL-X sim] GPU VRAM: tile %.1f MB  tree-data %.1f MB  "
        "(free %.0f MB / total %.0f MB)\n",
        tile_vram / 1e6,
        (double)delta * per_tree / 1e6,
        free_vram / 1e6, total_vram / 1e6);

    // ── Allocate GPU tree-data buffers ────────────────────────────────────────
    short* d_euler  = nullptr;
    short* d_sublc  = nullptr;
    short* d_spmin  = nullptr;
    short* d_spsub  = nullptr;
    int*   d_focc   = nullptr;
    int*   d_lcount = nullptr;

    SIM_CUDA_CHECK(cudaMalloc(&d_euler,  (long long)delta * E_max * sizeof(short)));
    SIM_CUDA_CHECK(cudaMalloc(&d_sublc,  (long long)delta * E_max * sizeof(short)));
    SIM_CUDA_CHECK(cudaMalloc(&d_spmin,  (long long)delta * LOG * E_max * sizeof(short)));
    SIM_CUDA_CHECK(cudaMalloc(&d_spsub,  (long long)delta * LOG * E_max * sizeof(short)));
    SIM_CUDA_CHECK(cudaMalloc(&d_focc,   (long long)delta * n * sizeof(int)));
    SIM_CUDA_CHECK(cudaMalloc(&d_lcount, delta * sizeof(int)));

    // ── Allocate GPU tile buffers ─────────────────────────────────────────────
    double* d_tile_num = nullptr;
    double* d_tile_den = nullptr;
    SIM_CUDA_CHECK(cudaMalloc(&d_tile_num, (long long)B * B * sizeof(double)));
    SIM_CUDA_CHECK(cudaMalloc(&d_tile_den, (long long)B * B * sizeof(double)));

    // Pinned host tile buffers for fast DMA download
    double* h_tile_num = nullptr;
    double* h_tile_den = nullptr;
    SIM_CUDA_CHECK(cudaMallocHost(&h_tile_num, (long long)B * B * sizeof(double)));
    SIM_CUDA_CHECK(cudaMallocHost(&h_tile_den, (long long)B * B * sizeof(double)));

    // ── Progress tracking ─────────────────────────────────────────────────────
    int    total_work       = num_batches * num_tiles;
    int    work_done        = 0;
    double t_start          = sim_now_sec();
    double t_last_print     = t_start - progressInterval;
    double last_pct_printed = -1.0;
    const bool step_mode    = (progressMaxSteps > 0);
    int    use_color        = sim_use_color();

    // ── Outer loop: tree batches ──────────────────────────────────────────────
    for (int t0 = 0; t0 < numTrees; t0 += delta) {
        int dt = (t0 + delta > numTrees) ? numTrees - t0 : delta;

        // Upload euler arrays
        SIM_CUDA_CHECK(cudaMemcpy(d_euler,
            h_euler + (long long)t0 * E_max,
            (long long)dt * E_max * sizeof(short), cudaMemcpyHostToDevice));
        SIM_CUDA_CHECK(cudaMemcpy(d_sublc,
            h_sublc + (long long)t0 * E_max,
            (long long)dt * E_max * sizeof(short), cudaMemcpyHostToDevice));

        // Upload sparse tables
        SIM_CUDA_CHECK(cudaMemcpy(d_spmin,
            h_spmin + (long long)t0 * LOG * E_max,
            (long long)dt * LOG * E_max * sizeof(short), cudaMemcpyHostToDevice));
        SIM_CUDA_CHECK(cudaMemcpy(d_spsub,
            h_spsub + (long long)t0 * LOG * E_max,
            (long long)dt * LOG * E_max * sizeof(short), cudaMemcpyHostToDevice));

        // Upload leaf maps
        SIM_CUDA_CHECK(cudaMemcpy(d_focc,
            h_focc + (long long)t0 * n,
            (long long)dt * n * sizeof(int), cudaMemcpyHostToDevice));
        SIM_CUDA_CHECK(cudaMemcpy(d_lcount,
            h_lcount + t0,
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
                    d_euler, d_sublc, d_spmin, d_spsub,
                    d_focc, d_lcount,
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
                    t_last_print     = now;
                    last_pct_printed = pct;
                }
            }
        }
    }

    // ── Release GPU buffers ───────────────────────────────────────────────────
    cudaFree(d_euler);  cudaFree(d_sublc);
    cudaFree(d_spmin);  cudaFree(d_spsub);
    cudaFree(d_focc);   cudaFree(d_lcount);
    cudaFree(d_tile_num); cudaFree(d_tile_den);
    cudaFreeHost(h_tile_num); cudaFreeHost(h_tile_den);

    // ── Release Java array references ─────────────────────────────────────────
    env->ReleaseShortArrayElements(j_euler_depths, h_euler,  JNI_ABORT);
    env->ReleaseShortArrayElements(j_euler_sublc,  h_sublc,  JNI_ABORT);
    env->ReleaseShortArrayElements(j_sparse_min,   h_spmin,  JNI_ABORT);
    env->ReleaseShortArrayElements(j_sparse_sublc, h_spsub,  JNI_ABORT);
    env->ReleaseIntArrayElements  (j_first_occ,    h_focc,   JNI_ABORT);
    env->ReleaseIntArrayElements  (j_euler_len,    h_elen,   JNI_ABORT);
    env->ReleaseIntArrayElements  (j_leaf_count,   h_lcount, JNI_ABORT);
    // Commit output arrays back to Java
    env->ReleaseDoubleArrayElements(j_num_sum_out, h_num, 0);
    env->ReleaseDoubleArrayElements(j_den_sum_out, h_den, 0);
}
