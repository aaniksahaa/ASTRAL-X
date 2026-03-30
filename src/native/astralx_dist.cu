/**
 * astralx_dist.cu
 * ===============
 * GPU distance-matrix computation via Euler tour + O(1) sparse-table RMQ.
 *
 * DESIGN OVERVIEW
 * ───────────────
 * For k gene trees and n taxa we need:
 *   dist_sum[a][b] += depth(a,t) + depth(b,t) - 2·depth(LCA(a,b), t)
 *   cooccur [a][b] += 1
 * summed over every tree t containing both a and b.
 *
 * GPU memory budget:
 *   Δ-tree batching : tree data (euler + sparse + firstOcc + leafDepth) ← O(Δ n log n)
 *   B-pair tiling   : output tile (dist_tile + cooccur_tile)             ← O(B²)
 *
 * Both Δ and B are auto-computed from free VRAM.
 * Default tile size: B = min(n, ceil(sqrt(n * k))) so B² ≈ n·k.
 * When k ≥ n the whole n×n output fits in one tile (no tiling loop).
 *
 * Control flow:
 *   outer:  tree batches  (k/Δ uploads)
 *   inner:  B×B pair tiles (each tile: zero → kernel → download → CPU accumulate)
 *
 * Kernel threadblock: 32 × 32 = 1024 threads.
 * Thread (da, db) in a tile starting at (a0, b0) computes all Δ-tree contributions
 * for the pair  (a0+da, b0+db)  in the current batch.
 *
 * Memory-access pattern:
 *   leafDepth[t·n + a]   : coalesced across warp (consecutive a)
 *   leafDepth[t·n + b]   : broadcast (same b within warp)
 *   firstOcc [t·n + a/b] : same pattern
 *   sparseMin[t·LOG·E + …]: gather (RMQ positions vary), but same tree t → L2 reuse
 *
 * No atomics needed: each (da, db) pair is owned by exactly one thread per kernel launch.
 */

#include <cuda_runtime.h>
#include <jni.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

// ── Utility: timing ──────────────────────────────────────────────────────────

static double dist_now_sec() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

static void dist_fmt_duration(double sec, char* buf, int bufsz) {
    int s = (int)sec, m = s / 60;
    s %= 60;
    if (m >= 60) snprintf(buf, bufsz, "%dh%02dm%02ds", m/60, m%60, s);
    else         snprintf(buf, bufsz, "%02d:%02d",     m, s);
}

// ── Utility: CUDA error check ─────────────────────────────────────────────────

#define DIST_CUDA_CHECK(call) \
    do { \
        cudaError_t _e = (call); \
        if (_e != cudaSuccess) { \
            fprintf(stderr, "[ASTRAL-X dist] CUDA error %s at %s:%d: %s\n", \
                    #call, __FILE__, __LINE__, cudaGetErrorString(_e)); \
            return; \
        } \
    } while(0)

#define DIST_CUDA_CHECK_RET(call, retval) \
    do { \
        cudaError_t _e = (call); \
        if (_e != cudaSuccess) { \
            fprintf(stderr, "[ASTRAL-X dist] CUDA error %s at %s:%d: %s\n", \
                    #call, __FILE__, __LINE__, cudaGetErrorString(_e)); \
            return (retval); \
        } \
    } while(0)

// ── GPU Kernel ───────────────────────────────────────────────────────────────

/**
 * compute_tile_kernel
 * ───────────────────
 * One thread per (da, db) = position within the B×B pair tile.
 * Global taxa: a = a0+da, b = b0+db.
 * Iterates over delta trees in the current batch and accumulates:
 *   tile_dist  [da * bB + db] += sum of topological distances
 *   tile_cooccur[da * bB + db] += co-occurrence count
 *
 * Layout of sparseMin (on device): [delta][LOG][E_max]
 *   Index: t*LOG*E_max + lvl*E_max + pos
 */
__global__ void compute_tile_kernel(
    const short* __restrict__ euler_depths,  // [delta * E_max]
    const short* __restrict__ sparse_min,    // [delta * LOG * E_max]
    const short* __restrict__ leaf_depth,    // [delta * n]    (-1 absent)
    const int*   __restrict__ first_occ,     // [delta * n]    (-1 absent)
    int delta, int n, int E_max, int LOG,
    int a0, int b0, int bA, int bB,          // tile start + actual dims
    double* __restrict__ tile_dist,          // [bA * bB] — zeroed before kernel
    int*    __restrict__ tile_cooccur        // [bA * bB] — zeroed before kernel
) {
    int da = blockIdx.x * blockDim.x + threadIdx.x;
    int db = blockIdx.y * blockDim.y + threadIdx.y;
    if (da >= bA || db >= bB) return;

    int a = a0 + da;
    int b = b0 + db;
    if (a >= n || b >= n || a >= b) return;   // upper triangle; diag handled by CPU

    double local_sum = 0.0;
    int    local_cnt = 0;

    for (int t = 0; t < delta; t++) {
        // Depths of leaves a and b in this tree (int16; -1 = absent)
        short da_depth = leaf_depth[(long)t * n + a];
        short db_depth = leaf_depth[(long)t * n + b];
        if (da_depth < 0 || db_depth < 0) continue;

        // Euler tour positions of a and b (for RMQ range)
        int fa = first_occ[(long)t * n + a];
        int fb = first_occ[(long)t * n + b];
        // fa, fb both ≥ 0 because da_depth ≥ 0 guarantees presence

        // RMQ: min depth in euler_depths[t][min(fa,fb) .. max(fa,fb)]
        int l = (fa < fb) ? fa : fb;
        int r = (fa < fb) ? fb : fa;
        int len = r - l + 1;
        // k = floor(log2(len))  via count-leading-zeros
        int k_lvl = 31 - __clz(len);

        const short* sp_t = sparse_min + (long)t * LOG * E_max;
        int lca_d = (int)min(
            sp_t[(long)k_lvl * E_max + l],
            sp_t[(long)k_lvl * E_max + (r - (1 << k_lvl) + 1)]
        );

        local_sum += (int)da_depth + (int)db_depth - 2 * lca_d;
        local_cnt++;
    }

    // Write to tile (unique location per thread, no atomics needed)
    long tidx = (long)da * bB + db;
    tile_dist  [tidx] += local_sum;
    tile_cooccur[tidx] += local_cnt;
}

// ── JNI entry point ───────────────────────────────────────────────────────────

extern "C" JNIEXPORT void JNICALL
Java_astralx_gpu_GPUDistanceMatrix_computeDistancesGPU(
    JNIEnv*  env,
    jclass   cls,
    jshortArray j_euler_depths,
    jshortArray j_sparse_min,
    jshortArray j_leaf_depth,
    jintArray   j_first_occ,
    jintArray   j_euler_len,
    jint     numTrees,
    jint     n,
    jint     E_max,
    jint     LOG,
    jint     tileSizeB,
    jdouble  progressInterval,
    jint     progressMaxSteps,
    jdoubleArray j_dist_sum_out,
    jintArray    j_cooccur_out
) {
    // ── Pin Java arrays ───────────────────────────────────────────────────────
    jboolean isCopy;
    jshort* h_euler  = env->GetShortArrayElements(j_euler_depths, &isCopy);
    jshort* h_sparse = env->GetShortArrayElements(j_sparse_min,   &isCopy);
    jshort* h_lddepth= env->GetShortArrayElements(j_leaf_depth,   &isCopy);
    jint*   h_focc   = env->GetIntArrayElements  (j_first_occ,    &isCopy);
    jint*   h_elen   = env->GetIntArrayElements  (j_euler_len,    &isCopy);
    jdouble* h_dist  = env->GetDoubleArrayElements(j_dist_sum_out, &isCopy);
    jint*    h_cooc  = env->GetIntArrayElements   (j_cooccur_out,  &isCopy);

    // ── Query free VRAM ───────────────────────────────────────────────────────
    size_t free_vram = 0, total_vram = 0;
    cudaMemGetInfo(&free_vram, &total_vram);

    // ── Determine tile size B ─────────────────────────────────────────────────
    // Default: B = min(n, ceil(sqrt(n * k)))
    // This makes B² ≈ n·k (amount downstream phases use anyway).
    // Cap so that B² × 12 ≤ 40% of free VRAM.
    int B = tileSizeB;
    if (B <= 0) {
        B = (int)ceil(sqrt((double)n * numTrees));
        if (B > n) B = n;
    }
    // Ensure B² × 12 fits within 40% of free VRAM
    while (B > 1 && (long long)B * B * 12 > (long long)(free_vram * 0.40)) B /= 2;
    if (B < 1) B = 1;

    // ── Determine tree batch size Δ ───────────────────────────────────────────
    // Remaining VRAM after tile output buffer: use 50% of free for tree data.
    size_t tile_vram = (size_t)B * B * (sizeof(double) + sizeof(int));
    size_t remaining = (free_vram > tile_vram + 32*1024*1024)
                     ? (size_t)((free_vram - tile_vram) * 0.50)
                     : 16*1024*1024ULL;  // fallback: 16 MB

    // Per-tree bytes: euler + sparse + leaf_depth + first_occ + euler_len
    size_t per_tree = (size_t)E_max * sizeof(short)
                    + (size_t)LOG * E_max * sizeof(short)
                    + (size_t)n * sizeof(short)
                    + (size_t)n * sizeof(int)
                    + sizeof(int);
    int delta = (per_tree > 0) ? (int)(remaining / per_tree) : numTrees;
    if (delta < 1)       delta = 1;
    if (delta > numTrees) delta = numTrees;

    int num_batches = (numTrees + delta - 1) / delta;
    int num_tiles_side = (n + B - 1) / B;
    int num_tiles = num_tiles_side * (num_tiles_side + 1) / 2;  // upper-triangle count

    fprintf(stderr,
        "\n[ASTRAL-X dist] GPU distance matrix: n=%d  k=%d  "
        "tile B=%d  tree-batch Δ=%d  (%d batches × %d tiles)\n",
        n, numTrees, B, delta, num_batches, num_tiles);
    fprintf(stderr,
        "[ASTRAL-X dist] GPU VRAM budget: tile %.1f MB  tree-data %.1f MB  "
        "(free %.0f MB total %.0f MB)\n",
        tile_vram / 1e6,
        (double)delta * per_tree / 1e6,
        free_vram / 1e6, total_vram / 1e6);

    // ── Allocate GPU buffers for one tree batch ───────────────────────────────
    short *d_euler  = nullptr, *d_sparse = nullptr,
          *d_lddepth= nullptr;
    int   *d_focc   = nullptr;

    DIST_CUDA_CHECK(cudaMalloc(&d_euler,   (long long)delta * E_max * sizeof(short)));
    DIST_CUDA_CHECK(cudaMalloc(&d_sparse,  (long long)delta * LOG * E_max * sizeof(short)));
    DIST_CUDA_CHECK(cudaMalloc(&d_lddepth, (long long)delta * n * sizeof(short)));
    DIST_CUDA_CHECK(cudaMalloc(&d_focc,    (long long)delta * n * sizeof(int)));

    // ── Allocate GPU tile buffers ─────────────────────────────────────────────
    double* d_tile_dist    = nullptr;
    int*    d_tile_cooccur = nullptr;
    DIST_CUDA_CHECK(cudaMalloc(&d_tile_dist,    (long long)B * B * sizeof(double)));
    DIST_CUDA_CHECK(cudaMalloc(&d_tile_cooccur, (long long)B * B * sizeof(int)));

    // Pinned host buffers for fast tile download
    double* h_tile_dist    = nullptr;
    int*    h_tile_cooccur = nullptr;
    DIST_CUDA_CHECK(cudaMallocHost(&h_tile_dist,    (long long)B * B * sizeof(double)));
    DIST_CUDA_CHECK(cudaMallocHost(&h_tile_cooccur, (long long)B * B * sizeof(int)));

    // ── Progress tracking ─────────────────────────────────────────────────────
    int    total_work   = num_batches * num_tiles;  // (batch, tile) pairs
    int    work_done    = 0;
    double t_start      = dist_now_sec();
    double t_last_print = t_start - progressInterval;  // print immediately
    double last_pct_printed = -1.0;
    const bool step_mode = (progressMaxSteps > 0);

    // ── Outer loop: tree batches ──────────────────────────────────────────────
    for (int t0 = 0; t0 < numTrees; t0 += delta) {
        int dt = delta;
        if (t0 + dt > numTrees) dt = numTrees - t0;

        // Upload batch: euler, sparse, leaf_depth, first_occ
        DIST_CUDA_CHECK(cudaMemcpy(d_euler,
            h_euler   + (long long)t0 * E_max,
            (long long)dt * E_max * sizeof(short), cudaMemcpyHostToDevice));
        DIST_CUDA_CHECK(cudaMemcpy(d_sparse,
            h_sparse  + (long long)t0 * LOG * E_max,
            (long long)dt * LOG * E_max * sizeof(short), cudaMemcpyHostToDevice));
        DIST_CUDA_CHECK(cudaMemcpy(d_lddepth,
            h_lddepth + (long long)t0 * n,
            (long long)dt * n * sizeof(short), cudaMemcpyHostToDevice));
        DIST_CUDA_CHECK(cudaMemcpy(d_focc,
            h_focc    + (long long)t0 * n,
            (long long)dt * n * sizeof(int), cudaMemcpyHostToDevice));

        // ── Inner loop: B×B pair tiles (upper triangle) ───────────────────────
        for (int a0 = 0; a0 < n; a0 += B) {
            int bA = B;  if (a0 + bA > n) bA = n - a0;
            for (int b0 = a0; b0 < n; b0 += B) {
                int bB = B;  if (b0 + bB > n) bB = n - b0;

                // Zero tile accumulators on GPU
                DIST_CUDA_CHECK(cudaMemset(d_tile_dist,    0, (long long)bA * bB * sizeof(double)));
                DIST_CUDA_CHECK(cudaMemset(d_tile_cooccur, 0, (long long)bA * bB * sizeof(int)));

                // Launch kernel
                dim3 block(32, 32);
                dim3 grid((bA + 31) / 32, (bB + 31) / 32);
                compute_tile_kernel<<<grid, block>>>(
                    d_euler, d_sparse, d_lddepth, d_focc,
                    dt, n, E_max, LOG,
                    a0, b0, bA, bB,
                    d_tile_dist, d_tile_cooccur
                );
                DIST_CUDA_CHECK(cudaDeviceSynchronize());

                // Download tile to pinned host memory
                DIST_CUDA_CHECK(cudaMemcpy(h_tile_dist,
                    d_tile_dist,    (long long)bA * bB * sizeof(double), cudaMemcpyDeviceToHost));
                DIST_CUDA_CHECK(cudaMemcpy(h_tile_cooccur,
                    d_tile_cooccur, (long long)bA * bB * sizeof(int),    cudaMemcpyDeviceToHost));

                // CPU accumulation: add tile into full n×n arrays
                // Handle both on-diagonal tile (a0==b0) and off-diagonal.
                for (int da = 0; da < bA; da++) {
                    int a = a0 + da;
                    for (int db = 0; db < bB; db++) {
                        int b = b0 + db;
                        if (a >= b) continue;  // skip lower triangle and diagonal
                        long tidx = (long)da * bB + db;
                        double v = h_tile_dist   [tidx];
                        int    c = h_tile_cooccur[tidx];
                        if (c == 0) continue;
                        h_dist[a * n + b] += v;
                        h_dist[b * n + a] += v;
                        h_cooc[a * n + b] += c;
                        h_cooc[b * n + a] += c;
                    }
                }

                // Progress
                work_done++;
                double now  = dist_now_sec();
                double pct  = (total_work > 0) ? 100.0 * work_done / total_work : 100.0;
                bool is_last = (work_done == total_work);
                bool should_print = false;
                if (step_mode)
                    should_print = is_last || (pct - last_pct_printed >= 100.0 / progressMaxSteps);
                else
                    should_print = is_last || (now - t_last_print >= progressInterval);

                if (should_print) {
                    char elapsed_buf[32];
                    dist_fmt_duration(now - t_start, elapsed_buf, sizeof(elapsed_buf));
                    int bar_w = 28, filled = (int)(bar_w * pct / 100.0);
                    char bar[64]; int bi = 0;
                    bar[bi++] = '[';
                    for (int i = 0; i < bar_w; i++) bar[bi++] = (i < filled) ? '#' : '.';
                    bar[bi++] = ']'; bar[bi] = '\0';
                    fprintf(stderr,
                        "     ▸  Distance matrix (GPU)  %s  %4d/%-4d  %5.1f%%  elapsed: %-8s  "
                        "batch: %d/%d\r",
                        bar,
                        work_done, total_work, pct,
                        elapsed_buf,
                        (t0 / delta) + 1, num_batches);
                    fflush(stderr);
                    t_last_print       = now;
                    last_pct_printed   = pct;
                }
            }
        }
    }
    fprintf(stderr, "\n");  // end progress line

    // ── Release GPU buffers ───────────────────────────────────────────────────
    cudaFree(d_euler);
    cudaFree(d_sparse);
    cudaFree(d_lddepth);
    cudaFree(d_focc);
    cudaFree(d_tile_dist);
    cudaFree(d_tile_cooccur);
    cudaFreeHost(h_tile_dist);
    cudaFreeHost(h_tile_cooccur);

    // ── Release Java array references ────────────────────────────────────────
    env->ReleaseShortArrayElements(j_euler_depths, h_euler,   JNI_ABORT);
    env->ReleaseShortArrayElements(j_sparse_min,   h_sparse,  JNI_ABORT);
    env->ReleaseShortArrayElements(j_leaf_depth,   h_lddepth, JNI_ABORT);
    env->ReleaseIntArrayElements  (j_first_occ,    h_focc,    JNI_ABORT);
    env->ReleaseIntArrayElements  (j_euler_len,     h_elen,    JNI_ABORT);
    // Commit output arrays back to Java
    env->ReleaseDoubleArrayElements(j_dist_sum_out, h_dist, 0);
    env->ReleaseIntArrayElements   (j_cooccur_out,  h_cooc, 0);
}
