/**
 * astralx_similarity.cu
 * =====================
 * GPU similarity-matrix computation reproducing ASTRAL-MP's
 * SimilarityMatrix.populateByQuartetDistance byte-for-byte.
 *
 * FORMULA (bridge identity, validated 1141/1141 pairs):
 *   same_side_T(a, b)  =  C2(kt − 2)  −  QD_T(a, b)
 *
 * O(1) closed form for QD via Euler tour + RMQ:
 *   QD_T(x, y) = ½ · [ (F(x) − F(cx)) + (F(y) − F(cy))
 *                     + (cxS − 1)·Z + (cyS − 1)·Z ]
 * where w = LCA(x, y), cx = child of w on x-side, cy = child of w on y-side,
 *   cxS = s(cx),  cyS = s(cy),   Z = kt − cxS − cyS,
 *   F(v) the root→v path prefix described in EulerTourBuilder.
 *
 * O(1) LCA + child-payload query via Euler tour RMQ:
 *   fa = firstOcc[a],  fb = firstOcc[b];   l = min(fa,fb), r = max(...)
 *   k_lvl = floor(log2(r − l + 1));  l2 = r − 2^k_lvl + 1
 *   dL = sparseMin[k_lvl][l];  dR = sparseMin[k_lvl][l2]
 *   leftWins = (dL <= dR)   (left-biased argmin, matches sparseMin build)
 *   The selected position is the INTERMEDIATE visit of LCA(x,y), where the
 *   payload sparse tables hold s(LCA.left), F(LCA.left), s(LCA.right),
 *   F(LCA.right). The leaf with the smaller firstOcc is in LCA.left.
 *
 * ARCHITECTURE:
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

// ── GPU Kernel ────────────────────────────────────────────────────────────────

/**
 * sim_tile_kernel
 * ───────────────
 * One thread per (da, db) = position within the B×B pair tile.
 * Global taxa: a = a0+da, b = b0+db.
 *
 * For each tree t in the current batch:
 *   1. Presence check via firstOcc.
 *   2. Skip if kt < 4 (C2(kt-2) = 0).
 *   3. O(1) RMQ → LCA's intermediate-position child payloads.
 *   4. Compute twoQD = 2·QD via the closed form.
 *   5. ss = C2(kt-2) − twoQD/2;  numAcc += ss;  denAcc += C2(kt-2).
 */
__global__ void sim_tile_kernel(
    const short*  __restrict__ euler_depths,        // [delta * E_max]
    const double* __restrict__ euler_F,             // [delta * E_max]
    const short*  __restrict__ sparse_min,          // [delta * LOG * E_max]
    const short*  __restrict__ sparse_left_child_s, // [delta * LOG * E_max]
    const double* __restrict__ sparse_left_child_f, // [delta * LOG * E_max]
    const short*  __restrict__ sparse_right_child_s,
    const double* __restrict__ sparse_right_child_f,
    const int*    __restrict__ first_occ,           // [delta * n]
    const int*    __restrict__ leaf_count,          // [delta]
    int delta, int n, int E_max, int LOG,
    int a0, int b0, int bA, int bB,
    double* __restrict__ tile_num,
    double* __restrict__ tile_den
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
        long focc_off = (long)t * n;
        int fa = first_occ[focc_off + a];
        int fb = first_occ[focc_off + b];
        if (fa < 0 || fb < 0) continue;

        int kt = leaf_count[t];
        long long cc = (long long)(kt - 2) * (kt - 3) / 2;   // C2(kt-2)
        if (cc <= 0) continue;

        // ── O(1) RMQ: select left-biased argmin position ────────────────────
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

        bool pickLeft = (dL <= dR);
        long pickIdx  = pickLeft ? ol : ol2;

        int    leftS = (int)   sparse_left_child_s [pickIdx];
        double leftF = (double)sparse_left_child_f [pickIdx];
        int    rightS= (int)   sparse_right_child_s[pickIdx];
        double rightF= (double)sparse_right_child_f[pickIdx];

        // Map (leftLeaf, rightLeaf) by tour order back to (a, b).
        // The leaf with the smaller firstOcc is in the LCA's LEFT child.
        int    aS;   double aF;
        int    bS;   double bF;
        if (fa <= fb) {
            aS = leftS;   aF = leftF;
            bS = rightS;  bF = rightF;
        } else {
            aS = rightS;  aF = rightF;
            bS = leftS;   bF = leftF;
        }

        long ed_off = (long)t * E_max;
        double Fa = euler_F[ed_off + fa];
        double Fb = euler_F[ed_off + fb];

        long long Z = (long long)(kt - aS - bS);
        // twoQD = (Fa - aF) + (Fb - bF) + (aS - 1)*Z + (bS - 1)*Z
        double twoQD = (Fa - aF) + (Fb - bF)
                     + (double)((long long)(aS - 1) * Z)
                     + (double)((long long)(bS - 1) * Z);

        double ss = (double)cc - twoQD * 0.5;     // same-side count
        local_num += ss;
        local_den += (double)cc;
    }

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
    jdoubleArray j_euler_F,
    jshortArray  j_euler_left_child_s,
    jdoubleArray j_euler_left_child_f,
    jshortArray  j_euler_right_child_s,
    jdoubleArray j_euler_right_child_f,
    jshortArray  j_sparse_min,
    jshortArray  j_sparse_left_child_s,
    jdoubleArray j_sparse_left_child_f,
    jshortArray  j_sparse_right_child_s,
    jdoubleArray j_sparse_right_child_f,
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
    jboolean isCopy;
    jshort*  h_euler   = env->GetShortArrayElements (j_euler_depths,         &isCopy);
    jdouble* h_eulerF  = env->GetDoubleArrayElements(j_euler_F,              &isCopy);
    jshort*  h_eLcS    = env->GetShortArrayElements (j_euler_left_child_s,   &isCopy);
    jdouble* h_eLcF    = env->GetDoubleArrayElements(j_euler_left_child_f,   &isCopy);
    jshort*  h_eRcS    = env->GetShortArrayElements (j_euler_right_child_s,  &isCopy);
    jdouble* h_eRcF    = env->GetDoubleArrayElements(j_euler_right_child_f,  &isCopy);
    jshort*  h_spmin   = env->GetShortArrayElements (j_sparse_min,           &isCopy);
    jshort*  h_sLcS    = env->GetShortArrayElements (j_sparse_left_child_s,  &isCopy);
    jdouble* h_sLcF    = env->GetDoubleArrayElements(j_sparse_left_child_f,  &isCopy);
    jshort*  h_sRcS    = env->GetShortArrayElements (j_sparse_right_child_s, &isCopy);
    jdouble* h_sRcF    = env->GetDoubleArrayElements(j_sparse_right_child_f, &isCopy);
    jint*    h_focc    = env->GetIntArrayElements   (j_first_occ,            &isCopy);
    jint*    h_elen    = env->GetIntArrayElements   (j_euler_len,            &isCopy);
    jint*    h_lcount  = env->GetIntArrayElements   (j_leaf_count,           &isCopy);
    jdouble* h_num     = env->GetDoubleArrayElements(j_num_sum_out,          &isCopy);
    jdouble* h_den     = env->GetDoubleArrayElements(j_den_sum_out,          &isCopy);

    size_t free_vram = 0, total_vram = 0;
    cudaMemGetInfo(&free_vram, &total_vram);

    int B = tileSizeB;
    if (B <= 0) {
        B = (int)ceil(sqrt((double)n * numTrees));
        if (B > n) B = n;
    }
    while (B > 1 && 2LL * B * B * (long long)sizeof(double) > (long long)(free_vram * 0.40)) B /= 2;
    if (B < 1) B = 1;

    size_t tile_vram = 2ULL * B * B * sizeof(double);
    size_t remaining = (free_vram > tile_vram + 32*1024*1024ULL)
                     ? (size_t)((free_vram - tile_vram) * 0.50)
                     : 16*1024*1024ULL;

    // Per-tree bytes: see Java side comment in SimilarityMatrixBuilder.buildGPU.
    //   euler arrays:  E_max × (2 + 2 + 2 + 8 + 8 + 8) = E_max × 30
    //   sparse tables: LOG × E_max × (2 + 2 + 2 + 8 + 8) = LOG × E_max × 22
    //   firstOcc:      n × 4
    //   leafCount:     4
    size_t per_tree = (size_t)E_max * 30
                    + (size_t)LOG * E_max * 22
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
    short*  d_euler   = nullptr;
    double* d_eulerF  = nullptr;
    short*  d_eLcS    = nullptr;
    double* d_eLcF    = nullptr;
    short*  d_eRcS    = nullptr;
    double* d_eRcF    = nullptr;
    short*  d_spmin   = nullptr;
    short*  d_sLcS    = nullptr;
    double* d_sLcF    = nullptr;
    short*  d_sRcS    = nullptr;
    double* d_sRcF    = nullptr;
    int*    d_focc    = nullptr;
    int*    d_lcount  = nullptr;

    long long sz_short_e  = (long long)delta * E_max * sizeof(short);
    long long sz_double_e = (long long)delta * E_max * sizeof(double);
    long long sz_short_s  = (long long)delta * LOG * E_max * sizeof(short);
    long long sz_double_s = (long long)delta * LOG * E_max * sizeof(double);

    SIM_CUDA_CHECK(cudaMalloc(&d_euler,  sz_short_e));
    SIM_CUDA_CHECK(cudaMalloc(&d_eulerF, sz_double_e));
    SIM_CUDA_CHECK(cudaMalloc(&d_eLcS,   sz_short_e));
    SIM_CUDA_CHECK(cudaMalloc(&d_eLcF,   sz_double_e));
    SIM_CUDA_CHECK(cudaMalloc(&d_eRcS,   sz_short_e));
    SIM_CUDA_CHECK(cudaMalloc(&d_eRcF,   sz_double_e));
    SIM_CUDA_CHECK(cudaMalloc(&d_spmin,  sz_short_s));
    SIM_CUDA_CHECK(cudaMalloc(&d_sLcS,   sz_short_s));
    SIM_CUDA_CHECK(cudaMalloc(&d_sLcF,   sz_double_s));
    SIM_CUDA_CHECK(cudaMalloc(&d_sRcS,   sz_short_s));
    SIM_CUDA_CHECK(cudaMalloc(&d_sRcF,   sz_double_s));
    SIM_CUDA_CHECK(cudaMalloc(&d_focc,   (long long)delta * n * sizeof(int)));
    SIM_CUDA_CHECK(cudaMalloc(&d_lcount, delta * sizeof(int)));

    double* d_tile_num = nullptr;
    double* d_tile_den = nullptr;
    SIM_CUDA_CHECK(cudaMalloc(&d_tile_num, (long long)B * B * sizeof(double)));
    SIM_CUDA_CHECK(cudaMalloc(&d_tile_den, (long long)B * B * sizeof(double)));

    double* h_tile_num = nullptr;
    double* h_tile_den = nullptr;
    SIM_CUDA_CHECK(cudaMallocHost(&h_tile_num, (long long)B * B * sizeof(double)));
    SIM_CUDA_CHECK(cudaMallocHost(&h_tile_den, (long long)B * B * sizeof(double)));

    int    total_work       = num_batches * num_tiles;
    int    work_done        = 0;
    double t_start          = sim_now_sec();
    double t_last_print     = t_start - progressInterval;
    double last_pct_printed = -1.0;
    const bool step_mode    = (progressMaxSteps > 0);
    int    use_color        = sim_use_color();

    for (int t0 = 0; t0 < numTrees; t0 += delta) {
        int dt = (t0 + delta > numTrees) ? numTrees - t0 : delta;

        long long off_e = (long long)t0 * E_max;
        long long bytes_short_e  = (long long)dt * E_max * sizeof(short);
        long long bytes_double_e = (long long)dt * E_max * sizeof(double);
        long long off_s = (long long)t0 * LOG * E_max;
        long long bytes_short_s  = (long long)dt * LOG * E_max * sizeof(short);
        long long bytes_double_s = (long long)dt * LOG * E_max * sizeof(double);

        SIM_CUDA_CHECK(cudaMemcpy(d_euler,  h_euler  + off_e, bytes_short_e,  cudaMemcpyHostToDevice));
        SIM_CUDA_CHECK(cudaMemcpy(d_eulerF, h_eulerF + off_e, bytes_double_e, cudaMemcpyHostToDevice));
        SIM_CUDA_CHECK(cudaMemcpy(d_eLcS,   h_eLcS   + off_e, bytes_short_e,  cudaMemcpyHostToDevice));
        SIM_CUDA_CHECK(cudaMemcpy(d_eLcF,   h_eLcF   + off_e, bytes_double_e, cudaMemcpyHostToDevice));
        SIM_CUDA_CHECK(cudaMemcpy(d_eRcS,   h_eRcS   + off_e, bytes_short_e,  cudaMemcpyHostToDevice));
        SIM_CUDA_CHECK(cudaMemcpy(d_eRcF,   h_eRcF   + off_e, bytes_double_e, cudaMemcpyHostToDevice));

        SIM_CUDA_CHECK(cudaMemcpy(d_spmin, h_spmin + off_s, bytes_short_s,  cudaMemcpyHostToDevice));
        SIM_CUDA_CHECK(cudaMemcpy(d_sLcS,  h_sLcS  + off_s, bytes_short_s,  cudaMemcpyHostToDevice));
        SIM_CUDA_CHECK(cudaMemcpy(d_sLcF,  h_sLcF  + off_s, bytes_double_s, cudaMemcpyHostToDevice));
        SIM_CUDA_CHECK(cudaMemcpy(d_sRcS,  h_sRcS  + off_s, bytes_short_s,  cudaMemcpyHostToDevice));
        SIM_CUDA_CHECK(cudaMemcpy(d_sRcF,  h_sRcF  + off_s, bytes_double_s, cudaMemcpyHostToDevice));

        SIM_CUDA_CHECK(cudaMemcpy(d_focc,
            h_focc + (long long)t0 * n,
            (long long)dt * n * sizeof(int), cudaMemcpyHostToDevice));
        SIM_CUDA_CHECK(cudaMemcpy(d_lcount,
            h_lcount + t0,
            dt * sizeof(int), cudaMemcpyHostToDevice));

        for (int a0 = 0; a0 < n; a0 += B) {
            int bA = (a0 + B > n) ? n - a0 : B;
            for (int b0 = a0; b0 < n; b0 += B) {
                int bB_tile = (b0 + B > n) ? n - b0 : B;

                SIM_CUDA_CHECK(cudaMemset(d_tile_num, 0,
                    (long long)bA * bB_tile * sizeof(double)));
                SIM_CUDA_CHECK(cudaMemset(d_tile_den, 0,
                    (long long)bA * bB_tile * sizeof(double)));

                dim3 block(32, 32);
                dim3 grid((bA + 31) / 32, (bB_tile + 31) / 32);
                sim_tile_kernel<<<grid, block>>>(
                    d_euler, d_eulerF,
                    d_spmin, d_sLcS, d_sLcF, d_sRcS, d_sRcF,
                    d_focc, d_lcount,
                    dt, n, E_max, LOG,
                    a0, b0, bA, bB_tile,
                    d_tile_num, d_tile_den
                );
                SIM_CUDA_CHECK(cudaDeviceSynchronize());

                SIM_CUDA_CHECK(cudaMemcpy(h_tile_num, d_tile_num,
                    (long long)bA * bB_tile * sizeof(double), cudaMemcpyDeviceToHost));
                SIM_CUDA_CHECK(cudaMemcpy(h_tile_den, d_tile_den,
                    (long long)bA * bB_tile * sizeof(double), cudaMemcpyDeviceToHost));

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

    cudaFree(d_euler);  cudaFree(d_eulerF);
    cudaFree(d_eLcS);   cudaFree(d_eLcF);
    cudaFree(d_eRcS);   cudaFree(d_eRcF);
    cudaFree(d_spmin);
    cudaFree(d_sLcS);   cudaFree(d_sLcF);
    cudaFree(d_sRcS);   cudaFree(d_sRcF);
    cudaFree(d_focc);   cudaFree(d_lcount);
    cudaFree(d_tile_num); cudaFree(d_tile_den);
    cudaFreeHost(h_tile_num); cudaFreeHost(h_tile_den);

    env->ReleaseShortArrayElements (j_euler_depths,        h_euler,  JNI_ABORT);
    env->ReleaseDoubleArrayElements(j_euler_F,             h_eulerF, JNI_ABORT);
    env->ReleaseShortArrayElements (j_euler_left_child_s,  h_eLcS,   JNI_ABORT);
    env->ReleaseDoubleArrayElements(j_euler_left_child_f,  h_eLcF,   JNI_ABORT);
    env->ReleaseShortArrayElements (j_euler_right_child_s, h_eRcS,   JNI_ABORT);
    env->ReleaseDoubleArrayElements(j_euler_right_child_f, h_eRcF,   JNI_ABORT);
    env->ReleaseShortArrayElements (j_sparse_min,          h_spmin,  JNI_ABORT);
    env->ReleaseShortArrayElements (j_sparse_left_child_s, h_sLcS,   JNI_ABORT);
    env->ReleaseDoubleArrayElements(j_sparse_left_child_f, h_sLcF,   JNI_ABORT);
    env->ReleaseShortArrayElements (j_sparse_right_child_s,h_sRcS,   JNI_ABORT);
    env->ReleaseDoubleArrayElements(j_sparse_right_child_f,h_sRcF,   JNI_ABORT);
    env->ReleaseIntArrayElements   (j_first_occ,           h_focc,   JNI_ABORT);
    env->ReleaseIntArrayElements   (j_euler_len,           h_elen,   JNI_ABORT);
    env->ReleaseIntArrayElements   (j_leaf_count,          h_lcount, JNI_ABORT);
    env->ReleaseDoubleArrayElements(j_num_sum_out, h_num, 0);
    env->ReleaseDoubleArrayElements(j_den_sum_out, h_den, 0);
}
