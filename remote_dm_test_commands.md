# ASTRAL-X Distance Matrix + Autocomplete — Remote Test Commands

All commands run from the ASTRAL-X root directory.
Build everything first:
```bash
bash test/run_tests.sh --no-build   # or just run below to rebuild
bash build.sh && bash build_native.sh
```

---

## Step 0 — Generate incomplete gene trees from simphy complete trees

Run once to create incomplete versions at various fractions.

```bash
# n=1000, k=1000  →  30% missing
python3 test/gen_incomplete.py \
    simphy/data/t_1000_g_1000_sb_0.000001_spmin_100000_spmax_200000/R1/all_gt.tre \
    /tmp/inc_n1000_k1000_f30.tre \
    --fraction 0.30 --seed 42 --stats

# n=1000, k=1000  →  50% missing
python3 test/gen_incomplete.py \
    simphy/data/t_1000_g_1000_sb_0.000001_spmin_100000_spmax_200000/R1/all_gt.tre \
    /tmp/inc_n1000_k1000_f50.tre \
    --fraction 0.50 --seed 42 --stats

# n=5000, k=1000  →  30% missing
python3 test/gen_incomplete.py \
    simphy/data/t_5000_g_1000_sb_0.000001_spmin_100000_spmax_200000/R1/all_gt.tre \
    /tmp/inc_n5000_k1000_f30.tre \
    --fraction 0.30 --seed 42 --stats

# n=5000, k=1000  →  50% missing
python3 test/gen_incomplete.py \
    simphy/data/t_5000_g_1000_sb_0.000001_spmin_100000_spmax_200000/R1/all_gt.tre \
    /tmp/inc_n5000_k1000_f50.tre \
    --fraction 0.50 --seed 42 --stats
```

---

## Step 1 — DM correctness: CPU vs GPU on moderate sizes

Dumps the distance matrix and compares Python, CPU, GPU line by line.
These should all be exact matches (max_diff = 0).

```bash
# n=1000, k=1000, 30% incomplete
java -Djava.library.path=native -cp build astralx.Main \
    -i /tmp/inc_n1000_k1000_f30.tre --verify-distance-matrix --cpu -q \
    > /tmp/dm_cpu_n1000.txt

java -Djava.library.path=native -cp build astralx.Main \
    -i /tmp/inc_n1000_k1000_f30.tre --verify-distance-matrix --gpu -q \
    > /tmp/dm_gpu_n1000.txt

python3 test/verify_dm.py /tmp/inc_n1000_k1000_f30.tre \
    > /tmp/dm_py_n1000.txt

python3 test/verify_dm.py --compare \
    /tmp/dm_py_n1000.txt /tmp/dm_cpu_n1000.txt /tmp/dm_gpu_n1000.txt
```

Repeat for n=5000 (CPU may be slow — use GPU only for the final check):
```bash
# n=5000, k=1000, 30% incomplete
java -Djava.library.path=native -cp build astralx.Main \
    -i /tmp/inc_n5000_k1000_f30.tre --verify-distance-matrix --gpu -q \
    > /tmp/dm_gpu_n5000.txt

# CPU reference (slow, skip if time-constrained)
java -Djava.library.path=native -cp build astralx.Main \
    -i /tmp/inc_n5000_k1000_f30.tre --verify-distance-matrix --cpu -q \
    > /tmp/dm_cpu_n5000.txt

python3 test/verify_dm.py --compare \
    /tmp/dm_cpu_n5000.txt /tmp/dm_gpu_n5000.txt
```

---

## Step 2 — DM scalability: timing CPU vs GPU

Time the distance matrix build for increasing n, k.
No correctness check — just timing and memory monitoring.

```bash
# n=1000, k=1000 — CPU
time java -Xmx64g -Djava.library.path=native -cp build astralx.Main \
    -i /tmp/inc_n1000_k1000_f30.tre --verify-distance-matrix --cpu -q \
    > /dev/null

# n=1000, k=1000 — GPU
time java -Xmx64g -Djava.library.path=native -cp build astralx.Main \
    -i /tmp/inc_n1000_k1000_f30.tre --verify-distance-matrix --gpu -q \
    > /dev/null

# n=5000, k=1000 — CPU (expect several minutes)
time java -Xmx128g -Djava.library.path=native -cp build astralx.Main \
    -i /tmp/inc_n5000_k1000_f30.tre --verify-distance-matrix --cpu -q \
    > /dev/null

# n=5000, k=1000 — GPU
time java -Xmx128g -Djava.library.path=native -cp build astralx.Main \
    -i /tmp/inc_n5000_k1000_f30.tre --verify-distance-matrix --gpu -q \
    > /dev/null

# n=5000, k=1000 — GPU, 50% incomplete
time java -Xmx128g -Djava.library.path=native -cp build astralx.Main \
    -i /tmp/inc_n5000_k1000_f50.tre --verify-distance-matrix --gpu -q \
    > /dev/null
```

---

## Step 3 — End-to-end autocomplete pipeline

Run ASTRAL-X with `--autocomplete-incomplete-gene-trees` and compare RF distance
to the true species tree. Compare three conditions:
  (a) Incomplete trees, NO autocomplete
  (b) Incomplete trees, WITH autocomplete
  (c) Complete trees (gold standard)

```bash
TRUE_N1000=simphy/data/t_1000_g_1000_sb_0.000001_spmin_100000_spmax_200000/R1/s_tree.trees
COMPLETE_N1000=simphy/data/t_1000_g_1000_sb_0.000001_spmin_100000_spmax_200000/R1/all_gt.tre

# (a) Incomplete, no autocomplete
java -Xmx128g -Djava.library.path=native -cp build astralx.Main \
    -i /tmp/inc_n1000_k1000_f30.tre --gpu -q \
    -o /tmp/out_n1000_noac.tre

# (b) Incomplete, with autocomplete (GPU distance matrix)
java -Xmx128g -Djava.library.path=native -cp build astralx.Main \
    -i /tmp/inc_n1000_k1000_f30.tre --gpu --autocomplete-incomplete-gene-trees -q \
    -o /tmp/out_n1000_ac.tre

# (c) Complete trees, gold standard
java -Xmx128g -Djava.library.path=native -cp build astralx.Main \
    -i $COMPLETE_N1000 --gpu -q \
    -o /tmp/out_n1000_complete.tre

# RF distances (requires ete3 or similar; or use astral-mp's RF tool)
# bash test_rf.sh $TRUE_N1000 /tmp/out_n1000_noac.tre
# bash test_rf.sh $TRUE_N1000 /tmp/out_n1000_ac.tre
# bash test_rf.sh $TRUE_N1000 /tmp/out_n1000_complete.tre
```

Same for n=5000:
```bash
TRUE_N5000=simphy/data/t_5000_g_1000_sb_0.000001_spmin_100000_spmax_200000/R1/s_tree.trees
COMPLETE_N5000=simphy/data/t_5000_g_1000_sb_0.000001_spmin_100000_spmax_200000/R1/all_gt.tre

# (a) Incomplete, no autocomplete
java -Xmx256g -Djava.library.path=native -cp build astralx.Main \
    -i /tmp/inc_n5000_k1000_f30.tre --gpu -q \
    -o /tmp/out_n5000_noac.tre

# (b) Incomplete, with autocomplete
java -Xmx256g -Djava.library.path=native -cp build astralx.Main \
    -i /tmp/inc_n5000_k1000_f30.tre --gpu --autocomplete-incomplete-gene-trees -q \
    -o /tmp/out_n5000_ac.tre

# (c) Complete gold standard
java -Xmx256g -Djava.library.path=native -cp build astralx.Main \
    -i $COMPLETE_N5000 --gpu -q \
    -o /tmp/out_n5000_complete.tre
```

---

## Step 4 — Vary missing fraction: accuracy vs completeness tradeoff

For n=1000, compare autocomplete at 10%, 30%, 50%, 70% missing:
```bash
for FRAC in 0.10 0.30 0.50 0.70; do
    TAG="f$(echo $FRAC | tr -d '.')"
    python3 test/gen_incomplete.py \
        simphy/data/t_1000_g_1000_sb_0.000001_spmin_100000_spmax_200000/R1/all_gt.tre \
        /tmp/inc_n1000_${TAG}.tre --fraction $FRAC --seed 42

    java -Xmx128g -Djava.library.path=native -cp build astralx.Main \
        -i /tmp/inc_n1000_${TAG}.tre --gpu --autocomplete-incomplete-gene-trees -q \
        -o /tmp/out_n1000_${TAG}_ac.tre

    echo "FRAC=$FRAC done → /tmp/out_n1000_${TAG}_ac.tre"
done
```

---

## Step 5 — GPU tile size sensitivity (optional)

Test different tile sizes B for the GPU distance matrix kernel.
Larger B = fewer kernel launches but more VRAM.

```bash
for B in 64 128 256 512; do
    echo "=== tile B=$B ==="
    time java -Xmx128g -Djava.library.path=native -cp build astralx.Main \
        -i /tmp/inc_n1000_k1000_f30.tre \
        --verify-distance-matrix --gpu --gpu-dist-tile-size $B -q \
        > /tmp/dm_gpu_B${B}.txt
done

# All tile sizes should produce identical matrices
python3 test/verify_dm.py --compare \
    /tmp/dm_gpu_B64.txt /tmp/dm_gpu_B128.txt /tmp/dm_gpu_B256.txt /tmp/dm_gpu_B512.txt
```

---

## What to look for

| Test | Expected |
|------|----------|
| DM correctness (Step 1) | Python == CPU == GPU, max_diff = 0 |
| GPU speedup (Step 2) | GPU >> CPU for n=5000 (expect 10–50×) |
| Autocomplete RF (Step 3) | RF(with AC) ≤ RF(no AC), approaching RF(complete) |
| Fraction sweep (Step 4) | RF degrades gracefully as fraction increases |
| Tile sensitivity (Step 5) | Identical outputs, time varies |
