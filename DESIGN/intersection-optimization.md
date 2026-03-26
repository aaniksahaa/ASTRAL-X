# Deriving the 3×3 Intersection Matrix from 4 (or 5) Values

## 1. Setup and Notation

Let **L** be the total taxon (leaf) set with |L| = n.

Let **g** be a gene tree with leaf set **L_g ⊆ L**. An internal node of g induces a tripartition of L_g into three parts. Let this gene tree tripartition be:

> **(X | Y | Z)**   where   X ∪ Y ∪ Z = L_g,   pairwise disjoint.

Let **(A | B | C)** be a candidate species tree tripartition of the full taxon set L:

> A ∪ B ∪ C = L,   pairwise disjoint.

We need to compute the **3×3 intersection matrix**:

```
         A           B           C
X    |X ∩ A|     |X ∩ B|     |X ∩ C|
Y    |Y ∩ A|     |Y ∩ B|     |Y ∩ C|
Z    |Z ∩ A|     |Z ∩ B|     |Z ∩ C|
```

That is, we need all 9 values  I_{ij} = |R_i ∩ S_j|  where R ∈ {X, Y, Z} and S ∈ {A, B, C}.

---

## 2. Key Observation: Row Constraints (Always Valid)

**Claim 1.** For any row R ∈ {X, Y, Z}:

> |R ∩ A| + |R ∩ B| + |R ∩ C| = |R|

**Proof.** Since A, B, C are pairwise disjoint and A ∪ B ∪ C = L, the sets (R ∩ A), (R ∩ B), (R ∩ C) form a partition of R (some parts may be empty). This is because every element r ∈ R satisfies r ∈ L (since R ⊆ L_g ⊆ L), and therefore r belongs to exactly one of A, B, or C.

More formally:
- **Disjointness:** For j₁ ≠ j₂, we have (R ∩ S_{j₁}) ∩ (R ∩ S_{j₂}) = R ∩ (S_{j₁} ∩ S_{j₂}) = R ∩ ∅ = ∅.
- **Coverage:** For any r ∈ R, since r ∈ L = A ∪ B ∪ C, we have r ∈ (R ∩ A) ∪ (R ∩ B) ∪ (R ∩ C).

Therefore:

```
|R ∩ C| = |R| − |R ∩ A| − |R ∩ B|
```

**This holds regardless of whether L_g = L or L_g ⊊ L**, because R ⊆ L_g ⊆ L, so every element of R is covered by the partition (A, B, C) of L. ∎

**Consequence:** Knowing any two entries in a row gives the third. So from the 3 × 3 matrix, we can recover the entire third column from the first two:

```
|X ∩ C| = |X| − |X ∩ A| − |X ∩ B|
|Y ∩ C| = |Y| − |Y ∩ A| − |Y ∩ B|
|Z ∩ C| = |Z| − |Z ∩ A| − |Z ∩ B|
```

This reduces the requirement from 9 values to 6 values (the first two columns).

---

## 3. Column Constraints: Case 1 (L_g = L, Complete Gene Trees)

**Claim 2.** If L_g = L (the gene tree has no missing taxa), then for any column S ∈ {A, B, C}:

> |X ∩ S| + |Y ∩ S| + |Z ∩ S| = |S|

**Proof.** Since X ∪ Y ∪ Z = L_g = L, the sets (X ∩ S), (Y ∩ S), (Z ∩ S) form a partition of S. Every element s ∈ S satisfies s ∈ L = L_g = X ∪ Y ∪ Z, so s belongs to exactly one of X, Y, Z.

- **Disjointness:** (X ∩ S) ∩ (Y ∩ S) = (X ∩ Y) ∩ S = ∅ ∩ S = ∅ (since X, Y are disjoint).
- **Coverage:** For any s ∈ S, since s ∈ L = X ∪ Y ∪ Z, we have s ∈ (X ∩ S) ∪ (Y ∩ S) ∪ (Z ∩ S).

Therefore:

```
|Z ∩ S| = |S| − |X ∩ S| − |Y ∩ S|
```
∎

**Consequence (Case 1):** When L_g = L, both row and column constraints hold. The 3×3 matrix has the following structure:

```
         A              B              C
X    |X∩A|          |X∩B|          |X|−|X∩A|−|X∩B|
Y    |Y∩A|          |Y∩B|          |Y|−|Y∩A|−|Y∩B|
Z    |A|−|X∩A|−|Y∩A|   |B|−|X∩B|−|Y∩B|   (by difference)
```

The bottom-right entry |Z ∩ C| can be obtained by either the row constraint or the column constraint (both give the same answer — we verify this consistency below).

**The 4 free values are: |X ∩ A|, |X ∩ B|, |Y ∩ A|, |Y ∩ B|.**

All other 5 entries are determined:

```
|X ∩ C| = |X| − |X ∩ A| − |X ∩ B|
|Y ∩ C| = |Y| − |Y ∩ A| − |Y ∩ B|
|Z ∩ A| = |A| − |X ∩ A| − |Y ∩ A|
|Z ∩ B| = |B| − |X ∩ B| − |Y ∩ B|
|Z ∩ C| = |Z| − |Z ∩ A| − |Z ∩ B|       ... (row constraint)
```

### Consistency Check for |Z ∩ C|

Via row constraint:
```
|Z ∩ C| = |Z| − |Z ∩ A| − |Z ∩ B|
         = |Z| − (|A| − |X∩A| − |Y∩A|) − (|B| − |X∩B| − |Y∩B|)
         = |Z| − |A| − |B| + |X∩A| + |Y∩A| + |X∩B| + |Y∩B|
```

Via column constraint:
```
|Z ∩ C| = |C| − |X ∩ C| − |Y ∩ C|
         = |C| − (|X| − |X∩A| − |X∩B|) − (|Y| − |Y∩A| − |Y∩B|)
         = |C| − |X| − |Y| + |X∩A| + |X∩B| + |Y∩A| + |Y∩B|
```

These are equal if and only if:
```
|Z| − |A| − |B| = |C| − |X| − |Y|
```

Rearranging: |X| + |Y| + |Z| = |A| + |B| + |C|. This holds because both sides equal |L| (= n) when L_g = L. ✓

---

## 4. Column Constraints: Case 2 (L_g ⊊ L, Incomplete Gene Trees)

When L_g ⊊ L, the gene tree is missing some taxa. Now X ∪ Y ∪ Z = L_g ⊊ L.

**Claim 3.** The column constraint from Claim 2 **fails** in general:

> |X ∩ S| + |Y ∩ S| + |Z ∩ S| ≠ |S|   in general

**Proof of failure.** Consider s ∈ S such that s ∉ L_g. Then s ∉ X, s ∉ Y, s ∉ Z, so s does not contribute to any of (X ∩ S), (Y ∩ S), (Z ∩ S). Such elements are "invisible" to the gene tree tripartition. ∎

**Corrected column constraint:**

The sets (X ∩ S), (Y ∩ S), (Z ∩ S) partition **S ∩ L_g** (not S itself). Formally:

- **Coverage:** For any s ∈ S ∩ L_g, we have s ∈ L_g = X ∪ Y ∪ Z, so s ∈ (X ∩ S) ∪ (Y ∩ S) ∪ (Z ∩ S).
- **Disjointness:** Same argument as before.

Therefore:

```
|X ∩ S| + |Y ∩ S| + |Z ∩ S| = |S ∩ L_g|
```

and so:

```
|Z ∩ S| = |S ∩ L_g| − |X ∩ S| − |Y ∩ S|
```

Applying this to each column:

```
|Z ∩ A| = |A ∩ L_g| − |X ∩ A| − |Y ∩ A|
|Z ∩ B| = |B ∩ L_g| − |X ∩ B| − |Y ∩ B|
|Z ∩ C| = |C ∩ L_g| − |X ∩ C| − |Y ∩ C|
```

**Note:** The row constraints from Claim 1 remain unchanged (they do not depend on L_g = L).

### What Do We Need to Know?

In Case 2, the **5 free values** are:

```
|X ∩ A|,  |X ∩ B|,  |Y ∩ A|,  |Y ∩ B|,   and one of {|A ∩ L_g|, |B ∩ L_g|, |C ∩ L_g|}
```

Actually, we can be more precise. Note that |A ∩ L_g| + |B ∩ L_g| + |C ∩ L_g| = |L_g| (since A, B, C partition L and hence also partition L_g). So knowing any two of these three gives the third. Furthermore, if we know |A ∩ L_g| and |B ∩ L_g|, we can derive |C ∩ L_g| = |L_g| − |A ∩ L_g| − |B ∩ L_g|.

In practice, the values |A ∩ L_g|, |B ∩ L_g|, |C ∩ L_g| can be precomputed once per gene tree per candidate tripartition (or more efficiently via the post-order traversal, since these are just the "total" row sums restricted to L_g).

### Full Reconstruction for Case 2

Given the 4 intersection values |X∩A|, |X∩B|, |Y∩A|, |Y∩B|, and the 3 restricted sizes |A∩L_g|, |B∩L_g|, |C∩L_g| (or equivalently, 2 of them plus |L_g|):

```
|X ∩ C| = |X| − |X ∩ A| − |X ∩ B|                              (row constraint)
|Y ∩ C| = |Y| − |Y ∩ A| − |Y ∩ B|                              (row constraint)
|Z ∩ A| = |A ∩ L_g| − |X ∩ A| − |Y ∩ A|                        (corrected column)
|Z ∩ B| = |B ∩ L_g| − |X ∩ B| − |Y ∩ B|                        (corrected column)
|Z ∩ C| = |Z| − |Z ∩ A| − |Z ∩ B|                              (row constraint on Z)
```

### Consistency Check for Case 2

Let us verify |Z ∩ C| via the corrected column constraint as well:

```
|Z ∩ C| = |C ∩ L_g| − |X ∩ C| − |Y ∩ C|
```

Substituting the row-derived values:
```
= |C ∩ L_g| − (|X| − |X∩A| − |X∩B|) − (|Y| − |Y∩A| − |Y∩B|)
= |C ∩ L_g| − |X| − |Y| + |X∩A| + |X∩B| + |Y∩A| + |Y∩B|
```

Via the row constraint on Z:
```
|Z ∩ C| = |Z| − |Z∩A| − |Z∩B|
= |Z| − (|A∩L_g| − |X∩A| − |Y∩A|) − (|B∩L_g| − |X∩B| − |Y∩B|)
= |Z| − |A∩L_g| − |B∩L_g| + |X∩A| + |Y∩A| + |X∩B| + |Y∩B|
```

These are equal iff:
```
|C ∩ L_g| − |X| − |Y| = |Z| − |A∩L_g| − |B∩L_g|
```

i.e.,
```
|A∩L_g| + |B∩L_g| + |C∩L_g| = |X| + |Y| + |Z|
```

Both sides equal |L_g|. ✓

---

## 5. Connection to ASTRAL-II's Post-Order Algorithm

Observe that ASTRAL-II's weight calculation algorithm (Algorithm 1 in the ASTRAL-II paper) exploits exactly the **row constraint** (Claim 1). At each internal node u with children u_left and u_right, the algorithm computes:

```
(x, y, z) = (C₁₁ + C₂₁,  C₁₂ + C₂₂,  C₁₃ + C₂₃)
```

These are the running intersection counts |subtree(u) ∩ A|, |subtree(u) ∩ B|, |subtree(u) ∩ C|, built incrementally. Then the complement (the Z-row, i.e., the taxa *outside* subtree(u)) is computed as:

```
(C₃₁, C₃₂, C₃₃) = (|X| − x,  |Y| − y,  |Z| − z)
```

Wait — looking more carefully at the ASTRAL-II algorithm, the complement is computed as:

```
(C₃₁, C₃₂, C₃₃) = (|A| − x,  |B| − y,  |C| − z)
```

**This is the column constraint!** It computes the third row (Z-row) using |S| − |X∩S| − |Y∩S|, which is valid only when L_g = L. When gene trees have missing data, this step requires the corrected formula |S ∩ L_g| − |X∩S| − |Y∩S|.

Actually, re-examining the algorithm more carefully: the variables (x, y, z) at node u represent (|subtree(u) ∩ A|, |subtree(u) ∩ B|, |subtree(u) ∩ C|). The "complement" node splits L_g into subtree(u) and L_g \ subtree(u). In the ASTRAL-II code:

```
(C₃₁, C₃₂, C₃₃) = (|A| − x,  |B| − y,  |C| − z)
```

This uses the **column constraint** and is correct when L_g = L. For incomplete gene trees, it would need to be:

```
(C₃₁, C₃₂, C₃₃) = (|A ∩ L_g| − x,  |B ∩ L_g| − y,  |C ∩ L_g| − z)
```

---

## 6. Formal Summary

**Theorem.** Let L be the total taxon set, g a gene tree with leaf set L_g ⊆ L, (X|Y|Z) a tripartition of L_g from g, and (A|B|C) a tripartition of L. Define the 3×3 intersection matrix I where I_{ij} = |R_i ∩ S_j| for R ∈ {X, Y, Z}, S ∈ {A, B, C}. Then:

**(a)** The row sums satisfy I_{i,1} + I_{i,2} + I_{i,3} = |R_i| for each i. This holds unconditionally (regardless of whether L_g = L).

**(b)** The column sums satisfy I_{1,j} + I_{2,j} + I_{3,j} = |S_j ∩ L_g| for each j. When L_g = L, this simplifies to |S_j|.

**(c)** The grand total satisfies Σ_{i,j} I_{ij} = |L_g|.

**(d) Case L_g = L:** The entire 3×3 matrix is determined by 4 values: I_{1,1}, I_{1,2}, I_{2,1}, I_{2,2} (the top-left 2×2 submatrix), together with the known marginals |X|, |Y|, |Z|, |A|, |B|, |C| (all derivable from the trees). Explicitly:

```
I_{1,3} = |X| − I_{1,1} − I_{1,2}
I_{2,3} = |Y| − I_{2,1} − I_{2,2}
I_{3,1} = |A| − I_{1,1} − I_{2,1}
I_{3,2} = |B| − I_{1,2} − I_{2,2}
I_{3,3} = |Z| − I_{3,1} − I_{3,2}   (equivalently, |C| − I_{1,3} − I_{2,3})
```

**(e) Case L_g ⊊ L:** The matrix is determined by 4 values I_{1,1}, I_{1,2}, I_{2,1}, I_{2,2}, plus the additional quantities |A ∩ L_g| and |B ∩ L_g| (from which |C ∩ L_g| = |L_g| − |A ∩ L_g| − |B ∩ L_g|). Explicitly:

```
I_{1,3} = |X| − I_{1,1} − I_{1,2}                           (row)
I_{2,3} = |Y| − I_{2,1} − I_{2,2}                           (row)
I_{3,1} = |A ∩ L_g| − I_{1,1} − I_{2,1}                     (corrected column)
I_{3,2} = |B ∩ L_g| − I_{1,2} − I_{2,2}                     (corrected column)
I_{3,3} = |Z| − I_{3,1} − I_{3,2}                           (row)
```

The quantities |A ∩ L_g|, |B ∩ L_g| can be precomputed once per gene tree per candidate tripartition.

**(f)** In both cases, the consistency of the two independent derivations of I_{3,3} (row vs. column) is guaranteed by the identity |X| + |Y| + |Z| = |A ∩ L_g| + |B ∩ L_g| + |C ∩ L_g| = |L_g|.

---


Also please see "intersection-optimization.md" for completeness...

## 7. Implications for Computational Efficiency

### 7.1 Why This Matters

In the weight calculation for ASTRAL-like methods, computing QI(T, T') requires all 9 intersection counts. Naively computing each intersection independently (e.g., via bitset AND operations) costs O(n) per intersection, or O(9n) = O(n) total. But if we can compute just the 2×2 top-left submatrix — 4 intersections — the remaining 5 are obtained by O(1) arithmetic. This doesn't change the asymptotic cost (each intersection is still O(n) with bitsets), but it saves constant factors.

### 7.2 Where It Truly Helps: Tuple-Based Representations

In STELAR-X's integer-tuple representation, each intersection |X ∩ A| costs O(min(|X|, |A|)) rather than O(n). The optimization is more impactful here:

- **Without the optimization:** 4 intersection computations for the 2×2 bipartition case (STELAR-X already does this — Equation 2 in the STELAR-X paper computes 4 intersections for M(x,y)).
- **With the optimization for tripartitions:** If one were to extend STELAR-X to handle tripartitions (a 3×3 matrix), computing only 4 intersections instead of 9 would save more than half the work, with the remaining 5 values obtained in O(1).

### 7.3 Connection to ASTRAL-II's Algorithm

ASTRAL-II's post-order traversal (Algorithm 1) effectively computes only (x, y) = (|subtree(u) ∩ A|, |subtree(u) ∩ B|) at each node, and derives z = |subtree(u) ∩ C| via the row constraint. The complement row is obtained via the column constraint. So the algorithm already exploits this structure — it computes 2 values per node (x, y) incrementally, derives the third (z) for free, and uses the column constraint for the complement. The entire 3×3 matrix at each node costs O(1) incremental work (just additions from children), which is why ASTRAL-II achieves O(nk) per tripartition instead of O(n²k).

---

## 8. Extension to d-Partitions (Polytomies)

For completeness, consider a gene tree node that defines a **d-partition** M = (M₁|M₂|...|M_d) and a candidate tripartition T = (A|B|C). The intersection matrix is now 3 × d:

```
I_{j,i} = |S_j ∩ M_i|    for j ∈ {1,2,3} (rows: A,B,C),  i ∈ {1,...,d} (columns: M₁,...,M_d)
```

**Row constraint:** Σ_i I_{j,i} = |S_j ∩ L_g| for each j. (Or |S_j| when L_g = L.)

**Column constraint:** Σ_j I_{j,i} = |M_i| for each i.

In this case, the matrix has 3d entries. Knowing the first two rows (2d values) gives the third row via column constraints. The first two rows can be computed in O(d) total using the post-order traversal (each row-entry is an incremental sum from children). This is the basis for ASTRAL-III's O(d) QI computation.