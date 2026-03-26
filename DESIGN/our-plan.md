So, you will be given a STELAR-X codebase as reference, and you can take help from there but i want a cleaner and essential code for this project. Like the ref is kinda bloated, so let's go through all the steps our code will need perform. You will need to write modular and efficient Java files following best practices and also optimal cuda kernels and launchers. 

# Input
=======

A set of gene trees just as stelar-x takes. For now assume that the trees will be in newick format and they will be rooted trees, so essentially, the multithreaded tree parsing will be almost all the same as done in stelar-x. 

Here, 

n = number of taxa
k = number of gene trees

So, our first task is parsing and preprocessing the trees.

# Parsing and Preprocessing

Well, the parsing is just it is like normal Newick parsing as in stelar-x. 

Now let us come to preprocessing. 

See the stelar-x ref code impleemnts many different classes like RangeBipartition, MixedBipartition, etc many many things. 

See, here we will have a very essential core class that is Cluster. 

So, at first our first task is to create similar post order traversal arrays for the trees, just like done by stelar-x, 

so, say we have k=3 input gene trees like 

((A,(B,(C,D))),E)
((A,B),((C,E),D))
(A,(D,E)) 

Now they will be preprocessed into k arrays, ofc they will be represented by numbers, assume we have 0-based indexing like A=0 upto E=4

Then the arrays will be like, 

0,1,2,3,4
0,1,2,4,3
0,3,4,-1,-1

Note that, where we have missing taxa, we have kept -1 in the array.

Aside this, we will compute a position mapping, this is like, for each array, for each taxon, which index does it appear, so here it will be like,

0,1,2,3,4
0,1,2,4,3
0,-1,-1,1,2

Explanation: Since here in 2nd array, taxon 3 appears at index 4, and taxon 4 appears at index 3
And in 3rd array, taxon 1, and taxon 2 does not appear, -1, -1, and taxon 3,4 appears at positions 1,2 respectively. 

note that, similar processing is already actually done in stelar-x you may take help from there.

# Computing Single Taxon hashes

Now, a crucial step, see, we currently have integer identifiers for the taxa, that is fine, but to handle further collision in later processes, we need to obtain sparsified taxon signatures. 

By that I mean, let there be a hash function H(t,s), which takes taxon identifier (an integer, say 0 for A), and a seed s, and it should sparsify t into the a much larger space of size P. Say, if we use 64-bit SplitMix, then, the numbers from 0 to n-1 would need to be dispersed into the large space of size P, in fact in the space of size 2^64 actually.

in addition, we will use m different random seeds to get m such different sparsified representations, and we will store all these, so kind of an nxm array, where we store all the m hashed representations for the n taxa.

The usefulness of this will be understood better later.

```c
uint64_t mix64(uint64_t x) {
    x ^= x >> 30;
    x *= 0xbf58476d1ce4e5b9ULL;   // odd => invertible mod 2^64
    x ^= x >> 27;
    x *= 0x94d049bb133111ebULL;   // odd => invertible mod 2^64
    x ^= x >> 31;
    return x;
}

uint64_t map(uint64_t i, uint64_t seed) {
    return mix64(i + seed);
}
```

This is just an example code block to show how it may look like. 

So, thus, we can get hashed representations for each taxon.

keep this simple, understandable, efficient and correct.

# Representing Clusters 

First, understand that although the trees are given as rooted in input, they can be treated as rooted or unrooted, in fact best there be a flag as such...

now, understand that, say for the tree, 

((A,B),((C,E),D))

when it is treated as rooted, the clusters will be

{A,B,C,E,D}
{A,B}
{C,E,D}
{C,E}
and the leaves like {A}, {B}, as such...

But importantly, if it is treated unrooted, there can be other clusters, 

for instance, first of all, unrooted means, we do not know the actual root and so any edge could be the root, and it has just been given rooted at a random edge.

Then, say one case is it could be rooted on any leaf edge, then

with {A}, {B,C,D,E} would also be a cluster
etc

also say is it were rooted along the parent edge of the MRCA of C and E, then, at one side there would be CE, at other side ABD.

then they would also be clusters. 

But one significant observation is that, any cluster be rooted or unrooted corresponds to either a range or subarray or the complement of that subarray for that tree (note that, complement depends on the exact taxa set in that tree, since there may be missing taxa)

Now let us how we can find a compact integer tuple representation of any cluster (be rooted or unrooted) from our array indices.

basically, we must not represent the clusters or taxa sets explicitly as sets or not even bitsets since that would require prohibitively large amount of memory.

Rather we have invented an integer tuple representation.

let us look at rooted case first.

say for the tree, 

((A,B),((C,E),D))

when it is treated as rooted, the clusters will be

{A,B,C,E,D}
{A,B}
{C,E,D}
{C,E}
and the leaves like {A}, {B}, as such...

But note that, if this tree has tree index i, then, we can represent each cluster as (i,start,end), where start and end are indices in the array for the range.

say for {A,B,C,E,D}, the range is from index 0 to index 4
for {A,B}, the range is 0 to 1
for {C,E,D}, the range is 2 to 4

as such. 

That is great, but how do we represent clusters in case of unrooted treating.

Say then, 

{A} is a cluster and also {B,C,D,E} is also a cluster, but note that, the latter is the complement of the former. 

so in the representation, we may keep one more boolean flag, whether complement or not, then say for {C,E}, it would be (i,2,3,0) (not complement), and for {A,B,D}, it would be (i,2,3,1), in fact, 

when we treat the given tree as unrooted, we proceed like this, we move bottom up, for each internal node, we register the left and right subtree range as clusters, and also the compleemnts of them as clusters, and also the full taxa set of that tree as a cluster, 

note that, by storing complements i just mean, creating another cluster object with just those 3 indices and the flag tuned on, nothing else, like we never explicitly store any sets.

Okay fine, now, however, if we treat the tree as unrooted, then, it is simple, just like sttelar-x, just bottom up traversal and left and right both will be clusters, but not any complements, and also full taxa set of that tree will be cluster. 

# Computing Cluster hashes

Now comes the most interesting part. To hash the clusters, we basically need to hash different subsets of the set of taxa.

In this step, we need to implement so that we be able to hash any cluster very efficiently. But see, hashing an arbitrary subset is expensive, but fortunately, we do not need that, Carefully understand that, any cluster the tree (be it treated rooted or unrooted), must correspond to either a subarray or the complement of a subarray in some tree-traversal array. 

Therefore, given that we did the previous step correctly, we actually will only ever need to compute hash for a subarray of the arrays or the corresponding complement. 

Now see, we will first need associative hash functions so that we can reliably apply hash on sets. We choose sum and XOR modulo 2^64. Note that we are always working on 64 bit integers and modulo 2^64 thus happens automatically. I mean make sure it is so. And also, we want the modulo to be very consistent and so that it supports add, XOR and also the reverse of add, that is subtract. 

In other words, say at some point, we do a - b mod P, where a < b, then if we obtain negative, but what we try to match later is positive (say a-b+P mod P), then we may get into trouble. So like consistent non-negative hash values so that we can support add, minus, xor etc. maybe in case of XOR, this will maybe not be a problem, since in case of XOR, just the binary representation matters maybe. Please handle this carefully. 

So, basically, we previously obtained m different single taxon hashes. For example, let m=2, 

and let

A --> m1a, m2a
B --> m1b, m2b
C --> m1c, m2c
...

as such.

Now, there will essentially be 2m different hash values for any cluster. For a cluster or taxa set {t1,t2,...}

its cluster hashes will be m hashes for the sums of m different hash values (mod P), amd m hashes for the XORs of the m different hash values, 

Now the question is, how do we efficiently calculate range hashes?

We will use prefix hashes, just like stelar-x does. 

For each of the 2m cases, we will essentially compute prefix scan arrays for that particular hash function across all the k gene trees. 

In fact, here also carefully handle the cases of missing taxa, so that -1's don't get hashed, well we maybe can keep -1's there, idk

and, also, for each gene trees, we will need 2m values denoting the hashes for alll the taxa in that gene tree, technically this is actually the last valid value in the prefix hash scan arrays, like say if it has no missing taxa, then the last values of prefix scan arrays would denote hash for all taxa in that gene trees, 

Now see, given any range (l,r), we can calc its hash, but like prefHash upto r minus pref hash upto l-1, here carefully maintain index 0 cases etc, this is standard, like, what happens if l = 0 etc, 

and also if the cluster is actually a complement, then, we first find it, then subtract these hashes from the total hashes fo the taxa in that gene tree. 

This is why complements are necessary and this is why we computed hashes for alltaxa sets in gene trees.

# Making sure that these cluster set hashes are seamlessly addable, subtractable

Like say we have a cluster meaning a set of taxa A, and another little one A', which is say a subset of A, now say we want hash of A-A', we can actually do it like just subtracting or XORing for add or XOR as associative function respectively. so like making sure this works... hopefully with our scheme, it will work definitely, 

Well, actually understand, the property we want for our hashing scheme to hold is, 

if A, B, C are sets of taxa, then 

hash(A) = hash(B) + hash(C) if and only if A = B U C, 

well, one side is trivially true always, but the other side is non-trivial, yet we can prove that this will still hold the other way with enough probability..., this proof will be added to the paper, you do not need to worry, just go on assuming it will hold... like yes we can rely on such addition, subtraction as such

# Hashtable of clusters with hash, one corresponding example cluster, and cluster frequency counter, thereby extracting unique clusters

Note that, a cluster simply means a set of taxa, and that means, no matter where that set appears in no matter what order, it is actually the same. All we need is this, for each cluster we need to be able to compute a hash (collision resistant), which already we have described, and then, at some later points, we may actually need to map a hash back to the set of taxa (note that, mapping back to the exact cluster that it came from does not matter, albeit all the clusters having same hash are equivalent in terms of any set operations like intersections we will later perform). Therefore, what we can do is this, 

keep traversing the rooted tree (note here that the original trees may be unrooted, but as said, we will have rooted them anyway for computations), now for each node (except for the root), be it a leaf or internal node, only except for the root, we register 2 clusters, the subtree range under that node, and the complement of that subtree range, like even for a leaf node, that leaf node, one element subarray will be a cluster and its complement, note that, complement is stored in the very similar way, just with the complement flag turned on, like the complement of the subarray [i,j], is stored as left=i, right=j, complement flag = ON, as such, now note that, just during this traversal, we can also keep calculating the hash of that cluster (as explained before), the reason for keeping the calculation is running is that, we want to minimiza space usage, See, the gene trees are highly likely to contain similar clusters, in that case, we do not even want to store clusters with identical taxa sets twice. How do we ensure this, well we keep a hash table, hash, correspodning cluster (just one example cluster, so that we may sometime later recover the set of taxa), and frequency of the apppearance of this cluster, actually this cluster frequency will also not be needed maybe, but ok, let's keep it

Now, we keep traversing tree, generating clusters, calculating hashes, and only insert a new entry if the hash iis unique, if not, we only increase the freq (note that we do not add this cluster at all, since an example cluster  must already have been inserted.)

# Including the super-complements

Now an important thing, so far we have done like for each partition of the tree we have included it if it is unique and so on, but for our algorithm, we also need this, in this same set, (this is actually the cluster set X), also including the complements w.r.t the total taxa set, that is if the total taxa set is S, then S-A for each A \in X so far... note that, here also we need to calc hashes and similarly add in the hash table, note very carefully that, here calc complleemnt hashes is a bit different, previously we would subtract from the total hash of the taxa in taht gene tree, but here we will need to subtract from the hash of the set of all the taxa in all the gene trees, this is the important distinction, so kinda note that, if no trees have any missing taxa, then in fact, no new cluster will be added in this step, well that is a special case...

# Representing partitions

In case of STELAR we had subtree bipartitions, the bipartition itself was a class then, however, now since we have defined a cluster very generally, like a cluster itself contains the tree index, left, right index, and also, whether complement or not, we can actually representing any partition, always as a set of clusters, like it is very general, say for aa rooted tree, if A|B is a subtree bip, we can simply have 2 clusters to represent it, if A|B|C is a tripartition, even then, we can actually do with a set of 2 clusters, assuming the rest of the taxa in that gene tree will be in the other, but note that, in some later stages of the algo, we may also need like, in a partition like structure, one part comes from one tree, another part comes from another tree, and the other part means TotalTaxa - the earlier two (note TotalTaxa here means all the total unique taxa, note for a particular gene tree), In addition, we may also have polytomies later, like then, in a partition, there may be more than 2/3, like 4 or 5 parts...

We want the representation to be as general as possible, The current plan is this, there will be a single Partition class which will be used to represent all kinds of partitions whereever needed, here there will be a set of non-trivial clusters, like, in case of tripartition, 2 non-trivial, in case of polytomy with like 4 parts, 3 non-trivial, so a set of nonTTrivialClusters, and in addition, a marker for the whole set to denote, like some big set S, which kinda determines the other trivially implied part, like S - A - B for A|B as such...

now, see, we know this big set will either be the taxa set for a gene tree, or the whole total taxa set, so this you can keep in the representation hopefully, in short we are definiing a most general yet not over consuming representing, like the clusters will just be pointers in java basically, like referemnces to clusters in the cluster ahsh tablem you see, 


and also, note that, we are just defining the partition as a clas, not yet like we are enumerating all such explicitly, we will work on it later on, we have just defined a class kinda for later use.



# HashTable of partitions

See, previously we had hashtable of clusters, fine. Now, we will have hashtable of partitions, since later on we will need to calc weights for gene trees tripartiitions, and these trips are also prone to being common among gene trees, 

so we can actually easily, hash a partition, as simply the set of hashes of the constituent clusters, but note that, the order does not matter, like the partitions A|B|C, A|C|B are essentially the same, 

so for each internal node, we can hash the partitions to store the unique tripartitions, note that, for polytomy case, there we could have done like, having hashtables for each size of partitions, or like you know, in any other way, like if hash of a partitions is the unordered set of hashes of its clusters, then be the set length differs (for poolytomy), or be the set contents differ (different taxa sets), it is a different paritions and need to be taken care of for weight calc separately

here also, we need to have such example pointers, and also frequency very important, these frequencies will need to be multiplied in case of weight calc, essentially, like weight for this partition multiplies by its freuencu of occurrences


# Finding mapping from Clusters to Candidate Bipartitions and thereby building a map of the DP search space

OK, so now we are almost at the endgame, recall that our main task was to reconstruct the tree, which we in turn do by inferring bipartitions bottom-up, well then, essentially we will need to traverse a DP search space, it is better to kinda soft build that search space beforehand, how we will search by calculating scores, will be discussed later, 

so, now our task is this, for a cluster, we need to be able to map it to candidate bipartitions (essentially two clusters), note that, all clusters are in turn in X, 

so basically, what we want to build is like, 

cluster A ----maps to----> Cluster B U cluster C

as such...

So, now task is exactly this, so focus here, given the set X of the clusters, we need to find such partitionings of clusters into such... 

Note that, our X, by definition does not contain the all-taxa cluster.

The DP essentiqally starts with the allTaxaCluster, all Taxa means all across all gt, 

and you see, by construction, for each A in X, all - A is also in X, so basically, any A in X, will induce a possible such A|total-A, this is trivial, maybe this can as weel not be stored as well, or any thing you want, maybe allTaxaCluster can be defined prior as a special... anything that works is fine


but the hard part is this, 

how do we solve this problem, given the set of clusters X, 

for each A in X, find possible breakings of A into A'|(A-A'), so that both A' and (A-A') belongs to X?

ok, i have planned this, 

note that, we can use the sizes of the clusters to some binning first, 

let X be divided into bins as per the size of the clusters, say for a size p, all the clusters of that size are in that bin, the size of that cluster is easily gettable, by subtraction per se of indices...

now see, by our hash associativity assumption (valid), we can do this, for a fixed A, check all other B, and calc hash(A) - hash(B), and checking whether this hash exists anywhere else, and then if yes, we got such one mapping

we can further optimize this by that binning, 

note that, for a fixed A of size szA, we need only check all bins of size sz < szA, furthermore, actually checking less or equal half suffices, since the other one will lie on the upper half....

now, say we check for the bin of size sz < szA / 2, note that, we know that, if such breaking is indeed possible, the other part will have size szA - sz. So, we will calc Del = hash(A) - hash(B) for B in the bin(sz), and will look up the bin of bin(szA - sz) to check whether this hash Del exists there, if yes, we get an entry, 

now we could have done this in CPU, yes also keep that support, but interesting, we can actually use GPU for such hashmap as well, note that, this is simply hashmap, one hashmap for each bin for fast membership checking, that's it, an example working GPU kernel for hashmap is given in "ref-cuda" dir, we can do as such

thus this step will give us the total DP mapping basically. 


** Also please see the "optimized-determination-of-local-mappings.md" file for a different optimized perspective for a somewhat slacked case...

# Weight Calculation of Candidate Bipartitions

Well now that we have the DP search space, we will now at some point traverse this search space and will choose optimal solutions based on scores etc. For that reason, the only thing remaining to complete the whole inference is how to calculate score for a particular candidate bipartition, then we very similar to STELAR-X, accumulate scores and keep choosing optimal biparittions at correpsonding levels and theerby form the tree. 


Ok so, now it boils down to this, for one given candidate biparitition X|Y (note that, since we will build a rooted tree, we consider candidates as bipartitions), find the quartet score for this (as opposed to triplet score as we did earlier)

Well, quartet score as defined by ASTRAL requires tripartitions by default, 



According to a document from **2018** (ASTRAL-III), the quartet contribution between a **candidate tripartition** (T=(X\mid Y\mid Z)) and a **gene-tree internal node partition** (M=(M_1\mid\cdots\mid M_d)) is defined via (QI(T,M)), and the tripartition weight uses a (\tfrac12) factor:

[
w(T)=\sum_{g\in G}\ \sum_{M\in N(g)} \frac{1}{2},QI(T,M).
]


Define, for each part (M_i) of the gene-tree node,
[
a_i=\lvert X\cap M_i\rvert,\quad b_i=\lvert Y\cap M_i\rvert,\quad c_i=\lvert Z\cap M_i\rvert.
]
Then
[
QI(T,M)=\sum_{i\in[d]}\ \sum_{j\in[d]\setminus{i}}\ \sum_{k\in[d]\setminus{i,j}}
\frac{a_i+b_j+c_k-3}{2}; a_i b_j c_k.
]


### Specializing to a **gene-tree tripartition** (A\mid B\mid C)

If the gene-tree node is binary/unrooted with exactly three sides, take (d=3) and ((M_1,M_2,M_3)=(A,B,C)). With
[
a_1=\lvert X\cap A\rvert,\ a_2=\lvert X\cap B\rvert,\ a_3=\lvert X\cap C\rvert,\quad
b_1=\lvert Y\cap A\rvert,\ldots,\quad
c_1=\lvert Z\cap A\rvert,\ldots
]
the exact equation becomes the 6 ordered terms (all permutations of choosing one side for (X), a *different* side for (Y), and the remaining side for (Z)):

[
\begin{aligned}
QI\big((X|Y|Z),(A|B|C)\big)=
&\ \frac{a_1+b_2+c_3-3}{2},a_1b_2c_3

* \frac{a_1+b_3+c_2-3}{2},a_1b_3c_2\
  &+ \frac{a_2+b_1+c_3-3}{2},a_2b_1c_3
* \frac{a_2+b_3+c_1-3}{2},a_2b_3c_1\
  &+ \frac{a_3+b_1+c_2-3}{2},a_3b_1c_2
* \frac{a_3+b_2+c_1-3}{2},a_3b_2c_1; .
  \end{aligned}
  ]
  This is exactly Eq. (4) with (d=3).



For now, let us think without polytomy, but actually that can also be done, i mean, this weighting is actually a totally separate modular part that deos not quite depend upon the previous parts, given a specific candidate biparition, how do we find its weight, is upto us, 

assume for now, we are using quartet score and we do not have polytomies, 


then the above equations will be used, and note that, all we basically need is the intersection counts, just having the intersection counts is basically enough




So, now let us consider, one candidate bipartition X|Y (like say in the DP state we were in the cluster Z and from the DP state map, say one possibility is this breaking of Z into X|Y where Z = X U Y), ok now, we need to score this X|Y wrt all the gene trees. but see we already have a hashtable of all the tripartitions, so we know the frequenciies of them too. 

Thus essentally it boils down to being able to score X|Y with respect to one particular tripartition A|B|C of gt.

Let L = all taxa set, let Lg = taxa set of gt. then it is A|B|(Lg-A-B)

Note that, the species tree must contain all taxa, thus the internal node in rooted tree tthat induces X|Y, also actually induces the tripartition 

X|Y|L-X-Y

Thus, we have the problem, 

score(  X|Y|L-X-Y, A|B|(Lg-A-B)  )

actually this is already nicely defined in ASTRAL as mentioned earlier. But we can optimize a little. 

See, naively, here, we need to 3 intersections. But look at this, 

A and B must essentially be subsets of L we know, 


therefore, |A \int L-X-Y| must be |L| - |A \int X| - |A \int Y|
similarly for B.

So, see these 2 can be actually inferred as such... but note that, we cannot guarantee that X is a subset of Lg, since Lg can be small etc... so like out of the 9 counts, 2 could be optimized, and also, in the same dimenion another one can be optimized that is L-X-Y with Lg-A-B, since Lg-A-B, will also be in L.. 

So, we kinda need to compute 6 intersections, and may infer the other three with subtraction etc.


# Calculation of Intersections

One way is current one, iterating over smaller range, as is done in STELAR-X

Another way is to use Wavelet Matrix per pair of gene trees (Build: O(nlogn), Memory: O(nlogn), Query Time: O(logn)). We may use Wavelet Matrix in GPU (working code given in ref-cuda)

Note that, the build time here is not an issue, k^2nlogn is affordable, but the memory is too much, as a whole it would require k^2nlogn memory to store all the wavelet mattrices.

Well one workaround is this, 

Look, we need to do this, for each candidate bipartition, get its total score across all the unique gene tree triparititon. 

note that, the candidate X|Y, is kinda free, in the sense that X and Y may come from different trees, diff ranges, but we know that for a gene tree tripartiiton, A|B|Lg-A-B, all these belong to the same tree, (note that, it is true that, we do not retain which exact gene trees each gt trip cn come from, like we only store one example tripartition, from one gt, but that is enough, liek since they are equivalent)

Now our idea is that, since each unique gene tree trip will in turn contribute to the total scores for each candidate bip, well, let us very carefully do this to carefully handle GPU memory

let us first init 0 scores for each candidate bip, we will essentially need to output total aggregate scores for each of these as output from the GPU kernel as we do in STELAR-X. 

Now let us for each gene tree gt_i, build wavelet matrix for this gt and all the other gts, this takes nklogn memory, now, this enables us to find any intersection count between this gt_i any other gt, so, now, for all the candidate bips X|Y, we calc the intersections efficiently and add to the scores as needs to bbe done...

Importantly, after doing this, let us carefully free up this memory of the wavelet matrix and then again doing the same thing for other gene trees, in this way at somne point the scores for all the candidate bips will be computed and outputted...

# Inference DP

Once we know all the scores, we can run the inference DP on the cluster to partitioon that DP state space map... and do like ASTRAL or STELAR-X does, building a tree, 

notet one thing that, the topmost bipartition choice itself does not have a score actually, like, its own score is 0, but its children's scores count... 


the following will be done later:
# Handling Polytomies in gene trees

# GPU batching




