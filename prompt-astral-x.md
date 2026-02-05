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

In this step, we need to implement so that we be able to hash any cluster very efficiently. But see, hashing an arbitrary subset is expensive, but fortunately, we do not need that, Carefully understand that, any cluster the tree (be it treated rooted or unrooted), must correspond to either a subarray of the complement of a subarray in some tree-traversal array. 

Therefore, given that we did the previous step correctly, we actually will only ever need to compute hash for a subarray of the arrays or the corresponding complement. 

Now see, we will first need associative hash functions so that we can reliably apply hash on sets. We choose sum and XOR modulo 2^64. Note that we are always working on 64 bit integers and modulo 2^64 thus happens automatically. I mean make sure it is so. And also, we want the modulo to be very consistent and so that it supports add, XOR and also the reverse of add, that is subtract. 

In other words, say at some point, we do a - b mod P, where a < b, then if we obtain negative, but what we try to match later is positive (say a-b+P mod P), then we may get into trouble. So liek consistent non-negative hash values so that we can support add, minux, xor etc. maybe in case of XOR, this will maybe not be a problem, since in case of XOR, just the binary representation matters maybe. Please handle this carefully. 

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

# Representing Bipartitions

# Extracting Unique Gene tree bipartitions

# Finding Candidate Bipartitions

# Finding mapping from Clusters to Candidate Bipartitions

# Weight Calculation of Candidate Bipartitions

One way is current one, iterating over smaller range

Another way is to use Wavelet Matrix per pair of gene trees (Build: O(nlogn), Memory: O(nlogn), Query Time: O(logn)). We may use Wavelet Matrix in GPU. 

The strategy can be like, for each bipartitions in gene trees (like SBP or for unrooted), building wavelet matrix for this and all others (nklogn memory), and then for all candidate bipartitions, find intersection with its ranges (in logn time), and update the contribution to the weight for that candidate bip. 

# Inference DP




