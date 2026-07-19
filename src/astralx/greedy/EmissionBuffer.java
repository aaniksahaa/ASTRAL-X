package astralx.greedy;

import astralx.cluster.ClusterHash;

import java.util.Collection;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Thread-safe accumulator for polytomy-resolution emissions.
 *
 * Keyed by {@link ClusterHash} signature.  First-writer-wins: if two threads
 * emit the same signature, only the first record is retained.  This is the
 * In parallel polytomy resolution each task owns one instance, so adaptive-round
 * novelty cannot be contaminated by emissions from unrelated tasks.  Completed
 * task buffers are merged deterministically into the caller's instance.
 *
 * Phase 5 integration of these emissions into the global ClusterTable (with
 * exemplars, either by gene-tree lookup or by synthesizing multi-range
 * exemplars) happens in a separate later pass; this buffer is the canonical
 * record of WHAT was emitted by Part II.
 */
public final class EmissionBuffer {
    private final ConcurrentHashMap<ClusterHash, EmittedBipartition> emitted =
        new ConcurrentHashMap<>();

    /**
     * Add an emission.  Returns true if this signature is new to the buffer.
     * The {@code MultiRange} pointer of the LATEST attempt is NOT retained;
     * the first writer's descriptor stays.
     */
    public boolean add(EmittedBipartition b) {
        return emitted.putIfAbsent(b.signature, b) == null;
    }

    public boolean contains(ClusterHash sig) { return emitted.containsKey(sig); }
    public int size()                        { return emitted.size(); }
    public Collection<EmittedBipartition> all() { return emitted.values(); }
}
