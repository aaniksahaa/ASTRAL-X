for the cross tree recombination with GPU by hash table etc, if the GPU has to store all the output such pairs in buffer, that would hurt GPU VRAM like (nk)^1.73



** for incomplete trees, need to include h3 in the hash, since otherwise it would wrongly do this, if M1,M2 match, but the complement does not, it would be wrong, now fixed





