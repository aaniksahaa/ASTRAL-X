package astralx.partition;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public final class PartitionTable {
    public static final class Entry {
        public final Partition representative;
        public int frequency;

        public Entry(Partition representative) {
            this.representative = representative;
            this.frequency = 1;
        }
    }

    private final Map<PartitionKey, Entry> byKey = new HashMap<>();
    private final List<Entry> entries = new ArrayList<>();

    public void upsert(Partition partition) {
        PartitionKey key = PartitionKey.from(partition);
        Entry existing = byKey.get(key);
        if (existing == null) {
            Entry created = new Entry(partition);
            byKey.put(key, created);
            entries.add(created);
        } else {
            existing.frequency++;
        }
    }

    public List<Entry> entries() {
        return entries;
    }
}
