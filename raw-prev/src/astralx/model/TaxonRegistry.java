package astralx.model;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public final class TaxonRegistry {
    private final Map<String, Integer> labelToId = new HashMap<>();
    private final List<String> idToLabel = new ArrayList<>();

    public int getOrCreate(String label) {
        Integer existing = labelToId.get(label);
        if (existing != null) {
            return existing;
        }
        int id = idToLabel.size();
        labelToId.put(label, id);
        idToLabel.add(label);
        return id;
    }

    public int getIdOrThrow(String label) {
        Integer id = labelToId.get(label);
        if (id == null) {
            throw new IllegalArgumentException("Unknown taxon label: " + label);
        }
        return id;
    }

    public String getLabel(int id) {
        return idToLabel.get(id);
    }

    public int size() {
        return idToLabel.size();
    }
}
