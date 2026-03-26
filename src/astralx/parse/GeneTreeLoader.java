package astralx.parse;

import astralx.model.GeneTree;
import astralx.model.TaxonRegistry;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public final class GeneTreeLoader {
    public static final class LoadedGeneTrees {
        public final TaxonRegistry taxa;
        public final List<GeneTree> trees;

        public LoadedGeneTrees(TaxonRegistry taxa, List<GeneTree> trees) {
            this.taxa = taxa;
            this.trees = trees;
        }
    }

    public LoadedGeneTrees load(String inputPath) throws IOException {
        TaxonRegistry taxa = new TaxonRegistry();
        List<String> lines = new ArrayList<>();
        try (BufferedReader br = new BufferedReader(new FileReader(inputPath))) {
            String line;
            while ((line = br.readLine()) != null) {
                String t = line.trim();
                if (!t.isEmpty()) {
                    lines.add(t);
                }
            }
        }

        NewickParser parser = new NewickParser();
        List<GeneTree> trees = new ArrayList<>(lines.size());
        for (int i = 0; i < lines.size(); i++) {
            trees.add(parser.parseGeneTree(lines.get(i), i, taxa));
        }
        return new LoadedGeneTrees(taxa, trees);
    }
}
