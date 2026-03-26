package astralx.cli;

public final class CliParser {
    private CliParser() {}

    public static Config parse(String[] args) {
        String input = null;
        String output = "species_tree.newick";
        boolean unrooted = false;
        int m = 4;
        long seed = 42L;
        Config.IntersectionMode intersectionMode = Config.IntersectionMode.WAVELET;
        Config.WeightMode weightMode = Config.WeightMode.GPU;

        for (int i = 0; i < args.length; i++) {
            switch (args[i]) {
                case "-i":
                case "--input":
                    input = args[++i];
                    break;
                case "-o":
                case "--output":
                    output = args[++i];
                    break;
                case "--unrooted":
                    unrooted = true;
                    break;
                case "-m":
                case "--hash-replicates":
                    m = Integer.parseInt(args[++i]);
                    break;
                case "--seed":
                    seed = Long.parseLong(args[++i]);
                    break;
                case "--intersection":
                    String mode = args[++i].trim().toLowerCase();
                    if ("wavelet".equals(mode)) {
                        intersectionMode = Config.IntersectionMode.WAVELET;
                    } else if ("cpu".equals(mode)) {
                        intersectionMode = Config.IntersectionMode.CPU;
                    } else {
                        throw new IllegalArgumentException("Invalid intersection mode: " + mode + " (expected: wavelet|cpu)");
                    }
                    break;
                case "--weight-mode":
                    String wmode = args[++i].trim().toLowerCase();
                    if ("gpu".equals(wmode)) {
                        weightMode = Config.WeightMode.GPU;
                    } else if ("cpu".equals(wmode)) {
                        weightMode = Config.WeightMode.CPU;
                    } else {
                        throw new IllegalArgumentException("Invalid weight mode: " + wmode + " (expected: gpu|cpu)");
                    }
                    break;
                default:
                    throw new IllegalArgumentException("Unknown argument: " + args[i]);
            }
        }

        if (input == null) {
            throw new IllegalArgumentException("Missing required -i/--input argument");
        }
        if (m < 1) {
            throw new IllegalArgumentException("hash-replicates must be >= 1");
        }

        return new Config(input, output, unrooted, m, seed, intersectionMode, weightMode);
    }

    public static String usage() {
        return "Usage: java astralx.Main -i <gene_trees.newick> [-o out.newick] [--unrooted] [-m hashReplicates] [--seed S] [--intersection wavelet|cpu] [--weight-mode gpu|cpu]";
    }
}
