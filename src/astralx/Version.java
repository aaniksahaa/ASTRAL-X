package astralx;

/** Build and runtime version source. */
final class Version {
    static final String DEFAULT = "1.0.0";

    private Version() {}

    static String current() {
        String packaged = Main.class.getPackage().getImplementationVersion();
        return packaged == null || packaged.isBlank() ? DEFAULT : packaged;
    }
}
