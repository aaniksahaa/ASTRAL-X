package astralx.support;

/**
 * Local posterior probability and coalescent-unit branch length of a species-tree
 * branch from its three quartet frequencies (Sayyari &amp; Mirarab 2016, MBE 33:1654).
 *
 * <p>This is a faithful port of ASTRAL's {@code Posterior} class.  ASTRAL relies on
 * the Colt library for {@code logGamma} and the regularised incomplete beta
 * function; {@link ColtMath} reproduces those routines operation for operation,
 * so every printed value is byte-identical to ASTRAL's.
 *
 * <pre>
 *   h(x)  = B(x+1, n-x+2λ) · (1 − I_{1/3}(x+1, n-x+2λ))
 *   PP(1) = h(f1) / (h(f1) + 2^{f2-f1} h(f2) + 2^{f3-f1} h(f3))
 *   bl    = −ln(1.5 · (1 − f1/(n+2λ)))   when f1/(n+2λ) &gt; 1/3, else 0
 * </pre>
 */
public final class Posterior {

    private static final double LOG2 = Math.log(2.0);
    /** Colt's {@code Constants.MACHEP}. */
    private static final double MACHEP = 1.11022302462515654042E-16;
    private static final String MESSAGE = "This shouldn't have happened."
        + " Maybe you set lambda too high or too low? "
        + "Please report the error with the following numbers: ";

    private final double f1, f2, f3, n, lambda;
    private double posterior = -1;

    public Posterior(double f1, double f2, double f3, double n, double lambda) {
        this.f1 = f1; this.f2 = f2; this.f3 = f3; this.n = n; this.lambda = lambda;
    }

    public double getPost() {
        if (posterior == -1) posterior = post();
        return posterior;
    }

    private double pValue = -1;

    /** ASTRAL's polytomy-test p-value (chi-square with two degrees of freedom); -2 when n &lt;= 15. */
    public double getPvalue() {
        if (pValue == -1) pValue = pvalue();
        return pValue;
    }

    private double pvalue() {
        double fThird = n / 3.;
        double x = Math.pow((f1 - fThird), 2) / fThird + Math.pow((f2 - fThird), 2) / fThird
            + Math.pow((f3 - fThird), 2) / fThird;
        double p = ColtMath.chiSquareComplemented(2, x);
        if (n <= 15) return -2;
        return p;
    }

    private double betaRatio(double alpha1, double beta1, double alpha2, double beta2) {
        return logGamma(alpha1) + logGamma(beta1) - logGamma(alpha2) - logGamma(beta2);
    }

    private double G(double x, double nt) {
        double g = 1 - incompleteBeta(x + 1, nt - x + 2 * lambda, 1.0 / 3.0);
        if (g <= MACHEP) g = 0.0;
        return g;
    }

    private double r(double mi) {
        return Math.exp(LOG2 * (mi - f1)
            + betaRatio(mi + 1, n - mi + 2 * lambda, f1 + 1, n - f1 + 2 * lambda));
    }

    private double rG(double mi) {
        double x = G(mi, n) * r(mi);
        if (Double.isNaN(x) || Double.isInfinite(x)) {
            if (mi * 3 < n) {
                return 0;
            } else if (mi * 3 >= n) {
                if (f1 > mi) {
                    return 0;
                } else if (f1 < mi) {
                    return Double.POSITIVE_INFINITY;
                } else {
                    throw new IllegalStateException(MESSAGE + "\n" + f1 + " " + f2 + " " + f3
                        + " " + lambda + " " + n);
                }
            } else {
                return Double.POSITIVE_INFINITY;
            }
        }
        return x;
    }

    private double post() {
        double g2 = rG(f2);
        double g3 = rG(f3);
        double g = G(f1, n) / (G(f1, n) + g2 + g3);
        if (Double.isInfinite(g)) {
            throw new IllegalStateException(MESSAGE + "\n" + f1 + " " + f2 + " " + f3
                + " " + lambda + " " + n);
        }
        if (Double.isNaN(g)) {
            if (f1 * 3 < n) return 0;
            throw new IllegalStateException(MESSAGE + "\n" + f1 + " " + f2 + " " + f3
                + " " + lambda + " " + n);
        }
        return g;
    }

    public static double branchLength(double f, double all, double lambd) {
        if ((f / (all + 2 * lambd)) > 1.0 / 3) {
            return -Math.log(1.5 * (1.0 - (f / (all + 2 * lambd))));
        }
        return 0.0;
    }

    public double branchLength() { return branchLength(f1, n, lambda); }

    // ------------------------------------------------------------------ special functions
    // Bit-exact Colt/Cephes ports (see ColtMath) so that full-precision output is
    // byte-identical to ASTRAL's.

    static double logGamma(double x) { return ColtMath.logGamma(x); }

    static double incompleteBeta(double a, double b, double x) { return ColtMath.incompleteBeta(a, b, x); }
}
