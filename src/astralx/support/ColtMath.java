package astralx.support;

/**
 * Bit-exact port of the Colt 1.2.0 special functions that ASTRAL's branch
 * support relies on ({@code cern.jet.stat.Gamma} and
 * {@code cern.jet.stat.Probability}), which are themselves Java transliterations
 * of the Cephes library.  Operation order and constants follow Colt's bytecode
 * exactly so that posterior probabilities printed with full precision are
 * byte-identical to ASTRAL's.
 */
final class ColtMath {

    private ColtMath() {}

    static final double MACHEP = 1.11022302462515654042E-16;
    static final double MAXLOG = 7.09782712893383996732E2;
    static final double MINLOG = -7.451332191019412076235E2;
    static final double MAXGAM = 171.624376956302725;
    static final double SQTPI  = 2.50662827463100050242E0;
    static final double LOGPI  = 1.14472988584940017414;
    static final double big    = 4.503599627370496e15;
    static final double biginv = 2.22044604925031308085e-16;

    // ------------------------------------------------------------ polynomials

    static double polevl(double x, double[] coef, int N) {
        double ans = coef[0];
        for (int i = 1; i <= N; i++) ans = ans * x + coef[i];
        return ans;
    }

    static double p1evl(double x, double[] coef, int N) {
        double ans = x + coef[0];
        for (int i = 1; i < N; i++) ans = ans * x + coef[i];
        return ans;
    }

    // ------------------------------------------------------------ gamma

    private static final double[] GAMMA_P = {
        1.60119522476751861407E-4, 1.19135147006586384913E-3, 1.04213797561761569935E-2,
        4.76367800457137231464E-2, 2.07448227648435975150E-1, 4.94214826801497100753E-1,
        9.99999999999999996796E-1 };
    private static final double[] GAMMA_Q = {
        -2.31581873324120129819E-5, 5.39605580493303397842E-4, -4.45641913851797240494E-3,
        1.18139785222060435552E-2, 3.58236398605498653373E-2, -2.34591795718243348568E-1,
        7.14304917030273074085E-2, 1.00000000000000000320E0 };

    static double gamma(double x) {
        double p, z;
        double q = Math.abs(x);
        if (q > 33.0) {
            if (x < 0.0) {
                p = Math.floor(q);
                if (p == q) throw new ArithmeticException("gamma: overflow");
                z = q - p;
                if (z > 0.5) {
                    p += 1.0;
                    z = q - p;
                }
                z = q * Math.sin(Math.PI * z);
                if (z == 0.0) throw new ArithmeticException("gamma: overflow");
                z = Math.abs(z);
                z = Math.PI / (z * stirlingFormula(q));
                return -z;
            } else {
                return stirlingFormula(x);
            }
        }
        z = 1.0;
        while (x >= 3.0) {
            x -= 1.0;
            z *= x;
        }
        while (x < 0.0) {
            if (x == 0.0) {
                throw new ArithmeticException("gamma: singular");
            } else if (x > -1.E-9) {
                return (z / ((1.0 + 0.5772156649015329 * x) * x));
            }
            z /= x;
            x += 1.0;
        }
        while (x < 2.0) {
            if (x == 0.0) {
                throw new ArithmeticException("gamma: singular");
            } else if (x < 1.e-9) {
                return (z / ((1.0 + 0.5772156649015329 * x) * x));
            }
            z /= x;
            x += 1.0;
        }
        if ((x == 2.0) || (x == 3.0)) return z;
        x -= 2.0;
        p = polevl(x, GAMMA_P, 6);
        q = polevl(x, GAMMA_Q, 7);
        return z * p / q;
    }

    private static final double[] STIR = {
        7.87311395793093628397E-4, -2.29549961613378126380E-4, -2.68132617805781232825E-3,
        3.47222221605458667310E-3, 8.33333333333482257126E-2 };
    private static final double MAXSTIR = 143.01608;

    static double stirlingFormula(double x) {
        double w = 1.0 / x;
        double y = Math.exp(x);
        w = 1.0 + w * polevl(w, STIR, 4);
        if (x > MAXSTIR) {
            double v = Math.pow(x, 0.5 * x - 0.25);
            y = v * (v / y);
        } else {
            y = Math.pow(x, x - 0.5) / y;
        }
        y = SQTPI * y * w;
        return y;
    }

    // ------------------------------------------------------------ log gamma

    private static final double[] LG_A = {
        8.11614167470508450300E-4, -5.95061904284301438324E-4, 7.93650340457716943945E-4,
        -2.77777777730099687105E-3, 8.33333333333331927722E-2 };
    private static final double[] LG_B = {
        -1.37825152569120859100E3, -3.88016315134637840924E4, -3.31612992738871184744E5,
        -1.16237097492762307383E6, -1.72173700820839662146E6, -8.53555664245765465627E5 };
    private static final double[] LG_C = {
        -3.51815701436523470549E2, -1.70642106651881159223E4, -2.20528590553854454839E5,
        -1.13933444367982507207E6, -2.53252307177582951285E6, -2.01889141433532773231E6 };

    static double logGamma(double x) {
        double p, q, w, z;
        if (x < -34.0) {
            q = -x;
            w = logGamma(q);
            p = Math.floor(q);
            if (p == q) throw new ArithmeticException("lgam: Overflow");
            z = q - p;
            if (z > 0.5) {
                p += 1.0;
                z = p - q;
            }
            z = q * Math.sin(Math.PI * z);
            if (z == 0.0) throw new ArithmeticException("lgamma: Overflow");
            z = LOGPI - Math.log(z) - w;
            return z;
        }
        if (x < 13.0) {
            z = 1.0;
            while (x >= 3.0) {
                x -= 1.0;
                z *= x;
            }
            while (x < 2.0) {
                if (x == 0.0) throw new ArithmeticException("lgamma: Overflow");
                z /= x;
                x += 1.0;
            }
            if (z < 0.0) z = -z;
            if (x == 2.0) return Math.log(z);
            x -= 2.0;
            p = x * polevl(x, LG_B, 5) / p1evl(x, LG_C, 6);
            return Math.log(z) + p;
        }
        if (x > 2.556348e305) throw new ArithmeticException("lgamma: Overflow");
        q = (x - 0.5) * Math.log(x) - x + 0.91893853320467274178;
        if (x > 1.0e8) return q;
        p = 1.0 / (x * x);
        if (x >= 1000.0) {
            q += ((7.9365079365079365079365e-4 * p - 2.7777777777777777777778e-3) * p
                + 0.0833333333333333333333) / x;
        } else {
            q += polevl(p, LG_A, 4) / x;
        }
        return q;
    }

    // ------------------------------------------------------------ incomplete beta

    static double incompleteBeta(double aa, double bb, double xx) {
        double a, b, t, x, xc, w, y;
        boolean flag;
        if (aa <= 0.0 || bb <= 0.0) throw new ArithmeticException("ibeta: Domain error!");
        if ((xx <= 0.0) || (xx >= 1.0)) {
            if (xx == 0.0) return 0.0;
            if (xx == 1.0) return 1.0;
            throw new ArithmeticException("ibeta: Domain error!");
        }
        flag = false;
        if ((bb * xx) <= 1.0 && xx <= 0.95) {
            t = powerSeries(aa, bb, xx);
            return t;
        }
        w = 1.0 - xx;
        if (xx > (aa / (aa + bb))) {
            flag = true;
            a = bb;
            b = aa;
            xc = xx;
            x = w;
        } else {
            a = aa;
            b = bb;
            xc = w;
            x = xx;
        }
        if (flag && (b * x) <= 1.0 && x <= 0.95) {
            t = powerSeries(a, b, x);
            if (t <= MACHEP) t = 1.0 - MACHEP;
            else t = 1.0 - t;
            return t;
        }
        y = x * (a + b - 2.0) - (a - 1.0);
        if (y < 0.0) w = incompleteBetaFraction1(a, b, x);
        else w = incompleteBetaFraction2(a, b, x) / xc;
        y = a * Math.log(x);
        t = b * Math.log(xc);
        if ((a + b) < MAXGAM && Math.abs(y) < MAXLOG && Math.abs(t) < MAXLOG) {
            t = Math.pow(xc, b);
            t *= Math.pow(x, a);
            t /= a;
            t *= w;
            t *= gamma(a + b) / (gamma(a) * gamma(b));
            if (flag) {
                if (t <= MACHEP) t = 1.0 - MACHEP;
                else t = 1.0 - t;
            }
            return t;
        }
        y += t + logGamma(a + b) - logGamma(a) - logGamma(b);
        y += Math.log(w / a);
        if (y < MINLOG) t = 0.0;
        else t = Math.exp(y);
        if (flag) {
            if (t <= MACHEP) t = 1.0 - MACHEP;
            else t = 1.0 - t;
        }
        return t;
    }

    static double incompleteBetaFraction1(double a, double b, double x) {
        double xk, pk, pkm1, pkm2, qk, qkm1, qkm2;
        double k1, k2, k3, k4, k5, k6, k7, k8;
        double r, t, ans, thresh;
        int n;
        k1 = a; k2 = a + b; k3 = a; k4 = a + 1.0;
        k5 = 1.0; k6 = b - 1.0; k7 = k4; k8 = a + 2.0;
        pkm2 = 0.0; qkm2 = 1.0; pkm1 = 1.0; qkm1 = 1.0;
        ans = 1.0; r = 1.0; n = 0;
        thresh = 3.0 * MACHEP;
        do {
            xk = -(x * k1 * k2) / (k3 * k4);
            pk = pkm1 + pkm2 * xk;
            qk = qkm1 + qkm2 * xk;
            pkm2 = pkm1; pkm1 = pk; qkm2 = qkm1; qkm1 = qk;
            xk = (x * k5 * k6) / (k7 * k8);
            pk = pkm1 + pkm2 * xk;
            qk = qkm1 + qkm2 * xk;
            pkm2 = pkm1; pkm1 = pk; qkm2 = qkm1; qkm1 = qk;
            if (qk != 0) r = pk / qk;
            if (r != 0) {
                t = Math.abs((ans - r) / r);
                ans = r;
            } else {
                t = 1.0;
            }
            if (t < thresh) return ans;
            k1 += 1.0; k2 += 1.0; k3 += 2.0; k4 += 2.0;
            k5 += 1.0; k6 -= 1.0; k7 += 2.0; k8 += 2.0;
            if ((Math.abs(qk) + Math.abs(pk)) > big) {
                pkm2 *= biginv; pkm1 *= biginv; qkm2 *= biginv; qkm1 *= biginv;
            }
            if ((Math.abs(qk) < biginv) || (Math.abs(pk) < biginv)) {
                pkm2 *= big; pkm1 *= big; qkm2 *= big; qkm1 *= big;
            }
        } while (++n < 300);
        return ans;
    }

    static double incompleteBetaFraction2(double a, double b, double x) {
        double xk, pk, pkm1, pkm2, qk, qkm1, qkm2;
        double k1, k2, k3, k4, k5, k6, k7, k8;
        double r, t, ans, z, thresh;
        int n;
        k1 = a; k2 = b - 1.0; k3 = a; k4 = a + 1.0;
        k5 = 1.0; k6 = a + b; k7 = a + 1.0; k8 = a + 2.0;
        pkm2 = 0.0; qkm2 = 1.0; pkm1 = 1.0; qkm1 = 1.0;
        z = x / (1.0 - x);
        ans = 1.0; r = 1.0; n = 0;
        thresh = 3.0 * MACHEP;
        do {
            xk = -(z * k1 * k2) / (k3 * k4);
            pk = pkm1 + pkm2 * xk;
            qk = qkm1 + qkm2 * xk;
            pkm2 = pkm1; pkm1 = pk; qkm2 = qkm1; qkm1 = qk;
            xk = (z * k5 * k6) / (k7 * k8);
            pk = pkm1 + pkm2 * xk;
            qk = qkm1 + qkm2 * xk;
            pkm2 = pkm1; pkm1 = pk; qkm2 = qkm1; qkm1 = qk;
            if (qk != 0) r = pk / qk;
            if (r != 0) {
                t = Math.abs((ans - r) / r);
                ans = r;
            } else {
                t = 1.0;
            }
            if (t < thresh) return ans;
            k1 += 1.0; k2 -= 1.0; k3 += 2.0; k4 += 2.0;
            k5 += 1.0; k6 += 1.0; k7 += 2.0; k8 += 2.0;
            if ((Math.abs(qk) + Math.abs(pk)) > big) {
                pkm2 *= biginv; pkm1 *= biginv; qkm2 *= biginv; qkm1 *= biginv;
            }
            if ((Math.abs(qk) < biginv) || (Math.abs(pk) < biginv)) {
                pkm2 *= big; pkm1 *= big; qkm2 *= big; qkm1 *= big;
            }
        } while (++n < 300);
        return ans;
    }

    static double powerSeries(double a, double b, double x) {
        double s, t, u, v, n, t1, z, ai;
        ai = 1.0 / a;
        u = (1.0 - b) * x;
        v = u / (a + 1.0);
        t1 = v;
        t = u;
        n = 2.0;
        s = 0.0;
        z = MACHEP * ai;
        while (Math.abs(v) > z) {
            u = (n - b) * x / n;
            t *= u;
            v = t / (a + n);
            s += v;
            n += 1.0;
        }
        s += t1;
        s += ai;
        u = a * Math.log(x);
        if ((a + b) < MAXGAM && Math.abs(u) < MAXLOG) {
            t = gamma(a + b) / (gamma(a) * gamma(b));
            s = s * t * Math.pow(x, a);
        } else {
            t = logGamma(a + b) - logGamma(a) - logGamma(b) + u + Math.log(s);
            if (t < MINLOG) s = 0.0;
            else s = Math.exp(t);
        }
        return s;
    }

    // ------------------------------------------------------------ incomplete gamma

    static double incompleteGamma(double a, double x) {
        double ans, ax, c, r;
        if (x <= 0 || a <= 0) return 0.0;
        if (x > 1.0 && x > a) return 1.0 - incompleteGammaComplement(a, x);
        ax = a * Math.log(x) - x - logGamma(a);
        if (ax < -MAXLOG) return 0.0;
        ax = Math.exp(ax);
        r = a;
        c = 1.0;
        ans = 1.0;
        do {
            r += 1.0;
            c *= x / r;
            ans += c;
        } while (c / ans > MACHEP);
        return ans * ax / a;
    }

    static double incompleteGammaComplement(double a, double x) {
        double ans, ax, c, yc, r, t, y, z;
        double pk, pkm1, pkm2, qk, qkm1, qkm2;
        if (x <= 0 || a <= 0) return 1.0;
        if (x < 1.0 || x < a) return 1.0 - incompleteGamma(a, x);
        ax = a * Math.log(x) - x - logGamma(a);
        if (ax < -MAXLOG) return 0.0;
        ax = Math.exp(ax);
        y = 1.0 - a;
        z = x + y + 1.0;
        c = 0.0;
        pkm2 = 1.0;
        qkm2 = x;
        pkm1 = x + 1.0;
        qkm1 = z * x;
        ans = pkm1 / qkm1;
        do {
            c += 1.0;
            y += 1.0;
            z += 2.0;
            yc = y * c;
            pk = pkm1 * z - pkm2 * yc;
            qk = qkm1 * z - qkm2 * yc;
            if (qk != 0) {
                r = pk / qk;
                t = Math.abs((ans - r) / r);
                ans = r;
            } else {
                t = 1.0;
            }
            pkm2 = pkm1; pkm1 = pk; qkm2 = qkm1; qkm1 = qk;
            if (Math.abs(pk) > big) {
                pkm2 *= biginv; pkm1 *= biginv; qkm2 *= biginv; qkm1 *= biginv;
            }
        } while (t > MACHEP);
        return ans * ax;
    }

    /** Colt {@code Probability.chiSquareComplemented(v, x)}. */
    static double chiSquareComplemented(double v, double x) {
        if (x < 0.0 || v < 1.0) return 0.0;
        return incompleteGammaComplement(v / 2.0, x / 2.0);
    }
}
