//! The complementary error function and the standard normal distribution.
//!
//! `erfc` follows the algorithm and the coefficients of fdlibm's `s_erf.c`
//! (Sun Microsystems, 1993; the version distributed with FreeBSD's msun):
//! (the constants below are the same f64 values, written in their shortest
//! decimal form), rational approximations on `|x| < 0.84375`, `[0.84375, 1.25)`,
//! `[1.25, 1/0.35)` and `[1/0.35, 28)`, and an underflow to zero beyond 28.
//! fdlibm documents an error below one ulp; this port is checked against
//! tabulated values in its tests with the tolerances declared there, and no
//! accuracy beyond those tests is claimed. `exp` is the platform's `f64::exp`,
//! so results are deterministic per platform, not bit identical across
//! platforms.
//!
//! The notice of the original, preserved as its licence requires:
//!
//! ```text
//! ====================================================
//! Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved.
//!
//! Developed at SunPro, a Sun Microsystems, Inc. business.
//! Permission to use, copy, modify, and distribute this
//! software is freely granted, provided that this notice
//! is preserved.
//! ====================================================
//! ```

const ERX: f64 = 8.450_629_115_104_675e-1;
// erf on [0, 0.84375].
const PP0: f64 = 1.283_791_670_955_125_6e-1;
const PP1: f64 = -3.250_421_072_470_015e-1;
const PP2: f64 = -2.848_174_957_559_851e-2;
const PP3: f64 = -5.770_270_296_489_442e-3;
const PP4: f64 = -2.376_301_665_665_016_3e-5;
const QQ1: f64 = 3.979_172_239_591_553_5e-1;
const QQ2: f64 = 6.502_224_998_876_73e-2;
const QQ3: f64 = 5.081_306_281_875_766e-3;
const QQ4: f64 = 1.324_947_380_043_216_4e-4;
const QQ5: f64 = -3.960_228_278_775_368e-6;
// erf on [0.84375, 1.25].
const PA0: f64 = -2.362_118_560_752_659_4e-3;
const PA1: f64 = 4.148_561_186_837_483_3e-1;
const PA2: f64 = -3.722_078_760_357_013e-1;
const PA3: f64 = 3.183_466_199_011_617_5e-1;
const PA4: f64 = -1.108_946_942_823_966_8e-1;
const PA5: f64 = 3.547_830_432_561_823_6e-2;
const PA6: f64 = -2.166_375_594_868_791e-3;
const QA1: f64 = 1.064_208_804_008_442_3e-1;
const QA2: f64 = 5.403_979_177_021_71e-1;
const QA3: f64 = 7.182_865_441_419_627e-2;
const QA4: f64 = 1.261_712_198_087_616_4e-1;
const QA5: f64 = 1.363_708_391_202_905e-2;
const QA6: f64 = 1.198_449_984_679_910_7e-2;
// erfc on [1.25, 1/0.35].
const RA0: f64 = -9.864_944_034_847_148e-3;
const RA1: f64 = -6.938_585_727_071_818e-1;
const RA2: f64 = -1.055_862_622_532_329_1e1;
const RA3: f64 = -6.237_533_245_032_600_6e1;
const RA4: f64 = -1.623_966_694_625_734_7e2;
const RA5: f64 = -1.846_050_929_067_110_4e2;
const RA6: f64 = -8.128_743_550_630_66e1;
const RA7: f64 = -9.814_329_344_169_145;
const SA1: f64 = 1.965_127_166_743_925_7e1;
const SA2: f64 = 1.376_577_541_435_190_4e2;
const SA3: f64 = 4.345_658_774_752_292_3e2;
const SA4: f64 = 6.453_872_717_332_679e2;
const SA5: f64 = 4.290_081_400_275_678_3e2;
const SA6: f64 = 1.086_350_055_417_794_4e2;
const SA7: f64 = 6.570_249_770_319_282;
const SA8: f64 = -6.042_441_521_485_81e-2;
// erfc on [1/0.35, 28].
const RB0: f64 = -9.864_942_924_700_1e-3;
const RB1: f64 = -7.992_832_376_805_23e-1;
const RB2: f64 = -1.775_795_491_775_475_2e1;
const RB3: f64 = -1.606_363_848_558_219_2e2;
const RB4: f64 = -6.375_664_433_683_896e2;
const RB5: f64 = -1.025_095_131_611_077_2e3;
const RB6: f64 = -4.835_191_916_086_514e2;
const SB1: f64 = 3.033_806_074_348_246e1;
const SB2: f64 = 3.257_925_129_965_739e2;
const SB3: f64 = 1.536_729_586_084_437e3;
const SB4: f64 = 3.199_858_219_508_595_5e3;
const SB5: f64 = 2.553_050_406_433_164_4e3;
const SB6: f64 = 4.745_285_412_069_553_7e2;
const SB7: f64 = -2.244_095_244_658_582e1;

/// `1 / sqrt(2 pi)`.
pub(crate) const INV_SQRT_2PI: f64 = 0.398_942_280_401_432_7;

/// fdlibm's `R/S` pair for `1.25 <= ax < 28`, with `s = 1 / ax^2`.
fn tail_rational(ax: f64, s: f64) -> (f64, f64) {
    if ax < 1.0 / 0.35 {
        (
            RA0 + s * (RA1 + s * (RA2 + s * (RA3 + s * (RA4 + s * (RA5 + s * (RA6 + s * RA7)))))),
            1.0 + s
                * (SA1
                    + s * (SA2
                        + s * (SA3 + s * (SA4 + s * (SA5 + s * (SA6 + s * (SA7 + s * SA8))))))),
        )
    } else {
        (
            RB0 + s * (RB1 + s * (RB2 + s * (RB3 + s * (RB4 + s * (RB5 + s * RB6))))),
            1.0 + s * (SB1 + s * (SB2 + s * (SB3 + s * (SB4 + s * (SB5 + s * (SB6 + s * SB7)))))),
        )
    }
}

/// erfc(|x|) for 0.84375 <= |x| < 28.
fn erfc_tail(ax: f64) -> f64 {
    if ax < 1.25 {
        let s = ax - 1.0;
        let p = PA0 + s * (PA1 + s * (PA2 + s * (PA3 + s * (PA4 + s * (PA5 + s * PA6)))));
        let q = 1.0 + s * (QA1 + s * (QA2 + s * (QA3 + s * (QA4 + s * (QA5 + s * QA6)))));
        return 1.0 - ERX - p / q;
    }
    let s = 1.0 / (ax * ax);
    let (r, big_s) = tail_rational(ax, s);
    // z is ax with its 32 low bits cleared, so z * z is exact and
    // exp(-x^2) is computed as exp(-z^2) * exp((z - x)(z + x)) (fdlibm).
    let z = f64::from_bits(ax.to_bits() & 0xffff_ffff_0000_0000);
    (-z * z - 0.5625).exp() * ((z - ax) * (z + ax) + r / big_s).exp() / ax
}

/// The complementary error function, `1 - erf(x)`, computed directly so that
/// the upper tail keeps its relative precision. NaN maps to NaN.
#[must_use]
pub(crate) fn erfc(x: f64) -> f64 {
    if x.is_nan() {
        return x;
    }
    let ax = x.abs();
    if ax < 0.84375 {
        if ax < 2f64.powi(-56) {
            return 1.0 - x;
        }
        let z = x * x;
        let r = PP0 + z * (PP1 + z * (PP2 + z * (PP3 + z * PP4)));
        let s = 1.0 + z * (QQ1 + z * (QQ2 + z * (QQ3 + z * (QQ4 + z * QQ5))));
        let y = r / s;
        if x < 0.25 {
            return 1.0 - (x + x * y);
        }
        return 0.5 - (x - 0.5 + x * y);
    }
    if ax < 28.0 {
        let tail = erfc_tail(ax);
        return if x < 0.0 { 2.0 - tail } else { tail };
    }
    if x < 0.0 { 2.0 } else { 0.0 }
}

/// The scaled complementary error function `erfcx(x) = e^{x^2} erfc(x)`, for
/// `x >= 0` (the only arguments the pricer needs).
///
/// On `[1.25, 28)` it reuses fdlibm's rational approximations: there
/// `erfc(x) = e^{-x^2 - 0.5625 + R/S} / x` exactly in the approximation, so
/// `erfcx(x) = e^{R/S - 0.5625} / x` with no `e^{x^2}` to overflow. Beyond 28
/// it is the asymptotic series `sum_n (-1)^n (2n - 1)!! / (2x^2)^n / (x sqrt pi)`
/// to `n = 7` (Abramowitz and Stegun, 7.1.23), whose next term is below
/// 1e-17 relative at `x = 28`. Below 1.25,
/// `e^{x^2} erfc(x)` directly. NaN and negative arguments give NaN.
#[must_use]
pub(crate) fn erfcx(x: f64) -> f64 {
    if x.is_nan() || x < 0.0 {
        return f64::NAN;
    }
    if x < 1.25 {
        return (x * x).exp() * erfc(x);
    }
    if x < 28.0 {
        let s = 1.0 / (x * x);
        let (r, big_s) = tail_rational(x, s);
        return (r / big_s - 0.5625).exp() / x;
    }
    if x.is_infinite() {
        return 0.0;
    }
    let z = 1.0 / (2.0 * x * x);
    let series = 1.0
        - z * (1.0
            - 3.0
                * z
                * (1.0
                    - 5.0
                        * z
                        * (1.0 - 7.0 * z * (1.0 - 9.0 * z * (1.0 - 11.0 * z * (1.0 - 13.0 * z))))));
    series / (x * std::f64::consts::PI.sqrt())
}

/// The standard normal distribution function, `N(x) = erfc(-x / sqrt 2) / 2`.
/// The lower tail is computed directly (no `1 - N(-x)` cancellation).
#[must_use]
pub(crate) fn norm_cdf(x: f64) -> f64 {
    0.5 * erfc(-x / std::f64::consts::SQRT_2)
}

/// The standard normal density.
#[must_use]
pub(crate) fn norm_pdf(x: f64) -> f64 {
    INV_SQRT_2PI * (-0.5 * x * x).exp()
}

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::indexing_slicing
)]
mod tests {
    use super::*;

    /// Relative error, or absolute error for an expected value of 0.
    fn rel(got: f64, want: f64) -> f64 {
        if want == 0.0 {
            got.abs()
        } else {
            ((got - want) / want).abs()
        }
    }

    // erfc at the exact f64 arguments (each literal is the f64 nearest to
    // the reference, written in its shortest form):
    // 1 - erf(x) with erf from its Maclaurin series (Abramowitz and Stegun,
    // 7.1.5) in 1200-digit decimal arithmetic
    // (`scripts/erfc_reference.py`). Tolerance: 4e-16 relative for normal
    // results; erfc(27) is subnormal and is compared to its nearest f64.
    const ERFC_TABLE: &[(f64, f64)] = &[
        (0.0, 1.0),
        (0.1, 8.875_370_839_817_15e-1),
        (0.5, 4.795_001_221_869_535e-1),
        (0.84375, 2.327_743_387_676_583_8e-1),
        (1.0, 1.572_992_070_502_851_3e-1),
        (1.25, 7.709_987_174_354_177e-2),
        (2.0, 4.677_734_981_047_266e-3),
        (1.0 / 0.35, 5.331_231_138_832_279_5e-5),
        (3.0, 2.209_049_699_858_544e-5),
        (5.0, 1.537_459_794_428_035e-12),
        (8.0, 1.122_429_717_298_292_6e-29),
        (10.0, 2.088_487_583_762_545e-45),
        (26.0, 5.663_192_408_856_143e-296),
        (27.0, 5.237_05e-319),
        (-1.0, 1.842_700_792_949_714_8),
        (-0.5, 1.520_499_877_813_046_5),
    ];

    #[test]
    fn test_erfc_tabulated_values_within_declared_tolerance() {
        for &(x, want) in ERFC_TABLE {
            let got = erfc(x);
            // A subnormal result has fewer significant bits: allow two
            // units of the smallest subnormal there.
            let ok = if want < f64::MIN_POSITIVE {
                (got - want).abs() <= 2.0 * 5e-324
            } else {
                rel(got, want) <= 4e-16
            };
            assert!(ok, "erfc({x}) = {got:e}, want {want:e}");
        }
    }

    #[test]
    fn test_erfc_beyond_28_underflows_to_zero_and_two() {
        assert_eq!(erfc(28.0), 0.0);
        assert_eq!(erfc(38.0), 0.0);
        assert_eq!(erfc(-38.0), 2.0);
        assert_eq!(erfc(f64::INFINITY), 0.0);
        assert_eq!(erfc(f64::NEG_INFINITY), 2.0);
        assert!(erfc(f64::NAN).is_nan());
    }

    #[test]
    fn test_norm_cdf_tails_are_direct_and_symmetric() {
        // N(-8) and N(-38) (subnormal) from the same high-precision erfc.
        // N(x) = erfc(-x / sqrt 2) / 2: the rounding of x / sqrt 2 (half an
        // ulp) moves erfc by a relative 2 z^2 eps/2 at z = -x / sqrt 2, so the
        // tolerance is (z^2 + 1) * 4 eps: 3e-14 at x = -8, 7e-13 at x = -38.
        let tol = |x: f64| (x * x / 2.0 + 1.0) * 4.0 * f64::EPSILON;
        assert!(rel(norm_cdf(-8.0), 6.220_960_574_271_784e-16) <= tol(-8.0));
        let n38 = norm_cdf(-38.0);
        assert!(n38 > 0.0 && n38 < f64::MIN_POSITIVE, "N(-38) = {n38:e}");
        // Subnormal (about 49 significant bits remain at 2.9e-316), well
        // inside the argument-rounding tolerance.
        assert!(rel(n38, 2.885_428_35e-316) <= tol(-38.0));
        assert!((norm_cdf(8.0) - (1.0 - 6.220_960_574_271_784e-16)).abs() <= f64::EPSILON);
        assert_eq!(norm_cdf(40.0), 1.0);
        assert_eq!(norm_cdf(-40.0), 0.0);
        for x in [-3.0, -1.0, -0.1, 0.0, 0.3, 2.0] {
            assert!((norm_cdf(x) + norm_cdf(-x) - 1.0).abs() <= 2e-16, "x = {x}");
        }
    }

    // erfcx(x) = e^(x^2) erfc(x) in 400-digit decimal arithmetic
    // (`scripts/black_reference.py`); tolerance 1e-15 relative.
    const ERFCX_TABLE: &[(f64, f64)] = &[
        (0.0, 1.0),
        (0.5, 0.6156903441929259),
        (1.0, 0.427583576155807),
        (1.25, 0.3678229164523611),
        (2.0, 0.25539567631050575),
        (5.0, 0.11070463773306863),
        (10.0, 0.05614099274382259),
        (27.0, 0.02088160799042094),
        (28.0, 0.020136801964214277),
        (30.0, 0.01879588886141675),
        (100.0, 0.005641613782989433),
        (10000000000.0, 5.641895835477563e-11),
    ];

    #[test]
    fn test_erfcx_tabulated_values_within_declared_tolerance() {
        for &(x, want) in ERFCX_TABLE {
            let got = erfcx(x);
            assert!(
                rel(got, want) <= 1e-15,
                "erfcx({x}) = {got:e}, want {want:e}"
            );
        }
        assert!(erfcx(-1.0).is_nan() && erfcx(f64::NAN).is_nan());
        assert_eq!(erfcx(f64::INFINITY), 0.0);
    }

    #[test]
    fn test_norm_pdf_reference_values() {
        assert!(rel(norm_pdf(0.0), 3.989_422_804_014_327e-1) <= 2e-16);
        assert!(rel(norm_pdf(1.0), 2.419_707_245_191_433_4e-1) <= 4e-16);
        assert!(rel(norm_pdf(-1.0), norm_pdf(1.0)) == 0.0);
        assert_eq!(norm_pdf(40.0), 0.0);
    }
}
