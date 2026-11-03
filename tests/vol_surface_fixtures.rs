//! Synthetic volatility surface fixtures: a known surface the static
//! diagnostics accept, surfaces built to break them, and the shock
//! scenarios with their own diagnostics.

#![cfg(feature = "options")]

use market_maker_rs::options::surface::{
    DiagnosticGrid, DiagnosticsConfig, KRange, Region, ShortExpiryPolicy, SurfaceShock, SviSlice,
    SviSurface, VolSurface, diagnose, run_scenarios, standard_scenarios,
};

const YEAR_DAYS: f64 = 365.0;
const EXPIRY_DAYS: [f64; 5] = [7.0, 30.0, 90.0, 180.0, 365.0];

/// The known surface: at-the-money volatility from 50% at 7 days to 45%
/// at one year, skew rho = -0.4, b growing with sqrt(tau), calibrated on
/// k in [-0.6, 0.6].
fn known_slice(tau: f64) -> SviSlice {
    let (b, rho, m, sigma) = (0.12 * tau.sqrt(), -0.4, 0.02, 0.25);
    let atm_vol = 0.5 - 0.05 * tau.min(1.0);
    let atm_w = atm_vol * atm_vol * tau;
    let a = atm_w - b * (rho * (0.0 - m) + (0.0_f64 - m).hypot(sigma));
    let range = KRange::new(-0.6, 0.6).unwrap();
    SviSlice::new(tau, a, b, rho, m, sigma, range).unwrap()
}

fn known_surface(policy: ShortExpiryPolicy) -> SviSurface {
    let slices = EXPIRY_DAYS
        .iter()
        .map(|d| known_slice(d / YEAR_DAYS))
        .collect();
    SviSurface::new(slices, policy).unwrap()
}

/// Every node, three points inside each segment, and the wings to |k| = 1.5.
fn grid() -> DiagnosticGrid {
    let mut tau = Vec::new();
    for pair in EXPIRY_DAYS.windows(2) {
        let [lo, hi] = [pair[0] / YEAR_DAYS, pair[1] / YEAR_DAYS];
        tau.extend([
            lo,
            lo + 0.25 * (hi - lo),
            lo + 0.5 * (hi - lo),
            lo + 0.75 * (hi - lo),
        ]);
    }
    tau.push(EXPIRY_DAYS[4] / YEAR_DAYS);
    DiagnosticGrid::uniform_k(-1.5, 1.5, 121, tau).unwrap()
}

#[test]
fn test_known_surface_diagnostics_are_clean_including_wings_and_short_expiry() {
    let surface = known_surface(ShortExpiryPolicy::Reject);
    assert!(surface.covers_nodes(&grid()));
    let report = diagnose(&surface, &grid(), DiagnosticsConfig::default()).unwrap();
    assert!(report.is_clean(), "{report:?}");
    assert_eq!(report.butterfly.points, 121 * 17);
    // Margins, not just signs: the fixture is well inside the conditions.
    assert!(report.butterfly.worst.unwrap().value > 0.2);
    assert!(report.calendar_gap.worst.unwrap().value > 0.0);
}

#[test]
fn test_known_surface_wings_are_extrapolated_and_short_expiry_opt_in() {
    let surface = known_surface(ShortExpiryPolicy::Reject);
    let tau = 30.0 / YEAR_DAYS;
    assert_eq!(surface.region(0.0, tau).unwrap(), Region::Calibrated);
    assert_eq!(surface.region(1.2, tau).unwrap(), Region::Extrapolated);
    assert!(surface.total_variance(0.0, 1.0 / YEAR_DAYS).is_err());

    let extended = known_surface(ShortExpiryPolicy::ConstantVol);
    let one_day = 1.0 / YEAR_DAYS;
    assert_eq!(extended.region(0.0, one_day).unwrap(), Region::Extrapolated);
    // Constant implied volatility per k below the first expiry.
    let first = known_slice(7.0 / YEAR_DAYS);
    let (w7, _, _) = first.eval(0.1).unwrap();
    let w1 = extended.total_variance(0.1, one_day).unwrap().w();
    assert!(((w1 / one_day).sqrt() - (w7 / first.tau()).sqrt()).abs() < 1e-12);
    let short_grid = DiagnosticGrid::uniform_k(
        -1.0,
        1.0,
        41,
        vec![0.5 / YEAR_DAYS, one_day, 3.0 / YEAR_DAYS],
    )
    .unwrap();
    assert!(
        diagnose(&extended, &short_grid, DiagnosticsConfig::default())
            .unwrap()
            .is_clean()
    );
}

#[test]
fn test_broken_surfaces_are_flagged_not_repaired() {
    let range = KRange::new(-1.5, 1.5).unwrap();
    // Butterfly: Axel Vogt's slice (Gatheral and Jacquier 2014).
    let vogt = |tau| SviSlice::new(tau, -0.0410, 0.1331, 0.3060, 0.3586, 0.4153, range).unwrap();
    let butterfly = SviSurface::new(vec![vogt(1.0), vogt(2.0)], ShortExpiryPolicy::Reject).unwrap();
    let report = diagnose(
        &butterfly,
        &DiagnosticGrid::uniform_k(-1.5, 1.5, 121, vec![1.0]).unwrap(),
        DiagnosticsConfig::default(),
    )
    .unwrap();
    assert!(!report.butterfly.is_clean());

    // Calendar: the 90 day slice above the 180 day one.
    let mut slices: Vec<SviSlice> = EXPIRY_DAYS
        .iter()
        .map(|d| known_slice(d / YEAR_DAYS))
        .collect();
    let tau90 = 90.0 / YEAR_DAYS;
    let s90 = known_slice(tau90);
    slices[2] = SviSlice::new(
        tau90,
        s90.a() + 0.08,
        s90.b(),
        s90.rho(),
        s90.m(),
        s90.sigma(),
        s90.k_range(),
    )
    .unwrap();
    let calendar = SviSurface::new(slices, ShortExpiryPolicy::Reject).unwrap();
    let report = diagnose(&calendar, &grid(), DiagnosticsConfig::default()).unwrap();
    assert!(!report.calendar_gap.is_clean());
    assert!(!report.calendar_slope.is_clean());
    // The surface itself is unchanged: diagnostics never repair.
    assert!((calendar.slices()[2].a() - (s90.a() + 0.08)).abs() < 1e-15);
}

#[test]
fn test_scenarios_carry_their_own_diagnostics() {
    let base = known_surface(ShortExpiryPolicy::Reject);
    let mut shocks = standard_scenarios().unwrap();
    shocks.push(SurfaceShock::new("skew-violent", 0.0, -1.5, 0.0, 30.0 / YEAR_DAYS).unwrap());
    let reports = run_scenarios(&base, &shocks, &grid(), DiagnosticsConfig::default()).unwrap();
    assert_eq!(reports.len(), shocks.len());
    for report in &reports {
        assert_eq!(
            report.diagnostics.butterfly.points + report.diagnostics.evaluation_failures.len(),
            121 * 17
        );
    }
    let parallel = reports
        .iter()
        .find(|r| r.shock.name() == "vol-up-5")
        .unwrap();
    assert!(
        parallel.diagnostics.is_clean(),
        "{:?}",
        parallel.diagnostics
    );
    let violent = reports
        .iter()
        .find(|r| r.shock.name() == "skew-violent")
        .unwrap();
    assert!(!violent.diagnostics.is_clean());
}
