# A bounded reproduction of the quadratic rough Heston model (#111)

Everything here is synthetic: Monte Carlo paths of the model at the SPX parameters of the
papers, calibrated to nothing here and never standing for BTC or any other market. Only the
SPX part is in scope. OptionStratLib has no VIX pricing or calibration under the model
(joaquinbejar/OptionStratLib#967, #968), so the VIX instruments and the joint SPX/VIX
calibration are out.

The papers are Rosenbaum and Zhang, *Multi-asset market making under the quadratic rough
Heston* (arXiv:2212.10164v1), and its reference [28], *Deep calibration of the quadratic rough
Heston model* (arXiv:2107.01611), the source of its kernel.

- `criteria.json` fixes the checks of step 1, and the papers' values they compare with,
  before anything runs.
- `criteria_1b.json` fixes the checks of step 1b, which followed step 1's failure: the time
  step sweep, every reading of the initial variance in the papers' text, and Monte Carlo
  tolerances for two step 1 checks that had none. It was committed before step 1b ran.
- `qrh_check` is the experiment crate that runs them on OptionStratLib 0.24's simulation. It
  has its own workspace, it is not a member of this crate's build, and it is not packaged.

From `qrh_check`:

```
cargo run --release -- ../criteria.json      # step 1
cargo run --release -- ../criteria_1b.json   # step 1b
cargo test --release                         # the checks' own arithmetic
```

Each run prints one JSON report and takes seconds on one core. The reports are deterministic
and reproduce byte for byte.

## Results

Both steps fail. The reproduction stops at step 1b, as `criteria_1b.json` says it must, and the
market making comparison of #111 does not start.

**The papers' parameters cannot be reproduced from their text.** At 2212.10164's parameters (a
0.265, b 0.246, c 0.0001, and its z0), the initial variance V0 = a (Z0 - b)^2 + c, with the
Z0 := sum c_i z0_i that both papers define, is 0.253 per year, against the printed 0.18.

- Rounding every printed value within its last decimal keeps V0 within [0.2475, 0.2586].
- OptionStratLib's own V0 equals this to the last digit, so the gap is not in the library.
- The readings the text does not support give 0.610 (no 1/Gamma(alpha) in the kernel), 0.009
  (z0 in the reverse order) and 0.008 (no weights). None is adopted.
- The parameter set is 2107.01611's calibration of 10 September 2019, with lambda 1.717
  printed as 1.7. Lambda does not enter V0.

**The scheme is not converged in the time step at short maturities.**

- At dt 0.0000625 years the at-the-money calls at T 0.03 and 0.05 still move by more than 2
  standard errors when dt is halved, and so do the calls at log-moneyness +0.05 at T 0.03,
  0.05 and 0.07.
- At T 0.03 the at-the-money call falls from 23.73 at dt 0.0005 to 21.23 at dt 0.0000625.
- A noise-free path shows the cause. The fastest factors (gamma up to 542 per year) take V
  from 0.253 to about 0.012 within a fraction of a step, while the spot step uses the
  left-point V_k for the whole step.
- Decaying the factors exactly instead of by the scheme's implicit step gives the same
  numbers. The error comes from the step size at the start, not from the factor update.

What passes:

- OptionStratLib's kernel weights equal the closed form of 2107.01611 Appendix A.
- Put-call parity holds within Monte Carlo error.
- Calls lie within the no-arbitrage bounds, up to 3 standard errors.
- At every maturity, the implied volatility at log-moneyness -0.10 is above the at-the-money
  one by more than 3 standard errors.

Two failures are reported and not judged again:

- The optimal ratio x*_10 of 2107.01611 Figure A.1 (3.92 at alpha 0.51) is not reproduced:
  the closed form L2(0, 0.1) distance here has no minimum below 21.
- The at-the-money implied volatility is about 0.10 to 0.12, against sqrt(V0) = 0.50. This
  follows from the initial transient above.

The full reports are on #111.
