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

## Stage A: the control under Approximation 3.1

Synthetic: simulated paths at the paper's SPX/VIX parameters, calibrated to nothing and never
standing for BTC. The design is `bridges/python/experiments/m5_07/design.json` (version 4,
sha256 `48c8ea5a...42fa`), frozen before any run on the paper's examples. Results are in
`bridges/python/experiments/m5_07/results/stage_a.json`.

**Decision: do not adopt.** The rule rests on example 2's primary comparison at x1 with both its
checks passing. That comparison is "worse", and both checks fail, so it carries the label
"control not validated". The scope is the control with the paper's delta on SPX/VIX parameters;
sigma 0.2267 per second is our unit choice, since the paper gives no conversion.

| Check (our thresholds) | Example 1 | Example 2 |
|---|---|---|
| 1, shape of v_hat against v, at most 5 % | inconclusive: error 1.09 %, but doubling the bound moves v by 0.66 %, over the 0.5 % limit | fail: max relative error 39.99 |
| 2, paired P&L within 5 % of exact, per kappa | fail: 4 of 7 pass, kappa 0.1, 0.3, 1 fail | fail: 4 of 7 pass, kappa 0.03, 0.1, 0.3 fail |

Check 1 fixes C and so validates shape only. Example 2's 39.99 is at r = 3.912, where v crosses zero (v = -0.14 there, range -60.5 to 9.9 on
tested cells), so the relative error is a limit of the rule at that cell. It is not the whole
story: in a diagnostic after the run, outside the rule and without any path, the same maximum is
4.85 on cells with |v| at least 1 % of max|v|, and 1.14 at 10 %. The shape does not agree away
from the crossing either.

Primary comparison, QRH greedy minus the `GreeksRiskManager` gate, 5000 paths, paired 90 % interval.
Both rows carry the label "control not validated", since no example passed checks 1 and 2:

| | Mean QRH | Mean gate | Interval | Std QRH | Std gate | Verdict |
|---|---|---|---|---|---|---|
| Example 1, x1 | 47.95 | 49.33 | [-1.75, -0.99] | 11.56 | 23.22 | worse |
| Example 2, x1 | 6.24 | 9.00 | [-3.10, -2.41] | 1.73 | 15.14 | worse |

- The rule compares mean P&L only. QRH runs at much lower standard deviation in both examples;
  the two arms are not at matched risk, and a mean-risk comparison was not the frozen rule.
- Example 1 at x1 is QRH against "always quote within Q", not against a delta gate, and cannot
  support adoption. Its x0.5 is the only place the gate binds there.
- Example 2's gate does not bind at x1 either: its x1 and x2 tables are identical, since limit R
  equals the grid's bound. The comparison on which the decision rests is therefore QRH against
  "always quote within R", not against a binding delta gate.
- The x0.5 and x2 gate limits and the 252 x 6.5 h year are all "worse" too, outside the rule.
- Scaled counts as hold (share 0 at x1 and x2; 4.6 % and 10.6 % at x0.5). That can only reduce
  the gate's trading, which favours QRH if those orders would have earned.
- The circuit breaker of `GreeksRiskManager` is not modelled: it is reset at each state.

Secondary cells, reported only. Example 1 against (greedy, joint): decoupled greedy has interval
[-0.13, 0.04]; the decoupled gate equals the joint gate at x1, neither binding. **Example 2's
decoupled cell is not runnable**: its delta is not a function of the net-risk state, as declared
before the run. Example 3 is coded but **was not run**, since checks 1 and 2 did not pass; it
would not be a reproduction of the paper's figure 4.10 (four assets, frozen delta, a 30-per-asset
bound of ours).

Stage B is pending on joaquinbejar/OptionStratLib#974: the delta along the Brownian does not
exist there. Stage C was not run: it is gated on stage A passing. margin-rs stress scenarios and
arbitrage-rs#6 are not covered.

Disclosures:

- The work was handed over before the run. Design version 3 was objected to because check 2 had no
  sample or interval and the decision no rule; version 4 fixed these before any run on the
  paper's examples, and also fixed the secondary cells.
- The attempt log records two starts. The first had no gate tables, by an operator error, and was
  stopped within seconds with no result written or read; the second used the one recovery allowed.
- The paper's greedy net-risk formula omits a factor and a coefficient; the derived form
  `2 phi D r delta + D delta^2 m <= D^j/2` is used, from v_hat differences.
  This explicit algebraic form was not written in design version 4, which says "the greedy
  quoting rule from v_hat"; it was checked as covered by that text only after the run.
- The backtest is Approximation 3.1 as frozen: one Brownian and Poisson fills, not the paper's
  QRH-driven simulation. kappa_j = kappa/2 across example 1's grid, as in the paper.

Reproduce: export the gate tables with `examples/qrh_gate_table`, then
`python experiments/m5_07/run_stage_a.py <out> <tables>` from `bridges/python`. The runner refuses
a second attempt without a recovery reason. Gate table hashes are in `results/gate_tables.sha256`.

