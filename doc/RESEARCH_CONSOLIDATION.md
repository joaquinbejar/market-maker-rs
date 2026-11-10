# Research consolidation (#114)

The decisions below are proposals for review, not adopted decisions. Nothing
here authorizes deployment or live trading. The closing
comment on each closed issue states its limits and governs over any summary here.

Commits are the merge commits on `main`. Detailed tables, hashes and commands stay in each
experiment's `README.md`, its `design.json` and its issue, and are not copied here.

## How to read the evidence

Four kinds of evidence are kept apart. One does not imply another.

| Kind | Meaning |
|---|---|
| Publication | A paper states a result. No row here takes a paper's claim as evidence. |
| Reproduction | A paper's setting was rebuilt here and its result compared. |
| Integration | The component exists in this crate and has its own tests. |
| Realistic capital | Behaviour under real capital, costs and execution. **No study here provides it.** |

Data labels: **SYNTHETIC** is simulated with made-up or paper parameters. **REAL** is the public
Hyperliquid Order Flow Dataset (Zenodo 18184441, BTC perpetual, December 2025). It holds no orders
of our own, so it cannot show a policy's fills or impact.

Common limits, which apply to every learned-policy row unless it says otherwise:

- One simulator family with made-up parameters, normally 5 holdout days per group.
- Intervals are 90 % block bootstrap, with no correction across the many comparisons.
- #102's main execution is in effect immediate fill on 10 s rows (#267). Later studies state the
  rational queue they used instead.
- Rows do not react to orders, except where a study says so. **Impact on learned policies was
  never measured** (see Gaps).
- A policy that never trades shows nothing about trading. Several learned arms did not trade.

## Evidence matrix

### M4: learning protocol

| Issue | Experiment | Paper | Commit | Data | Result | Proposed |
|---|---|---|---|---|---|---|
| #102 classical, pure RL, residual RL | `m4_05` | none named | `26f388ca1` | SYNTHETIC, QR regimes, 10 s | Maskable PPO learned not to trade: equivalent to no-trade, "revise". Residual SAC -27.95 a day pooled, "discard" against GLFT, +41.96 against the Avellaneda-Stoikov it corrects. No quoting arm made money. GLFT's exact quote was refused on 88.5 % of decisions (#264), so the discard against it is a weak comparison. | No-go. Closed with reduced scope by the owner. |
| #103 memory and frequency | `m4_06` | arXiv 2407.21025v1 | `e1191ec18` | SYNTHETIC, QR regimes, 1 s | At the primary fee no memory or frequency effect shown; of 12 learned pairs, 7 equivalent to no-trade, 5 worse, none better. At 0.0002, all 12 worse than no-trade. The fee rule's reading is disclosed. | No-go |
| #104 offline evaluation support | `m4_07` | arXiv 2408.02322v2 | `e143926b2` | SYNTHETIC; REAL for a state coverage diagnostic only | No-go on every target and log. Importance sampling's interval held the truth in 2 of 24 rows. On the capture, offline learning or evaluation is a no-go by construction. A correction after the holdout changed no decision. | No-go. Closed as a documented no-go. |
| #105 constrained RL | `m4_08` | arXiv 1705.10528v1 | `15fb184a6`, stress `4f247a150` | SYNTHETIC, QR regimes, shocks | Penalized PPO's lower cost is the cost of not trading (5 of 5 seeds never filled). CPO better than PPO on cost in 4 of 12 rows, with a net interval including 0. Under the margin-stress holdout, CPO has more non-healthy states than Avellaneda-Stoikov. Not a safety guarantee. | No-go |
| #280 offline training | `m4_11` | none named | `0458d929b` | SYNTHETIC | Stopped at its behaviour cloning sanity rule; holdout never read. | Closed by stop rule |
| #309 offline training, regime in input | `m4_11_regime` | none named | `246cd7aab` | SYNTHETIC | No offline policy better than an interactive one. PPO comparators never traded, undiagnosed. CQL's chosen alpha does not trade on the primary dataset; on `standard_eps10` it trades and loses. New experiment motivated by #280, not its continuation. | No-go |

### M5: components

| Issue | Experiment | Paper | Commit | Data | Result | Proposed |
|---|---|---|---|---|---|---|
| #106 MDQR generator | `m5_01` | arXiv 2501.08822v1 | `049e8fcba` | S0 SYNTHETIC; main study REAL, reused holdout, exploratory | Discarded by its frozen rule: best-queue correlation fails against both QR arms, and the transfer cannot pass because all 13 policies have zero fills on the QR arms' rows. All 9 primary queue-size comparisons pass, against QR arms that are transient or saturated on this epoch. | Discard. Closed with waivers by the owner. |
| #107 HSN | `m5_02` | arXiv 2610.16524v1 | `811b3063a` | SYNTHETIC | Paper's exponential tie reproduced; power-law gain not reproduced. A fixed spread does as well as any memory there. On #103's protocol and Hawkes flow, HSN helps nowhere. | No-go |
| #108 adversarial robustness | `m5_03`, `m5_10` | arXiv 2609.22785v1 | `64aa49036`, `f6c094c20` | SYNTHETIC: perturbed QR regimes; #107 C's Hawkes flow with impact | QR regimes: every quoting arm loses much more under the adversary's flow; PPO barely trades. Hawkes, primary: at impact 0.25 and 1.0 every quoting arm loses heavily under its adversary and under impact alone, below no-trade; at impact 0 the surviving stresses cost little. No limit breach. See below. | Descriptive: no decision rule; not evidence of robustness. Closed. |
| #109 consistent time travel | `time_travel_queues` example | arXiv 2408.02322v2 | `c9ebe4f96` | SYNTHETIC with a truth; REAL without one | On synthetic data, time travel's error is about that of an independent path. On the capture it changes outcomes a lot, and that is not evidence of impact. | No-go; keep chronological replay |
| #110 quote duration (Relaver) | `m4_10` | arXiv 2505.12465v1 | `dd4e73d05` | SYNTHETIC, QR regimes, 100 ms | Duration was not learned: 1-row duration on over 99.7 % of decisions. Its one "helps" compares not trading with losing a little. | No-go on the order book |
| #289 frequent batch auction toy | `m5_05_batch` | arXiv 2505.12465v1 | `8771c2549` | SYNTHETIC, a toy with other rules | Weak duration effect at one latency of three at fee 0. At 0.0002 the duration arm is worse than without duration at every latency, each interval below 0. Avellaneda-Stoikov's mean exceeds both learned arms. Never comparable with the order book results. | No claim beyond the toy |
| #111 quadratic rough Heston | `m5_07` | arXiv 2212.10164v1, 2107.01611 | `4bb36db9b` | SYNTHETIC at SPX/VIX parameters | Do not adopt: control checks not passed and the greedy rule is worse in mean P&L than always quoting within the bounds, where the gate does not bind. Lower standard deviation, not at matched risk. Stage B pending on OptionStratLib#974; stage C not run. The paper's V0 and x* were not reproduced. | Do not adopt. Closed on stage A alone by the owner. |
| #112 regime adaptation and transfer | `m5_08` | arXiv 1901.08938v1 | `5da0d96f6` | SYNTHETIC only | No-go on both comparisons, every simulator and execution group. | No-go |
| #281 adversarial training | `m5_05` | none named | `4eca6e6dd` | SYNTHETIC | No gain shown: A minus B in CVaR5 loss +6.39 [-2.94, 13.98], at a cost of 3.32 a day of net. Holdout resumed once after an external kill, disclosed. | No-go on this evidence. Closed. |
| #282 policies under impact | `m5_04` | none named | `e88bdf6b7` | SYNTHETIC | **Did not complete**: 7617 of 8981 jobs, no freeze, no holdout. | No result |
| #113 multi-agent feasibility | none, API reading | arXiv 2407.21025v1 | none | none | Prototype postponed; nothing run. | Postpone |
| arbitrage-rs#6 deep hedging | outside this repo | see that issue | not reviewed | not reviewed | **Not reviewed in this draft.** | Open |

**#108, read with these limits.** The Hawkes study's frozen rate test removed 14 of 24 points in
every setting before any policy ran, the fixed stress included, so the primary adversary searched
only the 10 surviving points, and the fixed stress is reported not run
under the frozen rule. An analysis of the removed points is secondary and exploratory, and feeds
no decision. The policies are #107 C's frozen ones, and the agent's own fills do not move the mid.
Part 2, #281's impact condition through #282's reactive runner, did not complete: its second and
last attempt stopped at its limit with 142 of 300 seed-days, and no estimand is drawn from them.
**Margin was not measured**: the Hawkes simulator has no capital, and part 1's only figure is a
requirement against a made-up capital of 250 that #102's notional limit makes a shortfall of 0 by
construction.

## Product matrix

"Integration" names what exists in the crate, by its documented modules; this draft did not
re-audit their tests. "Policies tested" is research evidence only.

| | Spot | Dated futures | Perpetuals | Options |
|---|---|---|---|---|
| Valuation | Mid and marks | Contract specifications; basis and calendar edge (`instruments`) | Contract specifications | European pricing, SVI surfaces, upstream surfaces (`options`) |
| Lifecycle | Backtest engine | Several dated futures on one ledger, contract lifecycle, roll plan (`backtest::contracts`, `instruments::roll`) | Funding in contract runs and in the kernel's hedge lane | Settlement in a `margin-rs` ledger (`options::settlement`); inverse options refused |
| Margin | Balance gate | Initial margin before an order (#230) | Initial margin before an order; a `margin-rs` maintenance assessment recorded in #105's stress only | Short option initial margin (#98), single contract, no netting |
| Hedging | None | Basis and roll tools | Hedge lane in a second instrument (#256), attribution (#266) | Delta hedging and `GreeksRiskManager`; delta along the model's Brownian missing (OptionStratLib#974) |
| Simulation | Queue-reactive, Hawkes and the Hyperliquid replay, as abstract rows | Abstract rows only | Hyperliquid BTC perpetual replay (REAL, no own orders) | Stage A of #111 only, under its frozen approximation |
| Policies actually tested | Classical and learned, on abstract synthetic rows with no product semantics | None product-specific | None on real fills. #105's stress assesses one linear position, recorded and never acted on | Greedy QRH against a gate that does not bind (#111); no learned policy |
| Realistic capital | Not tested | Not tested | Not tested | Not tested |

The learned studies use one abstract instrument, so they are **not evidence for any product**.

## Proposed go or no-go

| Component | Proposal | Basis and limit |
|---|---|---|
| Avellaneda-Stoikov, GLFT, fixed spread | Keep as baselines | Comparators only. They lost in the order book simulators at these fees; Avellaneda-Stoikov is positive in #289's toy at fee 0, and a fixed spread scores above the learned arms in #107's part A. Neither is evidence of profitability on a market. |
| Pure RL, residual RL | No-go for adoption | #102. The research code stays as evidence. |
| Memory, HSN, decision frequency | No-go | #103, #107 |
| Learned quote duration | No-go on the order book | #110. #297's venue-time expiry remains a crate feature, with no study of a policy using it. |
| Constrained RL as a safety mechanism | No-go | #105. Hard limits remain the crate's authority, independently of any policy. |
| Offline evaluation and training | No-go | #104, #280, #309. The capture has no actions. |
| Consistent time travel | No-go; chronological replay is the reference | #109 |
| MDQR | Discard on this epoch | #106, exploratory on a reused holdout |
| Regime detector and transfer | No-go as tested | #112, a finding about this detector on 1 s rows |
| Adversarial training | No gain shown | #281, flow stress only |
| Quadratic rough Heston control | Do not adopt | #111, on SPX/VIX parameters and our unit choice. BTC is not evaluable. |
| Multi-agent prototype | Postpone | #113 |

Any adoption later gets its own implementation issue.

## Archived hypotheses

Each is closed for the setting stated, not in general. A new study needs a new design.

- A learned market maker beats a classical quote at these fees and simulators.
- Recurrent or mixture-of-exponentials memory improves a quoting policy here.
- A policy learns when to let a quote rest, from warm-started PPO, on this order book.
- Off-policy evaluation can be trusted from logs of these behaviours.
- A pooled offline policy can clone a regime-dependent behaviour without the regime.
- Consistent time travel approximates a market's reaction at the touch.
- MDQR improves on the QR arms on every required dependency of this capture.
- An EWMA and VPIN detector at lambda 0.9 adapts on 1 s rows without false alarms.
- Training against this flow adversary lowers the held-out tail.

## Gaps

These stay visible, and none is closed by this document.

| Gap | State |
|---|---|
| #282 | Closed by the owner's decision with **impact unmeasured**: the selection stopped at its budget, with no freeze and no holdout. |
| #112 | **No-go with H not run**: the Hyperliquid capture was not evaluated, and no real-data claim is made. S3, the neural simulator, also not run. |
| #281 | **Impact condition not completed**: 13 held-out conditions, not 14. In #108's part 2 the first attempt failed at start with no day read, and the second stopped at its limit with 142 of 300 seed-days; nothing is drawn from them. |
| #108 | **Margin not measured.** Fixed stress not run under the frozen rule; the primary adversary searched only the points the rate test kept. |
| #111 | Closed on stage A alone. Stages B and C not run; Example 3 coded, not run; Example 2's decoupled cell not runnable. |
| #106 | Waived by the owner given the discard: snapshot and restore compatibility untested; determinism shown only by synthetic tests, with no real-data repeat; event size laws not compared against QR; toolchain versions not recorded. The Hawkes arm failed as unstable; the transfer's consistency check is uninformative for maskable PPO. |
| #102 | Closed with reduced scope. Alpha and hedge ablations not run; queue and latency ablations degenerate (#267, #268) and not rerun after those fixes; markout not measured; rejections not in the posted tables; capital is peak notional and inventory, without margin. The conclusions do not extend to binding queues, real latency or real data. |
| #104 | Logs have no explicit terminal or per-decision execution column; censoring not analysed separately; outputs recorded by sha256, not committed. A go would need new logged data with recorded propensities. |
| Margin | margin-rs stress scenarios unused by #111; #105 reports a recorded assessment only. |
| Real execution | No study has our own orders, fills or queue position on a real venue. |
| arbitrage-rs#6 | Deep hedging not reviewed here. |

## Sources

Paper links are the arXiv versions cited in each issue; paper PDFs are not redistributed. The
capture is public under its own license at Zenodo 18184441. No statement here attributes the
proprietary use of a model to any firm.
