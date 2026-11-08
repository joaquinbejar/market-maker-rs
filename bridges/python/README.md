# mm-rl-bridge

A Python bridge for an external RL trainer over the `market-maker-rs` RL spaces. It is a
nested crate outside the root build, so neither Python nor the trainer reach the library.
It is not published.

## What it holds

- `src/worker.rs`: one environment on its own thread. The thread owns the dataset and borrows
  the kernel inside its own function, with requests and replies over `std` channels. `close`
  and drop join the thread and free the dataset, and nothing continues after an error or a
  panic.
- `src/python.rs`: the extension module, behind the feature `python`.
- `python/mm_rl_bridge/envs.py`: thin `gymnasium.Env` classes. `MaskableMarketMakingEnv` is
  discrete, for `MaskablePPO`, `BoxMarketMakingEnv` is continuous, for `SAC`, and
  `ResidualMarketMakingEnv` is a bounded correction to a classical quote, for `SAC`.
- `python/mm_rl_bridge/expiry.py`: `ExpiringQuote`, a quote that expires by itself.
- `python/mm_rl_bridge/training.py`: training, checkpoints with their metadata, inference
  only loading and a protocol of train, select and one holdout evaluation.

The trainer is pinned: `stable-baselines3==2.9.2` and `sb3-contrib==2.9.2`, on the CPU with
one thread per run. `MaskablePPO` here has no Box actions and no recurrence, and a Box
cannot state a mask, so the continuous encoding uses the reject policy.

With `"hedge_actions": true` in the configuration (off by default, which keeps a
configuration's specification and fingerprint), the action space has hedges in the quoted
instrument, as the library's `HedgeActions::MarketableAtTouch` states: a marketable order at the
observed touch on the side that reduces the delivered position, of a size from the grid of
sizes and never larger than that position; a flat position allows none. The maskable encoding
adds a sixth value, 0 for no hedge and `i + 1` for the size `i`, masked by `hedge_mask`; the
continuous one adds a sixth value, above zero for a hedge of the size at bin `(2 v - 1)`, and a
hedge outside the mask is a charged hold. Either way a hedge replaces the resting orders, so the
other values are not read. A residual environment takes no hedge actions and refuses the flag. A
checkpoint loads only into an environment with the same flag.

A residual's baseline is `avellaneda_stoikov`, `glft`, or `scheduled`: `{"kind": "scheduled",
"points": [...], "changes": [[row, index], ...]}` quotes with `points[0]` from the start and,
from each change on, with the point it names. A change at row `r` applies from the first decision
whose observation holds row `r` or a later one, so it never acts on a row the agent has not
observed. The points are Avellaneda-Stoikov or GLFT configurations. Change rows must increase
strictly, and each change must name a point other than the one in force.

With `"cancel_action": true` (off by default, with the same guarantee), the action space has
the library's cancel-only action, `CancelActions::CancelOnly`, of action space version 4: every
resting or pending order is cancelled and nothing is sent. The maskable encoding gives the first
value a third choice, 2, always allowed; the continuous one adds a last value, above zero for a
cancel, read after a hedge and before a quote. The bridge drives version 4 only with the flag,
so a checkpoint of one space is refused by the other, and a residual environment refuses it.

With `"margin": {"base_deposit": "<lots>"}` (absent by default, which keeps a
configuration's kernel fingerprint), the kernel runs its balance gate, the library's
`KernelMargin`, after the hard limits: a buy reserves its notional at its limit price plus its
largest fee from the initial capital, and a sell its quantity from the base deposit, so an
order beyond the balance is cut to what it pays for or refused. It is the spot, fully
collateralised rule (one unit of balance per unit of notional), not a leveraged margin rule.
The spaces do not change, so a specification does not either; the kernel fingerprint in a
checkpoint's metadata tells the two apart. Every step's `info["gates"]` counts the orders the
gate cut (`reduced`) or refused (`rejected`, a cut off the lot grid included) by the check that
decided: `limits`, `combined` (a hedged kernel's) and `margin`. A hold submits nothing, so the
gate decides nothing on it.

`ChosenDuration(env, choices=(1, 5, 10, 30))` wraps a Box environment so that the policy
chooses, with each action, how many rows its quote rests before it decides again: one more value
in `[-1, 1]` at the end of the action, binned onto `choices`. Between decisions the resting
orders stay and nothing is sent; the reward is the sum over the rows, and `info` holds `rows`
and the chosen `duration`. `DecisionInterval(env, every)` is the same with a fixed interval.
A cancel takes effect after the cancel latency, so fills in flight stand.

`ExpiringQuote(env, rows=d)` wraps such an environment so that a quote expires by itself: after
a quote the environment applied it holds for `d - 1` decisions and cancels at the decision `d`
rows after the quote, with no decision of the policy, which decides again on the next row. In
venue time the quote rests `d` rows plus the cancel latency less the submission latency, so
exactly `d` rows when the two are equal: with no latency it can fill on rows `t + 1` to `t + d`
and not after. A fill before the cancel arrives stands, and a cancel that reaches an order
already filled is rejected. A cancel scheduled in the kernel, exact in venue time whatever the
latencies, is not modelled (#297).

Money crosses the boundary only as decimal strings and rows as integer ticks and lots. The
`f32` reward is the kernel's derived value, and `info` holds the exact reward and net result,
together with the proposed, decoded and applied action.

## The contract with the trainer

The specification comes from `market_maker_rs::rl`. Passing `expected_spec` makes an unknown
one fail before any step. Under `DummyVecEnv`, `done` is `terminated or truncated`, the
`TimeLimit.truncated` flag is set only on a truncation, and `terminal_observation` is the last
observation of the episode, which the tests check.

## Running

The extension is built against Python's stable ABI for 3.11, so it loads on 3.11 and 3.12;
both are tested.

```sh
uv venv --python 3.12 .venv && source .venv/bin/activate
uv pip install "maturin>=1.16,<2"
maturin develop --uv --extras test
cargo test
python -m pytest
```

## What this does not claim

Fixtures are synthetic, and a smoke run says the contract holds, not that a policy earns
anything. The environment's determinism is asserted by the tests. Training with one seed
twice is compared and printed for one platform and version set, not claimed across them, and
resuming a training run exactly is not claimed. A checkpoint's `commit` is declared by the
caller and not checked. Step cost is measured by `examples/step_cost.rs` and
`tests_py/step_cost.py` on one machine, and no batching is promised.
