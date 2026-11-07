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
