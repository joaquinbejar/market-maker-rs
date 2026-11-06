"""What a step costs from Python. `python tests_py/step_cost.py`. It measures one machine, once."""

import time

import numpy as np
from conftest import config, walk

from mm_rl_bridge import MaskableMarketMakingEnv

ROWS = 100_000


def main():
    env = MaskableMarketMakingEnv(config(max_steps=ROWS), walk(ROWS))
    env.reset(seed=1)
    quote = np.asarray([1, 1, 1, 0, 0])
    hold = np.asarray([0, 0, 0, 0, 0])
    steps, done = 0, False
    start = time.perf_counter()
    while not done:
        _, _, terminated, truncated, _ = env.step(quote if steps % 3 == 0 else hold)
        env.action_masks()
        done = terminated or truncated
        steps += 1
    elapsed = time.perf_counter() - start
    env.close()
    print(f"gymnasium step and mask: {elapsed / steps * 1e6:.1f} us per step, {steps} steps")


if __name__ == "__main__":
    main()
