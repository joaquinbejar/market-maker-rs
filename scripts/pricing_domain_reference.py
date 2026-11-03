"""Random reference cases over the whole domain of the Black helper and of
theta, for tests/options_pricing_domain.rs.

- black: F and K log-uniform in (e^-700, e^700), near the money or far from
  it, w log-uniform in [1e-38, 1e5]; the out-of-the-money side in 600-digit
  arithmetic and the in-the-money side by parity, kept when the price is in
  (1e-300, 1e300).
- theta: Black-Scholes-Merton theta on a flat surface at decimal inputs, in
  120-digit arithmetic, kept when |theta| >= 1e-18.

Run: python3 -I scripts/pricing_domain_reference.py (from the repository
root); it prints the two Rust tables.
"""

import math
import random
import sys
from decimal import Decimal as D
from decimal import getcontext
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import black_reference as br  # noqa: E402


def otm(f, k, w):
    f, k, w = D(f), D(k), D(w)
    v = w.sqrt()
    d1 = ((f / k).ln() + w / 2) / v
    d2 = d1 - v
    if f < k:
        return f * br.N(d1) - k * br.N(d2)
    return k * br.N(-d2) - f * br.N(-d1)


def black_rows():
    getcontext().prec = 600
    rng = random.Random(20261103)
    out = []
    while len(out) < 400:
        lf = rng.uniform(-690, 690)
        mode = rng.random()
        if mode < 0.4:
            delta = rng.choice([-1, 1]) * 10 ** rng.uniform(-15, 0.5)
        elif mode < 0.8:
            delta = rng.uniform(-60, 60)
        else:
            delta = rng.uniform(-1400, 1400)
        lk = lf + delta
        if not -700 < lk < 700:
            continue
        w = 10 ** rng.uniform(-38, 5)
        f, k = math.exp(lf), math.exp(lk)
        o = otm(f, k, w)
        if o == 0:
            continue
        call = o if f < k else (D(f) - D(k)) + o
        put = o if f >= k else (D(k) - D(f)) + o
        for kind, val in (("true", call), ("false", put)):
            if D("1e-300") < val < D("1e300"):
                out.append(f"    ({f!r}, {k!r}, {w!r}, {kind}, {val:.20e}),")
    return out


def dec(x, places):
    return format(D(x).quantize(D(1).scaleb(-places)), "f")


def theta_rows():
    getcontext().prec = 120
    rng = random.Random(7)
    out = []
    n = 0
    while n < 300:
        s = dec(10 ** rng.uniform(-3, 9), 6)
        if rng.random() < 0.5:
            k = dec(D(s) * D(1 + rng.choice([-1, 1]) * 10 ** rng.uniform(-12, -1)), 12)
        else:
            k = dec(D(s) * D(10 ** rng.uniform(-4, 4)), 6)
        if D(k) <= 0 or D(s) <= 0:
            continue
        tau = dec(10 ** rng.uniform(-3, 1.3), 6)
        r = rng.choice(["0", "0.05", dec(rng.uniform(-0.02, 0.15), 4)])
        q = rng.choice(["0", r, dec(rng.uniform(-0.02, 0.15), 4)])
        sigma = dec(10 ** rng.uniform(-12, 0.5), 14)
        if D(sigma) <= 0:
            continue
        call = rng.choice([True, False])
        n += 1
        th = br.theta(s, k, tau, r, q, sigma, call)
        if abs(th) >= D("1e-18"):
            out.append(f'    ("{s}", "{k}", "{tau}", "{r}", "{q}", "{sigma}", {str(call).lower()}, {th:.20e}),')
    return out


if __name__ == "__main__":
    print("// black: (F, K, w, call, price)")
    print("\n".join(black_rows()))
    print("// theta: (S, K, tau, r, q, sigma, call, theta)")
    print("\n".join(theta_rows()))
