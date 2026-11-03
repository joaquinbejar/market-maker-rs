"""Reference values for the erfc and normal distribution tests in
src/options/special.rs, and the high-precision Black-Scholes-Merton prices
of the reference table in tests/options_pricing.rs.

erfc(x) = 1 - erf(x), with erf from its Maclaurin series (Abramowitz and
Stegun, 7.1.5), evaluated in 1200-digit decimal arithmetic at the exact
value of each f64 argument; pi from Machin's formula. Enough digits for
|x| <= 27, where the largest series term is about 1e316.

Run: python3 -I scripts/erfc_reference.py
"""

from decimal import Decimal as D
from decimal import getcontext

getcontext().prec = 1200


def arctan_inv(n: int) -> D:
    x = D(1) / n
    x2, s, t, k = x * x, x, x, 1
    eps = D(10) ** (-(getcontext().prec + 5))
    while True:
        t, k = -t * x2, k + 2
        term = t / k
        if abs(term) < eps:
            return s
        s += term


PI = 4 * (4 * arctan_inv(5) - arctan_inv(239))
SQRT_PI = PI.sqrt()


def erf(x: D) -> D:
    s, n, term = D(0), 0, x  # term = x^(2n+1) / n!
    eps = D(10) ** (-1150)
    while True:
        add = term / (2 * n + 1)
        s += add if n % 2 == 0 else -add
        if n > 10 and abs(add) < eps:
            return 2 * s / SQRT_PI
        n += 1
        term = term * x * x / n


def erfc(x: D) -> D:
    return 1 - erf(x)


def bsm(spot, strike, tau, rate, carry, sigma, call) -> D:
    s, k, t, r, q, v = (D(repr(x)) for x in (spot, strike, tau, rate, carry, sigma))
    n = lambda x: erfc(-x / D(2).sqrt()) / 2  # noqa: E731
    fwd = s * ((r - q) * t).exp()
    vol = v * t.sqrt()
    d1 = ((fwd / k).ln() + vol * vol / 2) / vol
    d2 = d1 - vol
    df = (-r * t).exp()
    return df * (fwd * n(d1) - k * n(d2)) if call else df * (k * n(-d2) - fwd * n(-d1))


def main() -> None:
    for x in (0.0, 0.1, 0.5, 0.84375, 1.0, 1.25, 2.0, 1 / 0.35, 3.0, 5.0, 8.0, 10.0, 26.0, 27.0, -1.0, -0.5):
        print(f"erfc({x!r}) = {erfc(D(x)):.20e}")
    sqrt2 = D(2).sqrt()
    for x in (-8, -38):
        print(f"N({x}) = {erfc(-D(x) / sqrt2) / 2:.20e}")
    print(f"n(0) = {1 / (2 * PI).sqrt():.20e}")
    print(f"n(1) = {(-D(1) / 2).exp() / (2 * PI).sqrt():.20e}")
    getcontext().prec = 60
    for args in (
        (60.0, 65.0, 0.25, 0.08, 0.0, 0.3, True),
        (100.0, 95.0, 0.5, 0.10, 0.05, 0.2, False),
        (75.0, 70.0, 0.5, 0.10, 0.05, 0.35, False),
        (42.0, 40.0, 0.5, 0.10, 0.0, 0.2, True),
        (42.0, 40.0, 0.5, 0.10, 0.0, 0.2, False),
    ):
        print(f"bsm{args} = {bsm(*args):.18f}")


if __name__ == "__main__":
    main()
