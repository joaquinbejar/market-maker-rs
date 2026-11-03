"""Reference normalised Black call b(x, s) = e^{x/2} N(x/s + s/2) - e^{-x/2} N(x/s - s/2)
and erfcx, in high-precision decimal arithmetic."""
from decimal import Decimal as D, getcontext
import sys
getcontext().prec = 400
def arctan_inv(n):
    x = D(1)/n; x2 = x*x; s = x; t = x; k = 1
    eps = D(10)**(-(getcontext().prec+5))
    while True:
        t = -t*x2; k += 2; term = t/k
        if abs(term) < eps: return s
        s += term
PI = 4*(4*arctan_inv(5)-arctan_inv(239)); SQRTPI = PI.sqrt(); SQRT2 = D(2).sqrt()
def erfc(x):
    x = D(x)
    if x < 0: return 2 - erfc(-x)
    if x > 6:
        # continued fraction (Lentz), A&S 7.1.14, enough terms for x > 6
        # erfc(x) = exp(-x^2)/sqrt(pi) * 1/(x + 1/2/(x + 1/(x + 3/2/(x + ...))))
        n = 400; f = x
        for k in range(n, 0, -1):
            f = x + (D(k)/2)/f
        return (-x*x).exp()/SQRTPI/f
    s = D(0); n = 0; term = x
    while True:
        add = term/(2*n+1)
        s += add if n % 2 == 0 else -add
        if n > 10 and abs(add) < D(10)**(-380): break
        n += 1; term = term*x*x/n
    return 1 - 2*s/SQRTPI
def N(z): return erfc(-D(z)/SQRT2)/2
def b(x, s):
    x, s = D(x), D(s); h = x/s; t = s/2
    return (x/2).exp()*N(h+t) - (-x/2).exp()*N(h-t)
def erfcx(x):
    x = D(x)
    if x > 6:
        f = x
        for k in range(400, 0, -1):
            f = x + (D(k)/2)/f
        return 1/SQRTPI/f
    return (x*x).exp()*erfc(x)
def n(z): z = D(z); return (-z*z/2).exp()/(2*PI).sqrt()
def black(f, k, w, call):
    # Undiscounted Black at the exact values of the f64 arguments.
    f, k, w = D(f), D(k), D(w); v = w.sqrt()
    d1 = ((f/k).ln() + w/2)/v; d2 = d1 - v
    return f*N(d1) - k*N(d2) if call else k*N(-d2) - f*N(-d1)
def theta(s, k, tau, r, q, sigma, call):
    # theta = -dV/dtau of Black-Scholes-Merton at fixed sigma, per year;
    # strike, rates and sigma are decimal strings (the Decimal inputs).
    s, k, tau, r, q, sigma = (D(x) for x in (s, k, tau, r, q, sigma))
    v = sigma*tau.sqrt(); d1 = ((s/k).ln() + (r - q + sigma*sigma/2)*tau)/v; d2 = d1 - v
    dfr, dfq = (-r*tau).exp(), (-q*tau).exp()
    diffusion = s*dfq*n(d1)*sigma/(2*tau.sqrt())
    if call: return q*s*dfq*N(d1) - r*k*dfr*N(d2) - diffusion
    return r*k*dfr*N(-d2) - q*s*dfq*N(-d1) - diffusion
if __name__ == "__main__":
    # Lines: "b x s", "bscaled x s", "erfcx x", "black F K w call|put" (f64 arguments) or
    # "theta S K tau r q sigma call|put" (decimal strings).
    for line in sys.stdin:
        kind, *args = line.split()
        if kind == "b": print(format(b(float(args[0]), float(args[1])), '.25e'))
        elif kind == "bscaled":
            # b(x, s) e^{h^2/2}, h = x / s: representable for every h.
            x, s = D(float(args[0])), D(float(args[1])); h = x/s
            print(format(b(float(args[0]), float(args[1]))*(h*h/2).exp(), '.20e'))
        elif kind == "black": print(format(black(*(float(a) for a in args[:3]), args[3] == "call"), '.25e'))
        elif kind == "theta": print(format(theta(*args[:6], args[6] == "call"), '.25e'))
        else: print(format(erfcx(float(args[0])), '.25e'))
