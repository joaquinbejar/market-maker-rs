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
if __name__ == "__main__":
    for line in sys.stdin:
        kind, *args = line.split()
        if kind == "b": print(format(b(float(args[0]), float(args[1])), '.25e'))
        else: print(format(erfcx(float(args[0])), '.25e'))
