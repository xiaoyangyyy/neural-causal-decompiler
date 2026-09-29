"""Exact rational outward intervals, including checked transcendental tails.

Binary64/32 parameters are interpreted as exact rationals. This does not certify
additional device inference rounding. No floating solver status is a proof.
"""
from dataclasses import dataclass
from fractions import Fraction as Q
from math import isqrt

BITS = 80
DEN = 1 << BITS

def rational(x):
    if isinstance(x, bool):
        raise ValueError("Boolean is not a rational parameter")
    return Q(x)

def down(x):
    x = rational(x)
    return Q(x.numerator * DEN // x.denominator, DEN)

def up(x):
    return -down(-rational(x))

@dataclass(frozen=True)
class Interval:
    lo: Q
    hi: Q

    def __post_init__(self):
        object.__setattr__(self, 'lo', rational(self.lo))
        object.__setattr__(self, 'hi', rational(self.hi))
        if self.lo > self.hi:
            raise ValueError("Reversed interval")

    @classmethod
    def point(cls, x):
        return cls(rational(x), rational(x))

    def __add__(self, other):
        other = as_interval(other)
        return Interval(down(self.lo + other.lo), up(self.hi + other.hi))
    __radd__ = __add__

    def __neg__(self):
        return Interval(-self.hi, -self.lo)

    def __sub__(self, other):
        return self + -as_interval(other)

    def __rsub__(self, other):
        return as_interval(other) - self

    def __mul__(self, other):
        other = as_interval(other)
        v = [a * b for a in (self.lo, self.hi) for b in (other.lo, other.hi)]
        return Interval(down(min(v)), up(max(v)))
    __rmul__ = __mul__

    def __truediv__(self, other):
        other = as_interval(other)
        if other.lo <= 0 <= other.hi:
            raise ValueError("Denominator contains zero")
        return self * Interval(down(1 / other.hi), up(1 / other.lo))

    def square(self):
        values = (self.lo ** 2, self.hi ** 2)
        return Interval(down(0 if self.lo <= 0 <= self.hi else min(values)), up(max(values)))

    def abs(self):
        return Interval(0 if self.lo <= 0 <= self.hi else min(abs(self.lo), abs(self.hi)),
                        max(abs(self.lo), abs(self.hi)))

    def clip(self, low, high):
        low, high = rational(low), rational(high)
        if low > high:
            raise ValueError("Invalid clipping range")
        return Interval(min(high, max(low, self.lo)), min(high, max(low, self.hi)))

    def sqrt(self):
        if self.lo < 0:
            raise ValueError("Negative square root domain")
        def bound(x):
            n = isqrt(x.numerator * DEN * DEN // x.denominator)
            return Q(n, DEN), Q(n + (n*n*x.denominator != x.numerator*DEN*DEN), DEN)
        return Interval(bound(self.lo)[0], bound(self.hi)[1])

    def to_dict(self):
        return [str(self.lo), str(self.hi)]

    @classmethod
    def from_dict(cls, x):
        if not isinstance(x, list) or len(x) != 2:
            raise ValueError("Expected closed rational interval")
        return cls(*x)


def as_interval(x):
    return x if isinstance(x, Interval) else Interval.point(x)


def exp_point(x):
    """Taylor tail bounded by a geometric series after range reduction."""
    x = rational(x)
    if x < 0:
        positive = exp_point(-x)
        return Interval.point(1) / positive
    squarings = 0
    while x > Q(1, 8):
        x /= 2
        squarings += 1
        if squarings > 24:
            raise ValueError("Exponential domain exceeds proof budget")
    term = total = Q(1)
    n = 0
    while True:
        nxt = term * x / (n + 1)
        tail = nxt / (1 - x / (n + 2))
        if tail <= Q(1, DEN * 16):
            result = Interval(down(total), up(total + tail))
            break
        n += 1
        term = nxt
        total += term
    for _ in range(squarings):
        result = result.square()
    return result


def exponential(x):
    x = as_interval(x)
    return Interval(exp_point(x.lo).lo, exp_point(x.hi).hi)


def tanh_point(x):
    x = rational(x)
    if x < 0:
        return -tanh_point(-x)
    if x >= BITS:
        # exp(2*x) >= 2**BITS since e >= 2.
        return Interval(1 - Q(2, DEN), 1)
    e = exp_point(2 * x)
    return Interval.point(1) - Interval.point(2) / (e + 1)


def hyperbolic_tangent(x):
    x = as_interval(x)
    return Interval(max(Q(-1), tanh_point(x.lo).lo), min(Q(1), tanh_point(x.hi).hi))


def log_point(x):
    x = rational(x)
    if x <= 0:
        raise ValueError("Logarithm requires positive input")
    power = 0
    while x >= 2:
        x /= 2
        power += 1
    while x < 1:
        x *= 2
        power -= 1
    def reduced(v):
        t = (v - 1) / (v + 1)
        total = Q(0)
        term = t
        n = 0
        while True:
            total += 2 * term / (2*n + 1)
            nxt = term * t*t
            tail = 2 * nxt / ((2*n + 3) * (1 - t*t))
            if tail <= Q(1, DEN * 16):
                return Interval(down(total), up(total + tail))
            term = nxt
            n += 1
    return reduced(x) + reduced(Q(2)) * power


def logarithm(x):
    x = as_interval(x)
    return Interval(log_point(x.lo).lo, log_point(x.hi).hi)


def affine(weights, bias, inputs):
    if len(weights) != len(bias) or any(len(row) != len(inputs) for row in weights):
        raise ValueError("Affine dimensions disagree")
    return [sum((as_interval(x) * rational(w) for x, w in zip(inputs, row)),
                Interval.point(b)) for row, b in zip(weights, bias)]


def softmax(inputs):
    values = [exponential(x) for x in inputs]
    total = sum(values, Interval.point(0))
    return [v / total for v in values]


def attention(query, keys, values):
    if not query or len(keys) != len(values) or not keys:
        raise ValueError("Invalid attention dimensions")
    scale = Interval.point(len(query)).sqrt()
    if any(len(k) != len(query) for k in keys) or len({len(v) for v in values}) != 1:
        raise ValueError("Invalid attention rows")
    scores = [sum((as_interval(q)*as_interval(k) for q,k in zip(query,row)), Interval.point(0))/scale for row in keys]
    probs = softmax(scores)
    return [sum((p*as_interval(row[j]) for p,row in zip(probs,values)), Interval.point(0)) for j in range(len(values[0]))]


def trig_point(x, cosine=False):
    x = rational(x)
    if abs(x)>100:
        raise ValueError("Trigonometric domain exceeds proof budget")
    # Taylor's theorem: every derivative of sin/cos has absolute value <= 1.
    total = Q(0)
    degree = 0 if cosine else 1
    term = Q(1) if cosine else x
    from math import factorial
    while True:
        total += term
        remainder = abs(x)**(degree+1)/factorial(degree+1)
        if remainder <= Q(1,DEN*16):
            return Interval(max(Q(-1),down(total-remainder)),min(Q(1),up(total+remainder)))
        term *= -x*x/((degree+1)*(degree+2))
        degree += 2


def trigonometric(x, cosine=False):
    x=as_interval(x)
    middle=(x.lo+x.hi)/2
    center=trig_point(middle,cosine)
    radius=(x.hi-x.lo)/2
    # sin and cos are globally 1-Lipschitz.
    return Interval(max(Q(-1),down(center.lo-radius)),min(Q(1),up(center.hi+radius)))


def protected_division(a,b,epsilon='1/100000000'):
    a,b=as_interval(a),as_interval(b)
    e=rational(epsilon)
    if e<=0:
        raise ValueError("Positive protection threshold required")
    parts=[]
    if b.lo<=-e:
        parts.append(Interval(b.lo,min(b.hi,-e)))
    if b.lo<0 and b.hi>-e:
        parts.append(Interval.point(-e))
    if b.hi>=0 and b.lo<e:
        parts.append(Interval.point(e))
    if b.hi>=e:
        parts.append(Interval(max(b.lo,e),b.hi))
    values=[a/p for p in parts]
    return Interval(min(x.lo for x in values),max(x.hi for x in values))
