"""A frequent batch auction toy (#289): orders collected for 500 ms, then one uniform price.

SYNTHETIC and reported apart from the continuous order book: its execution rules are not the
crate's, so nothing here stands for a result there, nor the reverse.

Prices are whole ticks and sizes whole lots, so the clearing is exact; money is `Decimal`.

The clearing rule, for the orders of one batch:

- The price is the order price that maximises the matched volume `min(D(p), S(p))`, where `D(p)`
  is the lots bid at `p` or higher and `S(p)` the lots offered at `p` or lower. Ties go to the
  smallest imbalance `|D(p) - S(p)|`, then to the price nearest the reference (the last clearing
  price), then to the lower price. No price with a positive volume means no trade, and the
  reference stays.
- Each side fills from its most aggressive price level down. A level that fits in what is left
  fills in full; the level where the volume runs out shares what is left pro rata by size,
  rounded down to whole lots, and the lots the rounding leaves go one at a time by arrival,
  earliest first (then by order id).

The timing rule: a message sent at time `t` with delay `d` arrives at `t + d` and acts in the
batch whose close is strictly after its arrival; a message arriving exactly at a close acts in the
next batch. A cancel that arrives after a close cannot stop that batch's fill.

Limits, stated: one instrument, no queue priority inside a batch, no hidden or iceberg orders,
no self-trade prevention (the agent's own bid and ask may meet each other; they are counted as
fills of both), background orders that expire at their batch's clearing, and a fee charged on
the executed notional at one rate.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Order:
    """One order: `side` +1 buys and -1 sells, at `price` ticks for `lots` lots."""

    order_id: int
    side: int
    price: int
    lots: int
    arrival_ms: int
    owner: str = "background"

    def __post_init__(self):
        if self.side not in (1, -1):
            raise ValueError(f"side is +1 or -1, not {self.side}")
        if self.lots <= 0 or self.price <= 0:
            raise ValueError(f"an order has positive price and lots: {self}")


@dataclass(frozen=True)
class Clearing:
    """The outcome of one batch: its price (None with no trade), the volume and imbalance at
    that price, and the lots each order filled (orders that did not fill are left out)."""

    price: int | None
    volume: int
    imbalance: int
    fills: dict


def _curves(orders, price):
    demand = sum(o.lots for o in orders if o.side == 1 and o.price >= price)
    supply = sum(o.lots for o in orders if o.side == -1 and o.price <= price)
    return demand, supply


def clearing_price(orders, reference):
    """The price, the matched volume and the signed imbalance `D - S` at it, by the rule."""
    best = None
    for price in sorted({o.price for o in orders}):
        demand, supply = _curves(orders, price)
        volume = min(demand, supply)
        if volume <= 0:
            continue
        key = (-volume, abs(demand - supply), abs(price - reference), price)
        if best is None or key < best[0]:
            best = (key, price, volume, demand - supply)
    if best is None:
        return None, 0, 0
    _, price, volume, imbalance = best
    return price, volume, imbalance


def _allocate(levels, volume):
    """Fills for one side: `levels` from the most aggressive price, each a list of orders."""
    fills, left = {}, volume
    for level in levels:
        if left <= 0:
            break
        total = sum(o.lots for o in level)
        if total <= left:
            for o in level:
                fills[o.order_id] = o.lots
            left -= total
            continue
        shares = {o.order_id: left * o.lots // total for o in level}
        rest = left - sum(shares.values())
        for o in sorted(level, key=lambda o: (o.arrival_ms, o.order_id)):
            if rest == 0:
                break
            if shares[o.order_id] < o.lots:
                shares[o.order_id] += 1
                rest -= 1
        if rest:
            raise AssertionError("pro rata left lots unassigned")
        fills.update({k: v for k, v in shares.items() if v > 0})
        left = 0
    return fills


def clear(orders, reference):
    """Clears one batch of orders against the reference price, by the module's rule."""
    ids = [o.order_id for o in orders]
    if len(set(ids)) != len(ids):
        raise ValueError("order ids repeat in a batch")
    price, volume, imbalance = clearing_price(orders, reference)
    if price is None:
        return Clearing(None, 0, 0, {})
    buys = [o for o in orders if o.side == 1 and o.price >= price]
    sells = [o for o in orders if o.side == -1 and o.price <= price]

    def levels(side_orders, aggressive_first):
        prices = sorted({o.price for o in side_orders}, reverse=aggressive_first)
        return [[o for o in side_orders if o.price == p] for p in prices]

    fills = _allocate(levels(buys, True), volume)
    fills.update(_allocate(levels(sells, False), volume))
    return Clearing(price, volume, imbalance, fills)


def batch_of(arrival_ms, batch_ms):
    """The index of the batch a message arriving at `arrival_ms` acts in: the first whose close
    `(k + 1) * batch_ms` is strictly after the arrival."""
    if arrival_ms < 0:
        raise ValueError("arrivals are not negative")
    return arrival_ms // batch_ms
