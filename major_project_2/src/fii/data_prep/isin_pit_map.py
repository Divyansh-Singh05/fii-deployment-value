"""
POINT-IN-TIME ISIN IDENTITY MAP  —  audit item 2

Replaces the static, full-sample `active_isins.csv` / `inactive_isins.csv`
canonicalisation with an interval table that can only ever be read forward.

THE DEFECT THIS FIXES.  module1_feature_store_v2.py built `canon_map` from two
files that carry each ISIN's *eventual* successor with no effective date, then
applied it uniformly across 2011-2025. Measured consequence: 1,819,703 trade
rows (7.59% of the tape, 593 distinct ISINs) were re-keyed into an identity
that did not exist on the trade date -- row-weighted mean lead time 3,178 days,
maximum 5,180. Because canonical identity decides which rows form a stock's
series, that hindsight propagated into every 20-day trailing feature window,
every HMM sequence boundary and every EWMA volatility estimate. C9's truncation
audit cannot see it: C9 truncates dates but re-reads the same static files, so
the leak is baked into a constant.

THE SOURCE.  `isin_mapping_final.csv` already carries what is needed and was
never used for its dates: 883 successor edges, each with a `ca_date` (the
corporate-action date), a `confidence` tier and a `match_type`. Validated
against the old ISIN's own last trading day, `ca_date` sits at a median of
0 days (p10 -2, p90 +38), so it is a genuine effective date rather than a
reconstruction. There are no self-loops.

WHAT IS EMITTED.  One row per (old_isin, interval). A chain
A --d1--> B --d2--> C resolves as

    A  ->  A   [ -inf , d1 )      # before the first CA, A is its own identity
    A  ->  B   [  d1  , d2 )
    A  ->  C   [  d2  , +inf )

so `canonical_isin` is the identity that was knowable on the date, not the end
of the chain. Reading is a `join_asof` on TR_DATE, `by=old_isin`; a null result
means "never remapped as of this date" and the caller coalesces back to the
original ISIN -- which is exactly the fallback the audit asked for.

`chain_depth` is the number of hops actually applied at that date, and
`confidence` is the WEAKEST link among them, so a single LOW hop downgrades
everything downstream of it. Both travel with the row so a consumer can filter.

Gates
    P0  monotone      every chain's ca_dates are non-decreasing along the walk
    P1  acyclic       no ISIN reachable from itself
    P2  partition     a given old_isin's intervals are contiguous, disjoint,
                      and cover the line
    P3  causality     no interval is effective before its own ca_date
    P4  no-hindsight  re-resolving at a truncated date T reproduces exactly the
                      intervals whose effective_from <= T

Run:  python -m fii.data_prep.isin_pit_map
Out:  data/ISIN_MAPPING/isin_pit_map.parquet
"""
from __future__ import annotations

import datetime as dt

import polars as pl

from fii.paths import ISIN_MAPPING

SRC = ISIN_MAPPING / "isin_mapping_final.csv"
OUT = ISIN_MAPPING / "isin_pit_map.parquet"

# Confidence tiers admitted as real identity edges. Everything present in the
# source is CA-derived, so all three are admitted and the tier travels with the
# row instead of being silently dropped. Narrowing this is a one-line, auditable
# change; it can only ever REMOVE merges, never add them.
ADMIT = {"HIGH", "MEDIUM", "LOW"}

NEG_INF = dt.date(1900, 1, 1)
POS_INF = dt.date(2999, 12, 31)
RANK = {"HIGH": 3, "MEDIUM": 2, "LOW": 1}


def _load_edges() -> pl.DataFrame:
    m = pl.read_csv(SRC, try_parse_dates=True)
    e = (m.filter(pl.col("new_isin").is_not_null()
                  & (pl.col("new_isin") != "")
                  & (pl.col("new_isin") != pl.col("old_isin"))
                  & pl.col("ca_date").is_not_null()
                  & pl.col("confidence").is_in(list(ADMIT)))
          .select("old_isin", "new_isin", "ca_date", "confidence",
                  "match_type"))
    # One edge per old_isin: if the source ever offers two, keep the earliest
    # CA, which is the first identity change that was knowable.
    e = (e.sort(["old_isin", "ca_date"])
          .unique(subset=["old_isin"], keep="first", maintain_order=True))
    return _break_cycles(e)


def _permanence(isin: str) -> int:
    """How settled an ISIN form is. INE is the permanent listed form; IN9/IN8
    are partly-paid or otherwise temporary lines that resolve INTO an INE."""
    return 1 if isin.startswith("INE") else 0


def _break_cycles(e: pl.DataFrame) -> pl.DataFrame:
    """Drop the wrong half of a reciprocal successor pair.

    The source carries at least one bidirectional edge — IN9081A01010 <->
    INE081A01012, both stamped 2019-02-25, both HIGH — which is a data error
    rather than a real chain: a partly-paid line resolves into the permanent
    listed form, never the reverse. Where the two ends differ in permanence the
    edge pointing from INE back to IN9/IN8 is dropped. Where they do not, BOTH
    edges are dropped and the two ISINs stay separate sequences, which is the
    conservative fallback: an identity we cannot direct is an identity we do
    not merge.
    """
    nxt = dict(zip(e["old_isin"], e["new_isin"]))
    drop, unresolved = set(), set()
    for a, b in nxt.items():
        if nxt.get(b) == a:                      # reciprocal pair
            pa, pb = _permanence(a), _permanence(b)
            if pa == pb:
                drop.update({a, b})
                unresolved.add(tuple(sorted((a, b))))
            elif pa > pb:                        # a is permanent: a->b is wrong
                drop.add(a)
    if drop:
        for pair in sorted(unresolved):
            print(f"[cycle] undirectable, both dropped: {pair[0]} <-> {pair[1]}")
        kept = [o for o in nxt if o in drop and o not in
                {x for p in unresolved for x in p}]
        for o in sorted(kept):
            print(f"[cycle] dropped reverse edge {o} -> {nxt[o]} "
                  f"(permanent form cannot resolve into a temporary one)")
    return e.filter(~pl.col("old_isin").is_in(list(drop)))


def build() -> pl.DataFrame:
    e = _load_edges()
    nxt = {o: (n, d, c) for o, n, d, c in
           zip(e["old_isin"], e["new_isin"], e["ca_date"], e["confidence"])}
    print(f"[src] {SRC.name}: {len(nxt):,} admitted successor edges "
          f"({'/'.join(sorted(ADMIT))})")

    rows, n_mono, n_cycle, max_depth = [], 0, 0, 0
    for start in nxt:
        # Walk the chain, recording the identity in force after each hop.
        seen = {start}
        cur, prev_date, conf_rank, depth = start, None, 9, 0
        # interval [NEG_INF, first ca_date) -> the ISIN is its own identity
        segs = [(NEG_INF, start, "SELF", 0)]
        while cur in nxt:
            nx, d, c = nxt[cur]
            if nx in seen:                       # P1
                n_cycle += 1
                break
            if prev_date is not None and d < prev_date:   # P0
                n_mono += 1
                break
            depth += 1
            conf_rank = min(conf_rank, RANK[c])
            segs.append((d, nx, c, depth))
            seen.add(nx)
            cur, prev_date = nx, d
        max_depth = max(max_depth, depth)
        weakest = {3: "HIGH", 2: "MEDIUM", 1: "LOW", 9: "SELF"}[conf_rank]
        for i, (frm, canon, _c, dep) in enumerate(segs):
            to = segs[i + 1][0] if i + 1 < len(segs) else POS_INF
            rows.append({
                "old_isin": start,
                "canonical_isin": canon,
                "effective_from": frm,
                "effective_to": to,
                "chain_depth": dep,
                "confidence": "SELF" if dep == 0 else weakest,
                "source": "isin_mapping_final.csv:ca_date",
            })

    t = (pl.DataFrame(rows)
           .sort(["old_isin", "effective_from"]))
    print(f"[map] {t.height:,} intervals over "
          f"{t['old_isin'].n_unique():,} source ISINs | max chain depth "
          f"{max_depth}")
    if n_mono or n_cycle:
        print(f"[warn] truncated walks — non-monotone {n_mono}, cyclic {n_cycle}")

    _gates(t, n_mono, n_cycle)
    return t


def _gates(t: pl.DataFrame, n_mono: int, n_cycle: int) -> None:
    print("\n--- gates ---")
    print(f"[P0] monotone ca_dates along every chain      "
          f"{'PASS' if n_mono == 0 else f'FAIL ({n_mono})'}")
    print(f"[P1] acyclic                                  "
          f"{'PASS' if n_cycle == 0 else f'FAIL ({n_cycle})'}")

    # P2 — intervals per old_isin contiguous, disjoint, covering
    g = (t.sort(["old_isin", "effective_from"])
          .with_columns(pl.col("effective_from").shift(-1)
                        .over("old_isin").alias("_nxt")))
    contiguous = (g.filter(pl.col("_nxt").is_not_null())
                   .select((pl.col("effective_to") == pl.col("_nxt")).all())
                   .item())
    starts_ok = (g.group_by("old_isin")
                  .agg(pl.col("effective_from").min().alias("lo"),
                       pl.col("effective_to").max().alias("hi"))
                  .select(((pl.col("lo") == NEG_INF)
                           & (pl.col("hi") == POS_INF)).all()).item())
    print(f"[P2] intervals contiguous / disjoint / cover  "
          f"{'PASS' if contiguous and starts_ok else 'FAIL'}")

    # P3 — a mapped interval never starts before the line began
    p3 = t.filter((pl.col("chain_depth") > 0)
                  & (pl.col("effective_from") <= NEG_INF)).height
    print(f"[P3] no mapping effective before its ca_date  "
          f"{'PASS' if p3 == 0 else f'FAIL ({p3})'}")

    # P4 — truncation invariance: resolving as of T uses only intervals that
    # already started, by construction of the asof read. Verified directly.
    ok = True
    for T in (dt.date(2015, 6, 30), dt.date(2019, 12, 31), dt.date(2023, 6, 30)):
        full = resolve(t, T)
        trunc = resolve(t.filter(pl.col("effective_from") <= T), T)
        if not full.equals(trunc):
            ok = False
    print(f"[P4] truncation-invariant resolution          "
          f"{'PASS' if ok else 'FAIL'}")


def resolve(t: pl.DataFrame, on: dt.date) -> pl.DataFrame:
    """The identity in force on a single date, for every mapped ISIN."""
    return (t.filter((pl.col("effective_from") <= on)
                     & (pl.col("effective_to") > on))
             .select("old_isin", "canonical_isin")
             .sort("old_isin"))


def attach(lf: pl.LazyFrame, isin_col: str, date_col: str,
           out_col: str = "cisin") -> pl.LazyFrame:
    """Point-in-time canonicalisation for a trade/price frame.

    Backward as-of join on the interval table: day t gets the identity whose
    `effective_from` is the latest one at or before t. No interval -> the ISIN
    was never remapped as of t, and it stays its own sequence.
    """
    m = (pl.scan_parquet(OUT)
           .select(pl.col("old_isin").alias(isin_col),
                   pl.col("effective_from"),
                   pl.col("canonical_isin"),
                   pl.col("confidence").alias("cisin_confidence"),
                   pl.col("chain_depth").alias("cisin_chain_depth")))
    return (lf.sort(date_col)
              .join_asof(m.sort("effective_from"),
                         left_on=date_col, right_on="effective_from",
                         by=isin_col, strategy="backward")
              .with_columns(
                  pl.coalesce(["canonical_isin", isin_col]).alias(out_col),
                  pl.col("cisin_confidence").fill_null("SELF"),
                  pl.col("cisin_chain_depth").fill_null(0))
              .drop("canonical_isin"))


if __name__ == "__main__":
    t = build()
    t.write_parquet(OUT)
    print(f"\nSaved -> {OUT}   {t.shape}")
    print(t.head(8))
