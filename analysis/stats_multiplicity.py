#!/usr/bin/env python3
"""Per-condition paired tests with Holm's correction within each research question (Section 5.7).

The computation lives in data.holm_table() so that figures and this listing use
the same family definition; see there.

    .venv/bin/python -m analysis.stats_multiplicity
"""
from analysis.data import contrast, family, final_net, holm_table


def main():
    table = holm_table()
    for fam in ("RQ1", "RQ2"):
        rows = sorted((c for c in table if family(c) == fam), key=lambda c: table[c][0])
        print(f"\n{fam}: {len(rows)} conditions")
        for c in rows:
            x, b = contrast(c)
            d = [p - q for p, q in zip(final_net(c, x), final_net(c, b))]
            p, pa = table[c]
            flag = "  <- significant only unadjusted" if p < 0.05 <= pa else ""
            print(f"  {c:<36} {x} - {b}: {sum(d) / len(d):+.3f}  p={p:.2g}  p_Holm={pa:.2g}{flag}")
        print(f"  significant unadjusted: {sum(table[c][0] < 0.05 for c in rows)}, "
              f"after Holm: {sum(table[c][1] < 0.05 for c in rows)}")


if __name__ == "__main__":
    main()
