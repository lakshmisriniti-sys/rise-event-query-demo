"""Baseline without AI: brute-force search over simple sequence queries.

Tries every ordered sequence of 1 or 2 event TYPES (no attribute conditions)
with window sizes None, 3, 5, 10, and keeps the one with the best training F1.
"""

import itertools

from query import score


def search(train: list) -> dict:
    types = sorted({e["type"] for t in train for e in t["events"]} - {"Login", "Logout"})
    candidates = [[a] for a in types] + [list(p) for p in itertools.product(types, repeat=2)]
    best, best_f1 = None, -1.0
    for seq in candidates:
        for within in [None, 3, 5, 10]:
            if len(seq) == 1 and within is not None:
                continue
            q = {"steps": [{"type": t} for t in seq], "within": within}
            f1 = score(train, q)["f1"]
            if f1 > best_f1:
                best, best_f1 = q, f1
    return best
