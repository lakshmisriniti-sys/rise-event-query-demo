"""A tiny event query language, a matcher, and scoring.

A query is JSON like:
{"steps": [{"type": "Retry", "where": {"service": ["==", "payment"]}},
           {"type": "Timeout"}],
 "within": 5}

It matches a trace if the steps occur IN ORDER (other events may come in between),
each step's conditions hold, and the whole match spans at most `within` events (null = no limit).
"""

OPS = {
    "==": lambda a, b: a == b, "!=": lambda a, b: a != b,
    ">": lambda a, b: a > b, "<": lambda a, b: a < b,
    ">=": lambda a, b: a >= b, "<=": lambda a, b: a <= b,
}


def step_ok(event: dict, step: dict) -> bool:
    if event["type"] != step.get("type"):
        return False
    for attr, cond in (step.get("where") or {}).items():
        if attr not in event:
            return False
        op, value = cond
        try:
            if not OPS[op](event[attr], value):
                return False
        except (KeyError, TypeError):
            return False
    return True


def matches(trace: list, query: dict) -> bool:
    steps = query.get("steps") or []
    if not steps:
        return False
    within = query.get("within")

    def search(step_i, pos, start):
        if step_i == len(steps):
            return True
        for j in range(pos, len(trace)):
            if within is not None and start is not None and j - start + 1 > within:
                return False
            if step_ok(trace[j], steps[step_i]):
                if search(step_i + 1, j + 1, j if start is None else start):
                    return True
        return False

    return search(0, 0, None)


def score(traces: list, query: dict) -> dict:
    tp = fp = fn = 0
    fps, fns = [], []
    for t in traces:
        hit = matches(t["events"], query)
        if hit and t["failure"]:
            tp += 1
        elif hit:
            fp += 1
            fps.append(t)
        elif t["failure"]:
            fn += 1
            fns.append(t)
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return {"precision": p, "recall": r, "f1": f1, "tp": tp, "fp": fp, "fn": fn,
            "false_positives": fps, "false_negatives": fns}


def show(trace: list) -> str:
    """Compact one-line text form of a trace, for the LLM to read."""
    parts = []
    for e in trace:
        attrs = ",".join(f"{k}={v}" for k, v in e.items() if k != "type")
        parts.append(f"{e['type']}({attrs})" if attrs else e["type"])
    return " > ".join(parts)
