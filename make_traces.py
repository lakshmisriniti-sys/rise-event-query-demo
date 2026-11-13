"""Generates synthetic system event logs for 3 scenarios, each with a HIDDEN failure pattern.

Each trace is a short sequence of events from one session (logins, requests, retries,
database queries...). Failing traces contain the hidden pattern; normal traces never do,
but they contain look-alike "distractor" events. 3% of labels are flipped to mimic noisy labels.
The hidden patterns are saved separately so the discovered queries can be compared with them.
"""

import json
import random

from query import matches

random.seed(7)
SERVICES = ["search", "payment", "profile", "cart"]


def rand_event():
    kind = random.choices(
        ["Request", "Heartbeat", "CacheMiss", "DBQuery", "Retry", "Timeout", "DiskWarning"],
        weights=[30, 15, 10, 15, 8, 5, 4])[0]
    if kind in ("Request", "Retry"):
        return {"type": kind, "service": random.choice(SERVICES)}
    if kind == "DBQuery":
        return {"type": kind, "latency": random.choice([random.randint(5, 300), random.randint(300, 1200)])}
    if kind == "DiskWarning":
        return {"type": kind, "usage": random.randint(70, 99)}
    return {"type": kind}


SCENARIOS = {
    "S1_disk": {
        "description": "A disk warning above 90% usage is later followed by a failed write.",
        "pattern": {"steps": [{"type": "DiskWarning", "where": {"usage": [">", 90]}}, {"type": "WriteFail"}],
                    "within": None},
        "inject": lambda: [{"type": "DiskWarning", "usage": random.randint(91, 99)}, {"type": "WriteFail"}],
        "distract": lambda: random.choice([[{"type": "DiskWarning", "usage": random.randint(70, 90)}, {"type": "WriteFail"}],
                                           [{"type": "WriteFail"}]]),
    },
    "S2_payment_retries": {
        "description": "Two payment retries followed by a timeout, all within 6 events.",
        "pattern": {"steps": [{"type": "Retry", "where": {"service": ["==", "payment"]}},
                              {"type": "Retry", "where": {"service": ["==", "payment"]}},
                              {"type": "Timeout"}], "within": 6},
        "inject": lambda: [{"type": "Retry", "service": "payment"}, {"type": "Retry", "service": "payment"},
                           {"type": "Timeout"}],
        "distract": lambda: random.choice([[{"type": "Retry", "service": "payment"}, {"type": "Timeout"}],
                                           [{"type": "Retry", "service": "cart"}, {"type": "Retry", "service": "cart"},
                                            {"type": "Timeout"}]]),
    },
    "S3_slow_db": {
        "description": "A cache miss followed within 4 events by a database query slower than 800 ms.",
        "pattern": {"steps": [{"type": "CacheMiss"}, {"type": "DBQuery", "where": {"latency": [">", 800]}}],
                    "within": 4},
        "inject": lambda: [{"type": "CacheMiss"}, {"type": "DBQuery", "latency": random.randint(801, 2000)}],
        "distract": lambda: random.choice([[{"type": "CacheMiss"}, {"type": "DBQuery", "latency": random.randint(20, 700)}],
                                           [{"type": "DBQuery", "latency": random.randint(801, 2000)}]]),
    },
}


def insert_spread(base, items, max_gap):
    """Insert items into base in order, with up to max_gap other events between them."""
    trace = list(base)
    pos = random.randint(0, len(trace))
    for k, item in enumerate(items):
        trace.insert(pos, item)
        pos += 1 + (random.randint(0, max_gap) if k < len(items) - 1 else 0)
        pos = min(pos, len(trace))
    return trace


def make_trace(sc, failure):
    while True:
        base = [{"type": "Login"}] + [rand_event() for _ in range(random.randint(8, 20))]
        if random.random() < 0.6:
            base = insert_spread(base, sc["distract"](), 2)
        if failure:
            trace = insert_spread(base, sc["inject"](), 1)
        else:
            trace = base
        trace.append({"type": "Logout"})
        if matches(trace, sc["pattern"]) == failure:
            return trace


def make_split(sc, n):
    traces = []
    for i in range(n):
        failure = random.random() < 0.25
        events = make_trace(sc, failure)
        label = failure if random.random() > 0.03 else not failure  # 3% label noise
        traces.append({"id": i, "events": events, "failure": label})
    return traces


if __name__ == "__main__":
    data = {}
    for name, sc in SCENARIOS.items():
        data[name] = {"description": sc["description"], "hidden_pattern": sc["pattern"],
                      "train": make_split(sc, 300), "test": make_split(sc, 300)}
        n_fail = sum(t["failure"] for t in data[name]["train"])
        print(f"{name}: 300 train ({n_fail} labelled failures), 300 test")
    with open("data/traces.json", "w", encoding="utf-8") as f:
        json.dump(data, f)
