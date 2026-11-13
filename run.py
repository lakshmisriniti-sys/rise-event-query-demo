"""Runs the experiment on all 3 scenarios and saves results/results.json.

Methods, all evaluated on held-out TEST traces the LLM never saw:
  search      brute-force over 1-2 event types, no attribute conditions (no AI)
  llm_first   the LLM's first proposal, with no feedback
  llm_loop    the best query (by training F1) after up to 5 rounds of tool feedback
  llm_stats   like llm_loop, but the LLM also gets a data inspection tool's output
              (features over-represented in failing traces)
  hidden      the true pattern used to generate the data (the upper bound)

Finished parts are saved, so re-running only does what is missing.
"""

import json
import os

import baseline
import llm_discover
from query import score

OUT = "results/results.json"


def evaluate(test_traces, q):
    s = score(test_traces, q) if q.get("steps") else {"precision": 0, "recall": 0, "f1": 0}
    return {"query": q, "precision": s["precision"], "recall": s["recall"], "f1": s["f1"]}


def save(results):
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)


def main():
    os.makedirs("results", exist_ok=True)
    with open("data/traces.json", encoding="utf-8") as f:
        data = json.load(f)
    results = {}
    if os.path.exists(OUT):
        with open(OUT, encoding="utf-8") as f:
            results = json.load(f)

    print(f"Model: {llm_discover.MODEL}")
    for name, d in data.items():
        r = results.setdefault(name, {"description": d["description"], "test": {}})
        print(f"{name}: {d['description']}")
        if "history" not in r:
            print("  LLM + feedback:")
            r["history"] = llm_discover.discover(d["train"])
            save(results)
        if "history_stats" not in r:
            print("  LLM + feedback + data inspection tool:")
            r["history_stats"] = llm_discover.discover(d["train"], with_stats=True)
            save(results)
        queries = {
            "search": baseline.search(d["train"]),
            "llm_first": r["history"][0]["query"],
            "llm_loop": max(r["history"], key=lambda h: h["f1"])["query"],
            "llm_stats": max(r["history_stats"], key=lambda h: h["f1"])["query"],
            "hidden": d["hidden_pattern"],
        }
        r["test"] = {m: evaluate(d["test"], q) for m, q in queries.items()}
        for m, t in r["test"].items():
            print(f"   test F1 {m:9s} {t['f1']:.2f}  {json.dumps(t['query'])}")
        save(results)
    print("Saved results/results.json. Now run: python make_chart.py")


if __name__ == "__main__":
    main()
