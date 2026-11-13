"""Offline test with a fake model (no API key). Writes results/test_run/results.json."""
import json
import os

import llm_discover
import run

calls = {"n": 0}


def fake_ask(prompt):
    calls["n"] += 1
    if "Previous attempts" not in prompt:  # first try: a too-simple query
        return '{"reasoning": "timeouts look bad", "query": {"steps": [{"type": "Timeout"}], "within": null}}'
    return ('{"reasoning": "refined", "query": {"steps": [{"type": "CacheMiss"}, '
            '{"type": "DBQuery", "where": {"latency": [">", 800]}}], "within": 4}}')


llm_discover.ask = fake_ask
os.makedirs("results/test_run", exist_ok=True)
run.OUT = "results/test_run/results.json"
if os.path.exists(run.OUT):
    os.remove(run.OUT)
run.main()
print("fake calls:", calls["n"])
print(json.dumps(json.load(open(run.OUT))["S3_slow_db"]["test"]["llm_loop"]["f1"]))
