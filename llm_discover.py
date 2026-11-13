"""LLM query discovery: Gemini looks at example traces, proposes a query, gets scored by a tool,
sees its mistakes, and refines the query. Only TRAINING traces are ever shown to the LLM."""

import json
import os
import random
import re
import time

from dotenv import load_dotenv

from query import score, show

load_dotenv()
MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
CALL_DELAY = float(os.getenv("CALL_DELAY", "6"))
MAX_ROUNDS = 5

LANGUAGE = """Query language (JSON):
{"steps": [{"type": "<EventType>", "where": {"<attribute>": ["<op>", <value>]}}, ...], "within": <int or null>}
- Steps must occur in this order in the trace; other events may appear in between.
- "where" is optional; ops are ==, !=, >, <, >=, <=. Numbers must be numbers, not strings.
- "within" = the maximum number of events the whole match may span (null = no limit)."""

_client = None


def ask(prompt: str) -> str:
    global _client
    from google import genai
    from google.genai import types
    if _client is None:
        if not os.getenv("GEMINI_API_KEY"):
            raise SystemExit("No GEMINI_API_KEY in .env")
        _client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    config = types.GenerateContentConfig(temperature=0.2, response_mime_type="application/json")
    for attempt in range(5):
        try:
            time.sleep(CALL_DELAY)
            return _client.models.generate_content(model=MODEL, contents=prompt, config=config).text or ""
        except Exception as err:
            code = getattr(err, "code", None)
            if code in (400, 401, 403, 404) or "per day" in str(err):
                raise SystemExit(f"Gemini error {code}: {str(err)[:300]}")
            wait = 30 * (attempt + 1)
            print(f"   API error ({str(err)[:70]}...). Waiting {wait}s.")
            time.sleep(wait)
    raise RuntimeError("Gemini kept failing.")


def parse(text: str) -> dict:
    m = re.search(r"\{.*\}", text, re.DOTALL)
    try:
        q = json.loads(m.group(0)) if m else {}
    except json.JSONDecodeError:
        return {}
    return q.get("query", q)  # accept {"query": {...}} or the query itself


def data_overview(train: list) -> str:
    types, attrs = {}, {}
    for t in train:
        for e in t["events"]:
            types[e["type"]] = types.get(e["type"], 0) + 1
            for k, v in e.items():
                if k != "type":
                    attrs.setdefault(f"{e['type']}.{k}", set()).add(v)
    attr_text = []
    for k, vals in sorted(attrs.items()):
        if all(isinstance(v, (int, float)) for v in vals):
            attr_text.append(f"{k}: number {min(vals)}-{max(vals)}")
        else:
            attr_text.append(f"{k}: one of {sorted(vals)}")
    rng = random.Random(0)
    fails = rng.sample([t for t in train if t["failure"]], 6)
    oks = rng.sample([t for t in train if not t["failure"]], 6)
    return ("Event types (count): " + ", ".join(f"{k} ({v})" for k, v in sorted(types.items())) +
            "\nAttributes: " + "; ".join(attr_text) +
            "\n\nFAILING traces:\n" + "\n".join("- " + show(t["events"]) for t in fails) +
            "\n\nNORMAL traces:\n" + "\n".join("- " + show(t["events"]) for t in oks))


def _pct(traces, test):
    return 100 * sum(1 for t in traces if test(t["events"])) / len(traces)


def data_stats(train: list) -> str:
    """A 'data inspection tool': contrasts failing and normal traces.
    Shows how often each event type, attribute value and ordered pair of event types
    (within 4 events) appears in failing vs normal traces."""
    fail = [t for t in train if t["failure"]]
    ok = [t for t in train if not t["failure"]]
    rows = []
    types = sorted({e["type"] for t in train for e in t["events"]})
    for ty in types:
        rows.append((f"has {ty}", _pct(fail, lambda ev: any(e["type"] == ty for e in ev)),
                     _pct(ok, lambda ev: any(e["type"] == ty for e in ev))))
        attrs = sorted({k for t in train for e in t["events"] if e["type"] == ty for k in e if k != "type"})
        for a in attrs:
            vals = sorted({e[a] for t in train for e in t["events"] if e["type"] == ty})
            if all(isinstance(v, (int, float)) for v in vals):
                cuts = sorted({vals[int(len(vals) * q)] for q in (0.25, 0.5, 0.75, 0.9)})
                for c in cuts:
                    rows.append((f"has {ty} with {a} > {c}",
                                 _pct(fail, lambda ev: any(e["type"] == ty and e[a] > c for e in ev)),
                                 _pct(ok, lambda ev: any(e["type"] == ty and e[a] > c for e in ev))))
            else:
                for v in vals:
                    for n in (1, 2):
                        rows.append((f"has >= {n} {ty} with {a} == {v}",
                                     _pct(fail, lambda ev: sum(e["type"] == ty and e.get(a) == v for e in ev) >= n),
                                     _pct(ok, lambda ev: sum(e["type"] == ty and e.get(a) == v for e in ev) >= n)))
    for a in types:
        for b in types:
            def pair(ev, a=a, b=b):
                return any(ev[i]["type"] == a and any(ev[j]["type"] == b for j in range(i + 1, min(i + 4, len(ev))))
                           for i in range(len(ev)))
            rows.append((f"{a} then {b} within 4 events", _pct(fail, pair), _pct(ok, pair)))
    rows.sort(key=lambda r: r[1] - r[2], reverse=True)
    lines = [f"- {name}: {f:.0f}% of failing vs {o:.0f}% of normal traces" for name, f, o in rows[:25]]
    return ("DATA INSPECTION TOOL OUTPUT (features most over-represented in failing traces):\n" + "\n".join(lines))


def discover(train: list, log=print, with_stats: bool = False) -> list:
    """Returns the history: one entry per round with the query and its TRAINING scores."""
    intro = ("You are analysing system event logs. Some sessions ended in a failure. Find a query that "
             "matches the failing traces and not the normal ones. Labels may contain a little noise, "
             "so 100% is not always possible.\n\n" + LANGUAGE + "\n\n" + data_overview(train) +
             ("\n\n" + data_stats(train) if with_stats else "") +
             '\n\nReply with JSON only: {"reasoning": "<one sentence>", "query": {...}}')
    history = []
    prompt = intro
    for round_no in range(1, MAX_ROUNDS + 1):
        reply = ask(prompt)
        q = parse(reply)
        s = score(train, q) if q.get("steps") else {"precision": 0, "recall": 0, "f1": 0, "tp": 0, "fp": 0, "fn": 0,
                                                     "false_positives": [], "false_negatives": []}
        reasoning = ""
        try:
            reasoning = json.loads(re.search(r"\{.*\}", reply, re.DOTALL).group(0)).get("reasoning", "")
        except Exception:
            pass
        history.append({"round": round_no, "query": q, "reasoning": reasoning,
                        **{k: s[k] for k in ["precision", "recall", "f1", "tp", "fp", "fn"]}})
        log(f"   round {round_no}: train F1 {s['f1']:.2f}  {json.dumps(q)}")
        if s["f1"] >= 0.97:
            break
        fp_text = "\n".join("- " + show(t["events"]) for t in s["false_positives"][:2]) or "(none)"
        fn_text = "\n".join("- " + show(t["events"]) for t in s["false_negatives"][:2]) or "(none)"
        prompt = (intro + "\n\nPrevious attempts and their scores on the training data:\n" +
                  "\n".join(f"- round {h['round']}: {json.dumps(h['query'])} -> precision {h['precision']:.2f}, "
                            f"recall {h['recall']:.2f}, F1 {h['f1']:.2f}" for h in history) +
                  f"\n\nNormal traces your last query wrongly matched:\n{fp_text}"
                  f"\n\nFailing traces your last query missed:\n{fn_text}"
                  "\n\nPropose an improved query.")
    return history
