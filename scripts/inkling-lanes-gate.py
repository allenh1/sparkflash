#!/usr/bin/env python3
"""inkling-lanes-gate.py PORT MODE: checks the served fused lanes on 127.0.0.1:PORT.

MODE full (the default pool): four conversations, each answered alone first (the reference), then
all four at once (each answer must equal its reference), then every conversation's second turn
interleaved (each must continue its own lane: cached > 0, and equal its own reply sent alone
afterwards on a cold lane is not required: greedy continued turns may differ from cold ones).
MODE alone: only the references, for another server to match.
MODE tiny (a pool one short sequence fills): two conversations in turn; the second takes the
first's memory, so the first's next turn is cold (cached 0) and still answered.
A third argument `sampled` sends temperature 1.0, top_p 0.95 and seed 12345 instead of greedy: a
seeded sample is a function of the row, the seed and the position, so it too must equal alone.
Prints RESULT lines and one VERDICT line."""
import concurrent.futures as cf, hashlib, json, sys, time, urllib.request

port, mode = int(sys.argv[1]), sys.argv[2]
SAMPLED = len(sys.argv) > 3 and sys.argv[3] == "sampled"
URL = "http://127.0.0.1:%d/v1/chat/completions" % port
TOPICS = [
    "Explain how a hash table handles collisions, with a small Python example.",
    "Summarize the causes of the French Revolution in two paragraphs.",
    "List the prime numbers below 100, one per line.",
    "Describe how photosynthesis works to a ten year old.",
]
FOLLOW = "Now say the same thing in exactly three sentences."


def ask(messages, tokens=160):
    request = {"model": "thinkingmachines/Inkling-2-Small", "temperature": 0,
               "max_tokens": tokens, "messages": messages}
    if SAMPLED:
        request.update({"temperature": 1.0, "top_p": 0.95, "seed": 12345})
    body = json.dumps(request).encode()
    req = urllib.request.Request(URL, data=body, headers={"content-type": "application/json"})
    t0 = time.time()
    try:
        r = json.load(urllib.request.urlopen(req, timeout=1800))
    except urllib.error.HTTPError as error:
        return {"error": "%s %s" % (error.code, error.read()[:300]), "wall": time.time() - t0}
    m = r["choices"][0]["message"]
    usage = r.get("usage", {})
    return {
        "reasoning": m.get("reasoning_content") or "",
        "content": m.get("content") or "",
        "tokens": usage.get("completion_tokens", 0),
        "prompt": usage.get("prompt_tokens", 0),
        "cached": (usage.get("prompt_tokens_details") or {}).get("cached_tokens", -1),
        "wall": time.time() - t0,
    }


def digest(a):
    return hashlib.sha256((a.get("reasoning", "") + "\0" + a.get("content", "")).encode()).hexdigest()[:12]


def show(label, a):
    if "error" in a:
        print("RESULT %s: ERROR %s" % (label, a["error"]), flush=True)
    else:
        print("RESULT %s: prompt %d cached %d, %d tokens %.1f s (%.1f tok/s) %s" % (
            label, a["prompt"], a["cached"], a["tokens"], a["wall"], a["tokens"] / max(a["wall"], 1e-9),
            digest(a)), flush=True)


def turn2(topic, first):
    return [{"role": "user", "content": topic},
            {"role": "assistant", "content": first["content"]},
            {"role": "user", "content": FOLLOW}]


ok = True
if mode == "alone":
    for i, topic in enumerate(TOPICS):
        show("alone c%d" % i, ask([{"role": "user", "content": topic + " (alone)"}]))
    sys.exit(0)
if mode == "full":
    # Each conversation alone, one after another, on whichever lane the host gives it.
    alone = []
    for i, topic in enumerate(TOPICS):
        a = ask([{"role": "user", "content": topic + " (alone)"}])
        show("alone c%d" % i, a)
        alone.append(a)
    # The same four at once: each must answer as it did alone.
    t0 = time.time()
    with cf.ThreadPoolExecutor(4) as pool:
        together = list(pool.map(lambda t: ask([{"role": "user", "content": t + " (alone)"}]), TOPICS))
    wall = time.time() - t0
    for i, a in enumerate(together):
        show("together c%d" % i, a)
        same = "error" not in a and digest(a) == digest(alone[i])
        ok &= same
        print("CHECK together c%d equals alone: %s" % (i, same), flush=True)
    total = sum(a.get("tokens", 0) for a in together)
    print("AGGREGATE four at once: %d tokens in %.1f s = %.1f tok/s (alone %.1f tok/s each on average)" % (
        total, wall, total / wall, sum(a["tokens"] / a["wall"] for a in alone) / 4), flush=True)
    # Four new conversations at once, then all their second turns at once: each continues its own.
    firsts = [None] * 4
    with cf.ThreadPoolExecutor(4) as pool:
        firsts = list(pool.map(lambda t: ask([{"role": "user", "content": t}]), TOPICS))
    for i, a in enumerate(firsts):
        show("turn1 c%d" % i, a)
    with cf.ThreadPoolExecutor(4) as pool:
        seconds = list(pool.map(lambda i: ask(turn2(TOPICS[i], firsts[i])), range(4)))
    for i, a in enumerate(seconds):
        show("turn2 c%d" % i, a)
        hit = "error" not in a and a["cached"] > 0
        ok &= hit
        print("CHECK turn2 c%d continues its lane: %s (cached %s of %s)" % (i, hit, a.get("cached"), a.get("prompt")), flush=True)
else:
    a1 = ask([{"role": "user", "content": TOPICS[0]}])
    show("tiny a1", a1)
    b1 = ask([{"role": "user", "content": TOPICS[1]}])
    show("tiny b1", b1)
    a2 = ask(turn2(TOPICS[0], a1))
    show("tiny a2", a2)
    good = all("error" not in x for x in (a1, b1, a2)) and a2["cached"] == 0
    ok &= good
    print("CHECK a2 is cold after b1 took a's memory, and answered: %s" % good, flush=True)
    # A conversation alone in the pool continues: b's next turn after a2 took b's memory is cold,
    # but a's third turn right after a2 continues a2's lane.
    a3 = ask(turn2(TOPICS[0], a1) + [{"role": "assistant", "content": a2.get("content", "")},
                                       {"role": "user", "content": "Thanks."}], 32)
    show("tiny a3", a3)
    good = "error" not in a3 and a3["cached"] > 0
    ok &= good
    print("CHECK a3 continues a2: %s" % good, flush=True)
print("VERDICT %s %s" % (mode, "PASS" if ok else "FAIL"), flush=True)
