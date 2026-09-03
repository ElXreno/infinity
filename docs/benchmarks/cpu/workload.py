"""Load generator for a running infinity server. Standard library only.

    workload.py [--port 17997] [--label run] [--out .] mixed|titles|embed-reference|score-reference

`mixed` is a general embedding + reranking mix, `titles` reranks 200 short texts per request
(the shape of a search-and-rerank workload). The `*-reference` workloads only dump embeddings or
rerank scores to `<out>/embeddings-<label>.json` / `<out>/scores-<label>.json` for `compare.py`,
without measuring throughput. Texts come from `corpus.py`.
"""

import argparse
import json
import random
import statistics
import threading
import time
import urllib.request

from corpus import REFERENCE_QUERY, REFERENCE_TEXTS, text, title

BASE = "http://127.0.0.1:17997"
LABEL = "run"
OUT = "."


def request(path, payload):
    body = json.dumps(payload).encode()
    req = urllib.request.Request(
        BASE + path, data=body, headers={"Content-Type": "application/json"}, method="POST"
    )
    with urllib.request.urlopen(req, timeout=600) as resp:
        assert resp.status == 200, resp.status
        return json.load(resp)


def timed(path, payload):
    t0 = time.perf_counter()
    request(path, payload)
    return time.perf_counter() - t0


def run(name, payloads, units_per_request, concurrency):
    latencies = []
    lock = threading.Lock()
    queue = list(payloads)

    def worker():
        while True:
            with lock:
                if not queue:
                    return
                path, payload = queue.pop(0)
            dt = timed(path, payload)
            with lock:
                latencies.append(dt)

    t0 = time.perf_counter()
    threads = [threading.Thread(target=worker) for _ in range(concurrency)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.perf_counter() - t0
    latencies.sort()
    p50 = statistics.median(latencies)
    p95 = latencies[int(len(latencies) * 0.95) - 1]
    units = len(payloads) * units_per_request
    print(
        f"{LABEL:14s} {name:22s} c={concurrency} req={len(payloads):4d} "
        f"p50={p50 * 1000:7.1f}ms p95={p95 * 1000:7.1f}ms wall={wall:6.1f}s "
        f"units/s={units / wall:7.1f}",
        flush=True,
    )


def embed_payloads(seed, n, batch, lo, hi):
    rng = random.Random(seed)
    return [
        ("/embeddings", {"model": "embed", "input": [text(rng, lo, hi) for _ in range(batch)]})
        for _ in range(n)
    ]


def rerank_payloads(seed, n, docs, lo, hi):
    rng = random.Random(seed)
    return [
        (
            "/rerank",
            {
                "model": "rerank",
                "query": text(rng, 3, 8),
                "documents": [text(rng, lo, hi) for _ in range(docs)],
                "return_documents": False,
            },
        )
        for _ in range(n)
    ]


def title_payloads(seed, n, docs):
    rng = random.Random(seed)
    return [
        (
            "/rerank",
            {
                "model": "rerank",
                "query": title(rng),
                "documents": [title(rng) for _ in range(docs)],
                "return_documents": False,
            },
        )
        for _ in range(n)
    ]


def scores(query, documents):
    results = request(
        "/rerank",
        {"model": "rerank", "query": query, "documents": documents, "return_documents": False},
    )["results"]
    by_index = {r["index"]: r["relevance_score"] for r in results}
    return [by_index[i] for i in range(len(documents))]


def dump_embeddings():
    data = request("/embeddings", {"model": "embed", "input": REFERENCE_TEXTS})["data"]
    with open(f"{OUT}/embeddings-{LABEL}.json", "w") as f:
        json.dump([d["embedding"] for d in data], f)
    print(f"{LABEL}: embeddings of {len(REFERENCE_TEXTS)} texts saved", flush=True)


def dump_scores():
    reference = scores(REFERENCE_QUERY, REFERENCE_TEXTS[:3])
    print(f"{LABEL:14s} reference scores {[round(s, 4) for s in reference]}", flush=True)
    rng = random.Random(2024)
    pairs = [scores(title(rng), [title(rng) for _ in range(20)]) for _ in range(10)]
    with open(f"{OUT}/scores-{LABEL}.json", "w") as f:
        json.dump(pairs, f)
    print(f"{LABEL}: scores of 10x20 pairs saved", flush=True)


def mixed():
    dump_embeddings()
    for path, payload in embed_payloads(1, 10, 8, 5, 60) + rerank_payloads(1, 5, 4, 40, 200):
        request(path, payload)
    run("embed short b8", embed_payloads(42, 150, 8, 3, 25), 8, 1)
    run("embed mixed b8", embed_payloads(43, 150, 8, 5, 120), 8, 1)
    run("embed mixed b8", embed_payloads(44, 200, 8, 5, 120), 8, 4)
    run("embed single", embed_payloads(45, 200, 1, 5, 60), 1, 1)
    run("rerank 8 docs", rerank_payloads(46, 40, 8, 40, 300), 8, 1)
    run("rerank 8 docs", rerank_payloads(47, 60, 8, 40, 300), 8, 4)
    run("rerank 4 long docs", rerank_payloads(48, 30, 4, 300, 900), 4, 1)


def titles():
    dump_scores()
    for path, payload in title_payloads(1, 3, 50):
        request(path, payload)
    run("rerank 200 titles", title_payloads(42, 12, 200), 200, 1)
    run("rerank 200 titles", title_payloads(43, 16, 200), 200, 2)
    run("rerank 200 titles", title_payloads(44, 16, 200), 200, 4)


WORKLOADS = {
    "mixed": mixed,
    "titles": titles,
    "embed-reference": dump_embeddings,
    "score-reference": dump_scores,
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=17997)
    parser.add_argument("--label", default="run")
    parser.add_argument("--out", default=".")
    parser.add_argument("workload", choices=sorted(WORKLOADS))
    args = parser.parse_args()
    BASE = f"http://127.0.0.1:{args.port}"
    LABEL = args.label
    OUT = args.out
    WORKLOADS[args.workload]()
