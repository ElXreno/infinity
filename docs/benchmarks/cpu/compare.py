"""Compare two reference dumps written by workload.py.

    compare.py embeddings <reference.json> <candidate.json>
    compare.py scores <reference.json> <candidate.json>

Embeddings are compared per text by cosine similarity and max absolute difference. Rerank
scores are compared per query by Spearman rank correlation, overlap of the top 5 documents,
max absolute score difference and max absolute logit difference.
"""

import json
import math
import sys


def rank(values):
    order = sorted(range(len(values)), key=lambda i: -values[i])
    ranks = [0] * len(values)
    for r, i in enumerate(order):
        ranks[i] = r
    return ranks


def spearman(a, b):
    ra, rb = rank(a), rank(b)
    n = len(a)
    d2 = sum((x - y) ** 2 for x, y in zip(ra, rb))
    return 1 - 6 * d2 / (n * (n * n - 1))


def logit(p):
    p = min(max(p, 1e-9), 1 - 1e-9)
    return math.log(p / (1 - p))


def top(values, k=5):
    return set(sorted(range(len(values)), key=lambda i: -values[i])[:k])


def compare_embeddings(reference, candidate):
    for i, (x, y) in enumerate(zip(reference, candidate)):
        dot = sum(p * q for p, q in zip(x, y))
        nx = math.sqrt(sum(p * p for p in x))
        ny = math.sqrt(sum(q * q for q in y))
        max_abs = max(abs(p - q) for p, q in zip(x, y))
        print(f"text {i}: cosine={dot / (nx * ny):.6f} max_abs_diff={max_abs:.5f}")


def compare_scores(reference, candidate):
    rhos, overlaps, max_score, max_logit = [], [], 0.0, 0.0
    for a, b in zip(reference, candidate):
        rhos.append(spearman(a, b))
        overlaps.append(len(top(a) & top(b)) / 5)
        max_score = max(max_score, max(abs(x - y) for x, y in zip(a, b)))
        max_logit = max(max_logit, max(abs(logit(x) - logit(y)) for x, y in zip(a, b)))
    print(
        f"spearman mean={sum(rhos) / len(rhos):.4f} min={min(rhos):.4f}  "
        f"top5 overlap={sum(overlaps) / len(overlaps):.2f}  "
        f"max|Δscore|={max_score:.4f}  max|Δlogit|={max_logit:.3f}"
    )


if __name__ == "__main__":
    kind, reference_path, candidate_path = sys.argv[1:4]
    with open(reference_path) as f:
        reference = json.load(f)
    with open(candidate_path) as f:
        candidate = json.load(f)
    {"embeddings": compare_embeddings, "scores": compare_scores}[kind](reference, candidate)
