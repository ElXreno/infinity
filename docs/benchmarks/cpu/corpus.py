"""Synthetic texts built from the words of `docs/assets/multilingual_calibration.utf8`.

The benchmark does not ship or need real data: every text is a random sequence of words from
that multilingual calibration set, so the token lengths and the script mix are realistic while
the content is meaningless. `BENCH_CORPUS` overrides the corpus file.
"""

import os
import random
from pathlib import Path

DEFAULT_CORPUS = Path(__file__).resolve().parents[2] / "assets" / "multilingual_calibration.utf8"


def load_words(path=None):
    path = Path(path or os.environ.get("BENCH_CORPUS") or DEFAULT_CORPUS)
    words = []
    with path.open(encoding="utf-8") as f:
        next(f)
        for line in f:
            tokens = line.split()
            if len(tokens) >= 3:
                words.extend(tokens)
    return words


WORDS = load_words()


def text(rng, lo, hi):
    return " ".join(rng.choice(WORDS) for _ in range(rng.randint(lo, hi)))


def title(rng):
    return text(rng, 3, 12)


REFERENCE_TEXTS = [text(random.Random(i), 4, 10) for i in range(5)]
REFERENCE_QUERY = text(random.Random(99), 2, 4)
