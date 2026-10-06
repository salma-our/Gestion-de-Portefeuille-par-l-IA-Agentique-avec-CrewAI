"""Lexicon-based headline sentiment. Coarse but deterministic: the LLM never scores anything."""

import re
from dataclasses import dataclass
from datetime import date

POSITIVE = frozenset(
    """
    beat beats surge surges soar soars rally rallies gain gains rise rises rose jump jumps jumped
    growth grow grows grew strong stronger record upgrade upgrades upgraded outperform outperforms
    bullish profit profits profitable boost boosts boosted win wins won optimistic optimism upbeat
    rebound rebounds recovery recovers expand expands expansion approval approved innovation
    breakthrough success successful improve improves improved exceed exceeds exceeded positive
    top tops higher climb climbs advance advances confidence opportunity
    """.split()
)
NEGATIVE = frozenset(
    """
    miss misses missed fall falls fell drop drops dropped plunge plunges slump slumps decline
    declines declined loss losses weak weaker downgrade downgrades downgraded underperform bearish
    lawsuit sue sues sued probe investigation fine fined fraud recall layoffs cut cuts warn warns
    warning risk risks concern concerns fear fears crash tumble tumbles slide slides sell lower
    worst trouble problem problems delay delays ban bans halt struggle struggles plummet plummets
    slowdown recession tariff tariffs penalty bankruptcy default scandal antitrust
    """.split()
)
NEGATORS = frozenset({"not", "no", "never", "without", "neither", "nor"})
NEGATION_WINDOW = 2
_TOKEN = re.compile(r"[a-z']+")


@dataclass(frozen=True)
class NewsItem:
    """One headline."""

    title: str
    publisher: str
    published: date


def score_headline(title: str) -> float:
    """Score in [-1, 1]: (positive - negative) / (positive + negative); 0.0 if no lexicon match."""
    tokens = _TOKEN.findall(title.lower())
    positive = negative = 0
    for i, token in enumerate(tokens):
        if token not in POSITIVE and token not in NEGATIVE:
            continue
        window = tokens[max(0, i - NEGATION_WINDOW) : i]
        negated = any(w in NEGATORS or w.endswith("n't") for w in window)
        is_positive = (token in POSITIVE) != negated
        positive += is_positive
        negative += not is_positive
    total = positive + negative
    return (positive - negative) / total if total else 0.0


def aggregate(scores: list[float]) -> dict[str, float | int | None]:
    """Mean score and share of positive / negative headlines (None when there is no headline)."""
    n = len(scores)
    if n == 0:
        return {"count": 0, "mean": None, "positive_share": None, "negative_share": None}
    return {
        "count": n,
        "mean": sum(scores) / n,
        "positive_share": sum(s > 0 for s in scores) / n,
        "negative_share": sum(s < 0 for s in scores) / n,
    }
