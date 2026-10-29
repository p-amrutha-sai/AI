"""Tabular off-policy Q-learning over five recommendation strategies."""
from collections import Counter
from pathlib import Path
import json
import random

STRATEGIES = ("popular", "personalized", "diverse", "fair", "explore")


def encode_state(user, catalog, beliefs, intent="familiar"):
    means = [beliefs.preferences[user.id, c].mean for c in catalog.categories]
    favorite = max(range(len(means)), key=lambda n: means[n])
    uncertain = int(sum(beliefs.preferences[user.id, c].std for c in catalog.categories)
                    / len(means) > 0.17)
    recent = user.recent_categories[-4:]
    repetitive = int(bool(recent) and max(Counter(recent).values()) >= 3)
    satisfaction = 0 if user.satisfaction < 0.4 else (1 if user.satisfaction < 0.7 else 2)
    exposure = [m.exposure for m in catalog.merchants.values()]
    imbalanced = int(max(exposure, default=0) > 2 * (sum(exposure) / max(1, len(exposure)) + 1))
    popularity = [i.popularity for i in catalog.items.values()]
    concentrated = int(max(popularity, default=0) / max(1, sum(popularity)) > 0.15)
    intent_group = int(intent == "introduce_category") + 2 * int(intent == "lower_cost_alternative")
    return ":".join(map(str, (favorite, uncertain, satisfaction, int(user.budget < 35),
                              repetitive, imbalanced, concentrated, intent_group)))


class QPolicy:
    def __init__(self, seed=7, alpha=0.12, gamma=0.90):
        self.q: dict[str, list[float]] = {}
        self.visits: dict[str, list[int]] = {}
        self.rng = random.Random(seed)
        self.alpha, self.gamma = alpha, gamma
        self.metadata = {}

    def choose(self, state: str, epsilon=0.0):
        if self.rng.random() < epsilon:
            return self.rng.randrange(len(STRATEGIES))
        if state not in self.q:
            return STRATEGIES.index("personalized")
        row = self.q[state]
        return max(range(len(row)), key=lambda a: (row[a], -a))

    def update(self, state, action, reward, next_state, terminal=False, elapsed=1):
        row = self.q.setdefault(state, [0.0] * len(STRATEGIES))
        visits = self.visits.setdefault(state, [0] * len(STRATEGIES))
        target = reward + (0.0 if terminal else self.gamma**elapsed * max(self.q.get(next_state, [0.0] * 5)))
        row[action] += self.alpha * (target - row[action])
        visits[action] += 1

    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps({"algorithm": "tabular_q_learning", "alpha": self.alpha,
            "gamma": self.gamma, "strategies": STRATEGIES, "q": self.q,
            "visits": self.visits, "metadata": self.metadata}, indent=2) + "\n")

    @classmethod
    def load(cls, path):
        d = json.loads(Path(path).read_text())
        if tuple(d["strategies"]) != STRATEGIES:
            raise ValueError("Incompatible policy action space")
        obj = cls(alpha=d["alpha"], gamma=d["gamma"])
        obj.q, obj.visits, obj.metadata = d["q"], d["visits"], d["metadata"]
        return obj
