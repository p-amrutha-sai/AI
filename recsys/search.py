"""Typed graph BFS followed by bounded-pool, exact soft-CSP slate search."""
from collections import defaultdict, deque, Counter
from dataclasses import dataclass, field
from .models import Catalog, Item, User


class RecommendationGraph:
    def __init__(self, catalog: Catalog):
        self.edges: dict[str, set[str]] = defaultdict(set)
        def link(a, b):
            self.edges[a].add(b)
            self.edges[b].add(a)
        for item in catalog.items.values():
            link("item:" + item.id, "category:" + item.category)
            link("item:" + item.id, "merchant:" + item.merchant)
        for a, b in catalog.related_categories:
            link("category:" + a, "category:" + b)
        for user in catalog.users.values():
            link("user:" + user.id, "category:" + user.declared_category)
        # Retain the last 100 interactions for compact demonstration graphs.
        for index, event in enumerate(catalog.interactions[-100:]):
            link("user:" + event.user, "item:" + event.item)
            if event.kind == "review":
                review = f"review:{index}"
                link(review, "user:" + event.user)
                link(review, "item:" + event.item)

    def retrieve(self, user_id: str, max_hops: int = 4) -> dict[str, list[str]]:
        start = "user:" + user_id
        queue = deque([(start, [start])])
        visited, found = {start}, {}
        while queue:
            node, path = queue.popleft()
            if node.startswith("item:"):
                found[node[5:]] = path
            if len(path) - 1 >= max_hops:
                continue
            for neighbor in sorted(self.edges[node]):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))
        return found


@dataclass
class Constraints:
    size: int = 3
    max_recent_repeats: int = 2
    total_budget: float | None = None
    min_categories: int = 2
    min_merchants: int = 2
    diversity_weight: float = 0.5
    fairness_weight: float = 0.35


@dataclass
class SlateResult:
    items: list[Item]
    score: float
    status: str
    explanations: list[str] = field(default_factory=list)
    rejected: dict[str, int] = field(default_factory=dict)
    explored: int = 0
    pool_size: int = 0


def hard_reasons(item: Item, user: User, constraints: Constraints) -> list[str]:
    reasons = []
    if item.price > user.budget:
        reasons.append("over per-item budget")
    if item.stock <= 0:
        reasons.append("out of stock")
    if user.region not in item.regions:
        reasons.append("delivery unavailable")
    if user.allowed_categories and item.category not in user.allowed_categories:
        reasons.append("outside mandatory category allow-list")
    if user.recent[-8:].count(item.id) >= constraints.max_recent_repeats:
        reasons.append("recent repetition limit")
    return reasons


def select_pool(items: list[Item], scores: dict[str, float], catalog: Catalog,
                limit: int = 12) -> list[Item]:
    """Preserve category and merchant representatives before score truncation."""
    ordered = sorted(items, key=lambda i: (-scores[i.id], i.id))
    chosen = {item.id: item for item in ordered[:3]}
    for category in catalog.categories:
        representative = next((i for i in ordered if i.category == category), None)
        if representative:
            chosen[representative.id] = representative
    for merchant in sorted(catalog.merchants.values(), key=lambda m: (m.exposure, m.id)):
        representative = next((i for i in ordered if i.merchant == merchant.id), None)
        if representative:
            chosen[representative.id] = representative
    for item in ordered:
        if len(chosen) >= limit:
            break
        chosen[item.id] = item
    return list(chosen.values())[:limit]


def solve_slate(items: list[Item], user: User, scores: dict[str, float],
                constraints: Constraints, catalog: Catalog,
                pool_limit: int | None = 12) -> SlateResult:
    if constraints.size < 1 or constraints.max_recent_repeats < 1:
        raise ValueError("Slate size and repetition limit must be positive")
    rejected = Counter()
    feasible = []
    for item in items:
        reasons = hard_reasons(item, user, constraints)
        rejected.update(reasons)
        if not reasons:
            feasible.append(item)
    pool = select_pool(feasible, scores, catalog, pool_limit) if pool_limit else feasible
    best, best_score, explored = [], float("-inf"), 0
    target = min(constraints.size, len(pool))

    def value(chosen):
        cats = len({i.category for i in chosen})
        merchants = len({i.merchant for i in chosen})
        return (sum(scores[i.id] for i in chosen)
                - constraints.diversity_weight * max(0, constraints.min_categories - cats)
                - constraints.fairness_weight * max(0, constraints.min_merchants - merchants))

    # Backtracking explores each combination once; forward checking prunes size
    # and optional cart-budget violations. Hard constraints are never relaxed.
    def visit(start, chosen, cost):
        nonlocal best, best_score, explored
        explored += 1
        if len(chosen) == target:
            score = value(chosen)
            if score > best_score:
                best, best_score = chosen[:], score
            return
        needed = target - len(chosen)
        for index in range(start, len(pool) - needed + 1):
            item = pool[index]
            if constraints.total_budget is not None and cost + item.price > constraints.total_budget:
                continue
            visit(index + 1, chosen + [item], cost + item.price)

    # Prefer full slates. If a total budget forbids them, return the largest
    # feasible smaller slate with an explicit shortfall, never an invalid cart.
    while target > 0:
        visit(0, [], 0.0)
        if best:
            break
        target -= 1
    explanations = []
    if len(best) < constraints.size:
        explanations.append(f"Only {len(best)}/{constraints.size} items fit the hard constraints.")
    if len({i.category for i in best}) < constraints.min_categories:
        explanations.append("Category diversity target unmet: feasibility or relevance trade-off.")
    if len({i.merchant for i in best}) < constraints.min_merchants:
        explanations.append("Merchant coverage target unmet: feasibility or relevance trade-off.")
    status = "infeasible" if not best else ("tradeoff" if explanations else "satisfied")
    return SlateResult(best, best_score if best else 0.0, status, explanations,
                       dict(rejected), explored, len(pool))
