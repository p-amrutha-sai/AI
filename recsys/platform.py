"""Integration: planner -> Q strategy -> graph -> beliefs -> CSP -> trace."""
from dataclasses import asdict
from .models import Catalog
from .planning import CampaignController
from .rl import QPolicy, STRATEGIES, encode_state
from .search import Constraints, RecommendationGraph, solve_slate
from .uncertainty import BeliefModel


class Platform:
    def __init__(self, catalog: Catalog, policy: QPolicy | None = None,
                 approach="learned", planning=True, uncertainty=True):
        self.catalog, self.policy, self.approach = catalog, policy or QPolicy(), approach
        self.beliefs = BeliefModel(catalog)
        self.planner = CampaignController()
        self.use_planning, self.use_uncertainty = planning, uncertainty

    def context(self, user, intent):
        return encode_state(user, self.catalog, self.beliefs, intent)

    def recommend(self, user, epsilon=0.0):
        recent = user.recent_categories[-4:]
        fatigue = len(recent) == 4 and len(set(recent)) == 1 and user.satisfaction < 0.55
        intent = self.planner.next_action(user, fatigue) if self.use_planning else "familiar"
        state = self.context(user, intent)
        if self.approach == "learned":
            action = self.policy.choose(state, epsilon)
            strategy = STRATEGIES[action]
        else:
            strategy = self.approach
            action = STRATEGIES.index(strategy)
        if intent in {"ask_preference", "wait"}:
            return {"intent": intent, "strategy": strategy, "state": state,
                    "action_index": action, "items": [], "status": "information_action",
                    "plan": self.planner.last_trace.get(user.id, {})}

        paths = RecommendationGraph(self.catalog).retrieve(user.id)
        candidates = [self.catalog.items[item_id] for item_id in paths]
        scores, predictions = {}, {}
        max_pop = max(i.popularity for i in self.catalog.items.values()) + 1
        min_exposure = min(m.exposure for m in self.catalog.merchants.values())
        favorite = max(self.catalog.categories, key=lambda c: self.beliefs.preferences[user.id, c].mean)
        for item in candidates:
            p = self.beliefs.predict(user, item)
            predictions[item.id] = p
            repetition = user.recent[-8:].count(item.id) * 0.10
            popularity = (item.popularity + 1) / max_pop
            fair = 1 / (1 + self.catalog.merchants[item.merchant].exposure - min_exposure)
            novelty = float(item.category not in user.recent_categories[-6:])
            risk = p["risk_lower_bound"] if self.use_uncertainty else 0.5
            relevance = p["purchase_probability"] * 2 + 0.3 * risk
            choices = {"popular": popularity,
                       "personalized": relevance - repetition,
                       "diverse": relevance + 0.4 * novelty - repetition,
                       "fair": relevance + 0.45 * fair - repetition,
                       "explore": relevance + 1.5 * p["uncertainty"] + 0.2 * novelty - repetition}
            score = choices[strategy]
            if self.approach == "popular":
                score = popularity
            elif self.approach == "personalized":
                score = p["purchase_probability"]
            elif intent == "introduce_category":
                score += 0.20 * float(item.category != favorite)
            elif intent == "lower_cost_alternative":
                score += 0.35 * (1 - item.price / max(1, user.budget))
            elif intent == "familiar":
                score += 0.12 * float(item.category == favorite)
            scores[item.id] = score
        constraints = Constraints(
            diversity_weight=0.55 if strategy == "diverse" else (0.20 if strategy in {"fair", "explore"} else 0),
            fairness_weight=0.55 if strategy == "fair" else 0)
        result = solve_slate(candidates, user, scores, constraints, self.catalog)
        # Present strongest score first; CSP itself selects an unordered slate.
        result.items.sort(key=lambda i: (-scores[i.id], i.id))
        return {"intent": intent, "strategy": strategy, "state": state, "action_index": action,
                "favorite_category": favorite,
                "items": [i.id for i in result.items], "status": result.status,
                "constraints": asdict(constraints), "slate": {**asdict(result), "items": [i.id for i in result.items]},
                "paths": {i.id: paths[i.id] for i in result.items},
                "predictions": {i.id: predictions[i.id] for i in result.items},
                "explanations": {i.id: f"{strategy} strategy; {intent} campaign action; "
                    f"estimated purchase={predictions[i.id]['purchase_probability']:.3f}; "
                    f"price={i.price:.2f} within budget={user.budget:.2f}." for i in result.items},
                "plan": self.planner.last_trace.get(user.id, {})}

    def observe(self, event):
        self.catalog.interactions.append(event)
        if self.use_uncertainty:
            self.beliefs.observe(event, self.catalog.items[event.item])
