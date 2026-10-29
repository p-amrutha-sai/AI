"""Discounted Beta evidence models; noisy actions carry fractional evidence."""
from dataclasses import dataclass
import math
from .models import Catalog, Interaction, Item, User


@dataclass
class BetaBelief:
    alpha: float = 2.0
    beta: float = 2.0
    prior_alpha: float = 2.0
    prior_beta: float = 2.0

    @property
    def mean(self):
        return self.alpha / (self.alpha + self.beta)

    @property
    def std(self):
        n = self.alpha + self.beta
        return math.sqrt(self.alpha * self.beta / (n * n * (n + 1)))

    def update(self, success: float, weight: float = 1.0):
        if not 0 <= success <= 1 or weight < 0:
            raise ValueError("Invalid fractional evidence")
        self.alpha += success * weight
        self.beta += (1 - success) * weight

    def decay(self, factor=0.97):
        self.alpha = self.prior_alpha + factor * (self.alpha - self.prior_alpha)
        self.beta = self.prior_beta + factor * (self.beta - self.prior_beta)

    def summary(self):
        # A normal approximation is disclosed; not an exact Beta interval.
        return {"mean": round(self.mean, 4), "std": round(self.std, 4),
                "approx_95_interval": [round(max(0, self.mean - 1.96 * self.std), 4),
                                       round(min(1, self.mean + 1.96 * self.std), 4)],
                "evidence_mass": round(self.alpha + self.beta, 3)}


class BeliefModel:
    def __init__(self, catalog: Catalog, decay=0.97):
        self.decay_factor = decay
        self.preferences = {}
        self.purchase = {}
        self.reliability = {m: BetaBelief(3, 2, 3, 2) for m in catalog.merchants}
        self.claims = {m: BetaBelief() for m in catalog.merchants}
        self.reviewers: dict[str, BetaBelief] = {}
        self.fulfillment_evidence_seen: set[tuple[str, str, int]] = set()
        for user in catalog.users.values():
            for category in catalog.categories:
                prior = (3.0, 2.0) if category == user.declared_category else (2.0, 2.0)
                self.preferences[user.id, category] = BetaBelief(*prior, *prior)
                self.purchase[user.id, category] = BetaBelief(1.5, 5.5, 1.5, 5.5)

    def new_merchant(self, merchant_id):
        self.reliability.setdefault(merchant_id, BetaBelief(3, 2, 3, 2))
        self.claims.setdefault(merchant_id, BetaBelief())

    def tick(self):
        for group in (self.preferences, self.purchase, self.reliability, self.claims, self.reviewers):
            for belief in group.values():
                belief.decay(self.decay_factor)

    def predict(self, user: User, item: Item):
        pref = self.preferences[user.id, item.category]
        propensity = self.purchase[user.id, item.category]
        reliability = self.reliability[item.merchant]
        # Claim quality is shrunk towards 0.5 according to independently
        # observed claim credibility. Merchant text alone never updates trust.
        credibility = self.claims[item.merchant].mean
        quality = 0.5 + credibility * (item.claimed_quality - 0.5)
        affordability = max(0.1, 1 - 0.55 * item.price / max(1, user.budget))
        probability = propensity.mean * (0.5 + pref.mean) * (0.5 + reliability.mean) * affordability
        return {"preference": pref.mean, "uncertainty": pref.std,
                "purchase_probability": min(0.95, max(0.01, probability)),
                "reliability": reliability.mean, "quality": quality,
                "risk_lower_bound": max(0, pref.mean - pref.std)}

    def clarify(self, user: User, category: str, interest: float):
        self.preferences[user.id, category].update(interest, 2.0)

    def observe(self, event: Interaction, item: Item):
        pref = self.preferences[event.user, item.category]
        purchase = self.purchase[event.user, item.category]
        signals = {"browse": (0.55, 0.1), "click": (0.7, 0.35),
                   "like": (0.9, 0.8), "purchase": (0.9, 1.0),
                   "reject": (0.05, 0.9), "ignore": (0.4, 0.08)}
        if event.kind in signals:
            value, weight = signals[event.kind]
            if event.kind == "purchase" and event.gift:
                weight *= 0.15
            if event.kind == "ignore" and event.attentive is False:
                weight = 0.0
            pref.update(value, weight)
        if event.kind in {"click", "like", "purchase", "reject", "ignore"}:
            # Purchase is known even when silence is ambiguous about preference.
            purchase.update(float(event.kind == "purchase"), 0.7)
        transaction_key = (event.user, event.item, event.round)
        if event.fulfilled is not None and transaction_key not in self.fulfillment_evidence_seen:
            self.reliability[item.merchant].update(float(event.fulfilled), 1.0)
            self.fulfillment_evidence_seen.add(transaction_key)
        if event.kind == "review" and event.rating is not None:
            if not 0 <= event.rating <= 1:
                raise ValueError("Review ratings are normalized to [0,1]")
            reviewer = self.reviewers.setdefault(event.user, BetaBelief())
            weight = (1.0 if event.verified else 0.15) * reviewer.mean
            if event.gift:
                weight *= 0.15
            pref.update(event.rating, weight)
            # Compare reviews with separately observed fulfillment, when known.
            # Without corroboration reviewer reputation does not self-reinforce.
            if event.fulfilled is not None:
                reviewer.update(1 - abs(event.rating - float(event.fulfilled)), 0.5)
                self.claims[item.merchant].update(1 - abs(item.claimed_quality - event.rating), weight)

    def user_summary(self, user_id, categories):
        return {c: self.preferences[user_id, c].summary() for c in categories}
