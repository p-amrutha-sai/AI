"""Synthetic user/merchant agents and paired, keyed-random dynamic scenarios.

This is a laboratory model, not a validated model of real shoppers. Hidden
preferences/quality are never passed to Platform.recommend or encode_state.
"""
from collections import Counter
from dataclasses import asdict
from hashlib import sha256
import math
import random
from .models import Item, Merchant, Interaction, load_catalog
from .platform import Platform
from .rl import QPolicy
from .search import Constraints, hard_reasons


def keyed(seed, *parts):
    digest = sha256((str(seed) + "|" + "|".join(map(str, parts))).encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def jain(values):
    return sum(values)**2 / (len(values) * sum(v*v for v in values)) if any(values) else 1.0


class Simulator:
    def __init__(self, seed=100, policy=None, approach="learned", planning=True,
                 uncertainty=True, dynamic=True):
        self.seed, self.dynamic = seed, dynamic
        self.catalog = load_catalog()
        self.platform = Platform(self.catalog, policy, approach, planning, uncertainty)
        self.hidden_preferences = {}
        self.hidden_quality = {}
        for user in self.catalog.users.values():
            self.hidden_preferences[user.id] = {
                c: min(0.95, (0.76 if c == user.declared_category else 0.16)
                       + 0.15 * keyed(seed, user.id, c)) for c in self.catalog.categories}
        for merchant in self.catalog.merchants:
            self.hidden_quality[merchant] = 0.65 + 0.28 * keyed(seed, merchant, "quality")
        self.trace, self.events, self.round_metrics = [], [], []
        self.records = []
        self.responded = set()
        self.actions = Counter()
        self.hard_violations, self.brier, self.impressions = 0, 0.0, 0
        self.clicks, self.purchases, self.total_reward, self.decision_count = 0, 0, 0.0, 0
        self.category_exposure = Counter()
        self.item_exposure = Counter()
        self.repeat_exposure = 0
        self.slate_diversity = []
        self.novelty = []
        self.q_seen, self.q_decisions = 0, 0

    def log_event(self, round_no, kind, **detail):
        self.events.append({"round": round_no, "kind": kind, **detail})

    def apply_events(self, round_no):
        if not self.dynamic:
            return
        cats = self.catalog.categories
        if round_no == 4:
            for item in self.catalog.items.values():
                if item.merchant == "m0":
                    item.price = round(item.price * 1.25, 2)
            self.log_event(round_no, "price_change", merchant="m0", multiplier=1.25)
        if round_no == 6:
            item = next(i for i in self.catalog.items.values() if i.merchant == "m0")
            event = Interaction("u0", item.id, "review", round_no, 0.1, True,
                                fulfilled=False)
            self.platform.observe(event)
            self.catalog.users["u0"].last_feedback = "negative_review"
            self.log_event(round_no, "negative_review", item=item.id, verified=True)
        if round_no == 8:
            self.catalog.users["u0"].budget = 12.0
            self.log_event(round_no, "budget_change", user="u0", budget=12.0)
        if round_no == 10:
            self.catalog.merchants["m5"] = Merchant("m5", "New independent merchant")
            self.hidden_quality["m5"] = 0.85
            self.platform.beliefs.new_merchant("m5")
            for n, category in enumerate(cats):
                item = Item(f"new{n}", f"New {category} essential", category, "m5",
                            10.0 + n*6, 30, ["north", "south"], 0, 0.9)
                self.catalog.items[item.id] = item
            self.log_event(round_no, "new_merchant", merchant="m5", new_items=4)
        if round_no == 12:
            for uid in ("u0", "u3"):
                user = self.catalog.users[uid]
                new_category = cats[(cats.index(user.declared_category) + 1) % len(cats)]
                self.hidden_preferences[uid] = {c: 0.88 if c == new_category else 0.20 for c in cats}
                user.declared_category = new_category
                self.log_event(round_no, "interest_shift", user=uid, declared_category=new_category)
        if round_no == 16:
            for item in self.catalog.items.values():
                if item.merchant == "m1":
                    item.stock = 0
            self.log_event(round_no, "stock_outage", merchant="m1")
        if round_no == 18:
            for item in self.catalog.items.values():
                if item.merchant == "m0":
                    item.popularity += 100
            self.log_event(round_no, "popularity_growth", merchant="m0", added_per_item=100)
        if round_no % 3 == 0:
            for idx, review in enumerate(self.catalog.interactions):
                if idx in self.responded or review.kind != "review" or review.rating is None or review.rating >= 0.4:
                    continue
                self.responded.add(idx)
                item = self.catalog.items[review.item]
                merchant = self.catalog.merchants[item.merchant]
                merchant.responses += 1
                item.price = round(item.price * 0.95, 2)
                item.description_version += 1
                # Actual service improves slightly, not inferred from a reply.
                self.hidden_quality[item.merchant] = min(0.98, self.hidden_quality[item.merchant] + 0.01)
                self.log_event(round_no, "merchant_response", merchant=item.merchant,
                               item=item.id, new_price=item.price, description_version=item.description_version)

    def step(self, user, round_no, epsilon=0.0, keep_trace=False):
        decision = self.platform.recommend(user, epsilon)
        self.decision_count += 1
        self.actions[decision["strategy"]] += 1
        intent = decision["intent"]
        before_satisfaction = user.satisfaction
        feedback, events, utilities, predictions = [], [], [], []
        old_recent = user.recent[-8:]
        if intent == "ask_preference":
            category = user.declared_category
            answer = min(1.0, max(0.0, self.hidden_preferences[user.id][category]
                                  + 0.12*(keyed(self.seed, round_no, user.id, "answer")-0.5)))
            if self.platform.use_uncertainty:
                self.platform.beliefs.clarify(user, category, answer)
            reward, success = -0.04, True
            feedback.append("clarified")
        elif intent == "wait":
            user.satisfaction = min(1, user.satisfaction + 0.08)
            # Rest reduces recent category repetition without erasing item caps.
            user.recent_categories = user.recent_categories[-2:]
            reward, success = -0.03, True
            feedback.append("rested")
        else:
            item_ids = decision["items"]
            self.q_decisions += 1
            self.q_seen += int(decision["state"] in self.platform.policy.q)
            exposure_before = {m.id: m.exposure for m in self.catalog.merchants.values()}
            least_exposure = min(exposure_before.values())
            new_categories = set()
            for rank, item_id in enumerate(item_ids):
                item = self.catalog.items[item_id]
                self.hard_violations += bool(hard_reasons(item, user, Constraints()))
                pref = self.hidden_preferences[user.id][item.category]
                repeat = old_recent.count(item_id)
                same_cat = user.recent_categories[-6:].count(item.category)
                fatigue = 0.10*repeat + 0.035*same_cat
                quality = self.hidden_quality[item.merchant]
                affinity = max(0.01, min(0.99, pref + 0.12*(quality-0.5) - fatigue))
                attention = keyed(self.seed, round_no, user.id, item_id, "attention") > 0.12
                p_click = (0.10 + 0.72*affinity) * (1 - 0.08*rank)
                p_purchase_given_click = 0.50 * quality * (1 - 0.5*item.price/max(1, user.budget))
                purchase = attention and keyed(self.seed, round_no, user.id, item_id, "purchase") < p_click*p_purchase_given_click
                residual_click = p_click*(1-p_purchase_given_click) / max(0.001, 1-p_click*p_purchase_given_click)
                click = purchase or (attention and keyed(self.seed, round_no, user.id, item_id, "click") < residual_click)
                gift = purchase and keyed(self.seed, round_no, user.id, item_id, "gift") < 0.12
                fulfilled = keyed(self.seed, round_no, user.id, item_id, "fulfilled") < quality if purchase else None
                if purchase:
                    kind = "purchase"
                elif click:
                    kind = "like" if keyed(self.seed, round_no, user.id, item_id, "like") < 0.30 else "click"
                elif attention and affinity < 0.25:
                    kind = "reject"
                else:
                    kind = "ignore"
                event = Interaction(user.id, item.id, kind, round_no, gift=gift,
                                    attentive=attention, fulfilled=fulfilled)
                self.platform.observe(event)
                events.append(asdict(event))
                feedback.append(kind)
                if purchase:
                    item.stock -= 1
                    self.purchases += 1
                    rating = min(1, max(0, 0.55*affinity + 0.45*float(fulfilled)))
                    review = Interaction(user.id, item.id, "review", round_no, rating, True,
                                         gift=gift, fulfilled=fulfilled)
                    # Preserve corroboration for credibility; BeliefModel
                    # deduplicates fulfillment evidence across purchase/review.
                    self.platform.observe(review)
                    events.append(asdict(review))
                self.clicks += click
                self.impressions += 1
                self.repeat_exposure += int(repeat > 0)
                prediction = decision["predictions"][item.id]["purchase_probability"]
                self.brier += (prediction - float(purchase))**2
                predictions.append({"prediction": prediction, "purchase": int(purchase)})
                utility = max(0, min(1, affinity + 0.10*float(purchase) - 0.15*float(fulfilled is False)))
                utilities.append(utility)
                self.novelty.append(float(item.category not in user.recent_categories[-6:]))
                new_categories.add(item.category)
                self.category_exposure[item.category] += 1
                self.item_exposure[item.id] += 1
                self.catalog.merchants[item.merchant].exposure += 1
                item.popularity += int(click) + 2*int(purchase)
            if item_ids:
                diversity = len(new_categories)/len(item_ids)
                self.slate_diversity.append(diversity)
                avg_utility = sum(utilities)/len(utilities)
                user.satisfaction = 0.80*before_satisfaction + 0.20*avg_utility
                clicks = sum(e["kind"] in {"click", "like", "purchase"} for e in events)
                purchases = sum(e["kind"] == "purchase" for e in events)
                fair_bonus = sum(exposure_before[self.catalog.items[i].merchant] <= least_exposure + 2 for i in item_ids)/len(item_ids)
                repeat_penalty = sum(i in old_recent for i in item_ids)/len(item_ids)
                reward = (0.25*clicks + purchases + 0.35*user.satisfaction
                          + 0.15*diversity + 0.15*fair_bonus - 0.30*repeat_penalty - 0.25*(1-avg_utility))
                success = clicks > 0
                positive_categories = {self.catalog.items[e['item']].category for e in events
                                       if e['kind'] in {'click', 'like', 'purchase'}}
                if intent == 'familiar':
                    success = decision['favorite_category'] in positive_categories
                elif intent == 'introduce_category':
                    success = any(c != decision['favorite_category'] for c in positive_categories)
                elif intent == 'compare':
                    success = clicks > 0 and len(item_ids) >= 2
                user.recent.extend(item_ids)
                user.recent_categories.extend(self.catalog.items[i].category for i in item_ids)
                user.recent = user.recent[-24:]
                user.recent_categories = user.recent_categories[-24:]
            else:
                reward, success = -0.2, False
                user.satisfaction = max(0, user.satisfaction - 0.03)
            if round_no >= 5:
                leave_probability = max(0, 0.40-user.satisfaction) * 0.30
                if keyed(self.seed, round_no, user.id, "leave") < leave_probability:
                    user.active = False
                    reward -= 0.8
            if user.active:
                reward += 0.05  # continued engagement, including silent sessions
        if self.platform.use_planning:
            self.platform.planner.observe(user.id, intent, success)
        user.last_feedback = "reject" if feedback and all(x in {"ignore", "reject"} for x in feedback) else "positive"
        user.turns += 1
        self.total_reward += reward
        record = {"round": round_no, "user": user.id, "intent": intent, "strategy": decision["strategy"],
                  "observed_effect_success": success,
                  "items": decision["items"], "feedback": feedback, "reward": reward,
                  "satisfaction": user.satisfaction, "active": user.active,
                  "diversity": len({self.catalog.items[i].category for i in decision["items"]}),
                  "predictions": predictions}
        self.records.append(record)
        if keep_trace:
            self.trace.append({**record, "decision": decision, "observations": events,
                "beliefs_after": self.platform.beliefs.user_summary(user.id, self.catalog.categories)})
        return decision, reward

    def run(self, rounds=24, epsilon=0.0, train=False, keep_trace=False):
        pending = {}
        for round_no in range(rounds):
            self.apply_events(round_no)
            self.platform.beliefs.tick()
            for user in self.catalog.users.values():
                if not user.active:
                    continue
                decision, reward = self.step(user, round_no, epsilon, keep_trace)
                if train:
                    previous = pending.get(user.id)
                    if decision["items"]:
                        if previous:
                            self.platform.policy.update(*previous[:3], decision["state"], elapsed=previous[3])
                        pending[user.id] = [decision["state"], decision["action_index"], reward, 1]
                    elif previous:
                        previous[2] += self.platform.policy.gamma**previous[3] * reward
                        previous[3] += 1
                    if not user.active and user.id in pending:
                        transition = pending.pop(user.id)
                        self.platform.policy.update(*transition[:3], "terminal", terminal=True)
            self.round_metrics.append({"round": round_no,
                "satisfaction": sum(u.satisfaction for u in self.catalog.users.values())/len(self.catalog.users),
                "retention": sum(u.active for u in self.catalog.users.values())/len(self.catalog.users),
                "reward": self.total_reward,
                "merchant_jain": jain([m.exposure for m in self.catalog.merchants.values()])})
        if train:
            for transition in pending.values():
                self.platform.policy.update(*transition[:3], "terminal", terminal=True)
        return self.metrics()

    def metrics(self):
        cats = list(self.category_exposure.values())
        total = max(1, sum(cats))
        entropy = -sum((n/total)*math.log(n/total) for n in cats if n) / math.log(len(self.catalog.categories))
        item_total = max(1, sum(self.item_exposure.values()))
        return {"seed": self.seed, "decisions": self.decision_count, "impressions": self.impressions,
            "click_rate": self.clicks/max(1, self.impressions),
            "purchase_rate": self.purchases/max(1, self.impressions),
            "cumulative_reward": self.total_reward,
            "reward_per_decision": self.total_reward/max(1, self.decision_count),
            "retention": sum(u.active for u in self.catalog.users.values())/len(self.catalog.users),
            "satisfaction": sum(u.satisfaction for u in self.catalog.users.values())/len(self.catalog.users),
            "slate_diversity": sum(self.slate_diversity)/max(1, len(self.slate_diversity)),
            "category_entropy": entropy,
            "merchant_jain": jain([m.exposure for m in self.catalog.merchants.values()]),
            "item_hhi": sum((n/item_total)**2 for n in self.item_exposure.values()),
            "repetition_rate": self.repeat_exposure/max(1, self.impressions),
            "novelty": sum(self.novelty)/max(1, len(self.novelty)),
            "purchase_brier": self.brier/max(1, self.impressions),
            "hard_violations": self.hard_violations,
            "replans": len(self.platform.planner.replans),
            "learned_state_coverage": self.q_seen/max(1, self.q_decisions),
            "merchant_exposures": {m.id: m.exposure for m in self.catalog.merchants.values()},
            "strategy_counts": dict(self.actions)}


def train_policy(episodes=200, rounds=24, seed=7, progress=None):
    policy = QPolicy(seed)
    rewards = []
    for episode in range(episodes):
        # All training environment seeds are disjoint from evaluation/demo seeds.
        simulator = Simulator(seed=500_000 + episode, policy=policy)
        epsilon = max(0.08, 0.75 * (1 - episode/max(1, episodes)))
        metrics = simulator.run(rounds, epsilon, train=True)
        rewards.append(metrics["cumulative_reward"])
        if progress and (episode+1) % 25 == 0:
            progress(episode+1, episodes)
    policy.metadata = {"episodes": episodes, "rounds": rounds, "agent_seed": seed,
                       "environment_seed_start": 500_000, "environment_seed_stop_exclusive": 500_000 + episodes,
                       "epsilon": "linear 0.75 to floor 0.08", "training_rewards": rewards,
                       "training_decisions": sum(sum(row) for row in policy.visits.values())}
    return policy
