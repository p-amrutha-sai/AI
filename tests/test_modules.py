from dataclasses import asdict
from itertools import combinations
import json
from pathlib import Path
import random
import tempfile
import unittest

from recsys.models import load_catalog, Interaction
from recsys.search import RecommendationGraph, Constraints, solve_slate, hard_reasons
from recsys.planning import Facts, ACTION_MAP, plan_campaign, goal, CampaignController
from recsys.uncertainty import BetaBelief, BeliefModel
from recsys.rl import QPolicy, encode_state
from recsys.platform import Platform
from recsys.simulation import Simulator, train_policy
from recsys.evaluation import evaluate


class SearchTests(unittest.TestCase):
    def setUp(self):
        self.catalog = load_catalog()
        self.user = self.catalog.users["u0"]

    def test_bfs_paths_are_edges_and_reach_all_items(self):
        graph = RecommendationGraph(self.catalog)
        paths = graph.retrieve("u0")
        self.assertEqual(set(paths), set(self.catalog.items))
        self.assertEqual(len(paths["i000"]), 3)
        for path in paths.values():
            self.assertLessEqual(len(path), 5)
            for a,b in zip(path, path[1:]):
                self.assertIn(b, graph.edges[a])

    def test_reviews_and_history_are_in_graph(self):
        self.catalog.interactions.append(Interaction("u0", "i110", "review", 1, .8, True))
        graph = RecommendationGraph(self.catalog)
        self.assertIn("item:i110", graph.edges["user:u0"])
        self.assertIn("item:i110", graph.edges["review:0"])

    def test_independent_hard_filters_and_no_silent_relaxation(self):
        item = self.catalog.items["i410"]
        self.user.region, self.user.budget = "south", 1
        self.user.allowed_categories = ["books"]
        self.user.recent = [item.id, item.id]
        item.stock = 0
        reasons = hard_reasons(item, self.user, Constraints())
        self.assertEqual(len(reasons), 5)
        result = solve_slate([item], self.user, {item.id:1}, Constraints(), self.catalog)
        self.assertEqual(result.status, "infeasible")
        self.assertEqual(result.items, [])

    def test_exact_csp_matches_independent_bruteforce(self):
        rng = random.Random(9)
        self.user.budget = 100
        items = list(self.catalog.items.values())[:12]
        for _ in range(10):
            scores = {i.id:rng.random() for i in items}
            c = Constraints(total_budget=70, diversity_weight=.4, fairness_weight=.3)
            result = solve_slate(items, self.user, scores, c, self.catalog, pool_limit=None)
            def objective(slate):
                return sum(scores[i.id] for i in slate)-.4*max(0,2-len({i.category for i in slate}))-.3*max(0,2-len({i.merchant for i in slate}))
            brute = max(objective(s) for s in combinations(items,3) if sum(i.price for i in s)<=70)
            self.assertAlmostEqual(result.score, brute)

    def test_cart_budget_and_short_slate(self):
        items = list(self.catalog.items.values())
        c = Constraints(total_budget=8)
        result = solve_slate(items, self.user, {i.id:1 for i in items}, c, self.catalog, pool_limit=None)
        self.assertEqual(len(result.items), 1)
        self.assertLessEqual(sum(i.price for i in result.items), 8)
        self.assertEqual(result.status, "tradeoff")

    def test_mandatory_categories_never_relaxed_for_diversity(self):
        self.user.allowed_categories = ["books"]
        items = list(self.catalog.items.values())
        result = solve_slate(items,self.user,{i.id:1 for i in items},Constraints(),self.catalog)
        self.assertTrue(all(i.category == "books" for i in result.items))
        self.assertTrue(any("diversity" in e for e in result.explanations))


class PlanningTests(unittest.TestCase):
    def test_plan_has_valid_preconditions_and_reaches_goal(self):
        for initial in (Facts(), Facts(affordable=False,rested=False), Facts(known=True, supported=True)):
            result = plan_campaign(initial)
            current = initial
            for name in result["actions"]:
                self.assertTrue(ACTION_MAP[name].applicable(current))
                current = ACTION_MAP[name].apply(current)
            self.assertTrue(goal(current))
        self.assertAlmostEqual(plan_campaign(Facts())["cost"], 3.9)

    def test_failed_effect_is_not_assumed_and_rejection_replans(self):
        user = load_catalog().users["u0"]
        controller = CampaignController()
        self.assertEqual(controller.next_action(user), "ask_preference")
        controller.observe(user.id,"ask_preference",True)
        self.assertEqual(controller.next_action(user), "familiar")
        controller.observe(user.id,"familiar",False)
        self.assertFalse(controller.facts[user.id].supported)
        user.last_feedback = "reject"
        controller.next_action(user)
        self.assertIn("reject",controller.replans[-1]["reasons"])

    def test_budget_and_interest_changes_invalidate_plan(self):
        user = load_catalog().users["u0"]
        controller = CampaignController()
        controller.next_action(user)
        controller.observe(user.id,"ask_preference",True)
        user.budget = 10
        self.assertEqual(controller.next_action(user), "lower_cost_alternative")
        self.assertIn("budget_change",controller.replans[-1]["reasons"])
        user.declared_category = "sports"
        self.assertEqual(controller.next_action(user), "ask_preference")
        self.assertIn("new_interest",controller.replans[-1]["reasons"])


class UncertaintyTests(unittest.TestCase):
    def setUp(self):
        self.catalog = load_catalog()
        self.model = BeliefModel(self.catalog)
        self.item = self.catalog.items["i000"]

    def test_beta_update_and_decay_towards_prior(self):
        belief = BetaBelief()
        belief.update(1,2)
        self.assertAlmostEqual(belief.mean,2/3)
        belief.decay(.5)
        self.assertEqual((belief.alpha,belief.beta),(3,2))

    def test_click_is_weak_gift_is_discounted(self):
        prior = self.model.preferences["u0","books"].mean
        self.model.observe(Interaction("u0",self.item.id,"click",0),self.item)
        click_gain = self.model.preferences["u0","books"].mean-prior
        gift = BeliefModel(self.catalog)
        gift.observe(Interaction("u0",self.item.id,"purchase",0,gift=True),self.item)
        real = BeliefModel(self.catalog)
        real.observe(Interaction("u0",self.item.id,"purchase",0),self.item)
        self.assertGreater(click_gain,0)
        self.assertLess(gift.preferences["u0","books"].mean,real.preferences["u0","books"].mean)
        self.assertGreater(real.preferences["u0","books"].std,.15)

    def test_unattended_silence_not_preference_dislike(self):
        before = asdict(self.model.preferences["u0","books"])
        self.model.observe(Interaction("u0",self.item.id,"ignore",0,attentive=False),self.item)
        self.assertEqual(before,asdict(self.model.preferences["u0","books"]))

    def test_verified_review_has_more_evidence(self):
        weak = BeliefModel(self.catalog)
        strong = BeliefModel(self.catalog)
        weak.observe(Interaction("u0",self.item.id,"review",0,.1,False),self.item)
        strong.observe(Interaction("u0",self.item.id,"review",0,.1,True),self.item)
        self.assertLess(strong.preferences["u0","books"].mean,weak.preferences["u0","books"].mean)

    def test_merchant_claims_do_not_create_trust(self):
        before = self.model.claims["m0"].mean
        self.item.claimed_quality = 1
        self.model.predict(self.catalog.users["u0"],self.item)
        self.assertEqual(before,self.model.claims["m0"].mean)
        self.model.observe(Interaction("u0",self.item.id,"review",0,.1,True,fulfilled=False),self.item)
        self.assertLess(self.model.claims["m0"].mean,before)
        self.assertLess(self.model.reliability["m0"].mean,.6)

    def test_review_corroboration_does_not_double_count_transaction(self):
        purchase = Interaction("u0",self.item.id,"purchase",1,fulfilled=True)
        review = Interaction("u0",self.item.id,"review",1,.9,True,fulfilled=True)
        self.model.observe(purchase,self.item)
        reliability = asdict(self.model.reliability["m0"])
        self.model.observe(review,self.item)
        self.assertEqual(reliability,asdict(self.model.reliability["m0"]))
        self.assertGreater(self.model.claims["m0"].mean,.5)
        self.assertGreater(self.model.reviewers["u0"].mean,.5)


class RLTests(unittest.TestCase):
    def test_q_update_terminal_and_discounted_bootstrap(self):
        policy = QPolicy(alpha=.5,gamma=.9)
        policy.q["next"] = [2,1,0,0,0]
        policy.update("state",0,1,"next",elapsed=2)
        self.assertAlmostEqual(policy.q["state"][0],.5*(1+.81*2))
        policy.update("terminal_state",1,1,"next",terminal=True)
        self.assertAlmostEqual(policy.q["terminal_state"][1],.5)

    def test_saved_frozen_policy_roundtrip(self):
        policy = QPolicy()
        policy.update("x",2,3,"end",terminal=True)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"policy.json"
            policy.save(path)
            loaded = QPolicy.load(path)
            self.assertEqual(loaded.q,policy.q)
            self.assertEqual(loaded.choose("x"),2)

    def test_context_changes_with_budget_satisfaction_and_history(self):
        catalog = load_catalog()
        beliefs = BeliefModel(catalog)
        user = catalog.users["u0"]
        before = encode_state(user,catalog,beliefs)
        user.budget,user.satisfaction,user.recent_categories = 90,.2,["books"]*4
        self.assertNotEqual(before,encode_state(user,catalog,beliefs))


class IntegrationTests(unittest.TestCase):
    def test_deterministic_seed_and_dynamic_scenarios(self):
        a,b = Simulator(42),Simulator(42)
        first,second = a.run(24,keep_trace=True),b.run(24)
        self.assertEqual(first,second)
        self.assertEqual(first["hard_violations"],0)
        expected = {"price_change","negative_review","budget_change","new_merchant","interest_shift","stock_outage","popularity_growth","merchant_response"}
        self.assertTrue(expected <= {e["kind"] for e in a.events})
        traces = [t for t in a.trace if t["items"]]
        self.assertTrue(traces)
        self.assertTrue(all(t["decision"]["paths"] for t in traces))
        self.assertTrue(any("budget_change" in r["reasons"] for r in a.platform.planner.replans))

    def test_training_changes_policy_and_evaluation_is_frozen(self):
        policy = train_policy(episodes=2,rounds=8)
        self.assertTrue(policy.q)
        before = json.dumps(policy.q,sort_keys=True)
        Simulator(101,policy).run(8)
        self.assertEqual(before,json.dumps(policy.q,sort_keys=True))

    def test_evaluation_rejects_training_overlap(self):
        policy = QPolicy()
        policy.metadata = {"environment_seed_start":500000,"environment_seed_stop_exclusive":500002}
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"policy.json"
            policy.save(path)
            with self.assertRaises(ValueError):
                evaluate(path,folder,seeds=[500000],rounds=1)

    def test_hidden_preferences_are_not_part_of_public_catalog(self):
        simulator = Simulator(42)
        payload = json.dumps(asdict(simulator.catalog))
        self.assertNotIn("hidden_preferences",payload)
        self.assertFalse(hasattr(simulator.platform,"hidden_preferences"))

    def test_purchase_baseline_selects_highest_feasible_probabilities(self):
        catalog = load_catalog()
        platform = Platform(catalog,approach="personalized",planning=False)
        user = catalog.users["u0"]
        result = platform.recommend(user)
        feasible = [i for i in catalog.items.values() if not hard_reasons(i,user,Constraints())]
        top = sorted((platform.beliefs.predict(user,i)["purchase_probability"] for i in feasible),reverse=True)[:3]
        actual = sorted((result["predictions"][i]["purchase_probability"] for i in result["items"]),reverse=True)
        self.assertEqual(actual,top)

    def test_introduction_effect_requires_positive_alternative_category(self):
        simulator = Simulator(42)
        simulator.run(24,keep_trace=True)
        introductions = [row for row in simulator.trace if row['intent']=='introduce_category']
        self.assertTrue(introductions)
        for row in introductions:
            decision = row['decision']
            expected = any(simulator.catalog.items[e['item']].category != decision['favorite_category']
                           for e in row['observations'] if e['kind'] in {'click','like','purchase'})
            self.assertEqual(expected,row['observed_effect_success'])


if __name__ == "__main__":
    unittest.main()
