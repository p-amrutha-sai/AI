# Nine-person team guide and module walkthrough

This guide explains the delivered implementation and proposes responsibilities for nine group members. It does not assign past authorship to unnamed people. The implementation, experiments and documentation were produced with Codex assistance. Each member should record the work they actually perform while reviewing, validating, improving or extending it.

## Proposed allocation

| Member | Responsibility | Main files |
|---|---|---|
|  Sushanth| Problem formulation, entities, synthetic catalog and dataset assumptions | `recsys/models.py`, `data/catalog.json`, `data/DATA_CARD.md` |
|  Sai| Graph representation, BFS retrieval and recommendation paths | Graph/BFS portions of `recsys/search.py` |
|  Hari| Constraint satisfaction, slate optimization and infeasibility explanations | Constraints/CSP portions of `recsys/search.py` |
|  Hema Kiran| A* campaign planning, action effects and replanning | `recsys/planning.py` and execution checks in `recsys/simulation.py` |
|  Pranava sai| Bayesian preference evidence, purchase estimates and credibility | `recsys/uncertainty.py` |
|  Kalyan Ram  | Q-learning, state/action/reward design and training | `recsys/rl.py` and training/temporal credit in `recsys/simulation.py` |
|  Shubhan| Simulated user/merchant behavior and changing-world scenarios | `recsys/simulation.py` |
|  Raag| Tests, baselines, metrics and experiment interpretation | `tests/test_modules.py`, `recsys/evaluation.py`, `results/` |
|  Pranav| Integration, CLI, reproducibility, report and live demonstration | `recsys/platform.py`, `recsys/__main__.py`, `run_all.py`, `README.md`, `docs/` |

`CONTRIBUTIONS.md` specifies concrete evidence each person can produce. Member labels should be replaced with names once actual responsibilities are agreed. Nine team members and the six simulated shoppers are different concepts; there is no requirement for those numbers to match.

## Module 1: Search and Constraint Satisfaction

**Question answered:** Which products are reachable and relevant, and which combination is valid for this user?

I implemented a typed graph connecting users, interests/categories, products, merchants and reviews. Breadth-first search discovers products up to four edges away and saves the path used to reach each one. For example, a route can connect a user's declared books interest to the related tech category and then to a tech product. The saved path explains how the product entered consideration; it is not itself proof that the user will like it.

I then implemented a weighted soft CSP that chooses up to three distinct items. Mandatory rules exclude products above the per-item budget, out of stock, unavailable in the delivery region, outside a mandatory category allow-list, or already shown twice in the preceding eight item impressions. An optional total budget additionally constrains the sum of slate prices.

The solver uses backtracking with size and total-budget pruning. It scores feasible combinations using product scores and penalties for missing category/merchant coverage targets. Diversity and merchant coverage are soft objectives, so they cannot override mandatory user rules. It explains shortfalls and returns a shorter or empty slate when needed.

For efficiency, normal execution uses at most 12 candidates, preserving top-scored items and category/merchant representatives. Search is exact within that pool, not globally optimal across the catalog. Small correctness tests disable the pool cap and compare the solver with independent exhaustive enumeration.

**Example:** With budget 25, an item priced 30 is excluded even if it is popular or would increase diversity. If only books are allowed, the system can return books while reporting an unmet category-diversity target.

**Owners:** Members 2 and 3, supported by Member 1's data-model work.

## Module 2: Recommendation Planning

**Question answered:** What should the platform do next in a multi-step interaction?

I implemented an A* planner over seven Boolean facts: preferences known, existing interest supported, an alternative category introduced, alternatives compared, engagement observed, affordability established, and readiness after rest. Actions have preconditions, costs and expected effects.

The actions are asking a structured preference question, showing familiar products, introducing another category, comparing products, offering a lower-cost alternative, and waiting. A nominal campaign is: clarify -> familiar -> introduce category -> compare. The planner searches for a low-cost sequence reaching the symbolic goal.

I also implemented execution monitoring. A planned effect is not automatically treated as achieved: familiar recommendations require positive feedback in the favored category; introduction requires positive feedback in an alternative category; comparison requires engagement with at least two presented alternatives. Failed effects trigger replanning. Budget changes, rejection, negative reviews, changed declared interests and repetitive low-satisfaction sessions can invalidate parts of a plan.

**Example:** If the user lowers their budget, the controller marks affordability unresolved and plans a lower-cost alternative instead of continuing the old campaign unchanged.

This planner is optimal only in its small deterministic abstraction. Actual users can respond differently. The experiments do not establish that the planner improves reward or satisfaction over every baseline; questions/rest also consume turns.

**Owner:** Member 4, with Member 7 checking observed effects in the simulator.

## Module 3: Reasoning under Uncertainty

**Question answered:** What does the platform currently believe, and how much evidence supports that belief?

I implemented Beta evidence models for user-category preferences, purchase propensity, merchant fulfillment reliability, reviewer credibility and merchant-claim credibility. Belief means and uncertainty feed ranking and exploration. Purchase estimates combine preference, purchase propensity, reliability and affordability.

Different actions contribute different amounts of evidence. A click carries weaker preference evidence than a normal purchase. Gift purchases/reviews are discounted. Ignoring an item has little preference weight, and observably unattended silence has none. Verified reviews carry more weight than unverified reviews. Round-wise decay discounts older evidence so the model can adapt to changing interests.

For trust, merchant-written quality claims cannot independently improve credibility. Fulfillment observations update reliability, while corroborated reviews can update reviewer/claim beliefs. A transaction key ensures that a purchase and its review do not count the same fulfillment twice.

**Example:** One click on a tech product slightly raises estimated interest in tech, but it does not make the system certain the user likes tech. A gift purchase contributes much less preference evidence than an ordinary purchase.

These are hand-designed fractional evidence models, not calibrated estimates from real shoppers. Attention, gift and fulfillment annotations are assumptions of the synthetic experiment.

**Owner:** Member 5.

## Module 4: Reinforcement Learning

**Question answered:** Which recommendation strategy should the platform use in the current situation?

I implemented tabular Q-learning with five actions: favor popular items, favor personalized relevance, increase diversity, support less-exposed merchants, or explore uncertain interests. The learner selects a ranking strategy; graph search and the constraint solver still determine the feasible products.

The observed state includes the highest-mean preference category, uncertainty, satisfaction, budget, recent category repetition, merchant-exposure imbalance, popularity concentration and campaign intent. Hidden preference and merchant-quality vectors are kept inside the simulator and are not passed into the policy.

Rewards combine clicks, purchases, satisfaction, diversity, low-exposure merchant coverage and continued engagement, with costs for repetition, irrelevant suggestions, information/rest turns and dropout. Training uses epsilon-greedy exploration and a discounted Q update. Clarification/rest rewards between recommendation decisions are included in temporal credit; evaluation keeps the learned Q-table frozen.

I trained the delivered policy for 200 synthetic episodes and saved its Q-table and provenance. It contains 143 learned state bins and 24,557 recommendation updates. Evaluation uses separate environment seeds from training. The learned strategy is compared with popularity-only ranking, highest predicted purchase probability, fixed diversity, and two ablations.

**Example:** If recommendations are repetitive and merchant exposure is uneven, the learner can select a diverse or fair strategy rather than always exploiting the highest purchase estimate. Which action it actually takes depends on its learned Q-values; no such action is guaranteed for every state.

**Owner:** Member 6, supported by Members 7 and 8 for simulation and evaluation.

## How the complete project works

1. The planner chooses the next campaign action. Asking a question and resting are actual non-recommendation turns.
2. On a recommendation turn, Q-learning selects a ranking strategy.
3. BFS retrieves candidate products and their graph paths.
4. Beliefs provide preference, purchase and risk estimates for scoring.
5. The CSP selects a feasible slate and explains objective shortfalls.
6. User/merchant interaction changes beliefs, plans, stock, prices, popularity, satisfaction and exposure. Q-values also update during training, but remain frozen during evaluation.

I implemented a programmed simulator with six shoppers, five initial merchants, 40 initial products and four categories. It introduces changed budgets/interests, price changes, new products/merchants, negative reviews, merchant responses, stock outages and popularity growth. These are synthetic agents, not Generative AI agents.

The delivered verification includes 24 passing tests, 72 held-out evaluation runs (six variants x twelve seeds), zero observed hard-constraint violations in those runs, a reproducible clean-folder demo, raw results, a report and an evaluation guide. Those checks establish behavior in this synthetic setting, not real-world behavioral validity or universal algorithm superiority.

Generative AI remains excluded from the project at the user's request.
