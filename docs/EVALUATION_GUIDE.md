# Evaluation walkthrough and viva guide

## Before the evaluation

Use Python 3.10+ and open a terminal in the project root. The included policy and synthetic fixture allow immediate execution without internet access. Run `python3 -m unittest discover -s tests -v`. Review the report, especially negative results and simulator limitations. Each team member should complete the truthful contribution record and be able to explain the code they submit. The evaluation date was not supplied; all deliverables were prepared for immediate review.

## A 7-minute demonstration

1. **Explain the problem (45 seconds).** Six synthetic shoppers receive repeated recommendations. Budgets, inventory, prices, preferences and merchant exposure change. The objective is useful recommendations with explicit diversity/fairness trade-offs. Generative AI is excluded for this evaluation stage.
2. **Show a valid slate (60 seconds).** Run `python3 -m recsys recommend --user u0 --budget 25 --strategy diverse`. Point to the graph path, predicted purchase probability, campaign intent, and constraint rejection counts. A slate is a set of alternatives; the ordinary budget is per item.
3. **Show infeasibility (30 seconds).** Run `python3 -m recsys recommend --user u0 --budget 1`. The empty result explains why. It does not bypass the budget to satisfy a platform diversity target.
4. **Run the whole pipeline (90 seconds).** Run `python3 -m recsys demo`. Open `results/demo_excerpt.json` or the full trace. Show clarification, A* action sequence, feedback, updated beliefs and a replan after a budget or interest change. All explanations are fixed templates grounded in actual module outputs.
5. **Show dynamics (45 seconds).** In `results/demo.json`, inspect `events`: price change (4), negative review (6), budget change (8), new merchant (10), interest shifts (12), stock outage (16), popularity growth (18). Merchant responses change prices and product description versions. Round numbers start at zero.
6. **Discuss measured results (90 seconds).** Show `results/summary.md`, the report plots, and `results/evaluation.json`. Explain separate training/test seeds, frozen evaluation, baselines, ablations, and a result where the learner loses. Jain exposure equality is not proof of equitable treatment conditional on relevance or delivery eligibility.
7. **Explain one calculation (45 seconds).** Calculate a Beta update and a Q-learning update. Show a failed planned action leaving its goal facts false. Explain why the 12-item pool means catalog-wide optimality is not guaranteed.

The saved example and measured results can be inspected without retraining. A small fresh experiment is available with `python3 -m recsys evaluate --seeds 2 --seed-start 200 --output results/quick_check`.

## Questions the team should be ready to answer

- Why BFS? The graph edges are unweighted, so BFS finds shortest explanation paths within the hop bound. CSP slate optimization is a separate search over feasible item combinations.
- What is a CSP variable? Each of three slate positions selects a distinct candidate item. Unary feasibility restrictions apply per item; the optional cart constraint couples prices. Category/merchant shortfalls enter the soft objective, so this is a weighted soft CSP rather than pure feasibility alone.
- Is slate search globally optimal? It exhaustively searches the bounded candidate pool, not the full catalog. The tests compare unrestricted small instances against an independent combinations-based solution.
- Is planning real? A* searches symbolic states with preconditions, costs and effects. The first plan requires clarification before familiar items, then category introduction, then comparison. Budget/fatigue facts can require extra steps. Execution feedback determines whether effects occurred.
- Why not equate clicks with liking? A click receives 0.35 units of fractional preference evidence, compared with 1 unit for a normal purchase; gifts are discounted and unobserved/uncertain silence carries little evidence. This is an interpretable evidence model, not a calibrated human-behavior posterior.
- How is merchant trust protected? Merchant-written quality claims cannot update trust by themselves. Fulfillment observations update reliability; corroborated reviews can update claim/reviewer credibility. A purchase and its review share one transaction key, so the same fulfillment is counted once while still corroborating the review. Synthetic verification labels are assumptions, not a real fraud detector.
- What does RL control? It selects among five scoring/coverage strategies. Search and the CSP still choose feasible products, and the planner controls the campaign stage. This is sequential Q-learning, not five disconnected heuristics presented as learning.
- What happens during clarification/rest? They have explicit small costs. During training their discounted rewards are carried into the transition between successive recommendation decisions for that user, with discount raised to the elapsed turn count.
- Does the model see true preferences? No hidden preference or quality vector enters the public catalog or policy. It does observe the simulator's satisfaction feedback proxy, gift/attention annotations and fulfillment observations; those are strong assumptions for real-world transfer.
- Why might no planning score better? Questions and rest consume turns and incur costs. A short-horizon simulation can reward immediate item exposure; that is a meaningful limitation of this campaign model, not a result to conceal.
- Why might the learner lose to personalized ranking? Coarse state bins, finite training, changing dynamics, a hand-designed reward and the exploration/coverage trade-off can all matter. Experiments support only the measured synthetic setting; they do not establish a generally superior algorithm.
- What remains outside scope? Generative AI, real behavioral validation, production fairness guarantees and a browser interface. Actual names/individual contribution declarations must come from the team.

## Quick integrity check

The source code, data, saved policy, raw runs, summary and report are bundled. `results/VALIDATION.json` records actual validation and file hashes. If you modify source or regenerate results with other settings, rerun checks and update the report before quoting its numbers.
