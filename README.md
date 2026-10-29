# Group 13 - Dynamic Recommendation Platform

A complete, integrated implementation of **Modules 1-4** for the supplied Group 13 brief. **Module 5 (Generative AI) is deliberately excluded at the user's request.** The agents here are ordinary programmed user/merchant simulators. There are no LLMs, natural-language generation services, credentials, model downloads, or external datasets.

The system recommends affordable, available products to evolving synthetic users while measuring satisfaction, diversity and merchant exposure. It combines graph retrieval, a soft CSP, symbolic campaign planning, discounted Bayesian evidence and a learned sequential recommendation strategy.

## Run immediately

Requires **Python 3.10 or later**. Use a terminal in this project's root folder. Nothing needs to be installed with pip.

```sh
python3 -m recsys demo
python3 -m unittest discover -s tests -v
```

The first command runs all four modules across 24 rounds using the included trained policy and writes `results/demo.json`. It prints actual metrics. The second runs the correctness and integration checks.

For a single request, including an explanation of feasibility:

```sh
python3 -m recsys recommend --user u0 --budget 25 --strategy diverse
python3 -m recsys recommend --user u0 --budget 1
python3 -m recsys recommend --user u4 --allowed-categories books
python3 -m recsys recommend --user u0 --budget 40 --total-budget 25
```

Budget normally means the **maximum price of each recommended item**, since a slate contains alternatives, not a shopping basket. `--total-budget` explicitly demonstrates an additional cart budget. Items never bypass budget, stock, delivery, category allow-lists or repetition caps. An empty or shorter result is valid and explained.

## Reproduce the submitted experiments

```sh
python3 -m recsys train --episodes 200 --rounds 24 --seed 7 --output results/policy.json
python3 -m recsys evaluate --policy results/policy.json --seeds 12 --seed-start 100 --rounds 24 --output results
python3 -m recsys demo --policy results/policy.json --seed 42 --rounds 24 --output results/demo.json
```

Or use `python3 run_all.py` to run tests, training, evaluation and the demo in that order. It overwrites the corresponding generated result files. The checked-in report describes the submitted default settings; if settings or code change, its numbers must be updated too. Runtime depends on the machine; this is a small CPU-only experiment.

Training environment seeds are 500000-500199. Evaluation uses 100-111, and the demonstration uses 42. Evaluation freezes Q-values and rejects overlapping training seeds. Keyed random draws match exogenous user/item events between policies for the same seed; policy-dependent histories still differ. Repeating the same command and Python implementation produces the same numerical experiment outputs.

## Repository map

| Path | Purpose |
|---|---|
| `recsys/models.py` | Public users, items, merchants and interactions |
| `recsys/search.py` | Typed graph BFS, candidate pool and backtracking soft CSP |
| `recsys/planning.py` | A* campaign plans, preconditions/effects and replanning |
| `recsys/uncertainty.py` | Preference, purchase, reliability and credibility beliefs |
| `recsys/rl.py` | Q-learning state, five strategies, updates and persistence |
| `recsys/platform.py` | Integration and evidence-based, fixed-template explanations |
| `recsys/simulation.py` | Hidden response model, user/merchant agents and scenarios |
| `recsys/evaluation.py` | Baselines, ablations, metrics and paired seed comparisons |
| `data/catalog.json` | Included original synthetic fixture |
| `tests/test_modules.py` | Correctness, reproducibility and integration tests |
| `results/policy.json` | Trained Q-table, visit counts and training provenance |
| `results/evaluation.json` | Complete measured results and approximate intervals |
| `results/metrics.csv` | One row per policy and evaluation seed |
| `results/round_metrics.json` | Round-by-round reward, satisfaction, retention and fairness |
| `results/demo.json` | Full module-by-module demonstration trace |
| `docs/FINAL_REPORT.pdf` | Evaluation report |
| `docs/FINAL_REPORT.md` | Editable report source |
| `docs/EVALUATION_GUIDE.md` | Live demonstration and viva preparation |
| `docs/REQUIREMENTS.md` | Brief-to-implementation traceability |

All runtime dependencies are Python's standard library. `requirements.txt` documents that fact. Optional report generation uses ReportLab and is separate from the runnable project: `python3 tools/build_report.py` rebuilds the report and charts from saved results when ReportLab is installed.

## How the modules connect

1. The planner chooses the next campaign action from actual observed campaign facts. Clarification and rest are real non-recommendation turns.
2. Q-learning chooses a ranking strategy: popular, personalized, diverse, fair or exploratory. Fixed baselines choose a fixed strategy.
3. BFS discovers candidate items and returns paths through user, category, item, merchant and review nodes.
4. Bayesian evidence produces preference estimates, purchase probabilities and risk information.
5. The CSP filters mandatory constraints and searches an unordered slate. It optimizes item scores minus explicit shortfalls in soft category and merchant coverage targets.
6. Simulated feedback changes beliefs, histories, satisfaction, stock, popularity and merchant exposure. The planner checks observed effects and replans when needed. Training updates Q-values; evaluation never updates them.

Search is exact **within a maximum 12-item candidate pool**. The pool preserves the top three scored items, category representatives and merchant representatives before truncation. It does not claim global catalog optimality. `solve_slate(..., pool_limit=None)` disables this approximation for small exhaustive checks.

The A* planner is optimal only in its small deterministic symbolic model. It does not assume users will follow a plan: a failed action does not acquire its expected effects. A familiar step requires positive feedback in the current favored category; introduction requires positive feedback in an alternative category; comparison requires engagement with a slate of at least two items. Q-learning uses a coarse observed state and is not guaranteed to find an optimal policy in this partially observed, changing simulation.

## Inputs and sample output

Input records are included in `data/catalog.json`: 6 users, 5 initial merchants, 40 products and 4 categories. A deterministic scenario adds a sixth merchant and four products. Edit the fixture for different budgets, regions, allowed categories, prices, stock or declared interests; rebuild it with `python3 data/generate_catalog.py` to restore the original fixture.

`recommend` prints item IDs, graph paths, a campaign plan, belief predictions, constraint rejection counts, soft-objective shortfalls and fixed-template explanations. For budget 1, the result has `status: "infeasible"`, no items and an explicit hard-constraint shortfall. This is intentional.

The full demo includes `events`, `metrics` and a chronological `trace`. Each recommendation trace includes `decision.paths`, `decision.predictions`, `decision.slate`, `decision.plan`, observations and updated beliefs. Inspect `results/demo_excerpt.json` for a compact saved example.

## Evaluation interpretation

See `results/summary.md` and the report for measured comparisons. The suite compares popular-only ranking, highest predicted purchase probability, fixed diversity, learned Q-learning, and two ablations (no planning; no belief updates). All share hard constraints. The three fixed baselines share the same planner; popular and personalized baselines use their named ranking score without campaign score bonuses. The fixed-diversity policy uses the same campaign bonuses as the integrated model.

Metrics include cumulative reward, reward per decision, click/purchase rates, final satisfaction, retention, within-slate category ratio, category entropy, novelty, Jain merchant exposure equality, item exposure HHI, repetition, purchase Brier score and hard-constraint violations. Intervals are approximate and calculated across independent seeds, not individual impressions. Retention is simulated continued participation, not evidence from real people.

The learner is not required to win every metric. Read the paired differences and limitations rather than treating a single aggregate score as proof of real-world superiority. The no-planning/no-updates ablations apply a policy trained with those components, so part of their effect can be state-distribution shift.

## Scope and provenance

See `data/DATA_CARD.md`, `docs/ACKNOWLEDGMENTS.md` and `docs/CONTRIBUTIONS.md`. Code and synthetic data were created with AI coding assistance for this request. No real users, purchases, reviews or human study were used. Research inspired the formulation; no referenced project's source code, weights or datasets were copied. Team member names and individual contribution claims require the team's own truthful confirmation before submission.
