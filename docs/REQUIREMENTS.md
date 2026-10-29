# Requirements and evidence

The two supplied PDFs are source material for project requirements. The direct user request controls scope: finish all modules except Generative AI. The brief's Module 5 is therefore recorded as excluded, not silently counted as complete.

| Requirement | Implementation and demonstration evidence |
|---|---|
| Users, items, merchants, categories, reviews, interactions | `models.py`; typed nodes and historical edges in `search.py` |
| Relevant and alternative products, category paths | Breadth-first search up to four edges; saved `decision.paths` |
| Mandatory vs platform objectives | Hard budget/stock/region/allow-list/repetition filters; explicit soft coverage penalties |
| Explain incompatible objectives | `SlateResult.status`, `explanations` and `rejected`; budget-1 CLI example |
| Multi-step campaigns | A* state-space planner; ask, familiar, introduce, compare, lower-cost and wait actions |
| Preconditions and expected effects | `Facts`, `Action.requires` and `Action.effects`; failed effects remain false |
| Replan after rejection, changed budget, bad review or interest change | `CampaignController` monitoring; demo `decision.plan.reasons` and scenario events |
| Preferences and purchase probability | Discounted category-level Beta evidence and purchase propensity in `uncertainty.py` |
| Merchant/review/information reliability | Separate fulfillment, reviewer and claim-credibility beliefs |
| Click curiosity, gifts, ambiguous silence, small samples | Fractional click evidence, gift discounts, attention-aware silence, nontrivial priors and round-wise decay |
| Sequential learned policy | Q-learning over five slate-ranking strategies, delayed rewards and terminal dropout |
| Evolving state | Observed preference/uncertainty, satisfaction, budget, repetition, merchant imbalance, popularity concentration, campaign intent |
| Compare fixed approaches | Popular-only, highest predicted purchase probability and fixed-diversity baselines |
| Long-term effects | 24 rounds; satisfaction, retention, category entropy, repetition, HHI, merchant exposure and cumulative rewards |
| Integrated pipeline | `Platform.recommend`, `Platform.observe`, `Simulator.run`; full trace in `results/demo.json` |
| User/merchant dynamics | Programmed simulator responses, purchases/reviews, offers, product description versions, stock and new products |
| Reproducible code | Standard-library-only runtime, included fixture/Q-table, documented seeds, `run_all.py`, correctness tests |
| Report, README, demo | `README.md`, `FINAL_REPORT.pdf`/`.md`, `EVALUATION_GUIDE.md`, saved outputs |
| Research adaptation | `ACKNOWLEDGMENTS.md` and report explain adopted ideas and differences from primary sources |
| Contribution disclosure | `CONTRIBUTIONS.md` accurately separates delivered assistance from unknown individual human work |
| Generative AI | Excluded by the user's request; no LLM agents, generated-language interfaces or model calls |

All code modules are complete within the documented synthetic formulation. External validation with real shoppers, production deployment and a web interface are outside this implementation. A CLI is expressly acceptable in the course guidelines.
