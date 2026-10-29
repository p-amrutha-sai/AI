# Contribution record

The group has nine members. Their names and historical contributions were not supplied. This delivery contains AI-assisted implementation, experiment execution and documentation produced in this working session.

Before institutional submission, each actual member should record their name, the files or experiments they implemented/reviewed, an evidence reference (such as a commit or review note), and what they can explain during the viva. Do not claim independent human authorship for AI-assisted material or invent contributions.

The table below is a proposed responsibility and review allocation for all nine members, not a record of work already completed by those people. Replace Member 1-9 with actual names. Each person should complete their allocated work and record the specific review, changes or experiments they actually performed.

| Member | Proposed area | Concrete work and evidence to produce |
|---|---|---|
| Member 1 | Problem formulation and data model | Review users, products, merchants, categories and interactions in `models.py`; validate `data/catalog.json`; document inputs, outputs, decision variables, currency/budget assumptions and dataset limitations. Produce a data-validation note or a tested fixture improvement. |
| Member 2 | Graph search | Review typed graph construction and BFS in `search.py`; trace paths from user interests through categories to products; check the four-hop bound and historical/review edges. Produce annotated example paths and a retrieval test or improvement. |
| Member 3 | Constraint satisfaction | Review backtracking, candidate selection and slate objectives in `search.py`; check budget, stock, delivery, allowed categories and repetition; explain soft diversity/merchant objectives. Produce an independently checked feasible slate and an infeasible case. |
| Member 4 | Recommendation planning | Review A* search, campaign facts, preconditions, costs, effects and monitoring in `planning.py`; demonstrate budget-change and rejection replanning. Produce a plan trace and verify that an unobserved effect is not marked complete. |
| Member 5 | Uncertainty and trust | Review Beta evidence, noisy feedback weights, decay, purchase estimates and merchant/reviewer/claim credibility in `uncertainty.py`. Produce a numerical posterior example and check that purchase/review fulfillment evidence is counted once. |
| Member 6 | Reinforcement learning | Review state bins, five strategies, epsilon-greedy exploration, rewards, delayed credit and Q updates in `rl.py` and `simulation.py`. Reproduce training or conduct a separately labeled training experiment; explain the saved policy and its limits. |
| Member 7 | User/merchant simulation and dynamic scenarios | Review response equations and merchant behavior in `simulation.py`; trace price, budget, interest, stock, review, popularity and new-merchant events. Produce a scenario walkthrough and confirm that hidden preferences are not passed to the policy. |
| Member 8 | Testing and experimental evaluation | Run the test suite and held-out comparison in `evaluation.py`; inspect raw runs and metrics; explain seed separation, paired differences, fairness limitations and negative results. Produce test/experiment logs and a written interpretation. |
| Member 9 | Integration, reproducibility and documentation | Trace `platform.py` and the CLI from request to feedback; run the project from a clean folder; review the README, report and demo guide. Produce a reproducibility record, architecture explanation and live demonstration. |

Suggested final record format after the work is actually completed:

`Name | Work actually completed | Files/experiments reviewed or changed | Evidence reference | What I can explain in the viva`

Studying and independently validating AI-assisted work can be reported as review/validation. A member should claim implementation or experiment authorship only for work they actually performed. Every member should understand the overall pipeline in addition to their assigned area. This allocation is not a claim that effort or individual credit is equal.

The four software modules are implemented and tested. Generative AI remains excluded. See `TEAM_GUIDE.md` for the module explanations and the relationship between the nine responsibilities.
