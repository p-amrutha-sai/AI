# Measured synthetic evaluation

| Policy | reward_per_decision | click_rate | purchase_rate | satisfaction | retention | slate_diversity | merchant_jain | item_hhi | purchase_brier |
|---|---|---|---|---|---|---|---|---|---|
| popular | 0.3279 | 0.2857 | 0.0920 | 0.3534 | 0.7917 | 0.7707 | 0.1909 | 0.1482 | 0.0865 |
| personalized | 0.5103 | 0.3576 | 0.1172 | 0.4977 | 0.9583 | 0.5176 | 0.5688 | 0.0590 | 0.1117 |
| fixed_diversity | 0.6608 | 0.3579 | 0.1103 | 0.4529 | 0.9861 | 0.6659 | 0.5827 | 0.0487 | 0.1039 |
| learned | 0.6935 | 0.3696 | 0.1158 | 0.4943 | 1.0000 | 0.6029 | 0.7517 | 0.0382 | 0.1087 |
| no_planning | 0.7293 | 0.3793 | 0.1122 | 0.4949 | 0.9583 | 0.6093 | 0.6358 | 0.0433 | 0.1072 |
| no_belief_updates | 0.6071 | 0.3330 | 0.1064 | 0.4317 | 0.9861 | 0.6322 | 0.7861 | 0.0483 | 0.1012 |

See evaluation.json for seed-level values, approximate intervals and paired differences. All policies share the same hard constraints and symbolic planner except the explicit no_planning ablation.
