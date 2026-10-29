# Synthetic dataset card

This original fixture was generated for this project; no real or external user data is present. `generate_catalog.py` reconstructs `catalog.json` exactly. There are 6 users, 5 initial merchants, 40 items, 4 categories (books, tech, home, sports), two delivery regions and budgets from 24 to 85 arbitrary currency units. User u4 has a mandatory books/home allow-list. Merchant m4 delivers only to north.

User preferences and merchant fulfillment quality are sampled deterministically by the simulator from each environment seed. Public declared interests are distinct from hidden affinity vectors. The platform never receives those vectors or hidden quality. Satisfaction is an observed simulator feedback proxy derived from synthetic utility; it is not a human survey. Attention and gift flags are observable annotations in this experiment and would often be unavailable in practice.

Training: seeds 500000-500199, 200 episodes, up to 24 rounds each. Test: 12 independent seeds 100-111. Demo: seed 42. Policy-driven feedback and popularity updates create different trajectories across policies. Random draws are keyed by seed/round/user/item/event, so the same counterfactual event shares noise across policies.

The fixture deliberately introduces popularity imbalance and delivery constraints. The scenario adds a new merchant at round 10, changes budgets and interests, changes prices and stock, injects one verified negative review and allows rule-based merchant responses. All values are hand-designed simulation assumptions, not population estimates. There is no claim that the data are representative, demographically fair or behaviorally validated.

The response model uses attention probability 0.88; click propensity is a function of hidden affinity, price-independent relevance, rank and repetition; purchases additionally depend on price and hidden merchant quality. Gifts occur in 12% of purchases. Satisfaction follows an exponential moving average of hidden affinity-based utility. Dropout starts after round 5 when satisfaction is below 0.4. See `simulation.py` for the exact equations. Counterfactual pairing is imperfect because rank, inventory and histories are policy-dependent.

No separate validation set or hyperparameter search was used. All algorithm and reward constants were hand-set before the final reported run. Results assess this designed simulator, not a production recommender or the cited papers' datasets.
