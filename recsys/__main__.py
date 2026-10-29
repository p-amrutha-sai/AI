from argparse import ArgumentParser
from pathlib import Path
import json
from .models import load_catalog
from .platform import Platform
from .rl import QPolicy
from .simulation import Simulator, train_policy
from .evaluation import evaluate
from .search import Constraints, solve_slate


def main():
    parser = ArgumentParser(description="Group 13 recommendation laboratory - no Generative AI")
    sub = parser.add_subparsers(dest="command", required=True)
    train = sub.add_parser("train", help="Train Q-learning using synthetic sessions")
    train.add_argument("--episodes", type=int, default=200)
    train.add_argument("--rounds", type=int, default=24)
    train.add_argument("--seed", type=int, default=7)
    train.add_argument("--output", default="results/policy.json")
    evaluation = sub.add_parser("evaluate", help="Compare frozen policies on held-out seeds")
    evaluation.add_argument("--policy", default="results/policy.json")
    evaluation.add_argument("--seeds", type=int, default=12)
    evaluation.add_argument("--seed-start", type=int, default=100)
    evaluation.add_argument("--rounds", type=int, default=24)
    evaluation.add_argument("--output", default="results")
    demo = sub.add_parser("demo", help="Run complete scenario and save inspectable traces")
    demo.add_argument("--policy", default="results/policy.json")
    demo.add_argument("--seed", type=int, default=42)
    demo.add_argument("--rounds", type=int, default=24)
    demo.add_argument("--output", default="results/demo.json")
    recommend = sub.add_parser("recommend", help="Show a constrained recommendation")
    recommend.add_argument("--user", default="u0")
    recommend.add_argument("--budget", type=float)
    recommend.add_argument("--region")
    recommend.add_argument("--strategy", choices=["popular", "personalized", "diverse", "fair", "explore"], default="diverse")
    recommend.add_argument("--allowed-categories", nargs="*")
    recommend.add_argument("--total-budget", type=float)
    args = parser.parse_args()
    if hasattr(args, "rounds") and args.rounds < 1:
        parser.error("--rounds must be positive")
    if args.command == "train":
        if args.episodes < 1:
            parser.error("--episodes must be positive")
        policy = train_policy(args.episodes, args.rounds, args.seed,
                              lambda a,b: print(f"Training episode {a}/{b}", flush=True))
        policy.save(args.output)
        print(f"Saved {len(policy.q)} learned states to {args.output}")
    elif args.command == "evaluate":
        if args.seeds < 1:
            parser.error("--seeds must be positive")
        result = evaluate(args.policy, args.output, range(args.seed_start,args.seed_start+args.seeds),
                          args.rounds, lambda name: print(f"Evaluating {name}", flush=True))
        for name, metrics in result["aggregate"].items():
            print(f"{name:20} reward/decision={metrics['reward_per_decision']['mean']:.4f} "
                  f"satisfaction={metrics['satisfaction']['mean']:.4f} violations={metrics['hard_violations']['mean']:.0f}")
    elif args.command == "demo":
        policy = QPolicy.load(args.policy)
        simulator = Simulator(args.seed, policy)
        metrics = simulator.run(args.rounds, keep_trace=True)
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps({"seed": args.seed, "metrics": metrics,
            "events": simulator.events, "trace": simulator.trace}, indent=2)+"\n")
        print(json.dumps(metrics, indent=2))
        print(f"Full module traces: {output}")
    else:
        catalog = load_catalog()
        if args.user not in catalog.users:
            parser.error("Unknown user; choose "+", ".join(catalog.users))
        user = catalog.users[args.user]
        if args.budget is not None:
            if args.budget < 0:
                parser.error("--budget must be nonnegative")
            user.budget = args.budget
        if args.region:
            user.region = args.region
        if args.allowed_categories is not None:
            if any(c not in catalog.categories for c in args.allowed_categories):
                parser.error("Unknown category")
            user.allowed_categories = args.allowed_categories
        platform = Platform(catalog, approach=args.strategy)
        # This command demonstrates a clarified request; demo shows clarification.
        from .planning import Facts
        platform.planner.facts[user.id] = Facts(known=True)
        decision = platform.recommend(user)
        if args.total_budget is not None:
            if args.total_budget < 0:
                parser.error("--total-budget must be nonnegative")
            constraints = Constraints(total_budget=args.total_budget)
            items = list(catalog.items.values())
            scores = {i.id: platform.beliefs.predict(user, i)["purchase_probability"] for i in items}
            from dataclasses import asdict
            result = solve_slate(items, user, scores, constraints, catalog, pool_limit=None)
            decision = asdict(result)
        print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    main()
