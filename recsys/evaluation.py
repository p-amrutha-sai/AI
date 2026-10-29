"""Frozen-policy held-out evaluations and seed-level paired uncertainty."""
from pathlib import Path
import csv
import json
import math
import statistics
from .simulation import Simulator
from .rl import QPolicy

VARIANTS = {
    "popular": {"approach": "popular"},
    "personalized": {"approach": "personalized"},
    "fixed_diversity": {"approach": "diverse"},
    "learned": {"approach": "learned"},
    "no_planning": {"approach": "learned", "planning": False},
    "no_belief_updates": {"approach": "learned", "uncertainty": False},
}


def confidence(values):
    mean = statistics.mean(values)
    # Approximate normal interval across independent seeds, not impressions.
    se = statistics.stdev(values)/math.sqrt(len(values)) if len(values) > 1 else 0
    return {"mean": mean, "std": statistics.stdev(values) if len(values)>1 else 0,
            "approx_95_ci": [mean-1.96*se, mean+1.96*se]}


def evaluate(policy_path, output, seeds=range(100, 112), rounds=24, progress=None):
    policy = QPolicy.load(policy_path)
    seeds = list(seeds)
    training_start = policy.metadata.get("environment_seed_start", 500_000)
    training_stop = policy.metadata.get("environment_seed_stop_exclusive", training_start)
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("Evaluation seeds must be non-empty and unique")
    if any(training_start <= seed < training_stop for seed in seeds):
        raise ValueError("Evaluation seeds overlap policy training seeds")
    destination = Path(output)
    destination.mkdir(parents=True, exist_ok=True)
    rows, learning_curves, event_counts = [], [], {}
    for name, config in VARIANTS.items():
        if progress:
            progress(name)
        for seed in seeds:
            simulator = Simulator(seed=seed, policy=policy, **config)
            metrics = simulator.run(rounds)
            metrics["policy"] = name
            rows.append(metrics)
            learning_curves.extend({**point, "policy": name, "seed": seed} for point in simulator.round_metrics)
            event_counts[f"{name}:{seed}"] = dict(__import__('collections').Counter(e["kind"] for e in simulator.events))
    numeric = [k for k,v in rows[0].items() if isinstance(v, (int, float)) and k != "seed"]
    aggregate = {name: {metric: confidence([row[metric] for row in rows if row["policy"] == name])
                        for metric in numeric} for name in VARIANTS}
    paired = {}
    for baseline in ("popular", "personalized", "fixed_diversity", "no_planning", "no_belief_updates"):
        paired[baseline] = {}
        for metric in ("reward_per_decision", "satisfaction", "merchant_jain", "retention"):
            learned = {r["seed"]: r[metric] for r in rows if r["policy"] == "learned"}
            differences = [learned[r["seed"]]-r[metric] for r in rows if r["policy"] == baseline]
            paired[baseline][metric] = confidence(differences)
    result = {"settings": {"seeds": seeds, "rounds": rounds, "users": 6,
                "training_seeds": [training_start, training_stop], "frozen_policy": True,
                "interval": "normal approximation over independent seed-level values; exploratory, no multiple-testing correction",
                "fairness": "unweighted Jain exposure equality, not relevance-conditioned fairness",
                "synthetic": True}, "aggregate": aggregate, "paired_differences": paired,
                "runs": rows, "scenario_event_counts": event_counts}
    (destination/"evaluation.json").write_text(json.dumps(result, indent=2)+"\n")
    (destination/"round_metrics.json").write_text(json.dumps(learning_curves, indent=2)+"\n")
    with (destination/"metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, ["policy", "seed"]+numeric)
        writer.writeheader()
        writer.writerows({key: row[key] for key in ["policy", "seed"]+numeric} for row in rows)
    columns = ("reward_per_decision", "click_rate", "purchase_rate", "satisfaction", "retention", "slate_diversity", "merchant_jain", "item_hhi", "purchase_brier")
    table = ["| Policy | "+" | ".join(columns)+" |", "|---|"+"---|"*len(columns)]
    for name in VARIANTS:
        table.append("| "+name+" | "+" | ".join(f"{aggregate[name][m]['mean']:.4f}" for m in columns)+" |")
    (destination/"summary.md").write_text("# Measured synthetic evaluation\n\n"+"\n".join(table)+
        "\n\nSee evaluation.json for seed-level values, approximate intervals and paired differences. All policies share the same hard constraints and symbolic planner except the explicit no_planning ablation.\n")
    return result
