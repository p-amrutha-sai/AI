"""A* over symbolic campaign states, with monitored execution and replanning."""
from dataclasses import dataclass, asdict
from heapq import heappush, heappop
from itertools import count


@dataclass(frozen=True)
class Facts:
    known: bool = False
    supported: bool = False
    introduced: bool = False
    compared: bool = False
    engaged: bool = False
    affordable: bool = True
    rested: bool = True


@dataclass(frozen=True)
class Action:
    name: str
    requires: tuple[str, ...]
    effects: tuple[str, ...]
    cost: float

    def applicable(self, state: Facts):
        return all(getattr(state, key) for key in self.requires) and any(
            not getattr(state, key) for key in self.effects)

    def apply(self, state: Facts):
        values = asdict(state)
        values.update({key: True for key in self.effects})
        return Facts(**values)


ACTIONS = (
    Action("ask_preference", (), ("known",), 0.8),
    Action("lower_cost_alternative", ("known",), ("affordable",), 0.7),
    Action("wait", (), ("rested",), 0.6),
    Action("familiar", ("known", "affordable", "rested"), ("supported", "engaged"), 1.0),
    Action("introduce_category", ("known", "supported", "affordable", "rested"),
           ("introduced", "engaged"), 1.2),
    Action("compare", ("known", "introduced", "affordable", "rested"), ("compared",), 0.9),
)
ACTION_MAP = {a.name: a for a in ACTIONS}


def goal(state: Facts):
    return all(asdict(state).values())


def plan_campaign(initial: Facts) -> dict:
    """Optimal in the small deterministic abstraction, not in the real world.

    h=0.6*ceil(missing/2) is admissible: every action costs >=0.6 and
    changes at most two false goal facts. Observations may invalidate effects.
    """
    def heuristic(state):
        missing = sum(not v for v in asdict(state).values())
        return 0.6 * ((missing + 1) // 2)
    serial = count()
    frontier = [(heuristic(initial), 0.0, next(serial), initial, [])]
    costs = {initial: 0.0}
    expanded = 0
    while frontier:
        _, cost, _, state, sequence = heappop(frontier)
        if cost > costs[state]:
            continue
        expanded += 1
        if goal(state):
            return {"actions": sequence, "cost": round(cost, 3), "expanded": expanded,
                    "initial": asdict(initial), "goal": "all seven campaign facts true"}
        for action in ACTIONS:
            if not action.applicable(state):
                continue
            successor = action.apply(state)
            next_cost = cost + action.cost
            if next_cost < costs.get(successor, float("inf")):
                costs[successor] = next_cost
                heappush(frontier, (next_cost + heuristic(successor), next_cost,
                                    next(serial), successor, sequence + [action.name]))
    return {"actions": [], "cost": None, "expanded": expanded, "status": "unreachable"}


class CampaignController:
    def __init__(self):
        self.facts = {}
        self.pending = {}
        self.snapshots = {}
        self.replans = []
        self.last_trace = {}

    def next_action(self, user, fatigue=False):
        state = self.facts.get(user.id, Facts())
        previous = self.snapshots.get(user.id)
        reasons = []
        if previous and previous[0] != user.budget:
            state = Facts(**{**asdict(state), "affordable": False, "compared": False})
            reasons.append("budget_change")
        if previous and previous[1] != user.declared_category:
            state = Facts(**{**asdict(state), "known": False, "introduced": False, "compared": False})
            reasons.append("new_interest")
        if user.last_feedback in {"reject", "negative_review"}:
            state = Facts(**{**asdict(state), "supported": False, "engaged": False})
            reasons.append(user.last_feedback)
        if fatigue and state.rested:
            state = Facts(**{**asdict(state), "rested": False})
            reasons.append("fatigue")
        if goal(state):
            # Begin another campaign while retaining clarified preferences.
            state = Facts(known=state.known)
            reasons.append("campaign_complete")
        pending = self.pending.get(user.id, [])
        if pending and not ACTION_MAP[pending[0]].applicable(state):
            reasons.append("precondition_changed")
        if not pending or reasons:
            trace = plan_campaign(state)
            pending = trace["actions"][:]
            trace["reasons"] = reasons or ["initial_or_observed_effect_failed"]
            trace["user"] = user.id
            self.last_trace[user.id] = trace
            self.replans.append(trace)
        self.facts[user.id] = state
        self.pending[user.id] = pending
        self.snapshots[user.id] = (user.budget, user.declared_category)
        return pending[0] if pending else "wait"

    def observe(self, user_id, action_name, success):
        if success:
            self.facts[user_id] = ACTION_MAP[action_name].apply(self.facts[user_id])
            self.pending[user_id] = self.pending[user_id][1:]
        else:
            # Do not apply a planned effect when no engagement was observed.
            self.pending[user_id] = []
