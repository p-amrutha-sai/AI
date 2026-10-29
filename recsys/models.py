"""Public platform records. Hidden response parameters live only in simulation.py."""
from dataclasses import dataclass, field
from pathlib import Path
import json


@dataclass
class Item:
    id: str
    name: str
    category: str
    merchant: str
    price: float
    stock: int
    regions: list[str]
    popularity: int = 0
    claimed_quality: float = 0.8
    description_version: int = 1


@dataclass
class Merchant:
    id: str
    name: str
    exposure: int = 0
    responses: int = 0


@dataclass
class User:
    id: str
    budget: float
    region: str
    declared_category: str
    allowed_categories: list[str] = field(default_factory=list)
    recent: list[str] = field(default_factory=list)
    recent_categories: list[str] = field(default_factory=list)
    satisfaction: float = 0.65
    active: bool = True
    last_feedback: str = "none"
    turns: int = 0


@dataclass
class Interaction:
    user: str
    item: str
    kind: str
    round: int
    rating: float | None = None
    verified: bool = False
    gift: bool = False
    attentive: bool | None = None
    fulfilled: bool | None = None


@dataclass
class Catalog:
    categories: list[str]
    related_categories: list[list[str]]
    items: dict[str, Item]
    merchants: dict[str, Merchant]
    users: dict[str, User]
    interactions: list[Interaction] = field(default_factory=list)


def load_catalog(path: str | Path | None = None) -> Catalog:
    source = Path(path) if path else Path(__file__).resolve().parent.parent / "data/catalog.json"
    d = json.loads(source.read_text())
    catalog = Catalog(d["categories"], d["related_categories"],
                      {x["id"]: Item(**x) for x in d["items"]},
                      {x["id"]: Merchant(**x) for x in d["merchants"]},
                      {x["id"]: User(**x) for x in d["users"]})
    if not catalog.categories or not catalog.users or not catalog.items:
        raise ValueError("Catalog requires categories, users and items")
    for item in catalog.items.values():
        if item.category not in catalog.categories or item.merchant not in catalog.merchants:
            raise ValueError(f"Invalid references for {item.id}")
        if item.price < 0 or item.stock < 0 or not 0 <= item.claimed_quality <= 1:
            raise ValueError(f"Invalid values for {item.id}")
    for user in catalog.users.values():
        if user.budget < 0 or user.declared_category not in catalog.categories:
            raise ValueError(f"Invalid user {user.id}")
    return catalog
