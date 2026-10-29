"""Generate the original synthetic fixture. No external data or copied code."""
from pathlib import Path
import json

categories = ["books", "tech", "home", "sports"]
names = {
    "books": ["Learning journal", "Puzzle collection"],
    "tech": ["USB task light", "Compact keyboard"],
    "home": ["Desk organizer", "Reusable bottle"],
    "sports": ["Resistance band", "Training mat"],
}
items = []
for merchant in range(5):
    for category_index, category in enumerate(categories):
        for variant in range(2):
            price = [8, 28, 14, 18][category_index] + 4*variant + 2*merchant
            items.append({"id": f"i{merchant}{category_index}{variant}",
                "name": names[category][variant]+f" - seller {merchant}",
                "category": category, "merchant": f"m{merchant}", "price": float(price),
                "stock": 80, "regions": ["north", "south"] if merchant != 4 else ["north"],
                "popularity": 120 if merchant == 0 else 5+merchant*2+variant,
                "claimed_quality": 0.85 if merchant != 3 else 0.98})
users = [{"id": f"u{n}", "budget": [32,45,65,85,24,40][n],
          "region": "north" if n%2 == 0 else "south", "declared_category": categories[n%4],
          "allowed_categories": ["books", "home"] if n == 4 else []} for n in range(6)]
data = {"categories": categories,
        "related_categories": [["books","tech"],["tech","home"],["home","sports"],["sports","books"]],
        "merchants": [{"id": f"m{n}", "name": f"Synthetic seller {n}"} for n in range(5)],
        "items": items, "users": users}
Path(__file__).with_name("catalog.json").write_text(json.dumps(data, indent=2)+"\n")
