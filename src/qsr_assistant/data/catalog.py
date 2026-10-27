"""Fictional QSR catalog: menu, promos and ingredient bill-of-materials.

Everything here is synthetic. It is *inspired by* quick-service-restaurant apps and is
not affiliated with, or derived from, any real company's data.
"""
from __future__ import annotations

INGREDIENT_UNITS = {
    "bun": "pcs", "beef_patty": "pcs", "chicken_patty": "pcs", "fish_fillet": "pcs",
    "veg_patty": "pcs", "cheese_slice": "pcs", "lettuce": "kg", "tomato": "kg",
    "pickles": "kg", "potato": "kg", "nuggets": "pcs", "hashbrown": "pcs",
    "cola_syrup": "L", "coffee_beans": "kg", "milk": "L", "ice_cream_mix": "L",
    "chocolate_sauce": "L", "muffin": "pcs",
}

ALLERGENS = {
    "bun": "gluten", "cheese_slice": "milk", "fish_fillet": "fish", "nuggets": "gluten",
    "milk": "milk", "ice_cream_mix": "milk", "muffin": "gluten, egg",
    "hashbrown": "gluten",
}


def _m(id, name, category, price, pop, tags, desc, ing):
    return dict(id=id, name=name, category=category, price=price, popularity=pop,
                tags=tags, desc=desc, ingredients=ing)


MENU = [
    _m(1, "Classic Burger", "burger", 4.5, 10, ["beef", "kids"],
       "Grilled beef patty with pickles, lettuce and tomato on a toasted bun.",
       {"bun": 1, "beef_patty": 1, "lettuce": 0.03, "tomato": 0.03, "pickles": 0.02}),
    _m(2, "Double Cheese Burger", "burger", 6.2, 9, ["beef", "cheese"],
       "Two beef patties, two slices of melted cheese and pickles.",
       {"bun": 1, "beef_patty": 2, "cheese_slice": 2, "pickles": 0.02}),
    _m(3, "Spicy Chicken Burger", "burger", 5.4, 8, ["chicken", "spicy"],
       "Crispy spicy chicken fillet with lettuce and chilli mayo.",
       {"bun": 1, "chicken_patty": 1, "lettuce": 0.03}),
    _m(4, "Crispy Fish Burger", "burger", 5.6, 3, ["fish"],
       "Crumbed fish fillet with tartare sauce and a slice of cheese.",
       {"bun": 1, "fish_fillet": 1, "cheese_slice": 1}),
    _m(5, "Veggie Delight Burger", "burger", 4.8, 4, ["vegetarian"],
       "Plant-based patty with lettuce, tomato and herb mayo.",
       {"bun": 1, "veg_patty": 1, "lettuce": 0.03, "tomato": 0.03}),
    _m(6, "Chicken Nuggets 6pc", "sides", 4.2, 7, ["chicken", "shareable", "kids"],
       "Six golden chicken nuggets with a dipping sauce of your choice.",
       {"nuggets": 6}),
    _m(7, "Large Fries", "sides", 3.4, 12, ["vegetarian", "shareable"],
       "Hot, crispy salted fries in a large portion.",
       {"potato": 0.22}),
    _m(8, "Medium Fries", "sides", 2.8, 8, ["vegetarian"],
       "Hot, crispy salted fries in a medium portion.",
       {"potato": 0.15}),
    _m(9, "Hash Brown", "sides", 1.5, 3, ["vegetarian", "breakfast"],
       "Crispy golden shredded potato hash brown, served at breakfast.",
       {"hashbrown": 1}),
    _m(10, "Cola", "drinks", 2.2, 11, ["cold"],
       "Chilled fountain cola served over ice.",
       {"cola_syrup": 0.05}),
    _m(11, "Iced Coffee", "drinks", 3.2, 5, ["cold", "coffee"],
       "Cold brewed coffee with milk and ice, a refreshing pick on hot days.",
       {"coffee_beans": 0.02, "milk": 0.15}),
    _m(12, "Hot Coffee", "drinks", 2.5, 5, ["hot", "coffee", "breakfast"],
       "Freshly brewed hot coffee, perfect for cool mornings.",
       {"coffee_beans": 0.02, "milk": 0.05}),
    _m(13, "Soft Serve Cone", "dessert", 1.8, 6, ["cold", "sweet", "kids"],
       "Creamy vanilla soft serve in a crispy cone.",
       {"ice_cream_mix": 0.1}),
    _m(14, "Chocolate Sundae", "dessert", 3.4, 4, ["cold", "sweet"],
       "Vanilla soft serve topped with rich chocolate sauce.",
       {"ice_cream_mix": 0.15, "chocolate_sauce": 0.03}),
    _m(15, "Breakfast Muffin", "breakfast", 3.8, 3, ["breakfast", "hot"],
       "Toasted muffin with egg and melted cheese, served until 11am.",
       {"muffin": 1, "cheese_slice": 1}),
]

PROMOS = [
    dict(id="FRIES20", name="Fries Friday", item_id=7, tier="all",
         text="20% off a Large Fries when you order any burger. Valid on Fridays, all day.",
         terms="One redemption per order. Not combinable with other fries offers."),
    dict(id="COMBO5", name="Double Cheese Combo", item_id=2, tier="all",
         text="Save $1.50 on a Double Cheese Burger combo (burger + fries + drink).",
         terms="Valid daily from 11am. Medium fries and regular drink included."),
    dict(id="CONE1", name="Cone for a Dollar", item_id=13, tier="all",
         text="Soft Serve Cone for $1 with any order over $8.",
         terms="One cone per order. Valid in the app only."),
    dict(id="MORNING", name="Morning Coffee Deal", item_id=12, tier="all",
         text="Hot Coffee at half price between 6am and 10am.",
         terms="Valid on weekdays. One coffee per customer per day."),
    dict(id="GOLD2X", name="Gold Double Points", item_id=None, tier="Gold",
         text="Gold and Platinum rewards members earn 2x points on weekend orders.",
         terms="Points post within 24 hours. Excludes delivery fees."),
    dict(id="FAMILY", name="Family Bundle", item_id=1, tier="all",
         text="4 burgers, 2 Large Fries and 4 drinks for a fixed price of $22.",
         terms="Choose any 4 burgers except Double Cheese. Dine-in and pickup only."),
]

OFFER_IDS = [p["id"] for p in PROMOS]
ITEM_BY_ID = {m["id"]: m for m in MENU}
PROMO_BY_ID = {p["id"]: p for p in PROMOS}


def item_mix() -> dict[int, float]:
    """Share of items sold, proportional to popularity."""
    total = sum(m["popularity"] for m in MENU)
    return {m["id"]: m["popularity"] / total for m in MENU}


def ingredients_per_order(avg_items_per_order: float = 2.4) -> dict[str, float]:
    """Expected ingredient consumption per order given the item mix."""
    mix = item_mix()
    out: dict[str, float] = {k: 0.0 for k in INGREDIENT_UNITS}
    for m in MENU:
        for ing, qty in m["ingredients"].items():
            out[ing] += mix[m["id"]] * qty * avg_items_per_order
    return out


# Approximate calories per serving (synthetic, used in the knowledge base)
CALORIES = {1: 520, 2: 740, 3: 560, 4: 480, 5: 450, 6: 270, 7: 380, 8: 290, 9: 140,
            10: 150, 11: 120, 12: 60, 13: 190, 14: 340, 15: 410}

_STORE_NAMES = ["Riverside", "Harbor Point", "Maple Grove", "Old Town", "Lakeview", "Hillcrest",
                "Central Plaza", "Airport Road", "University Avenue", "Greenfield", "Sunset Mall",
                "Northgate", "Eastwood", "Westbrook", "Southpark", "Cedar Heights", "Bayview",
                "Oakridge", "Ironbridge", "Stonebridge"]
_STREETS = ["Market St", "Station Rd", "Park Ave", "High St", "Mill Lane", "Canal Way",
            "Orchard Rd", "Bridge St", "Church Rd", "Quarry Lane"]


def _store(i: int) -> dict:
    name = _STORE_NAMES[i - 1]
    open24 = i in (4, 9, 15)
    return dict(
        id=i, name=name, address=f"{10 + i * 7} {_STREETS[i % len(_STREETS)]}",
        phone=f"555-01{i:02d}", drive_thru=(i % 3 != 0), open_24h=open24,
        open_hour=0 if open24 else (6 if i % 2 else 7),
        close_hour=24 if open24 else (23 if i % 2 else 22),
        play_area=(i % 4 == 1), parking=20 + (i * 3) % 40, delivery_km=3 + i % 4,
        ev_charging=(i % 5 == 0))


STORES = [_store(i) for i in range(1, 21)]
STORE_BY_ID = {s["id"]: s for s in STORES}
