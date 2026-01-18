"""Portfolio baskets for multi-pair evaluation."""

from typing import List, Dict


BASKETS: Dict[str, List[str]] = {
    "basket_6": [
        "EURUSD",
        "USDJPY",
        "USDCAD",
        "AUDUSD",
        "EURGBP",
        "AUDJPY",
    ],
    "basket_8": [
        "EURUSD",
        "USDJPY",
        "USDCAD",
        "AUDUSD",
        "EURGBP",
        "AUDJPY",
        "GBPUSD",
        "NZDJPY",
    ],
}


def normalize_basket_name(name: str) -> str:
    key = (name or "").strip().lower().replace("-", "_")
    if key in {"6", "basket6", "basket_6", "sweet_spot_6", "six"}:
        return "basket_6"
    if key in {"8", "basket8", "basket_8", "eight"}:
        return "basket_8"
    return key


def get_basket(name: str) -> List[str]:
    """Return basket by name; raises KeyError if unknown."""
    key = normalize_basket_name(name)
    if key not in BASKETS:
        raise KeyError(f"Unknown basket '{name}'. Available: {', '.join(sorted(BASKETS))}")
    return list(BASKETS[key])


def get_default_basket() -> List[str]:
    return list(BASKETS["basket_6"])
