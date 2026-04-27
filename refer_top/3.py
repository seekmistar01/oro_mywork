import json
import logging
import re
import threading
import time
from collections import defaultdict
from os import getenv
from typing import Any
from urllib.parse import quote_plus

from src.agent.agent_interface import (
    Tool,
    create_dialogue_step,
    execute_tool_call,
)
from src.agent.proxy_client import ProxyClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ── Configuration ────────────────────────────────────────────────────────────

FALLBACKID: str = "0"

_inference_client = ProxyClient(timeout=90, max_retries=10)
_search_client = ProxyClient(timeout=30, max_retries=2)

_product_detail_cache: dict[str, dict] = {}
_oro_request_times: list[float] = []
_oro_rate_limit_lock = threading.Lock()
_ORO_MAX_REQUESTS_PER_MINUTE = 90
_ORO_WINDOW_SECONDS = 60.0
_ORO_MIN_INTERVAL_SECONDS = 0.7

TOOL_CALL_DELAY = 0.5
TOOL_CALL_MAX_RETRIES = 3
TOOL_CALL_BASE_BACKOFF = 1.0
_last_tool_call_time = 0.0

DEFAULT_PARSE_MODEL = "deepseek-ai/DeepSeek-V3.2-TEE"
DEFAULT_CHOOSE_PRODUCT_MODEL = "deepseek-ai/DeepSeek-V3.2-TEE"
FALLBACK_MODEL = "deepseek-ai/DeepSeek-V3.2-TEE"

_SCORING_STOPWORDS_SEQ = [
    "the",
    "a",
    "an",
    "for",
    "with",
    "from",
    "that",
    "this",
    "i",
    "me",
    "my",
    "looking",
    "show",
    "find",
    "want",
    "need",
    "get",
    "finish",
    "buy",
    "also",
    "and",
    "in",
    "is",
    "it",
    "am",
    "im",
    "priced",
    "pesos",
    "php",
    "price",
    "between",
    "than",
    "above",
    "below",
    "more",
    "less",
    "over",
    "under",
    "of",
    "to",
    "or",
    "on",
    "at",
    "by",
    "its",
    "be",
    "can",
    "has",
    "have",
    "will",
    "would",
    "should",
    "item",
    "items",
    "both",
    "these",
    "offering",
    "sells",
    "shop",
    "budget",
    "voucher",
    "discount",
    "first",
    "second",
    "third",
    "made",
    "using",
    "available",
    "support",
    "supports",
    "compatible",
    "please",
    "looking",
    "age",
    "size",
]
_SCORING_STOPWORDS: frozenset[str] = frozenset(_SCORING_STOPWORDS_SEQ)
_SCORING_STOPWORDS_SHOP: frozenset[str] = frozenset(_SCORING_STOPWORDS_SEQ[:-1])

_PRODUCT_PROMPT = """Extract search params as JSON. No markdown.
{"products":[{"keywords":"search query","price_range":"min-max"|null,"service":"official"|"freeShipping"|"COD"|"flashsale"|null}],"is_shop_voucher":bool}
- keywords: product type + brand + material + color + size + use. 3-6 words, include ALL qualifying terms. Additional:
    - If keywords are short, add a related word to make it longer.(e.g., shoe -> shoe products)
- price_range: "100-500", "100-", "0-500". null if none.
- service: LazMall=official, free shipping=freeShipping, COD=COD, flash sale=flashsale. null if none.
JSON only:"""


_SHOP_PROMPT = """Extract search params as JSON. No markdown. Find multi-products.
{"products":[{"keywords":"search query","price_range":"min-max"|null,"service":"official"|"freeShipping"|"COD"|"flashsale"|null}]}
- keywords: product type + brand + material + color + use. 3-6 words, include ALL qualifying terms. Keep full color descriptors including any qualifier. Drop opening/fastening mechanism terms.
- price_range: "100-500", "100-", "0-500". null if none.
- service: LazMall=official, free shipping=freeShipping, COD=COD, flash sale=flashsale. null if none.
- Same store must sell MULTIPLE differents (numbered First/Second/Also)
- Multi-product: one entry per product, preserve order.
JSON only:"""

_VOUCHER_PROMPT = """Extract search params as JSON. No markdown. Find products that fit within a budget after applying a voucher discount.
{"products":[{"keywords":"search query","price_range":"min-max"|null,"service":"official"|"freeShipping"|"COD"|"flashsale"|null}], "voucher": { "voucher_type": "platform|shop", "discount_type": "fixed|percentage", "discount_value": "fixed amount OR percentage number e.g. 42 for 42%", "threshold": "minimum total price for voucher to apply", "cap": "max discount for percentage vouchers, 0 if not mentioned or fixed type", "budget": "the user's maximum budget" }, "is_shop_voucher":bool}
- keywords: product type + brand + material + color + use. 3-6 words, include ALL qualifying terms that are explicitly mentioned in the query, but MUST never include service related terms in keywords.
- price_range: "100-500", "100-", "0-500". null if none.
- service: "LazMall / guaranteed authenticity / quick returns"→"official", "free shipping / free delivery" → "freeShipping", "cash on delivery / COD / payment on delivery" → "COD", "LazFlash / flash deal / limited-time deal" → "flashsale"; null if none. Multiple options available, combine them with ",".
- Multi-product: one entry per product, preserve order. Budget/voucher info are NOT products.
- is_shop_voucher: true if "same shop" voucher.
JSON only:"""

_TASK_EXTRACTION_PROMPTS: dict[str, str] = {
    "product": _PRODUCT_PROMPT,
    "shop": _SHOP_PROMPT,
    "voucher": _VOUCHER_PROMPT,
}

PRODUCT_JUDGE_MAX_RETRIES = 3

_PRODUCT_JUDGE_PROMPT = """You are choosing the best final product candidate for a shopping benchmark.

Pick the ONE candidate that most exactly satisfies the user request.

Priorities:
1. Prefer explicit structured evidence in attributes and sku_options.
2. Compatibility/model, material, function, theme, brand, service, and price constraints all matter.
3. Do not prefer a candidate just because its title is broader, more generic, or contains more common keywords.
4. If one candidate better matches the requested attributes, choose it even if another candidate has the same heuristic score.
5. Treat semantically equivalent value strings as the same match even if formatting differs slightly.
   Minor wording, spacing, punctuation, tokenization, or formatting differences should not change the decision by themselves.
6. Do not over-weight one appealing field when both candidates already satisfy it.
   If multiple candidates match the same color/model/service, prefer the candidate whose title + attributes + sku_options are more consistently aligned overall.
7. Prefer stronger overall agreement across independent constraints over a single more literal-looking phrase.

Return JSON only:
{"best_product_id":"...","reason":"short reason"}"""


_REGEX_STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "from",
    "that",
    "this",
    "are",
    "was",
    "can",
    "has",
    "have",
    "been",
    "will",
    "find",
    "finish",
    "looking",
    "show",
    "want",
    "need",
    "get",
    "buy",
    "product",
    "products",
    "search",
    "same",
    "shop",
    "within",
    "budget",
    "voucher",
    "discount",
    "price",
    "priced",
    "pesos",
    "php",
    "between",
    "than",
    "greater",
    "less",
    "more",
    "under",
    "over",
    "about",
    "also",
    "both",
    "these",
    "them",
    "each",
    "all",
    "any",
    "one",
    "two",
    "three",
    "four",
    "five",
    "offering",
    "sells",
    "using",
    "in",
    "is",
    "it",
    "its",
    "or",
    "at",
    "on",
    "by",
    "be",
    "do",
    "an",
    "my",
    "me",
    "im",
    "items",
    "item",
    "only",
    "just",
    "first",
    "second",
    "supports",
    "support",
    "compatible",
    "available",
    "made",
    "please",
    "like",
    "of",
    "above",
    "deals",
    "options",
    "option",
    "delivery",
    "shipping",
    "offers",
    "lazmall",
    "lazflash",
    "official",
    "cash",
    "payment",
    "pay",
    "cost",
    "costs",
    "via",
    "themed",
    "such",
    "those",
    "store",
    "stores",
    "focus",
    "category",
    "specifically",
    "guaranteed",
    "authenticity",
    "returns",
    "quick",
    "perks",
    "should",
    "help",
    "purchase",
    "type",
    "to",
    "named",
    "called",
    "family",
    "belongs",
    "comes",
    "another",
    "lastly",
    "benefits",
    "you",
    "weighing",
    "capacity",
    "size",
    "sized",
    "eu",
    "fits",
}

_MULTI_PRODUCT_SPLIT_RE = re.compile(
    r"(?:,?\s*and\s+also\s+|,?\s*also,?\s+|Second(?:ly)?,\s*|Third(?:ly)?,\s*"
    r"|First,\s*|\(\d+\)\s*|\d+\.\s*|Additionally,\s*"
    r"|[.]\s*Next,\s*|[.]\s*Lastly,\s*|[.]\s*Finally,\s*|[.]\s*Last,\s*)",
    re.IGNORECASE,
)

_BUDGET_SPLIT_RE = re.compile(r"(?:My budget|budget is|I have a voucher)", re.IGNORECASE)

_COLOR_TERMS = {
    "red",
    "blue",
    "green",
    "black",
    "white",
    "pink",
    "purple",
    "orange",
    "gray",
    "brown",
    "gold",
    "silver",
    "beige",
    "navy",
    "maroon",
    "teal",
    "coral",
    "cream",
    "ivory",
    "khaki",
    "lavender",
    "magenta",
    "olive",
    "rose",
    "tan",
    "turquoise",
    "violet",
}


def _strip_color_keywords(kw: str) -> str:
    """Remove color terms from keywords to broaden title-based search."""
    words = kw.split()
    stripped = [w for w in words if w.lower() not in _COLOR_TERMS]
    return " ".join(stripped) if stripped else kw


# ── ORO API rate limiting ───────────────────────────────────────────────────


def _wait_for_oro_request_slot() -> None:
    """Throttle ORO API requests to stay below the shared per-IP limit."""
    while True:
        sleep_for = 0.0
        with _oro_rate_limit_lock:
            now = time.monotonic()
            cutoff = now - _ORO_WINDOW_SECONDS

            while _oro_request_times and _oro_request_times[0] <= cutoff:
                _oro_request_times.pop(0)

            if _oro_request_times:
                since_last_request = now - _oro_request_times[-1]
                if since_last_request < _ORO_MIN_INTERVAL_SECONDS:
                    sleep_for = max(
                        sleep_for,
                        _ORO_MIN_INTERVAL_SECONDS - since_last_request,
                    )

            if len(_oro_request_times) >= _ORO_MAX_REQUESTS_PER_MINUTE:
                sleep_for = max(
                    sleep_for,
                    _ORO_WINDOW_SECONDS - (now - _oro_request_times[0]),
                )

            if sleep_for <= 0:
                _oro_request_times.append(now)
                return

        time.sleep(sleep_for)


def _oro_get(path: str, params: dict | None = None):
    """Perform a rate-limited GET against the ORO API."""
    _wait_for_oro_request_slot()
    return _search_client.get(path, params)


def safe_tool_call(tool_name: str, params: dict) -> dict:
    """Execute a tool call with a short inter-call delay and exponential backoff on failure."""
    global _last_tool_call_time
    elapsed = time.monotonic() - _last_tool_call_time
    if elapsed < TOOL_CALL_DELAY:
        time.sleep(TOOL_CALL_DELAY - elapsed)

    for attempt in range(TOOL_CALL_MAX_RETRIES):
        try:
            result = execute_tool_call(tool_name, params)
            _last_tool_call_time = time.monotonic()
            return result
        except Exception:
            if attempt == TOOL_CALL_MAX_RETRIES - 1:
                raise
            backoff = TOOL_CALL_BASE_BACKOFF * (2**attempt)
            logger.warning(
                "Tool call %s failed (attempt %d/%d), retrying in %.1fs",
                tool_name,
                attempt + 1,
                TOOL_CALL_MAX_RETRIES,
                backoff,
            )
            time.sleep(backoff)


# ── Registered tools ────────────────────────────────────────────────────────


@Tool
def find_product(
    q: str,
    page: int = 1,
    shop_id: str | None = None,
    price: str | None = None,
    sort: str | None = None,
    service: str | None = None,
) -> list[dict]:
    """Search for products matching query."""
    params = {
        "q": quote_plus(q),
        "page": page,
        "shop_id": shop_id,
        "price": price,
        "sort": sort,
        "service": service,
    }
    if params.get("sort") == "default":
        params.pop("sort")
    if params.get("service") == "default":
        params.pop("service")
    elif params.get("service") and "default" in params["service"]:
        params["service"] = ",".join(x for x in params["service"].split(",") if x != "default")
    result = _oro_get("/search/find_product", params)
    result = result if result is not None else []
    if shop_id and not result:
        retry = dict(params)
        retry.pop("service", None)
        result = _oro_get("/search/find_product", retry) or []
    return result


@Tool
def find_products_in_same_shop(product_queries: str) -> dict:
    """Find multiple products from the SAME shop."""
    try:
        specs = json.loads(product_queries) if isinstance(product_queries, str) else product_queries
    except json.JSONDecodeError:
        return {"found": False, "error": "Invalid JSON"}
    if not specs or not isinstance(specs, list):
        return {"found": False, "error": "Need non-empty list"}

    voucher = {}
    if isinstance(specs[-1], dict) and specs[-1].get("voucher"):
        voucher = json.loads(specs.pop()["voucher"]) or {}

    orig_query = ""
    if isinstance(specs[-1], dict) and specs[-1].get("_original_query"):
        orig_query = specs.pop()["_original_query"]

    broad_results = []
    for spec in specs:
        q = spec.get("q", "")
        params = {"q": quote_plus(q), "page": 1}
        if spec.get("price"):
            params["price"] = spec["price"]
        if spec.get("service"):
            params["service"] = spec["service"]

        p_list = []
        if voucher:
            for page in range(1, 3):
                params["page"] = page
                p_list.extend(_oro_get("/search/find_product", params) or [])
        else:
            p_list = _oro_get("/search/find_product", params) or []
        
        broad_results.append(p_list)

    if not any(broad_results):
        return {"found": False, "error": "No results for any product", "shops_tried": 0}

    shop_coverage = defaultdict(lambda: defaultdict(list))
    for idx, results in enumerate(broad_results):
        for prod in results:
            sid = str(prod.get("shop_id", ""))
            if sid:
                shop_coverage[sid][idx].append(prod)

    def _score_shop(shop_id: str):
        coverage = shop_coverage[shop_id]
        total_score = 0
        for idx, prods in coverage.items():
            q = specs[idx].get("q", "") if idx < len(specs) else ""
            total_score += max((_score_product(p, orig_query or q) for p in prods), default=0)
        return (len(coverage), total_score)

    candidates = sorted(shop_coverage.keys(), key=_score_shop, reverse=True)
    max_shops = 10

    for shop_id in candidates[:max_shops]:
        coverage = shop_coverage[shop_id]
        found = []
        ok = True
        for idx, spec in enumerate(specs):
            q = spec.get("q", "")
            score_q = orig_query or q
            if idx in coverage and coverage[idx]:
                best = _select_best_product(coverage[idx], q or score_q, prefer_cheaper=True)
                if best:
                    found.append(best)
                    continue
            got_it = False
            for _page in (1, 2):
                results = (
                    _oro_get(
                        "/search/find_product",
                        {"q": quote_plus(q), "page": _page, "shop_id": shop_id},
                    )
                    or []
                )
                if results:
                    best = _select_best_product(results, q or score_q, prefer_cheaper=True)
                    if best:
                        found.append(best)
                        got_it = True
                        break
            if not got_it:
                ok = False
                break

        if ok and len(found) == len(specs):
            if voucher:
                prices = [str(p.get("price", 0)) for p in found]
                voucher_result = safe_tool_call("calculate_voucher", {
                    "product_prices": ",".join(prices),
                    "voucher_type": voucher["discount_type"],
                    "discount_value": voucher["discount_value"],
                    "threshold": voucher["threshold"],
                    "budget": voucher["budget"],
                    "cap": voucher["cap"],
                })
                if not (voucher_result.get("result") and voucher_result["result"]["voucher_applied"] and voucher_result["result"]["total_after"] <= voucher["budget"]):
                    ok = False
            if ok:
                return {
                    "found": True,
                    "shop_id": shop_id,
                    "products": [
                        {
                            "product_id": p.get("product_id"),
                            "title": p.get("title", ""),
                            "price": p.get("price"),
                            "shop_id": p.get("shop_id"),
                        }
                        for p in found
                    ],
                    "shops_tried": candidates.index(shop_id) + 1,
                }

    return {
        "found": False,
        "error": f"No shop has all {len(specs)} products",
        "shops_tried": min(len(candidates), max_shops),
    }


@Tool
def calculate_voucher(
    product_prices: str,
    voucher_type: str,
    discount_value: float,
    threshold: float,
    budget: float,
    cap: float = 0,
) -> dict:
    """
    Calculate the final price after applying a voucher discount. Use this for voucher tasks to verify budget.

    Args:
        product_prices: Comma-separated product prices, e.g. "100,50,75"
        voucher_type: "fixed" for fixed discount, "percentage" for percentage discount
        discount_value: The discount amount (e.g. 18 for fixed, 42 for 42% percentage)
        threshold: Minimum total price for voucher to apply
        budget: Maximum budget the user has
        cap: Maximum discount amount for percentage vouchers (0 = no cap)

    Returns:
        Dict with: {total_before, discount_amount, total_after, within_budget, voucher_applied}
    """
    try:
        prices = [float(p.strip()) for p in str(product_prices).split(",")]
    except ValueError:
        return {"error": "Invalid product_prices format. Use comma-separated numbers."}

    total = sum(prices)
    discount = 0.0
    voucher_applied = False

    if total >= threshold:
        voucher_applied = True
        if voucher_type == "fixed":
            discount = discount_value
        elif voucher_type == "percentage":
            discount = total * (discount_value / 100.0)
            if cap > 0:
                discount = min(discount, cap)

    total_after = total - discount

    return {
        "prices": prices,
        "total_before": round(total, 2),
        "discount_amount": round(discount, 2),
        "total_after": round(total_after, 2),
        "within_budget": total_after <= budget,
        "voucher_applied": voucher_applied,
        "budget": budget,
    }


@Tool
def recommend_product(product_ids: str) -> str:
    """Recommend products to the user."""
    return f"Having recommended the products to the user: {product_ids}."


@Tool
def terminate(status: str = "success") -> str:
    """End the dialogue."""
    return f"The interaction has been completed with status: {status}"


# ── Product ranking ─────────────────────────────────────────────────────────


def _fetch_product_details(product_ids: list[str]) -> dict[str, dict]:
    """Fetch and cache product detail records for the given product IDs."""
    if not product_ids:
        return {}
    uncached = [pid for pid in product_ids if pid not in _product_detail_cache]
    for i in range(0, len(uncached), 10):
        batch = uncached[i : i + 10]
        result = _oro_get("/search/view_product_information", {"product_ids": ",".join(batch)})
        if result and isinstance(result, list):
            for p in result:
                _product_detail_cache[str(p.get("product_id", ""))] = p
    return {pid: _product_detail_cache[pid] for pid in product_ids if pid in _product_detail_cache}


def _score_product(product: dict, query_text: str, detail: dict = None) -> int:
    """Score how well a product title (and optionally its attributes) match a query."""
    title = product.get("title", "").lower()
    t_words = set(re.findall(r"\b\w+\b", title))
    q_words = list(
        dict.fromkeys(
            w
            for w in re.findall(r"\b\w+\b", query_text.lower())
            if w not in _SCORING_STOPWORDS and len(w) > 1
        )
    )

    score = 0
    for qw in q_words:
        if (
            qw in t_words
            or qw.endswith("s")
            and qw[:-1] in t_words
            or not qw.endswith("s")
            and (qw + "s") in t_words
            or len(qw) >= 3
            and any(tw.startswith(qw) for tw in t_words if len(tw) > len(qw))
        ):
            score += 2
        elif any(qw.startswith(tw) or tw.startswith(qw) for tw in t_words if len(tw) > 2):
            score += 1
        if any(c.isdigit() for c in qw) and qw in title:
            score += 2

    if detail:
        attr_text = ""
        exact_values = set()
        for k, vs in (detail.get("attributes") or {}).items():
            attr_text += " " + k.replace("_", " ")
            for v in vs if isinstance(vs, list) else [vs]:
                v_str = str(v).strip().lower()
                attr_text += " " + v_str
                exact_values.add(v_str)
        for _sku_id, opts in (detail.get("sku_options") or {}).items():
            if isinstance(opts, dict):
                for k, v in opts.items():
                    v_str = str(v).strip().lower()
                    attr_text += " " + k.replace("_", " ") + " " + v_str
                    exact_values.add(v_str)
        attr_words = set(re.findall(r"\b\w+\b", attr_text.lower()))
        for qw in q_words:
            if qw in exact_values:
                score += 3
            elif (qw + "#") in exact_values:
                score += 5
            elif qw in attr_words:
                score += 2

    return score


def _score_product_for_product_case(
    product: dict, query_text: str, detail: dict = None, prefer_cheaper: bool = False
) -> float:
    """Score how well a product title (and optionally its attributes) match a query."""
    title = product.get("title", "").lower()
    t_words = set(re.findall(r"\b\w+\b", title))
    q_words = list(
        dict.fromkeys(
            w
            for w in re.findall(r"\b\w+\b", query_text.lower())
            if w not in _SCORING_STOPWORDS and len(w) > 1
        )
    )

    score = 0
    for qw in q_words:
        if (
            qw in t_words
            or qw.endswith("s")
            and qw[:-1] in t_words
            or not qw.endswith("s")
            and (qw + "s") in t_words
            or len(qw) >= 3
            and any(tw.startswith(qw) for tw in t_words if len(tw) > len(qw))
        ):
            score += 2
        elif any(qw.startswith(tw) or tw.startswith(qw) for tw in t_words if len(tw) > 2):
            score += 1
        if any(c.isdigit() for c in qw) and qw in title:
            score += 2

    if detail:
        attr_text = ""
        exact_values = set()
        for k, vs in (detail.get("attributes") or {}).items():
            attr_text += " " + k.replace("_", " ")
            for v in vs if isinstance(vs, list) else [vs]:
                v_str = str(v).strip().lower()
                attr_text += " " + v_str
                exact_values.add(v_str)
        for _sku_id, opts in (detail.get("sku_options") or {}).items():
            if isinstance(opts, dict):
                for k, v in opts.items():
                    v_str = str(v).strip().lower()
                    attr_text += " " + k.replace("_", " ") + " " + v_str
                    exact_values.add(v_str)
        attr_words = set(re.findall(r"\b\w+\b", attr_text.lower()))
        for qw in q_words:
            if qw in exact_values:
                score += 3
            elif (qw + "#") in exact_values:
                score += 5
            elif qw in attr_words:
                score += 2

        # SKU precision bonus: prefer products whose SKU option values
        # are tighter matches to query words (fewer extraneous words).
        # This helps distinguish near-identical products where one SKU
        # value has extra prefixes like "for ".
        if prefer_cheaper:
            score -= (product.get("price", 0) or 0) / 100_000
            q_word_set = set(q_words)
            best_precision = 0.0
            for _sku_id, opts in (detail.get("sku_options") or {}).items():
                if isinstance(opts, dict):
                    val_words = set()
                    for v in opts.values():
                        for w in re.findall(r"\b\w+\b", str(v).lower()):
                            if len(w) > 1:
                                val_words.add(w)
                    if val_words:
                        overlap = len(val_words & q_word_set)
                        precision = overlap / len(val_words)
                        best_precision = max(best_precision, precision)
            score += best_precision * 4

    return score


def max_total_price(
    budget,
    min_required,
    discount_rate,
    discount_cap,
    discount_type: str = "percentage",
):
    if discount_type == "fixed":
        max_price = budget + discount_rate
        if max_price <= min_required:
            return min_required
        return max_price

    # Percentage: normalize LLM-style "42" (42%) to a fraction
    rate = discount_rate / 100.0 if discount_rate > 1 else discount_rate
    if rate <= 0 or rate >= 1:
        return None

    # Without a cap, max cart total such that total * (1 - rate) = budget
    max_price_no_cap = budget / (1 - rate)
    discount_no_cap = max_price_no_cap * rate

    if discount_cap and discount_cap > 0 and discount_no_cap > discount_cap:
        max_price = budget + discount_cap
    else:
        max_price = max_price_no_cap

    if max_price <= min_required:
        return None

    return max_price


def _voucher_feasibility_bonus(p: dict, voucher: dict | None) -> float:
    if not voucher:
        return 0.0

    pr = p.get("price")
    if not isinstance(pr, (int, float)) or pr <= 0:
        return 0.0

    min_price = voucher["threshold"]
    max_price = max_total_price(
        voucher["budget"],
        voucher["threshold"],
        voucher["discount_value"],
        voucher["cap"],
        voucher["discount_type"],
    )

    if pr >= min_price and (not max_price or pr <= max_price):
        return 72.0
    if pr + 1e-6 < min_price:
        return -28.0
    return 0.0


def _select_best_product(
    products: list,
    query_text: str,
    top_count: int = 10,
    prefer_cheaper: bool = False,
    voucher: dict | None = None,
) -> dict | None:
    """Return the product from `products` that best matches `query_text`."""
    if not products:
        return None

    def _prerank(p: dict) -> float:
        return _score_product(p, query_text) + _voucher_feasibility_bonus(p, voucher)

    top = sorted(products, key=_prerank, reverse=True)[:top_count]
    pids = [str(p.get("product_id", "")) for p in top if p.get("product_id")]
    details = _fetch_product_details(pids)

    def _final_score(p: dict) -> float:
        s = _score_product(p, query_text, details.get(str(p.get("product_id", ""))))
        if prefer_cheaper:
            s -= (p.get("price", 0) or 0) / 100_000
        if voucher:
            s += _voucher_feasibility_bonus(p, voucher)
        return s

    return max(top, key=_final_score)


def _parse_json_object_from_llm(content: str) -> dict | None:
    """Parse a single JSON object from LLM text; tolerant of fences and extra prose."""
    cleaned = re.sub(r"```json?\s*", "", content)
    cleaned = re.sub(r"```\s*$", "", cleaned).strip()
    try:
        out = json.loads(cleaned)
        return out if isinstance(out, dict) else None
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", content, re.DOTALL)
        if m:
            try:
                out = json.loads(m.group())
                return out if isinstance(out, dict) else None
            except json.JSONDecodeError:
                return None
    return None


def _extract_judge_json(content: str) -> dict | None:
    """Extract a JSON object from an LLM response, tolerant of code fences."""
    return _parse_json_object_from_llm(content)


def _product_judge_payload(
    product: dict, detail: dict | None, score: float, query_text: str
) -> dict:
    """Build a compact per-candidate payload for the product judge LLM.

    Ranks SKU options by query-word overlap so the most relevant variants
    appear first, and truncates to 8 to keep the prompt focused.
    """
    sku_options = (detail or {}).get("sku_options", {}) or {}
    query_words = set(
        w
        for w in re.findall(r"\b\w+\b", query_text.lower())
        if w not in _SCORING_STOPWORDS and len(w) > 1
    )

    ranked: list = []
    for opt in sku_options.values():
        if not isinstance(opt, dict):
            continue
        opt_words = set(
            w
            for w in re.findall(r"\b\w+\b", " ".join(str(v).lower() for v in opt.values()))
            if len(w) > 1
        )
        ranked.append((len(query_words & opt_words), opt))

    sku_preview: list[dict] = []
    seen_keys = set()
    for _overlap, opt in sorted(ranked, key=lambda item: item[0], reverse=True):
        key = json.dumps(opt, sort_keys=True, ensure_ascii=False)
        if key not in seen_keys:
            seen_keys.add(key)
            sku_preview.append(opt)

    return {
        "product_id": str(product.get("product_id", "")).strip(),
        "title": product.get("title", ""),
        "price": product.get("price"),
        "service": product.get("service", []),
        "attributes": (detail or {}).get("attributes", {}),
        "sku_options_preview": sku_preview[:8],
        "heuristic_score": score,
    }


def _llm_choose_tied_product(
    query_text: str,
    tied_candidates: list,
    details: dict[str, dict],
) -> dict | None:
    if len(tied_candidates) < 2:
        return tied_candidates[0][0] if tied_candidates else None

    payload = {
        "request": query_text,
        "candidates": [
            _product_judge_payload(p, details.get(str(p.get("product_id", ""))), s, query_text)
            for p, s in tied_candidates[:5]
        ],
    }
    user_content = json.dumps(payload, ensure_ascii=False)

    env_model = getenv("SANDBOX_MODEL")
    model_chain = [env_model] if env_model else [DEFAULT_CHOOSE_PRODUCT_MODEL, FALLBACK_MODEL]

    for model in model_chain:
        for attempt in range(1, PRODUCT_JUDGE_MAX_RETRIES + 1):
            result = _inference_client.post(
                "/inference/chat/completions",
                json_data={
                    "model": model,
                    "temperature": 0,
                    "stream": False,
                    "messages": [
                        {"role": "system", "content": _PRODUCT_JUDGE_PROMPT},
                        {"role": "user", "content": user_content},
                    ],
                },
            )
            if not (result and result.get("choices")):
                logger.warning(
                    "ProductJudge: %s no response (attempt %d/%d)",
                    model,
                    attempt,
                    PRODUCT_JUDGE_MAX_RETRIES,
                )
                continue

            content = result["choices"][0].get("message", {}).get("content", "")
            parsed = _extract_judge_json(content)
            if not isinstance(parsed, dict):
                logger.warning(
                    "ProductJudge: %s invalid JSON (attempt %d/%d)",
                    model,
                    attempt,
                    PRODUCT_JUDGE_MAX_RETRIES,
                )
                continue

            best_pid = str(parsed.get("best_product_id", "")).strip()
            for product, _score in tied_candidates[:5]:
                if str(product.get("product_id", "")).strip() == best_pid:
                    logger.info(
                        "ProductJudge: selected pid=%s via %s (attempt %d/%d)",
                        best_pid,
                        model,
                        attempt,
                        PRODUCT_JUDGE_MAX_RETRIES,
                    )
                    return product

            logger.warning(
                "ProductJudge: %s returned unknown pid=%s (attempt %d/%d)",
                model,
                best_pid,
                attempt,
                PRODUCT_JUDGE_MAX_RETRIES,
            )
        logger.warning(
            "ProductJudge: %s exhausted %d retries, trying next model",
            model,
            PRODUCT_JUDGE_MAX_RETRIES,
        )

    logger.warning("ProductJudge: all models exhausted, falling back to heuristic")
    return None


def _select_best_product_for_product_case(
    products: list, query_text: str, top_count: int = 10, prefer_cheaper: bool = False
) -> dict | None:

    if not products:
        return None

    top = sorted(
        products,
        key=lambda p: _score_product_for_product_case(p, query_text, None, prefer_cheaper),
        reverse=True,
    )[:top_count]
    pids = [str(p.get("product_id", "")) for p in top if p.get("product_id")]
    details = _fetch_product_details(pids)

    def _final_score(p: dict) -> float:
        s = _score_product_for_product_case(
            p, query_text, details.get(str(p.get("product_id", ""))), prefer_cheaper
        )
        return s

    scored = [(p, _final_score(p)) for p in top]
    if not scored:
        return None

    best_score = max(s for _, s in scored)

    near_tied = sorted(
        [(p, s) for p, s in scored if s == best_score],
        key=lambda x: x[1],
        reverse=True,
    )
    logger.info(f"near_tied: {[(p.get('product_id', ''), s) for p, s in near_tied]}")

    if len(near_tied) > 2:
        llm_choice = _llm_choose_tied_product(query_text, near_tied, details)
        logger.info(f"llm_choice: {llm_choice}")
        if llm_choice is not None:
            return llm_choice

    return near_tied[0][0]


# ── Query parsing ───────────────────────────────────────────────────────────


def _infer_task_type(query: str) -> str:
    """Heuristically determine whether the query is for a product, shop, or voucher."""
    q = query.lower()
    if "voucher" in q or "budget" in q or "discount" in q:
        return "voucher"
    if "shop" in q and any(w in q for w in ("both", "these", "offering", "sells", "same")):
        return "shop"
    return "product"


def _sanitize_keyword_text(text: str | None) -> str:
    if not text:
        return "product"
    filtered = [
        w
        for w in re.findall(r"\b\w+\b", str(text).lower())
        if w not in _SCORING_STOPWORDS and len(w) > 1
    ]
    if not filtered:
        return "product"
    return " ".join(dict.fromkeys(filtered))


def _sanitize_product_search_params(params: dict) -> dict:
    sanitized = dict(params)
    products: list[dict] = []
    for product in sanitized.get("products", []) or []:
        if not isinstance(product, dict):
            continue
        cleaned = dict(product)
        if "keywords" in cleaned:
            cleaned["keywords"] = _sanitize_keyword_text(cleaned.get("keywords"))
        if "q" in cleaned:
            cleaned["q"] = _sanitize_keyword_text(cleaned.get("q"))
        products.append(cleaned)
    if products:
        sanitized["products"] = products
    return sanitized


def _extract_query_params_llm(query: str, kw_task: str) -> dict:
    system_prompt = _TASK_EXTRACTION_PROMPTS.get(kw_task, _PRODUCT_PROMPT)
    model = getenv("SANDBOX_MODEL", DEFAULT_PARSE_MODEL)
    result = _inference_client.post(
        "/inference/chat/completions",
        json_data={
            "model": model,
            "temperature": 0,
            "stream": False,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query},
            ],
        },
    )
    if result and result.get("choices"):
        content = result["choices"][0].get("message", {}).get("content", "")
        parsed = _parse_json_object_from_llm(content)
        if parsed is not None:
            if kw_task == "product":
                return _sanitize_product_search_params(parsed)
            if kw_task == "shop":
                for p in parsed.get("products", []):
                    if p.get("keywords"):
                        p["keywords"] = " ".join(
                            w
                            for w in p["keywords"].split()
                            if w.lower() not in _SCORING_STOPWORDS_SHOP
                        )
            return parsed
        logger.warning("LLM extraction: %s returned unparseable response, trying next", model)
    else:
        logger.warning("LLM extraction: %s returned no response, trying next", model)

    logger.warning("LLM extraction failed on all models; falling back to regex")
    return _extract_query_params_regex(query)


def _extract_query_params_regex(query: str) -> dict:
    """Regex-based fallback to extract structured search parameters from a query."""
    task_type = _infer_task_type(query)

    def _extract_product_spec(text: str) -> dict:
        alpha_words = [
            w for w in re.findall(r"\b[a-zA-Z]{2,}\b", text.lower()) if w not in _REGEX_STOPWORDS
        ]
        alnum_tokens = re.findall(r"\b\d+[a-zA-Z]+\b|\b[a-zA-Z]+\d+[a-zA-Z]*\b", text.lower())
        words = alpha_words[:6]
        for t in alnum_tokens[:2]:
            if t not in words:
                words.append(t)
        for s in re.findall(r"(\d+)#", text)[:2]:
            if s not in words:
                words.append(s)
        keywords = " ".join(words) or "product"

        price_range = None
        m = re.search(
            r"(?:greater|more|over|above|>|cost[s]?\s+more)\s*(?:than\s*)?(\d+)", text, re.I
        )
        if m:
            price_range = f"{m.group(1)}-"
        else:
            m = re.search(r"(\d{1,6})\s*(?:to|and|-)\s*(\d{1,6})\s*(?:pesos|php)", text, re.I)
            if m:
                price_range = f"{m.group(1)}-{m.group(2)}"
            elif re.search(r"(?:price|pesos|php|cost)", text, re.I):
                m = re.search(r"(\d{1,6})\s+(?:to|and)\s+(\d{1,6})", text)
                if m:
                    price_range = f"{m.group(1)}-{m.group(2)}"

        service = None
        tl = text.lower()
        if "lazmall" in tl or "official" in tl:
            service = "official"
        if "free shipping" in tl or "free delivery" in tl:
            service = "freeShipping" if not service else f"{service},freeShipping"
        if "lazflash" in tl or "flash sale" in tl or "flashsale" in tl:
            service = "flashsale" if not service else f"{service},flashsale"
        if "cash on delivery" in tl or "cod" in tl:
            service = "COD" if not service else f"{service},COD"

        return {"keywords": keywords, "price_range": price_range, "service": service}

    product_text = _BUDGET_SPLIT_RE.split(query)[0].strip()
    if not product_text or len(product_text) < 15:
        product_text = query

    parts = [
        p.strip() for p in _MULTI_PRODUCT_SPLIT_RE.split(product_text) if p and len(p.strip()) > 10
    ]
    if not parts:
        parts = [query]

    products = [_extract_product_spec(p) for p in parts]
    products = [p for p in products if len(p["keywords"].split()) >= 2] or products
    is_shop = task_type == "shop" or (task_type == "voucher" and "same shop" in query.lower())

    return {"task_type": task_type, "products": products, "is_shop_voucher": is_shop}


# ── Execution flows ─────────────────────────────────────────────────────────


def _spec_to_find_product_params(product: dict, *, include_price: bool = True) -> dict[str, Any]:
    """Build `find_product` params from one `products[]` spec."""
    params: dict[str, Any] = {"q": product.get("keywords", "product")}
    if include_price and product.get("price_range"):
        params["price"] = product["price_range"]
    if product.get("service"):
        params["service"] = product["service"]
    return params


def _normalize_voucher_fields(raw: dict | None) -> dict:
    v = raw or {}
    return {
        "discount_type": v.get("discount_type", "percentage"),
        "discount_value": float(v.get("discount_value", 0)),
        "threshold": float(v.get("threshold", 0)),
        "cap": float(v.get("cap", 0)),
        "budget": float(v.get("budget", 0)),
    }


def _deduplicate_products(products: list) -> list:
    """Deduplicate products by product_id, preserving order."""
    seen: set = set()
    out: list = []
    for p in products:
        pid = str(p.get("product_id", ""))
        if pid and pid not in seen:
            seen.add(pid)
            out.append(p)
    return out


def _format_product_ids(ids: list, expected_order: list = None) -> str:
    """Deduplicate IDs, optionally sort by expected order, and join with commas."""
    seen = set()
    out = []
    for pid in ids:
        pid = str(pid).strip()
        if pid and pid not in seen:
            seen.add(pid)
            out.append(pid)
    if expected_order:
        rank = {pid: i for i, pid in enumerate(expected_order)}
        out = sorted(out, key=lambda p: rank.get(p, len(expected_order)))
    return ",".join(out) if out else FALLBACKID


def _append_step(think: str, tool_results: list, response: str, query: str, steps: list) -> None:
    steps.append(create_dialogue_step(think, tool_results, response, query, len(steps) + 1))


def _finish_session(product_ids: list, status: str, query: str, steps: list) -> None:
    """Always recommend before terminating the session."""
    rec = safe_tool_call(
        "recommend_product",
        {"product_ids": _format_product_ids(product_ids)},
    )
    term = safe_tool_call("terminate", {"status": status})
    _append_step("Done.", [rec, term], "Done.", query, steps)


def _run_single_product_search(params: dict, query: str, steps: list) -> None:
    """Execute a single-product search and recommend the best result."""
    prods = params.get("products", [{}])
    p = prods[0] if prods else {}
    search_params = _spec_to_find_product_params(p)
    kw = p.get("keywords", "product")

    all_results = []
    tool_results: list = []
    # Secondary search with color terms stripped — colors often live in SKUs, not titles.
    cheaper = False
    stripped_kw = _strip_color_keywords(kw)
    if stripped_kw != kw and len(stripped_kw.split()) >= 2:
        for page in range(1, 4):
            search_params["page"] = page
            result = safe_tool_call("find_product", search_params)
            all_results.extend(result["result"] or [])
            tool_results.append(result)
        broad_params = dict(search_params)
        broad_params["q"] = stripped_kw
        for page in range(1, 3):
            broad_params["page"] = page
            result = safe_tool_call("find_product", broad_params)
            all_results.extend(result["result"] or [])
            tool_results.append(result)
        cheaper = True
    else:
        for page in range(1, 2):
            search_params["page"] = page
            result = safe_tool_call("find_product", search_params)
            all_results.extend(result["result"] or [])
            tool_results.append(result)

    _append_step("Processing.", tool_results, "", query, steps)

    seen = set()
    unique: list[dict] = []
    for prod in all_results:
        pid = str(prod.get("product_id", ""))
        if pid and pid not in seen:
            seen.add(pid)
            unique.append(prod)

    best = (
        _select_best_product_for_product_case(unique, query, top_count=20, prefer_cheaper=cheaper)
        if unique
        else None
    )
    if best:
        _finish_session([str(best["product_id"])], "success", query, steps)
    else:
        _finish_session([FALLBACKID], "failure", query, steps)


def _run_same_shop_search(params: dict, query: str, steps: list, voucher: dict = None) -> None:
    """Search for multiple products from the same shop and recommend them."""
    queries = [_spec_to_find_product_params(p) for p in params.get("products", [])]
    if not queries:
        queries = [{"q": "product"}]
    queries.append({"_original_query": query})
    queries.append({"voucher": json.dumps(voucher)})

    result = safe_tool_call(
        "find_products_in_same_shop",
        {"product_queries": json.dumps(queries)},
    )
    _append_step("Processing.", [result], "", query, steps)

    shop_result = result["result"]
    pids = []
    if isinstance(shop_result, dict) and shop_result.get("found"):
        pids = [str(p["product_id"]) for p in shop_result["products"]]
    else:
        for p in params.get("products", []):
            try:
                r = safe_tool_call("find_product", _spec_to_find_product_params(p))
                _append_step("Processing.", [r], "", query, steps)
                if r["result"]:
                    best = _select_best_product(r["result"], query, prefer_cheaper=True)
                    if best:
                        pids.append(str(best["product_id"]))
            except Exception:
                logger.exception("_run_same_shop_search: fallback search failed for spec %s", p)

    if pids:
        _finish_session(pids, "success", query, steps)
    else:
        _finish_session([FALLBACKID], "failure", query, steps)


def _run_voucher_search(params: dict, query: str, steps: list) -> None:
    """Execute a voucher/budget search, delegating to same-shop flow when appropriate."""
    is_shop = params.get("is_shop_voucher", False) or "same shop" in query.lower()
    products = params.get("products", [])

    voucher = params.get("voucher", {})
    discount_type = voucher.get("discount_type", "percentage")
    discount_value = float(voucher.get("discount_value", 0))
    threshold = float(voucher.get("threshold", 0))
    cap = float(voucher.get("cap", 0))
    budget = float(voucher.get("budget", 0))

    if is_shop and len(products) > 1:
        _run_same_shop_search(params, query, steps, {"discount_type": discount_type, "discount_value": discount_value, "threshold": threshold, "budget": budget, "cap": cap})
        return

    voucher = _normalize_voucher_fields(params.get("voucher"))

    min_price = voucher["threshold"]
    max_price = max_total_price(
        voucher["budget"],
        voucher["threshold"],
        voucher["discount_value"],
        voucher["cap"],
        voucher["discount_type"],
    )

    pids = []
    for p in products:
        sp = _spec_to_find_product_params(p, include_price=False)
        result = safe_tool_call("find_product", sp)
        _append_step("Processing.", [result], "", query, steps)

        found = result["result"] or []

        if len(products) == 1:
            best = found[0] if found else None
            if best is not None and best["price"] >= voucher["threshold"]:
                voucher_result = safe_tool_call(
                    "calculate_voucher",
                    {
                        "product_prices": str(best["price"]),
                        "voucher_type": voucher["discount_type"],
                        "discount_value": voucher["discount_value"],
                        "threshold": voucher["threshold"],
                        "budget": voucher["budget"],
                        "cap": voucher["cap"],
                    },
                )
                if (
                    voucher_result.get("result")
                    and voucher_result["result"]["total_after"] <= voucher["budget"]
                ):
                    pids.append(str(best["product_id"]))
                    continue

            sp["price"] = f"{min_price}-{max_price}"

        for page in range(1, 4):
            sp["page"] = page
            result = safe_tool_call("find_product", sp)
            found.extend(result["result"] or [])

        found = _deduplicate_products(found)

        if found:
            kw = p.get("keywords", "product")
            score_q = kw if len(products) > 1 else query
            best = _select_best_product(
                found,
                score_q,
                top_count=20,
                prefer_cheaper=False,
                voucher=voucher if len(products) == 1 else None,
            )
            if best:
                pids.append(str(best["product_id"]))

    if pids:
        _finish_session(pids, "success", query, steps)
    else:
        _finish_session([FALLBACKID], "failure", query, steps)


# ── Entry point ─────────────────────────────────────────────────────────────


def agent_main(problem_data: dict) -> list[dict]:
    _product_detail_cache.clear()
    steps: list = []
    query: str = problem_data.get("query", "")
    logger.info("agent_main: started")

    try:
        kw_task = _infer_task_type(query)

        params = _extract_query_params_llm(query, kw_task)
        logger.info(
            "agent_main task=%s n_products=%d params=%s",
            kw_task,
            len(params.get("products", [])),
            params,
        )
        _append_step("Processing.", [], "", query, steps)

        if kw_task == "shop":
            _run_same_shop_search(params, query, steps)
        elif kw_task == "voucher":
            _run_voucher_search(params, query, steps)
        else:
            _run_single_product_search(params, query, steps)

    except Exception:
        logger.error("agent_main: unhandled exception", exc_info=True)
        try:
            _finish_session([FALLBACKID], "failure", query, steps)
        except Exception:
            steps.append(create_dialogue_step("Done.", [], "Done.", query, len(steps) + 1))

    if not steps:
        steps.append(create_dialogue_step("Done.", [], "Done.", query, 1))

    logger.info("agent_main: completed with %d steps", len(steps))
    return steps
