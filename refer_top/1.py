import json
import logging
import re
import threading
import time
from collections import defaultdict
from collections.abc import Sequence
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

Product = dict[str, Any]
SearchSpec = dict[str, Any]

DEFAULT_PRODUCT_QUERY = "product"

TOP_RELEVANCE_CANDIDATES = 10
CHEAPER_PRICE_TIEBREAK_DIVISOR = 100_000
SHOP_SCORE_THRESHOLD = 6.0

FALLBACK_PRODUCT_ID: str = "0"

_inference_client = ProxyClient(timeout=90, max_retries=20)
_search_client = ProxyClient(timeout=30, max_retries=5)

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

DEFAULT_PARSE_MODEL_FOR_PRODUCT = "deepseek-ai/DeepSeek-V3.2-TEE"
DEFAULT_PARSE_MODEL_FOR_VOUCHER = "deepseek-ai/DeepSeek-V3.2-TEE"
DEFAULT_CHOOSE_PRODUCT_MODEL = "openai/gpt-oss-120b-TEE"
FALLBACK_MODEL = "deepseek-ai/DeepSeek-V3.1-TEE"
DEFAULT_PARSE_MODEL_FOR_SHOP = "deepseek-ai/DeepSeek-V3.2-TEE"

_SCORING_STOPWORDS = [
    "the","a","an","for","with","from","that","this","i","me",
    "my","looking","show","find","want","need","get","finish",
    "buy","also","and","in","is","it","am","im","priced","pesos",
    "php","price","between","than","above","below","more","less",
    "over","under","of","to","or","on","at","by","its","be","can",
    "has","have","will","would","should","item","items","both","these",
    "offering","sells","shop","budget","voucher","discount","first","second",
    "third","brand","made","using","available","support","supports","compatible",
    "please","looking","age",
]

_PRODUCT_PROMPT = """Extract search params as JSON. No markdown.
{"products":[{"keywords":"search query","price_range":"min-max"|null,"service":"official"|"freeShipping"|"COD"|"flashsale"|null,"only_product_type":bool}],"is_shop_voucher":bool}
- keywords: product type + brand + material + color + quantity/units + dimensions + packaging/logistics + product/misc + capacity + sharp + fit + style + length + use. use 2-8 words. Include ALL qualifying terms. Keep full color descriptors including any qualifier. IMPORTANT RULE:
    - Must preserve the left-to-right order of terms exactly as they appear in the query.
    - MUST never include service related terms in keywords.
    - Must identify the price unit and the other one.
    - Compact any number+unit pair: remove the space and use the standard abbreviation. 
    - When a query has "any" immediately before a noun, keep BOTH in keywords as a phrase (e.g. "any season", "any weather")—never drop "any" as filler; include the full "any <word>" pair.
- price_range: "100-500", "100-", "0-500". null if none.
- only_product_type: true if the keywords are only nouns — even if it is a multi-word compound noun (the words together name the product, not describe it). Set false only when at least one additional word is a separate qualifier such as a brand name, color, material, numeric spec, or descriptive adjective added on top of the core product name.
- service: "LazMall / guaranteed authenticity / quick returns"→"official", "free shipping / free delivery" → "freeShipping", "cash on delivery / COD / payment on delivery" → "COD", "LazFlash / flash deal / limited-time deal" → "flashsale"; null if none. Multiple options available, combine them with ",".
JSON only:"""


_SHOP_PROMPT = """Extract search params as JSON. No markdown. Find multi-products.
{"products":[{"query":"the part of the raw query describing this product","keywords":"search query","price_range":"min-max"|null,"service":"official"|"freeShipping"|"COD"|"flashsale"|null,"only_product_type":bool}]}
- keywords: product type + brand + material + color + size + quantity/units + weight/volume + dimensions + packaging/logistics + product/misc + sharp + fit + style + length + selling unit + use. 2-8 words, include ALL qualifying terms. Keep full color descriptors including any qualifier. Drop opening/fastening mechanism terms. IMPORTANT RULE:
    - Must preserve the left-to-right order of terms exactly as they appear in the query.
    - MUST never include service related terms in keywords.
    - Compact any number+unit pair: remove the space and use the standard abbreviation. 
    - When a query has "any" immediately before a noun, keep BOTH in keywords as a phrase (e.g. "any season", "any weather")—never drop "any" as filler; include the full "any <word>" pair.
- price_range: "100-500", "100-", "0-500". null if none.
- service: "LazMall"→"official", "free shipping / free delivery" → "freeShipping", "cash on delivery / COD / payment on delivery" → "COD", "LazFlash / flash deal / limited-time deal" → "flashsale"; null if none. Multiple options available, combine them with ",".
- only_product_type: true if the keywords are the product type name alone — even if it is a multi-word compound noun (the words together name the product, not describe it). Set false only when at least one additional word is a separate qualifier such as a brand name, color, material, numeric spec, or descriptive adjective added on top of the core product name.
- Same store must sell MULTIPLE differents (numbered First/Second/Also)
- Multi-product: one entry per product, preserve order.
JSON only:"""

_VOUCHER_PROMPT = """Extract search params as JSON. No markdown. Find products that fit within a budget after applying a voucher discount.
{"products":[{"query": "corresponding part of the raw query for this product.", "keywords":"search query","price_range":"min-max"|null,"service":"official"|"freeShipping"|"COD"|"flashsale"|null,"only_product_type":bool}], "voucher": { "voucher_type": "platform|shop", "discount_type": "fixed|percentage", "discount_value": "fixed amount OR percentage number e.g. 42 for 42%", "threshold": "minimum total price for voucher to apply", "cap": "max discount for percentage vouchers, 0 if not mentioned or fixed type", "budget": "the user's maximum budget" }, "is_shop_voucher":bool}
- keywords: product type + brand + material + color + quantity/units + weight/volume + dimensions + packaging/logistics + product/misc + sharp + fit + style + length + use. 2-8 words, include qualifying terms that are explicitly mentioned in the query. IMPORTANT RULE:
    - Must preserve the left-to-right order of terms exactly as they appear in the query.
    - MUST never include service related terms and secondary keywords.
    - Keep full color descriptors including any qualifier.
    - Compact any number+unit pair: remove the space and use the standard abbreviation. 
    - When a query has "any" immediately before a noun, keep BOTH in keywords as a phrase (e.g. "any season", "any weather")—never drop "any" as filler; include the full "any <word>" pair.
- price_range: "100-500", "100-", "0-500". null if none.
- service: "LazMall / guaranteed authenticity / quick returns"→"official", "free shipping / free delivery" → "freeShipping", "cash on delivery / COD / payment on delivery" → "COD", "LazFlash / flash deal / limited-time deal" → "flashsale"; null if none. Multiple options available, combine them with ",".
- only_product_type: true if the keywords are the product type name alone — even if it is a multi-word compound noun (the words together name the product, not describe it). Set false only when at least one additional word is a separate qualifier such as a brand name, color, material, numeric spec, or descriptive adjective added on top of the core product name.
- Multi-product: one entry per product, preserve order. Budget/voucher info are NOT products.
- is_shop_voucher: true if "same shop" voucher.
JSON only:"""

_SHOP_SCORER_PROMPT = """You are scoring product candidates for a shopping benchmark.

Score EVERY candidate against the user request. Return a score for ALL of them.

Priorities:
1. Prefer explicit structured evidence in attributes and sku_options. In the user_content.request, "any" means "all."
2. Compatibility/model, material, function, theme, brand, quantity/units, weight/volume, dimensions, packaging/logistics, sharp, fit, style, length, use, service, and price constraints all matter.
3. Do not prefer a candidate just because its title is broader, more generic, or contains more common keywords.
4. Treat semantically equivalent value strings as the same match even if formatting differs slightly.
5. if "only_product_type" is true, account for the product_type + "only" option in the sku_option and attributes but not in title.
    Minor wording, spacing, punctuation, tokenization, or formatting differences should not change the decision by themselves.
6. Do not over-weight one appealing field when both candidates already satisfy it.
   If multiple candidates match the same color/model/service, prefer the candidate whose title + attributes + sku_options are more consistently aligned overall.
7. Prefer stronger overall agreement across independent constraints over a single more literal-looking phrase.
- score: integer from 0 (no match) to 10 (perfect match) per candidate.

Return JSON array only, one object per candidate in the same order received:
[{"product_id":"...","score":8},{"product_id":"...","score":3},...]"""

_TASK_EXTRACTION_PROMPTS: dict[str, str] = {
    "product": _PRODUCT_PROMPT,
    "shop": _SHOP_PROMPT,
    "voucher": _VOUCHER_PROMPT,
}

PRODUCT_JUDGE_MAX_RETRIES = 3

_PRODUCT_JUDGE_PROMPT = """You are choosing the best final product candidate for a shopping benchmark.

Pick the ONE candidate that most exactly satisfies the user request.

Priorities:
1. Prefer explicit structured evidence in attributes and sku_options. In the user_content.request (it is the requirement query), “any” does not refer to special terms or items; it means the same as “all.”
2. Compatibility/model, material, function, theme, brand,  quantity/units, weight/volume, dimensions, packaging/logistics, product/misc, sharp, fit, style, length, use, service, and price constraints all matter.
3. Do not prefer a candidate just because its title is broader, more generic, or contains more common keywords.
4. If one candidate better matches the requested attributes, choose it even if another candidate has the same heuristic score.
5. Treat semantically equivalent value strings as the same match even if formatting differs slightly.
   Minor wording, spacing, punctuation, tokenization, or formatting differences should not change the decision by themselves.
6. Do not over-weight one appealing field when both candidates already satisfy it.
   If multiple candidates match the same color/model/service, prefer the candidate whose title + attributes + sku_options are more consistently aligned overall.
7. Prefer stronger overall agreement across independent constraints over a single more literal-looking phrase.
8. Prefer cheaper product.
9. if "only_product_type" is true, Must account for the product_type + "only" option in the sku_option and attributes but not in title.

- relevance_score: integer from 0 (no match) to 10 (perfect match) reflecting how well the best candidate satisfies the request.

Return JSON only:
{"best_product_id":"...","reason":"short reason","relevance_score":8}"""


_THINK_NARRATOR_PROMPT = """You are an AI shopping assistant. Write 2–4 sentences of internal, first-person reasoning explaining what you are doing at this step.

You receive a JSON object with a "query" field and additional context. Identify which case applies from the keys present and write accordingly:

CASE 1 — "keywords" + "price_constraints" + "service_filters" present (query analysis / planning step):
You are analysing the user's request before searching. State what the user wants to buy, list the exact search keywords you will use, mention any price range and service type constraints. If "only_product_type" is true, explain that the query is a bare product type with no extra qualifiers so you will append "only" to the search to avoid unrelated products — quote the "only_product_type_reason" value if present. If "budget_constraint" is present, note the voucher discount type, threshold, and budget.

CASE 2 — "search_query" + "top_candidates" present (search results step):
You just ran a product search. State the exact search query and any price/service filters applied. Report how many results came back ("total_results"). Name the most relevant top candidates by their title and price from "top_candidates". State what you will evaluate next.

CASE 3 — "selected" + "constraints" present (product selection step):
You are choosing the best product. Name the selected product by its product_id and title. Explain which specific attributes, SKU options, or specs from "selected.attributes" and "selected.sku_options_sample" satisfy the constraints (price, service, keywords). Quote the "llm_reason" value if it is non-empty and explain why it is the best match.

CASE 4 — "product_count" + "products" present (multi-product shop planning step):
You are about to search for multiple products from the same shop. State how many products are needed and name each one using its keywords value. Mention the price range and service constraint for each product.

CASE 5 — "shop_id" + "selected_products" present (shop found step):
You found a shop carrying all required products. State the shop ID. For each entry in "selected_products", name its title and price. Confirm they collectively satisfy the query. If "llm_reasoning" is also present, reference the relevance scores that led to this shop being chosen.

CASE 6 — "budget_constraint" + "candidates_per_product" present (voucher candidate evaluation step):
You are checking which products fit within the voucher budget. State the voucher discount type, threshold, and the max allowed total from "max_allowed_total". For each entry in "candidates_per_product", name the keywords and the top product candidate's title and price.

CASE 7 — "selected_products" + "budget_constraint" present, no "candidates_per_product" (voucher selection confirmed step):
You selected products that fit the voucher budget. Name each product from "selected_products" by title and price. State the total price before discount from "total_before_discount" and confirm it is within the allowed budget. Quote "llm_reason" if present.

CASE 8 — "selected_products" + "total_spent" + "allowed_total" present (fixed-budget selection step):
You finalised products within a fixed spending limit. Name each product, state the exact total spent and the allowed maximum, and confirm the selection is within budget.

CASE 9 — "scoring_summary" + "score_threshold" present (LLM scoring / shop-coverage step):
You just LLM-scored all candidate products against the query. State the score threshold from "score_threshold". For each entry in "scoring_summary", report how many products were collected ("total_collected") and how many passed the threshold ("passed_threshold"), naming the top-scoring candidates by title and score from "top_candidates". State how many full-coverage shops were found using "full_coverage_shops_found" and what you will do next.

CASE 10 — "case_c_resolution" present (anchor-product fallback step):
No single shop covered all required products after score filtering. Describe the sub-case strategy. If "sub_case" is 4, explain how you evaluated partial-coverage shops and filled the missing spec by searching inside the winner shop. Otherwise, name the anchor product using its spec index, keywords, product_id, and shop_id, and explain that you searched the remaining specs within that shop to maximise coverage.

CASE 11 — "recommended_product_ids" + "status" present (final recommendation step):
You are finalising the session. State the product IDs you are recommending from "recommended_product_ids". Confirm the outcome using "status" (success or failure). If "llm_reason" is present, quote it to justify the choice. If "note" is present, mention it.

Rules:
- Always write in first person ("I searched…", "I selected…", "I found…", "I am planning to…").
- Reference actual values from the context: IDs, titles, prices, keywords, shop IDs, attributes, scores.
- Be specific and concrete — never vague or generic.
- Do NOT output JSON or markdown. Plain text only.
- 2–4 sentences maximum."""

_REGEX_STOPWORDS = {
    "the", "and", "for", "with", "from", "that", "this",
    "are", "was", "can", "has", "have", "been", "will",
    "find", "finish", "looking", "show", "want", "need",
    "get", "buy", "product", "products", "search", "same",
    "shop", "within", "budget", "voucher", "discount", "price",
    "priced", "pesos", "php", "between", "than", "greater", "less",
    "more", "under", "over", "about", "also", "both", "these",
    "them", "each", "all", "one", "two", "three", "four",
    "five", "offering", "sells", "using", "in", "is", "it", "its",
    "or", "at", "on", "by", "be", "do", "an", "my", "me", "im",
    "items", "item", "just", "first", "second", "supports",
    "support", "compatible", "available", "made", "please", "like",
    "of", "above", "deals", "options", "option", "delivery", "shipping", 
    "offers", "lazmall", "lazflash", "official", "cash", "payment", "pay",
    "cost", "costs", "via", "themed", "such", "those", "store", "stores",
    "focus", "category", "specifically", "guaranteed", "authenticity",
    "returns", "quick", "perks", "should", "help", "purchase", "type",
    "to", "named", "called", "family", "belongs", "comes", "another",
    "lastly", "benefits", "you", "weighing", "capacity", "size", "sized", "eu", "fits",
}

_MULTI_PRODUCT_SPLIT_RE = re.compile(
    r"(?:,?\s*and\s+also\s+|,?\s*also,?\s+|Second(?:ly)?,\s*|Third(?:ly)?,\s*"
    r"|First,\s*|\(\d+\)\s*|\d+\.\s*|Additionally,\s*"
    r"|[.]\s*Next,\s*|[.]\s*Lastly,\s*|[.]\s*Finally,\s*|[.]\s*Last,\s*)",
    re.IGNORECASE,
)

_BUDGET_SPLIT_RE = re.compile(r"(?:My budget|budget is|I have a voucher)", re.IGNORECASE)


def _wait_for_oro_request_slot() -> None:
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
    _wait_for_oro_request_slot()
    return _search_client.get(path, params)


def safe_tool_call(tool_name: str, params: dict) -> dict:
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


@Tool
def find_product(
    q: str,
    page: int = 1,
    shop_id: str | None = None,
    price: str | None = None,
    sort: str | None = None,
    service: str | None = None,
) -> list[dict]:
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


def _normalize_service(service: str | None) -> str | None:
    if not service:
        return service
    if service == "default":
        return None

    services = [
        part.strip() for part in service.split(",") if part.strip() and part.strip() != "default"
    ]
    return ",".join(services) or None


def _build_search_params(
    query: str,
    *,
    page: int = 1,
    shop_id: str | None = None,
    price: str | None = None,
    sort: str | None = None,
    service: str | None = None,
) -> dict[str, Any]:
    params: dict[str, Any] = {"q": quote_plus(query), "page": page}
    if shop_id:
        params["shop_id"] = shop_id
    if price:
        params["price"] = price
    if sort and sort != "default":
        params["sort"] = sort
    normalized_service = _normalize_service(service)
    if normalized_service:
        params["service"] = normalized_service
    return params


def _search_products(params: dict[str, Any]) -> list[Product]:
    return _search_client.get("/search/find_product", params) or []


def _search_products_for_spec(
    spec: SearchSpec,
    *,
    shop_id: str | None = None,
    include_price: bool = True,
    omit_service_from_api: bool = False,
) -> list[Product]:
    price = None
    if include_price:
        price = spec.get("price")
        if price is None:
            price = spec.get("price_range")

    service = None if omit_service_from_api else spec.get("service")

    found = []

    for page in range(1, 3):
        result = _search_products(
            _build_search_params(
                spec.get("q") or spec.get("keywords") or DEFAULT_PRODUCT_QUERY,
                page=page,
                shop_id=shop_id,
                price=price,
                service=service,
            )
        )
        found.extend(result or [])

    return found


def _group_products_by_shop(
    broad_results: Sequence[Sequence[Product]],
) -> dict[str, dict[int, list[Product]]]:
    shop_coverage: dict[str, dict[int, list[Product]]] = defaultdict(lambda: defaultdict(list))
    for index, products in enumerate(broad_results):
        for product in products:
            shop_id = str(product.get("shop_id", ""))
            if shop_id:
                shop_coverage[shop_id][index].append(product)
    return shop_coverage


def _product_matches_services(product: Product, service_spec: str | None) -> bool:
    if not service_spec:
        return True
    required = [part.strip() for part in str(service_spec).split(",") if part.strip()]
    if not required:
        return True
    offered = product.get("service") or []
    if not isinstance(offered, list):
        offered = []
    return all(req in offered for req in required)


def _filter_products_by_spec_services(
    products: Sequence[Product], spec: SearchSpec
) -> list[Product]:
    service_spec = spec.get("service")
    if not service_spec:
        return list(products)
    return [p for p in products if _product_matches_services(p, service_spec)]


def _pick_best_shop_by_llm_scores(
    shop_ids: list[str],
    shop_coverage: dict[str, dict[int, list[Product]]],
    specs: list[SearchSpec],
    query: str,
) -> tuple[str | None, dict[int, dict]]:
    best_shop_id: str | None = None
    best_total_score: float = -1.0
    best_chosen: dict[int, dict] = {}

    for shop_id in shop_ids:
        total_score = 0.0
        chosen_for_shop: dict[int, dict] = {}

        for spec_idx, spec in enumerate(specs):
            products = list((shop_coverage.get(shop_id) or {}).get(spec_idx) or [])
            if not products:
                continue

            spec_query = spec.get("query") or spec.get("keywords") or query
            pids = [str(p.get("product_id", "")) for p in products if p.get("product_id")]
            details = _fetch_product_details(pids)

            chosen = _llm_choose_product(
                spec_query,
                products,
                details,
                only_product_type=bool(spec.get("only_product_type", False)),
                model=FALLBACK_MODEL,
            )
            if chosen:
                score = float(chosen.get("_llm_relevance_score", 0))
                total_score += score
                chosen_for_shop[spec_idx] = {
                    "product_id": str(chosen.get("product_id", "")),
                    "reason": chosen.get("_llm_reason", ""),
                    "score": score,
                }
            elif products:
                chosen_for_shop[spec_idx] = {
                    "product_id": str(products[0].get("product_id", "")),
                    "reason": "",
                    "score": 0.0,
                }

        if total_score > best_total_score:
            best_total_score = total_score
            best_shop_id = shop_id
            best_chosen = chosen_for_shop

    logger.info(
        "_pick_best_shop_by_llm_scores: winner=%s total_score=%.1f chosen=%s",
        best_shop_id, best_total_score, best_chosen,
    )
    return best_shop_id, best_chosen


def _llm_score_products(
    query_text: str,
    candidates: list[Product],
    details: dict[str, dict],
    only_product_type: bool = False,
    model: str = DEFAULT_CHOOSE_PRODUCT_MODEL,
) -> list[tuple[Product, float]]:
    if not candidates:
        return []

    payload = {
        "request": query_text,
        "candidates": [
            _product_judge_payload(p, details.get(str(p.get("product_id", ""))), query_text)
            for p in candidates
        ],
        "only_product_type": only_product_type,
    }
    user_content = json.dumps(payload, ensure_ascii=False)

    env_model = getenv("SANDBOX_MODEL")
    model_chain = [env_model] if env_model else [model, FALLBACK_MODEL]

    for m in model_chain:
        for attempt in range(1, PRODUCT_JUDGE_MAX_RETRIES + 1):
            result = _inference_client.post(
                "/inference/chat/completions",
                json_data={
                    "model": m,
                    "temperature": 0.5,
                    "stream": False,
                    "messages": [
                        {"role": "system", "content": _SHOP_SCORER_PROMPT},
                        {"role": "user", "content": user_content},
                    ],
                },
            )
            if not (result and result.get("choices")):
                logger.warning(
                    "ShopScorer: %s no response (attempt %d/%d)", m, attempt, PRODUCT_JUDGE_MAX_RETRIES
                )
                continue

            content = result["choices"][0].get("message", {}).get("content", "")
            cleaned = re.sub(r"```json?\s*", "", content)
            cleaned = re.sub(r"```\s*$", "", cleaned).strip()
            parsed = None
            try:
                parsed = json.loads(cleaned)
            except json.JSONDecodeError:
                m_arr = re.search(r"\[.*\]", content, re.DOTALL)
                if m_arr:
                    try:
                        parsed = json.loads(m_arr.group())
                    except json.JSONDecodeError:
                        pass

            if not isinstance(parsed, list):
                logger.warning(
                    "ShopScorer: %s invalid JSON array (attempt %d/%d)", m, attempt, PRODUCT_JUDGE_MAX_RETRIES
                )
                continue

            pid_to_score: dict[str, float] = {}
            for item in parsed:
                if isinstance(item, dict):
                    pid = str(item.get("product_id", "")).strip()
                    try:
                        score = float(item.get("score", 0))
                    except (TypeError, ValueError):
                        score = 0.0
                    if pid:
                        pid_to_score[pid] = score

            scored = [
                (p, pid_to_score.get(str(p.get("product_id", "")).strip(), 0.0))
                for p in candidates
            ]
            scored.sort(key=lambda x: x[1], reverse=True)
            logger.info("ShopScorer: %s scored %d products", m, len(scored))
            return scored

    logger.warning("ShopScorer: all models exhausted, falling back to heuristic scoring")
    scored = [
        (p, 7.0 if _score_product_relevance(p, query_text) > 0 else 0.0)
        for p in candidates
    ]
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored


def _pick_winning_spec_by_depth(spec_indices: list[int], specs: list[SearchSpec]) -> int:
    def _raw(spec: SearchSpec) -> tuple[float, int, int]:
        kw_count = len((spec.get("keywords") or "").split())

        price_score = 0.0
        price_range = spec.get("price_range") or ""
        if price_range and "-" in price_range:
            parts = price_range.split("-", 1)
            lo, hi = parts[0].strip(), parts[1].strip()
            if lo and hi:
                price_score = 1.5   # exact range e.g. "30-50"
            elif lo or hi:
                price_score = 1.0   # open-ended e.g. "50-" or "-30"

        svc_count = len(
            [s.strip() for s in (spec.get("service") or "").split(",") if s.strip()]
        )
        return (price_score, kw_count, svc_count)

    raw = {idx: _raw(specs[idx]) for idx in spec_indices}
    max_kw  = max(d[1] for d in raw.values())
    max_svc = max(d[2] for d in raw.values())

    final: dict[int, float] = {}
    for idx, (price_s, kw, svc) in raw.items():
        score = price_s
        if kw  == max_kw:  score += 1.0
        if svc == max_svc: score += 1.0
        final[idx] = score

    max_score = max(final.values())
    winners = [idx for idx, s in final.items() if s == max_score]
    return winners[0]  # first wins if still tied


def _search_and_pick_spec_for_shop(
    spec: SearchSpec, shop_id: str, query: str
) -> Product | None:
    products = _search_products_for_spec(spec, shop_id=shop_id)
    if not products:
        products = _search_products_for_spec(spec, shop_id=shop_id, omit_service_from_api=True)
    if not products:
        return None
    pids = [str(p.get("product_id", "")) for p in products if p.get("product_id")]
    details = _fetch_product_details(pids)
    spec_query = spec.get("query") or spec.get("keywords") or query
    best = _llm_choose_product(
        spec_query, products[:10], details,
        only_product_type=bool(spec.get("only_product_type", False)),
        model=FALLBACK_MODEL,
    )
    return best if best is not None else (products[0] if products else None)


def _try_partial_shop_coverage(
    specs: list[SearchSpec],
    spec_scored: list[list[tuple[Product, float]]],
    shop_coverage: dict[str, dict[int, list[Product]]],
    query: str,
    n_specs: int,
) -> tuple[list[str] | None, dict]:
    target = n_specs - 1
    partial_shops = {
        sid: cov for sid, cov in shop_coverage.items() if len(cov) == target
    }
    if not partial_shops:
        return None, {}

    pid_to_score: dict[str, float] = {
        str(p.get("product_id", "")): score
        for scored in spec_scored
        for p, score in scored
    }

    def _shop_total(cov: dict) -> float:
        total = 0.0
        for spec_idx, products in cov.items():
            total += max(
                (pid_to_score.get(str(p.get("product_id", "")), 0.0) for p in products),
                default=0.0,
            )
        return total

    shop_scores = {sid: _shop_total(cov) for sid, cov in partial_shops.items()}
    max_score   = max(shop_scores.values())
    best_shops  = [sid for sid, s in shop_scores.items() if s == max_score]
    winner_shop = best_shops[0]  # ties → first candidate

    coverage = partial_shops[winner_shop]
    covered  = set(coverage.keys())
    missing_idx = next(i for i in range(n_specs) if i not in covered)

    pids: list[str | None] = [None] * n_specs
    for spec_idx in covered:
        shop_pids = {str(p.get("product_id", "")) for p in coverage[spec_idx]}
        best_p = next(
            (p for p, _ in spec_scored[spec_idx] if str(p.get("product_id", "")) in shop_pids),
            coverage[spec_idx][0] if coverage[spec_idx] else None,
        )
        if best_p:
            pids[spec_idx] = str(best_p.get("product_id", ""))

    best_missing = _search_and_pick_spec_for_shop(specs[missing_idx], winner_shop, query)
    if not best_missing:
        return None, {}
    pids[missing_idx] = str(best_missing.get("product_id", ""))

    if not all(pid is not None for pid in pids):
        return None, {}

    context = {
        "sub_case": 4,
        "partial_shops_evaluated": len(partial_shops),
        "winner_shop_id": winner_shop,
        "winner_shop_score": round(max_score, 2),
        "covered_spec_indices": sorted(covered),
        "missing_spec_idx": missing_idx,
        "missing_spec_keywords": specs[missing_idx].get("keywords", ""),
        "filled_missing_product": {
            "product_id": str(best_missing.get("product_id", "")),
            "title": best_missing.get("title", ""),
            "price": best_missing.get("price"),
        },
    }
    return pids, context


def _resolve_case_c(
    specs: list[SearchSpec],
    spec_scored: list[list[tuple[Product, float]]],
    shop_coverage: dict[str, dict[int, list[Product]]],
    query: str,
    n_specs: int,
) -> tuple[list[str] | None, dict]:
    if n_specs >= 3:
        pids, ctx = _try_partial_shop_coverage(specs, spec_scored, shop_coverage, query, n_specs)
        if pids:
            return pids, ctx

    global_max = max(
        (scored[0][1] for scored in spec_scored if scored),
        default=0.0,
    )
    if global_max <= 0:
        return None, {}

    top_by_spec: dict[int, list[Product]] = defaultdict(list)
    for spec_idx, scored in enumerate(spec_scored):
        for product, score in scored:
            if score >= global_max:
                top_by_spec[spec_idx].append(product)

    top_spec_indices = list(top_by_spec.keys())

    if len(top_spec_indices) == 1:
        spec_idx   = top_spec_indices[0]
        candidates = top_by_spec[spec_idx]
        if len(candidates) == 1:
            anchor     = candidates[0]
            sub_case   = 1
            tie_note   = "Single global top-scoring product; anchoring directly."
        else:
            anchor     = min(candidates, key=lambda p: p.get("price") or float("inf"))
            sub_case   = 2
            tie_note   = (
                f"{len(candidates)} products tied at score {global_max:.1f} "
                f"in spec[{spec_idx}]; picked cheapest as anchor."
            )
        anchor_spec_idx = spec_idx
    else:
        winning_spec_idx = _pick_winning_spec_by_depth(top_spec_indices, specs)
        candidates = top_by_spec[winning_spec_idx]
        anchor = (
            candidates[0]
            if len(candidates) == 1
            else min(candidates, key=lambda p: p.get("price") or float("inf"))
        )
        anchor_spec_idx = winning_spec_idx
        sub_case  = 3
        tie_note  = (
            f"Top score {global_max:.1f} tied across specs {top_spec_indices}; "
            f"depth scoring selected spec[{winning_spec_idx}] as anchor."
        )

    anchor_shop_id = str(anchor.get("shop_id", ""))
    if not anchor_shop_id:
        return None, {}

    logger.info(
        "_resolve_case_c: sub_case=%d anchor spec_idx=%d product_id=%s shop_id=%s score=%.1f",
        sub_case, anchor_spec_idx, anchor.get("product_id"), anchor_shop_id, global_max,
    )

    pids: list[str | None] = [None] * n_specs
    pids[anchor_spec_idx] = str(anchor.get("product_id", ""))
    filled_specs: list[dict] = []

    for i in range(n_specs):
        if i == anchor_spec_idx:
            continue
        best = _search_and_pick_spec_for_shop(specs[i], anchor_shop_id, query)
        if not best:
            logger.info(
                "_resolve_case_c: no product for spec[%d] in shop %s", i, anchor_shop_id
            )
            return None, {}
        pids[i] = str(best.get("product_id", ""))
        filled_specs.append({
            "spec_idx": i,
            "keywords": specs[i].get("keywords", ""),
            "product_id": str(best.get("product_id", "")),
            "title": best.get("title", ""),
            "price": best.get("price"),
            "llm_reason": best.get("_llm_reason", ""),
        })

    if not all(pid is not None for pid in pids):
        return None, {}

    context = {
        "sub_case": sub_case,
        "global_max_score": global_max,
        "tie_note": tie_note,
        "anchor": {
            "spec_idx": anchor_spec_idx,
            "keywords": specs[anchor_spec_idx].get("keywords", ""),
            "product_id": str(anchor.get("product_id", "")),
            "title": anchor.get("title", ""),
            "price": anchor.get("price"),
            "shop_id": anchor_shop_id,
        },
        "filled_specs": filled_specs,
    }
    return pids, context


def _extract_query_words(query_text: str) -> list[str]:
    return list(
        dict.fromkeys(
            word
            for word in re.findall(r"\b\w+\b", query_text.lower())
            if word not in _SCORING_STOPWORDS and len(word) > 1
        )
    )


def _build_detail_search_text(detail: Product) -> tuple[str, set[str]]:
    tokens: list[str] = []
    exact_values: set[str] = set()

    for key, values in (detail.get("attributes") or {}).items():
        tokens.append(key.replace("_", " "))
        for value in values if isinstance(values, list) else [values]:
            value_str = str(value).strip().lower()
            tokens.append(value_str)
            exact_values.add(value_str)

    for options in (detail.get("sku_options") or {}).values():
        if isinstance(options, dict):
            for key, value in options.items():
                value_str = str(value).strip().lower()
                tokens.append(key.replace("_", " "))
                tokens.append(value_str)
                exact_values.add(value_str)

    return " ".join(tokens).lower(), exact_values


def _score_product_relevance(
    product: Product,
    query_text: str,
    detail: Product | None = None,
) -> float:
    title = product.get("title", "").lower()
    title_words = set(re.findall(r"\b\w+\b", title))
    query_words = _extract_query_words(query_text)

    score = 0
    for query_word in query_words:
        if (
            query_word in title_words
            or query_word.endswith("s")
            and query_word[:-1] in title_words
            or not query_word.endswith("s")
            and f"{query_word}s" in title_words
            or len(query_word) >= 3
            and any(
                title_word.startswith(query_word)
                for title_word in title_words
                if len(title_word) > len(query_word)
            )
        ):
            score += 2
        elif any(
            query_word.startswith(title_word) or title_word.startswith(query_word)
            for title_word in title_words
            if len(title_word) > 2
        ):
            score += 1

        if any(char.isdigit() for char in query_word) and query_word in title:
            score += 2

    if detail:
        detail_text, exact_values = _build_detail_search_text(detail)
        detail_words = set(re.findall(r"\b\w+\b", detail_text))
        for query_word in query_words:
            if query_word in exact_values:
                score += 3
            elif f"{query_word}#" in exact_values:
                score += 5
            elif query_word in detail_words:
                score += 2

    return score


@Tool
def calculate_voucher(
    product_prices: str,
    voucher_type: str,
    discount_value: float,
    threshold: float,
    budget: float,
    cap: float = 0,
) -> dict:
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
    return f"Having recommended the products to the user: {product_ids}."


@Tool
def terminate(status: str = "success") -> str:
    return f"The interaction has been completed with status: {status}"


def _fetch_product_details(product_ids: list[str]) -> dict[str, dict]:
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


def _parse_price_range_str(price_range: str) -> tuple:
    if not price_range or not isinstance(price_range, str):
        return None, None
    parts = price_range.split("-", 1)
    try:
        lo = float(parts[0]) if parts[0].strip() else None
    except ValueError:
        lo = None
    try:
        hi = float(parts[1]) if len(parts) > 1 and parts[1].strip() else None
    except ValueError:
        hi = None
    return lo, hi


def _score_product_for_product_case(
    product: dict, query_text: str, detail: dict = None, parsed_spec: dict = None
) -> float:
    title = product.get("title", "").lower()
    t_words = set(re.findall(r"\b\w+\b", title))
    q_words = list(
        dict.fromkeys(
            w
            for w in re.findall(r"\b\w+\b", query_text.lower())
            if w not in _SCORING_STOPWORDS and len(w) > 1
        )
    )
    spec = parsed_spec or {}
    score = 0.0

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

    price = product.get("price")
    if isinstance(price, (int, float)) and spec.get("price_range"):
        lo, hi = _parse_price_range_str(spec["price_range"])
        if (lo is not None and price < lo) or (hi is not None and price > hi):
            score -= 25
        else:
            score += 5

    product_services = set(product.get("service") or [])
    if spec.get("service"):
        required = {s.strip() for s in spec["service"].split(",") if s.strip()}
        for svc in required:
            if svc in product_services:
                score += 5
            else:
                score -= 15
    else:
        if product_services:
            for svc in product_services:
                if svc not in ["COD", "official"]:
                    score -= 4

    if detail:
        exact_values = set()
        attr_words = set()
        brand_values = set()

        for k, vs in (detail.get("attributes") or {}).items():
            k_lower = k.lower()
            attr_words.update(re.findall(r"\b\w+\b", k_lower.replace("_", " ")))
            for v in vs if isinstance(vs, list) else [vs]:
                v_str = str(v).strip().lower()
                exact_values.add(v_str)
                attr_words.update(re.findall(r"\b\w+\b", v_str))

        for _sku_id, opts in (detail.get("sku_options") or {}).items():
            if isinstance(opts, dict):
                for k, v in opts.items():
                    v_str = str(v).strip().lower()
                    exact_values.add(v_str)
                    attr_words.update(re.findall(r"\b\w+\b", v_str))
                    attr_words.update(re.findall(r"\b\w+\b", k.lower().replace("_", " ")))

        for qw in q_words:
            if qw in brand_values:
                score += 8
            elif qw in exact_values or (qw + "#") in exact_values:
                score += 5
            elif qw in attr_words:
                score += 2

    return score


def voucher_max_total_price(voucher: dict) -> float | None:
    discount_type = voucher.get("discount_type", "percentage")
    discount_rate = float(voucher.get("discount_value", 0))
    min_required = float(voucher.get("threshold", 0))
    discount_cap = float(voucher.get("cap", 0))
    budget = float(voucher.get("budget", 0))

    if discount_type == "fixed":
        max_price = budget + discount_rate
        if max_price <= min_required:
            return min_required
        return max_price

    rate = discount_rate / 100.0 if discount_rate > 1 else discount_rate
    if rate <= 0 or rate >= 1:
        return None

    if discount_cap > 0 and budget / (1 - rate) > (budget + discount_cap):
        max_price = budget + discount_cap
    else:
        max_price = budget / (1 - rate)

    if max_price <= min_required:
        return min_required

    return max_price


def _select_best_product(
    products: Sequence[Product],
    query_text: str,
    *,
    prefer_cheaper: bool = False,
    exclude_ids: set[str] | None = None,
) -> Product | None:
    if not products:
        return None
    if exclude_ids:
        products = [
            product for product in products if str(product.get("product_id", "")) not in exclude_ids
        ]
    if not products:
        return None

    scored_products = sorted(
        products,
        key=lambda product: _score_product_relevance(product, query_text),
        reverse=True,
    )
    top_candidates = scored_products[:TOP_RELEVANCE_CANDIDATES]

    logger.info(f"===================Top Candidates: {top_candidates}=======================")
    details = _fetch_product_details(
        [
            str(product.get("product_id", ""))
            for product in top_candidates
            if product.get("product_id")
        ]
    )

    def final_score(product: Product) -> float:
        score = _score_product_relevance(
            product,
            query_text,
            details.get(str(product.get("product_id", ""))),
        )
        if prefer_cheaper:
            price = product.get("price", 0) or 0
            score -= price / CHEAPER_PRICE_TIEBREAK_DIVISOR
        return score

    scored = [(p, final_score(p)) for p in top_candidates]
    if not scored:
        return None

    best_score = max(s for _, s in scored)

    tied = [p for p, s in scored if s == best_score]

    if len(tied) >= 2:
        return max(tied, key=lambda x: x.get("sold_count", 0) or 0)
    else:
        return tied[0]


def _parse_json_object_from_llm(content: str) -> dict | None:
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
    return _parse_json_object_from_llm(content)


def _product_judge_payload(
    product: dict, detail: dict | None, query_text: str
) -> dict:
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
    }


def _llm_choose_product(
    query_text: str,
    candidates: list,
    details: dict[str, dict],
    only_product_type: bool = False,
    model: str = DEFAULT_CHOOSE_PRODUCT_MODEL,
) -> dict | None:
    payload = {
        "request": query_text,
        "candidates": [
            _product_judge_payload(p, details.get(str(p.get("product_id", ""))), query_text)
            for p in candidates[:10]
        ],
        "only_product_type": only_product_type
    }
    user_content = json.dumps(payload, ensure_ascii=False)

    env_model = getenv("SANDBOX_MODEL")
    model_chain = [env_model] if env_model else [model, FALLBACK_MODEL]

    for model in model_chain:
        for attempt in range(1, PRODUCT_JUDGE_MAX_RETRIES + 1):
            result = _inference_client.post(
                "/inference/chat/completions",
                json_data={
                    "model": model,
                    "temperature": 0.5,
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
            reason = str(parsed.get("reason", "")).strip()
            try:
                relevance_score = float(parsed.get("relevance_score", 0))
            except (TypeError, ValueError):
                relevance_score = 0.0
            logger.info(
                "---------------llm-----product_id:  %s ----- reason: %s  score: %.1f",
                best_pid, reason, relevance_score,
            )
            for product in candidates[:10]:
                if str(product.get("product_id", "")).strip() == best_pid:
                    result_product = dict(product)
                    result_product["_llm_reason"] = reason
                    result_product["_llm_relevance_score"] = relevance_score
                    return result_product

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
    products: list,
    query_text: str,
    top_count: int = 10,
    prefer_cheaper: bool = False,
    parsed_spec: dict = None,
) -> dict | None:

    if not products:
        return None

    top = sorted(
        products,
        key=lambda p: _score_product_for_product_case(p, query_text, parsed_spec=parsed_spec),
        reverse=True,
    )[:top_count]
    pids = [str(p.get("product_id", "")) for p in top if p.get("product_id")]
    details = _fetch_product_details(pids)
    if not top:
        return None
    llm_choice = _llm_choose_product(query_text, top, details, only_product_type=bool(parsed_spec.get("only_product_type", False)))
    logger.info(f"llm_choice: {llm_choice}")
    if llm_choice is not None:
        return llm_choice
    return max(
        top,
        key=lambda p: _score_product_for_product_case(
            p, query_text, details.get(str(p.get("product_id", ""))), parsed_spec=parsed_spec
        ),
    )


def _infer_task_type(query: str) -> str:
    q = query.lower()
    if "voucher" in q or "budget" in q or "discount" in q:
        return "voucher"
    if "shop" in q and any(w in q for w in ("both", "these", "offering", "offers", "sells", "same")):
        return "shop"
    return "product"


def _sanitize_keyword_text(text: str | None) -> str:
    if not text:
        return "product"
    filtered = [
        w
        for w in text.lower().split()
        if w not in _SCORING_STOPWORDS
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


def _extract_query_params_llm(query: str, task_type: str) -> dict:
    system_prompt = _TASK_EXTRACTION_PROMPTS.get(task_type, _PRODUCT_PROMPT)
    if task_type == "product":
        default_model = DEFAULT_PARSE_MODEL_FOR_PRODUCT
    elif task_type == "shop":
        default_model = DEFAULT_PARSE_MODEL_FOR_SHOP
    else:
        default_model = DEFAULT_PARSE_MODEL_FOR_VOUCHER

    model = getenv("SANDBOX_MODEL", default_model)
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
            if task_type == "product":
                return _sanitize_product_search_params(parsed)
            if task_type == "shop":
                for p in parsed.get("products", []):
                    if p.get("keywords"):
                        p["keywords"] = " ".join(
                            w for w in p["keywords"].split() if w.lower() not in _SCORING_STOPWORDS
                        )
            return parsed
        logger.warning("LLM extraction: %s returned unparseable response, trying next", model)
    else:
        logger.warning("LLM extraction: %s returned no response, trying next", model)

    logger.warning("LLM extraction failed on all models; falling back to regex")
    return _extract_query_params_regex(query)


def _extract_query_params_regex(query: str) -> dict:
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


def _spec_to_find_product_params(product: dict, *, include_price: bool = True) -> dict[str, Any]:
    keywords = product.get("keywords", "product")
    service = product.get("service")

    if not service and bool(product.get("only_product_type")):
        q = keywords + " only"
    else:
        q = keywords

    params: dict[str, Any] = {"q": q}
    if include_price and product.get("price_range"):
        params["price"] = product["price_range"]
    if service:
        params["service"] = service
    return params


def _parse_price_range(price_range: str | None) -> tuple[float | None, float | None]:
    if not price_range:
        return None, None
    s = str(price_range).strip()
    if "-" not in s:
        try:
            return None, float(s)
        except ValueError:
            return None, None
    idx = s.index("-")
    lo_str = s[:idx].strip()
    hi_str = s[idx + 1:].strip()
    lo = float(lo_str) if lo_str else None
    hi = float(hi_str) if hi_str else None
    return lo, hi


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
    seen: set = set()
    out: list = []
    for p in products:
        pid = str(p.get("product_id", ""))
        if pid and pid not in seen:
            seen.add(pid)
            out.append(p)
    return out


def _format_product_ids(ids: list, expected_order: list = None) -> str:
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
    return ",".join(out) if out else FALLBACK_PRODUCT_ID

def _enrich_products_for_reason(product_summaries: list[dict]) -> list[dict]:
    try:
        pids = [str(p.get("product_id", "")) for p in product_summaries]
        _fetch_product_details(pids)
    except Exception:
        logger.warning("_enrich_products_for_reason: detail fetch failed", exc_info=True)

    enriched = []
    for p in product_summaries:
        try:
            pid = str(p.get("product_id", ""))
            detail = _product_detail_cache.get(pid, {}) if isinstance(_product_detail_cache, dict) else {}
            entry: dict = {
                "product_id": pid,
                "title": p.get("title") or (detail.get("title", "") if isinstance(detail, dict) else ""),
                "price": p.get("price") if p.get("price") is not None else (detail.get("price") if isinstance(detail, dict) else None),
            }
            if isinstance(detail, dict):
                sku_options_raw = detail.get("sku_options") or []
                normalized_skus: list[dict] = []
                if isinstance(sku_options_raw, list):
                    for s in sku_options_raw:
                        if not isinstance(s, dict):
                            continue
                        vals = s.get("values", [])
                        if not isinstance(vals, list):
                            vals = list(vals.values()) if isinstance(vals, dict) else []
                        normalized_skus.append({"name": s.get("name"), "values": vals[:5]})
                elif isinstance(sku_options_raw, dict):
                    attr_values: dict[str, list] = {}
                    for variant in sku_options_raw.values():
                        if not isinstance(variant, dict):
                            continue
                        for attr_name, attr_val in variant.items():
                            attr_values.setdefault(attr_name, [])
                            if attr_val not in attr_values[attr_name]:
                                attr_values[attr_name].append(attr_val)
                    for attr_name, vals in attr_values.items():
                        normalized_skus.append({"name": attr_name, "values": vals[:5]})
                if normalized_skus:
                    entry["sku_options"] = normalized_skus[:3]

                attrs = detail.get("attributes") or {}
                if isinstance(attrs, dict) and attrs:
                    entry["attributes"] = {k: v for k, v in list(attrs.items())[:8]}

                services = detail.get("service_tags") or detail.get("services") or []
                if isinstance(services, list) and services:
                    entry["service_tags"] = services[:6]
        except Exception:
            logger.warning("_enrich_products_for_reason: failed for product_id=%s", p.get("product_id"), exc_info=True)
            entry = {
                "product_id": str(p.get("product_id", "")),
                "title": p.get("title", ""),
                "price": p.get("price"),
            }
        enriched.append(entry)
    return enriched

def _generate_think_text(query: str, context: dict, fallback: str) -> str:
    try:
        user_content = json.dumps({"query": query, **context}, ensure_ascii=False)
        model = getenv("SANDBOX_MODEL", FALLBACK_MODEL)
        result = _inference_client.post(
            "/inference/chat/completions",
            json_data={
                "model": model,
                "temperature": 0.3,
                "max_tokens": 500,
                "stream": False,
                "messages": [
                    {"role": "system", "content": _THINK_NARRATOR_PROMPT},
                    {"role": "user", "content": user_content},
                ],
            },
        )
        if result and result.get("choices"):
            text = result["choices"][0].get("message", {}).get("content", "").strip()
            if len(text) >= 20:
                return text
            if text:
                logger.warning(
                    "_generate_think_text: LLM returned suspiciously short response (%d chars), using fallback",
                    len(text),
                )
    except Exception:
        logger.warning("_generate_think_text: LLM call failed, using fallback", exc_info=True)
    return fallback


def _append_step(think: str, tool_results: list, response: str, query: str, steps: list) -> None:
    steps.append(create_dialogue_step(think, tool_results, response, query, len(steps) + 1))


def _finish_session(product_ids: list, status: str, query: str, steps: list, think: str = "", llm_reason: str = "") -> None:
    rec = safe_tool_call(
        "recommend_product",
        {"product_ids": _format_product_ids(product_ids)},
    )
    term = safe_tool_call("terminate", {"status": status})
    formatted_ids = _format_product_ids(product_ids)
    if not think:
        fallback_finish = (
            f"I am recommending product(s) {formatted_ids} for the query. "
            + (f"{llm_reason} " if llm_reason else "")
            + f"Status: {status}."
        )
        think = _generate_think_text(
            query,
            {
                "recommended_product_ids": formatted_ids,
                "status": status,
                **({"llm_reason": llm_reason} if llm_reason else {}),
                "note": "Finalising recommendation and terminating the session.",
            },
            fallback=fallback_finish,
        )
    _append_step(think, [rec, term], "Done.", query, steps)


def _run_single_product_search(params: dict, query: str, steps: list) -> None:
    prods = params.get("products", [{}])
    p = prods[0] if prods else {}
    search_params = _spec_to_find_product_params(p)

    all_results = []
    tool_results: list = []
    search_params["page"] = 1
    result = safe_tool_call("find_product", search_params)
    all_results.extend(result["result"] or [])
    tool_results.append(result)

    seen = set()
    unique: list[dict] = []
    for prod in all_results:
        pid = str(prod.get("product_id", ""))
        if pid and pid not in seen:
            seen.add(pid)
            unique.append(prod)

    top_candidates = [
        {"title": r.get("title", ""), "price": r.get("price"), "product_id": str(r.get("product_id", ""))}
        for r in unique[:5]
    ]
    fallback_search = (
        f"Searched for '{search_params.get('q', '')}' "
        f"(price={search_params.get('price', 'any')}, service={search_params.get('service', 'any')}). "
        f"Found {len(unique)} results. Top candidates: {top_candidates}."
    )
    think_search = _generate_think_text(
        query,
        {
            "search_query": search_params.get("q", ""),
            "price_filter": search_params.get("price"),
            "service_filter": search_params.get("service"),
            "total_results": len(unique),
            "top_candidates": top_candidates,
        },
        fallback=fallback_search,
    )
    _append_step(think_search, tool_results, "", query, steps)

    best = (
        _select_best_product_for_product_case(unique, query, top_count=10, parsed_spec=p)
        if unique
        else None
    )
    if best:
        pid = str(best.get("product_id", ""))
        detail = _product_detail_cache.get(pid, {})
        fallback_pick = (
            f"Selected product_id={pid} "
            f"title='{best.get('title', '')[:100]}' "
            f"price={best.get('price')} service={best.get('service')}. "
            + (f"LLM reason: {best.get('_llm_reason', '')}" if best.get("_llm_reason") else "Chosen by heuristic score.")
        )
        think_pick = _generate_think_text(
            query,
            {
                "selected": {
                    "product_id": pid,
                    "title": best.get("title", ""),
                    "price": best.get("price"),
                    "service": best.get("service"),
                    "attributes": detail.get("attributes", {}),
                    "sku_options_sample": list(detail.get("sku_options", {}).values())[:3],
                },
                "constraints": {
                    "price_range": p.get("price_range"),
                    "service": p.get("service"),
                    "keywords": p.get("keywords"),
                },
                "llm_reason": best.get("_llm_reason", ""),
            },
            fallback=fallback_pick,
        )
        _finish_session([pid], "success", query, steps, think=think_pick, llm_reason=best.get("_llm_reason", ""))
    else:
        _finish_session([FALLBACK_PRODUCT_ID], "failure", query, steps,
                        think="No suitable product found matching the query constraints.")


def _run_same_shop_search(params: dict, query: str, steps: list) -> None:
    specs = params.get("products", [])
    n_specs = len(specs)
    if not specs:
        _finish_session(
            [FALLBACK_PRODUCT_ID], "failure", query, steps,
            think="No product specs found in shop query.",
        )
        return

    kw_list = [s.get("keywords") or s.get("q", "") for s in specs]
    think_analyze = _generate_think_text(
        query,
        {
            "product_count": n_specs,
            "products": [
                {
                    "keywords": s.get("keywords"),
                    "price_range": s.get("price_range"),
                    "service": s.get("service"),
                }
                for s in specs
            ],
        },
        fallback=(
            f"Searching for {n_specs} products from the same shop. "
            f"Keywords: {kw_list}. "
            f"Price ranges: {[s.get('price_range') for s in specs]}. "
            f"Services: {[s.get('service') for s in specs]}."
        ),
    )

    all_results: list[list[Product]] = []
    search_tool_calls: list = []
    for spec in specs:
        sp = _spec_to_find_product_params(spec)
        found: list[Product] = []
        seen: set[str] = set()
        for page in range(1, 3):
            r = safe_tool_call("find_product", {**sp, "page": page})
            search_tool_calls.append(r)
            for p in r.get("result") or []:
                pid = str(p.get("product_id", ""))
                if pid and pid not in seen:
                    found.append(p)
                    seen.add(pid)
        all_results.append(found)

    _append_step(think_analyze, search_tool_calls, "", query, steps)

    spec_scored: list[list[tuple[Product, float]]] = []
    for spec_idx, (spec, products) in enumerate(zip(specs, all_results)):
        spec_query = spec.get("query") or spec.get("keywords") or query
        pids = [str(p.get("product_id", "")) for p in products if p.get("product_id")]
        details = _fetch_product_details(pids)
        scored = _llm_score_products(
            spec_query, products, details,
            only_product_type=bool(spec.get("only_product_type", False)),
        )
        filtered = [(p, s) for p, s in scored if s >= SHOP_SCORE_THRESHOLD]
        spec_scored.append(filtered)
        logger.info(
            "_run_same_shop_search: spec[%d] %d products → %d passed score >= %.1f",
            spec_idx, len(scored), len(filtered), SHOP_SCORE_THRESHOLD,
        )

    filtered_results: list[list[Product]] = [[p for p, _ in scored] for scored in spec_scored]
    shop_coverage = _group_products_by_shop(filtered_results)

    full_shops = [sid for sid, cov in shop_coverage.items() if len(cov) == n_specs]

    scoring_summary = [
        {
            "spec_idx": i,
            "keywords": specs[i].get("keywords", ""),
            "total_collected": len(all_results[i]),
            "passed_threshold": len(spec_scored[i]),
            "top_candidates": [
                {"title": p.get("title", ""), "price": p.get("price"), "score": s}
                for p, s in spec_scored[i][:3]
            ],
        }
        for i in range(n_specs)
    ]
    fallback_scoring = (
        f"LLM-scored products for {n_specs} specs (threshold={SHOP_SCORE_THRESHOLD}). "
        + " | ".join(
            f"spec[{i}]: {len(spec_scored[i])}/{len(all_results[i])} passed"
            for i in range(n_specs)
        )
        + f". Full-coverage shops found: {len(full_shops)}."
    )
    think_scoring = _generate_think_text(
        query,
        {
            "scoring_summary": scoring_summary,
            "score_threshold": SHOP_SCORE_THRESHOLD,
            "full_coverage_shops_found": len(full_shops),
        },
        fallback=fallback_scoring,
    )
    _append_step(think_scoring, [], "", query, steps)

    if len(full_shops) == 1:
        shop_id = full_shops[0]
        used_ids: set[str] = set()
        pids: list[str] = []
        for spec_idx in range(n_specs):
            for p in shop_coverage[shop_id].get(spec_idx, []):
                pid = str(p.get("product_id", ""))
                if pid and pid not in used_ids:
                    pids.append(pid)
                    used_ids.add(pid)
                    break
        if len(pids) == n_specs:
            enriched = _enrich_products_for_reason([{"product_id": pid} for pid in pids])
            think_found = _generate_think_text(
                query,
                {
                    "shop_id": shop_id,
                    "note": "Only one shop found covering all product specs.",
                    "selected_products": enriched,
                },
                fallback=(
                    f"Only one shop ({shop_id}) covers all {n_specs} specs. "
                    f"Recommending product IDs: {pids}."
                ),
            )
            _finish_session(pids, "success", query, steps, think=think_found)
            return

    if len(full_shops) > 1:
        shop_id, chosen = _pick_best_shop_by_llm_scores(
            full_shops, shop_coverage, specs, query
        )
        pids = [chosen[i]["product_id"] for i in range(n_specs) if i in chosen]
        if shop_id and len(pids) == n_specs:
            enriched = _enrich_products_for_reason([{"product_id": pid} for pid in pids])
            llm_reasoning = [
                {
                    "spec_index": i,
                    "product_id": chosen[i]["product_id"],
                    "reason": chosen[i]["reason"],
                    "relevance_score": chosen[i]["score"],
                }
                for i in range(n_specs) if i in chosen
            ]
            think_found = _generate_think_text(
                query,
                {
                    "shop_id": shop_id,
                    "note": (
                        f"Multiple full-coverage shops ({len(full_shops)}); "
                        "LLM relevance scoring selected the best shop."
                    ),
                    "selected_products": enriched,
                    "llm_reasoning": llm_reasoning,
                },
                fallback=(
                    f"Selected shop {shop_id} via LLM relevance scoring "
                    f"from {len(full_shops)} full-coverage candidates. "
                    f"Product IDs: {pids}."
                ),
            )
            _finish_session(pids, "success", query, steps, think=think_found)
            return

    logger.info(
        "_run_same_shop_search: Case C — no full-coverage shop "
        "(%d specs, %d shops after score filtering). Applying anchor-product strategy.",
        n_specs, len(shop_coverage),
    )
    pids_resolved, case_c_ctx = _resolve_case_c(specs, spec_scored, shop_coverage, query, n_specs)

    if pids_resolved and len(pids_resolved) == n_specs:
        sub_case = case_c_ctx.get("sub_case", 0)
        if sub_case == 4:
            fallback_case_c = (
                f"Sub-case 4: {case_c_ctx.get('partial_shops_evaluated', 0)} shops covering "
                f"{n_specs-1}/{n_specs} specs evaluated. "
                f"Winner shop {case_c_ctx.get('winner_shop_id')} "
                f"(score={case_c_ctx.get('winner_shop_score')}). "
                f"Filled missing spec[{case_c_ctx.get('missing_spec_idx')}] "
                f"('{case_c_ctx.get('missing_spec_keywords')}') by searching within that shop."
            )
        else:
            anchor = case_c_ctx.get("anchor", {})
            fallback_case_c = (
                f"Sub-case {sub_case}: {case_c_ctx.get('tie_note', '')} "
                f"Anchor: spec[{anchor.get('spec_idx')}] '{anchor.get('keywords')}' "
                f"product_id={anchor.get('product_id')} price={anchor.get('price')} "
                f"shop_id={anchor.get('shop_id')}. "
                f"Searched remaining specs within that shop."
            )
        think_case_c = _generate_think_text(
            query,
            {
                "case_c_resolution": case_c_ctx,
                "note": (
                    "No full-coverage shop after score filtering; "
                    "resolved via anchor-product strategy."
                ),
            },
            fallback=fallback_case_c,
        )
        _append_step(think_case_c, [], "", query, steps)

        enriched = _enrich_products_for_reason(
            [{"product_id": pid} for pid in pids_resolved]
        )
        think_found = _generate_think_text(
            query,
            {
                "shop_id": case_c_ctx.get("anchor", {}).get("shop_id")
                           or case_c_ctx.get("winner_shop_id", "resolved"),
                "selected_products": enriched,
                "llm_reasoning": case_c_ctx.get("filled_specs", []),
            },
            fallback=(
                f"Anchor strategy resolved. "
                f"Product IDs: {pids_resolved}."
            ),
        )
        _finish_session(pids_resolved, "success", query, steps, think=think_found)
        return

    _finish_session(
        [FALLBACK_PRODUCT_ID],
        "failure",
        query,
        steps,
        think=f"Could not find a single shop carrying all {n_specs} required products.",
    )


def _run_voucher_search(params: dict, query: str, steps: list) -> None:
    is_shop = params.get("is_shop_voucher", False) or "same shop" in query.lower()
    products = params.get("products", [])

    if is_shop and len(products) > 1:
        _run_same_shop_search(params, query, steps)
        return

    n_specs = len(products)
    if not products:
        _finish_session(
            [FALLBACK_PRODUCT_ID], "failure", query, steps,
            think="No product specs found in voucher query.",
        )
        return

    voucher = _normalize_voucher_fields(params.get("voucher"))
    allowed_total = voucher_max_total_price(voucher)
    if not allowed_total or allowed_total <= 0:
        _finish_session(
            [FALLBACK_PRODUCT_ID], "failure", query, steps,
            think="Could not calculate allowed total from voucher parameters.",
        )
        return

    kw_list = [p.get("keywords", "") for p in products]
    think_analyze = _generate_think_text(
        query,
        {
            "product_count": n_specs,
            "budget": voucher.get("budget"),
            "allowed_total": round(allowed_total, 2),
            "products": [
                {"keywords": p.get("keywords"), "price_range": p.get("price_range")}
                for p in products
            ],
        },
        fallback=(
            f"Voucher task: {n_specs} product(s). "
            f"Budget={voucher.get('budget')}, allowed_total={allowed_total:.2f}. "
            f"Keywords: {kw_list}."
        ),
    )

    scan_tool_calls: list = []
    max_prices: list[float] = []
    for spec in products:
        sp = _spec_to_find_product_params(spec, include_price=False)
        sp["price"] = f"1-{allowed_total:.0f}"
        sp["sort"] = "pricedesc"
        r = safe_tool_call("find_product", sp)
        scan_tool_calls.append(r)
        found = r.get("result") or []
        max_prices.append(float(found[0].get("price", 0)) if found else 0.0)

    _append_step(think_analyze, scan_tool_calls, "", query, steps)
    logger.info(
        "_run_voucher_search: allowed_total=%.2f n_specs=%d max_prices=%s",
        allowed_total, n_specs, max_prices,
    )

    remaining_order: list[int] = sorted(
        range(n_specs), key=lambda i: max_prices[i], reverse=True
    )
    logger.info("_run_voucher_search: initial processing order=%s", remaining_order)

    picked_products: list[Product] = []   # in processing order
    picked_orig_idx: list[int] = []       # original spec indices, same order
    budget_tool_calls: list = []

    while remaining_order:
        position = len(picked_products)
        is_anchor = position == 0
        spent = sum(float(p.get("price", 0)) for p in picked_products)

        found_valid = False
        for candidate_i in list(remaining_order):
            others = [j for j in remaining_order if j != candidate_i]
            reserved = sum(
                (lo or 0.0)
                for j in others
                for lo, _ in [_parse_price_range(products[j].get("price_range"))]
            )
            ceiling = allowed_total - spent - reserved
            if n_specs > 1:
                floor = allowed_total / n_specs if is_anchor else 1.0
            else:
                floor = 1.0

            orig_lo, orig_hi = _parse_price_range(products[candidate_i].get("price_range"))
            final_lo = max(orig_lo if orig_lo is not None else 0.0, floor)
            final_hi = min(orig_hi if orig_hi is not None else float("inf"), ceiling)

            if final_lo > final_hi:
                logger.info(
                    "_run_voucher_search: spec[%d] at position %d: empty intersection "
                    "(%.2f > %.2f), trying next candidate",
                    candidate_i, position, final_lo, final_hi,
                )
                continue

            sp = _spec_to_find_product_params(products[candidate_i], include_price=False)
            sp["price"] = f"{final_lo:.0f}-{final_hi:.0f}"
            cands: list[Product] = []
            seen_pids: set[str] = set()
            for page in range(1, 3):
                r = safe_tool_call("find_product", {**sp, "page": page})
                budget_tool_calls.append(r)
                for p in r.get("result") or []:
                    pid = str(p.get("product_id", ""))
                    if pid and pid not in seen_pids:
                        cands.append(p)
                        seen_pids.add(pid)

            if not cands:
                logger.info(
                    "_run_voucher_search: spec[%d] no results in range %.0f-%.0f, trying next",
                    candidate_i, final_lo, final_hi,
                )
                continue

            spec_q = (
                products[candidate_i].get("query")
                or products[candidate_i].get("keywords")
                or query
            )
            pids_cands = [str(p.get("product_id", "")) for p in cands if p.get("product_id")]
            details = _fetch_product_details(pids_cands)
            chosen = _llm_choose_product(
                spec_q, cands, details,
                only_product_type=bool(products[candidate_i].get("only_product_type", False)),
                model=FALLBACK_MODEL,
            )
            if chosen is None:
                chosen = cands[0]

            picked_products.append(chosen)
            picked_orig_idx.append(candidate_i)
            remaining_order = [j for j in remaining_order if j != candidate_i]
            found_valid = True
            logger.info(
                "_run_voucher_search: position %d → spec[%d] picked "
                "product_id=%s price=%.2f range=%.0f-%.0f",
                position, candidate_i,
                chosen.get("product_id"), float(chosen.get("price", 0)),
                final_lo, final_hi,
            )
            break

        if not found_valid:
            logger.warning(
                "_run_voucher_search: no valid candidate at position %d, failing", position
            )
            think_fail = _generate_think_text(
                query,
                {
                    "position": position,
                    "allowed_total": round(allowed_total, 2),
                    "spent_so_far": round(spent, 2),
                    "note": (
                        f"No product found for spec at processing position {position} "
                        f"that fits within the remaining voucher budget."
                    ),
                },
                fallback=(
                    f"I could not find a suitable product for spec at position {position} "
                    f"within the remaining budget (spent={spent:.2f}, "
                    f"allowed_total={allowed_total:.2f}). Aborting the voucher search."
                ),
            )
            _append_step(think_fail, budget_tool_calls, "", query, steps)
            _finish_session(
                [FALLBACK_PRODUCT_ID], "failure", query, steps,
                think=(
                    f"Could not find a product for spec at position {position} "
                    f"within the voucher budget constraints "
                    f"(allowed_total={allowed_total:.2f})."
                ),
            )
            return

    pid_map = {
        orig_idx: str(picked_products[k].get("product_id", ""))
        for k, orig_idx in enumerate(picked_orig_idx)
    }
    price_map = {
        orig_idx: float(picked_products[k].get("price", 0))
        for k, orig_idx in enumerate(picked_orig_idx)
    }
    pids = [pid_map[i] for i in range(n_specs)]
    total_price = sum(price_map.values())
    enriched = _enrich_products_for_reason([
        {
            "product_id": pid_map[i],
            "title": picked_products[picked_orig_idx.index(i)].get("title", ""),
            "price": picked_products[picked_orig_idx.index(i)].get("price"),
        }
        for i in range(n_specs)
    ])

    fallback_done = (
        f"Voucher search complete. "
        f"Total before discount: {total_price:.2f}, allowed_total={allowed_total:.2f}, "
        f"budget={voucher.get('budget')}. Product IDs: {pids}."
    )
    think_done = _generate_think_text(
        query,
        {
            "selected_products": enriched,
            "total_before_discount": round(total_price, 2),
            "budget_constraint": voucher,
        },
        fallback=fallback_done,
    )
    _append_step(think_done, budget_tool_calls, "", query, steps)
    _finish_session(pids, "success", query, steps)


def agent_main(problem_data: dict) -> list[dict]:
    _product_detail_cache.clear()
    steps: list = []
    query: str = problem_data.get("query", "")

    try:
        task_type = _infer_task_type(query)

        params = _extract_query_params_llm(query, task_type)
        logger.info(f"agent_main -> params: {params}")

        products_info = params.get("products", [])
        keywords_list = [p.get("keywords") or p.get("q", "") for p in products_info]
        price_list = [p.get("price_range") for p in products_info]
        service_list = [p.get("service") for p in products_info]
        fallback_init = (
            f"Query: '{query[:300]}'. "
            f"Search keywords: {keywords_list}. "
            f"Price constraints: {price_list}. "
            f"Service filters: {service_list}."
        )
        ctx_init: dict = {
            "keywords": keywords_list,
            "price_constraints": price_list,
            "service_filters": service_list,
        }
        if products_info:
            p0 = products_info[0]
            if bool(p0.get("only_product_type")):
                ctx_init["only_product_type"] = True
                ctx_init["only_product_type_reason"] = (
                    "The query refers to the product type alone with no additional qualifiers "
                    "(no brand, color, material, or numeric spec). "
                    "Appending 'only' to the search query narrows results to this exact product "
                    "type and avoids unrelated products that merely contain this term."
                )
        if params.get("voucher"):
            v = params["voucher"]
            ctx_init["budget_constraint"] = {
                "discount_type": v.get("discount_type"),
                "discount_value": v.get("discount_value"),
                "threshold": v.get("threshold"),
                "cap": v.get("cap"),
                "budget": v.get("budget"),
            }
        think_init = _generate_think_text(query, ctx_init, fallback=fallback_init)
        _append_step(think_init, [], "", query, steps)

        if task_type == "shop":
            _run_same_shop_search(params, query, steps)
        elif task_type == "voucher":
            _run_voucher_search(params, query, steps)
        else:
            _run_single_product_search(params, query, steps)

    except Exception:
        logger.error("agent_main: unhandled exception", exc_info=True)
        try:
            _finish_session([FALLBACK_PRODUCT_ID], "failure", query, steps)
        except Exception:
            steps.append(create_dialogue_step("Done.", [], "Done.", query, len(steps) + 1))

    if not steps:
        steps.append(create_dialogue_step("Done.", [], "Done.", query, 1))

    logger.info("agent_main: completed with %d steps", len(steps))
    return steps
