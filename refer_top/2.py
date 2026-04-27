import json
import logging
import re
import threading
import time
from collections import defaultdict
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Set, Tuple
from urllib.parse import quote_plus

from src.agent.agent_interface import Tool, create_dialogue_step, execute_tool_call
from src.agent.proxy_client import ProxyClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

Product = Dict[str, Any]
SearchSpec = Dict[str, Any]
FALLBACKID: str = "0"
LLM_MODEL = "deepseek-ai/DeepSeek-V3.1-Terminus-TEE"
LLM_MODEL_CANDIDATES = [
    "deepseek-ai/DeepSeek-V3.1-Terminus-TEE",
    "moonshotai/Kimi-K2.5-TEE",
    "Qwen/Qwen3-235B-A22B-Instruct-2507-TEE",
    "deepseek-ai/DeepSeek-V3.1-TEE",
    "MiniMaxAI/MiniMax-M2.5-TEE",
]
LLM_MODEL_ROUNDS = 3
PRODUCT_JUDGE_MODEL_CANDIDATES: Optional[List[str]] = ["moonshotai/Kimi-K2.5-TEE", "deepseek-ai/DeepSeek-V3.1-Terminus-TEE", "MiniMaxAI/MiniMax-M2.5-TEE"]
PRODUCT_JUDGE_MODEL_ROUNDS = LLM_MODEL_ROUNDS
PRODUCT_JUDGE_MAX_RETRIES = 3

SINGLE_EXTRACT_MODEL_CANDIDATES: Optional[List[str]] = [
    "deepseek-ai/DeepSeek-V3.1-Terminus-TEE",
    "moonshotai/Kimi-K2.5-TEE",
    "Qwen/Qwen3-235B-A22B-Instruct-2507-TEE",
    "deepseek-ai/DeepSeek-V3.1-TEE"
]
SINGLE_EXTRACT_MODEL_ROUNDS: Optional[int] = None
SINGLE_JUDGE_MODEL_CANDIDATES: Optional[List[str]] = [
    "moonshotai/Kimi-K2.5-TEE",
    "Qwen/Qwen3-235B-A22B-Instruct-2507-TEE",
    "deepseek-ai/DeepSeek-V3.1-TEE"
]
SINGLE_JUDGE_MODEL_ROUNDS: Optional[int] = None
TOP_RELEVANCE_CANDIDATES = 10
MAX_SHOPS_WIDE_QUERY = 6
MAX_SHOPS_FOR_TWO_OR_FEWER_SPECS = 8
TOOL_CALL_DELAY = 0.5
TOOL_CALL_MAX_RETRIES = 3
TOOL_CALL_BASE_BACKOFF = 1.0
DEFAULT_PRODUCT_QUERY = "product"
CHEAPER_PRICE_TIEBREAK_DIVISOR = 100_000


_inference_client = ProxyClient(timeout=30, max_retries=5)
_search_client = ProxyClient(timeout=25, max_retries=5)
_product_detail_cache: dict[str, dict] = {}
_oro_request_times: list[float] = []
_oro_rate_limit_lock = threading.Lock()
_ORO_MAX_REQUESTS_PER_MINUTE = 90
_ORO_WINDOW_SECONDS = 60.0
_ORO_MIN_INTERVAL_SECONDS = 0.7
_last_tool_call_time = 0.0

_SCORING_STOPWORDS: frozenset[str] = frozenset([
    "the","a","an","for","with","from","that","this","i","me","my","looking","show",
    "find","want","need","get","finish","buy","also","and","in","is","it","am","im",
    "priced","pesos","php","price","between","than","above","below","more","less",
    "over","under","of","to","or","on","at","by","its","be","can","has","have","will",
    "would","should","item","items","both","these","offering","sells","shop","budget",
    "voucher","discount","first","second","third","brand","made","using","available",
    "support","supports","compatible","please","looking","age",
])

_PRODUCT_PROMPT = """Extract search params as JSON. No markdown.
{"products":[{"keywords":"search query","price_range":"min-max"|null,"service":"official"|"freeShipping"|"COD"|"flashsale"|null}],"is_shop_voucher":bool}
- keywords: product type + brand + material + color + size + use(strict in order of appearance). 3-6 words, Drop any trailing term whose primary function is to attach, hold, or secure items together, when it appears after the core product name. Clean up the keywords by removing "fastener" / "clamp" / "clasp" / "holder" / or similar words.
- price_range: "100-500", "100-", "0-500". null if none.
- service: "LazMall / guaranteed authenticity / quick returns" → "official", "free shipping / free delivery" → "freeShipping", "cash on delivery / COD / payment on delivery" → "COD", "LazFlash / flash deal / limited-time deal" → "flashsale"; null if none. Multiple options available, combine them with ",".
JSON only:"""

_SHOP_PROMPT = """Extract search params as JSON. No markdown.
{"task_type":"product"|"shop"|"voucher","products":[{"keywords":"search query","price_range":"min-max"|null,"service":"official"|"freeShipping"|"COD"|"flashsale"|null}]"}
- keywords: product type + brand + material + color + size + use + purpose/style. 3-10 words, include ALL qualifying terms.
    if keywords less than 2, add a related word like 'product' to make it longer.(ex: shoe -> shoe products)
- price_range: "100-500", "100-", "0-500". null if none.
- service: LazMall=official, free shipping=freeShipping, COD=COD, flash sale=flashsale. null if none.
- task_type: product=single, shop=same-shop multi, voucher=budget/discount.
- Multi-product: one entry per product, preserve order. Budget/voucher info are NOT products.
- is_shop_voucher: true only if the query mentions "same shop" in voucher task.
JSON only:"""

_VOUCHER_PROMPT = """Extract search params as JSON. No markdown. Find products that fit within a budget after applying a voucher discount.
{"products":[{"keywords":"search query","price_range":"min-max"|null,"service":"official"|"freeShipping"|"COD"|"flashsale"|null}], "voucher": { "voucher_type": "platform|shop", "discount_type": "fixed|percentage", "discount_value": "fixed amount OR percentage number e.g. 42 for 42%", "threshold": "minimum total price for voucher to apply", "cap": "max discount for percentage vouchers, 0 if not mentioned or fixed type", "budget": "the user's maximum budget" }, "is_shop_voucher":bool}
- keywords: product type + brand + material + color + use. 3-6 words, include ALL qualifying terms that are explicitly mentioned in the query, but MUST never include service related terms in keywords.
- price_range: "100-500", "100-", "0-500". null if none.
- service: "LazMall / guaranteed authenticity / quick returns" → "official", "free shipping / free delivery" → "freeShipping", "cash on delivery / COD / payment on delivery" → "COD", "LazFlash / flash deal / limited-time deal" → "flashsale"; null if none. Multiple options available, combine them with ",".
- Multi-product: one entry per product, preserve order. Budget/voucher info are NOT products.
- is_shop_voucher: true if "same shop" voucher.
JSON only:"""

_TASK_EXTRACTION_PROMPTS = {"product": _PRODUCT_PROMPT, "shop": _SHOP_PROMPT, "voucher": _VOUCHER_PROMPT}

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

_MULTI_PRODUCT_SPLIT_RE = re.compile(
    r"(?:,?\s*and\s+also\s+|,?\s*also,?\s+|Second(?:ly)?,\s*|Third(?:ly)?,\s*"
    r"|First,\s*|\(\d+\)\s*|\d+\.\s*|Additionally,\s*"
    r"|[.]\s*Next,\s*|[.]\s*Lastly,\s*|[.]\s*Finally,\s*|[.]\s*Last,\s*)",
    re.IGNORECASE,
)
_BUDGET_SPLIT_RE = re.compile(r"(?:My budget|budget is|I have a voucher)", re.IGNORECASE)


_REGEX_STOPWORDS = {
    "the","and","for","with","from","that","this","are","was","can","has","have",
    "been","will","find","finish","looking","show","want","need","get","buy","product",
    "products","search","same","shop","within","budget","voucher","discount","price",
    "priced","pesos","php","between","than","greater","less","more","under","over",
    "about","also","both","these","them","each","all","any","one","two","three","four",
    "five","offering","sells","using","in","is","it","its","or","at","on","by","be",
    "do","an","my","me","im","items","item","only","just","first","second","supports",
    "support","compatible","available","made","please","like","of","above","deals",
    "options","option","delivery","shipping","offers","lazmall","lazflash","official",
    "cash","payment","pay","cost","costs","via","themed","such","those","store","stores",
    "focus","category","specifically","guaranteed","authenticity","returns","quick",
    "perks","should","help","purchase","type","to","named","called","family","belongs",
    "comes","another","lastly","benefits","you","weighing","capacity","size","sized",
    "eu","fits",
}

def _wait_for_oro_request_slot() -> None:
    while True:
        sleep_for = 0.0
        with _oro_rate_limit_lock:
            now = time.monotonic()
            cutoff = now - _ORO_WINDOW_SECONDS
            while _oro_request_times and _oro_request_times[0] <= cutoff:
                _oro_request_times.pop(0)
            if _oro_request_times:
                gap = now - _oro_request_times[-1]
                if gap < _ORO_MIN_INTERVAL_SECONDS:
                    sleep_for = max(sleep_for, _ORO_MIN_INTERVAL_SECONDS - gap)
            if len(_oro_request_times) >= _ORO_MAX_REQUESTS_PER_MINUTE:
                sleep_for = max(sleep_for, _ORO_WINDOW_SECONDS - (now - _oro_request_times[0]))
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
            time.sleep(TOOL_CALL_BASE_BACKOFF * (2 ** attempt))


@Tool
def find_product(
    q: str, page: int = 1, shop_id: str | None = None,
    price: str | None = None, sort: str | None = None, service: str | None = None,
) -> list[dict]:
    """Search for products matching query."""
    params = {"q": quote_plus(q), "page": page, "shop_id": shop_id,
              "price": price, "sort": sort, "service": service}
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
def find_product_for_shop(
    q: str, page: int = 1, shop_id: str | None = None,
    price: str | None = None, sort: str | None = None, service: str | None = None,
) -> list[dict]:
    """Search for products matching query."""
    params = _build_search_params(
        q,
        page=page,
        shop_id=shop_id,
        price=price,
        sort=sort,
        service=service,
    )
    result = _search_products(params)
    if shop_id and not result:
        retry_params = {**params}
        retry_params.pop("service", None)
        result = _search_products(retry_params)
    return result


def _normalize_service(service: Optional[str]) -> Optional[str]:
    if not service or service == "default":
        return None
    parts = [s.strip() for s in service.split(",") if s.strip() and s.strip() != "default"]
    return ",".join(parts) or None


def _build_search_params(query: str, *, page: int = 1, shop_id: Optional[str] = None,
                         price: Optional[str] = None, sort: Optional[str] = None,
                         service: Optional[str] = None) -> Dict[str, Any]:
    params: Dict[str, Any] = {"q": quote_plus(query), "page": page}
    if shop_id: params["shop_id"] = shop_id
    if price: params["price"] = price
    if sort and sort != "default": params["sort"] = sort
    svc = _normalize_service(service)
    if svc: params["service"] = svc
    return params


def _search_products(params: Dict[str, Any]) -> List[Product]:
    return _search_client.get("/search/find_product", params) or []


def _search_products_for_spec(spec: SearchSpec, *, shop_id: Optional[str] = None,
                              include_price: bool = True,
                              omit_service_from_api: bool = False) -> List[Product]:
    price = None
    if include_price:
        price = spec.get("price") or spec.get("price_range")
    service = None if omit_service_from_api else spec.get("service")
    return _search_products(_build_search_params(
        spec.get("q") or spec.get("keywords") or "product",
        shop_id=shop_id, price=price, service=service,
    ))


def _product_matches_services(product: Product, service_spec: Optional[str]) -> bool:
    if not service_spec:
        return True
    required = [s.strip() for s in str(service_spec).split(",") if s.strip()]
    if not required:
        return True
    offered = product.get("service") or []
    return all(r in offered for r in required)


def _group_products_by_shop(broad_results: Sequence[Sequence[Product]]) -> Dict[str, Dict[int, List[Product]]]:
    shop_coverage: Dict[str, Dict[int, List[Product]]] = defaultdict(lambda: defaultdict(list))
    for idx, products in enumerate(broad_results):
        for p in products:
            sid = str(p.get("shop_id", ""))
            if sid:
                shop_coverage[sid][idx].append(p)
    return shop_coverage


def _filter_products_by_spec_services(products: Sequence[Product], spec: SearchSpec) -> List[Product]:
    svc = spec.get("service")
    if not svc:
        return list(products)
    return [p for p in products if _product_matches_services(p, svc)]


def _pick_products_for_shop(
    shop_id: str,
    shop_coverage: Dict[str, Dict[int, List[Product]]],
    specs: Sequence[SearchSpec],
    original_query: str,
    *,
    broad_omit_service: bool
) -> Optional[List[Product]]:

    selected: List[Product] = []
    used_ids: Set[str] = set()
    coverage = shop_coverage.get(shop_id, {})

    product_lists = {}
    best_lists = {}
    used_lists = {}

    def fetch_products(idx: int, spec: SearchSpec) -> List[Product]:
        # Try coverage first
        products = list(coverage.get(idx) or [])
        if spec.get("service"):
            products = _filter_products_by_spec_services(products, spec)

        # Try search with different omit_service flags
        if not products:
            for omit_service in (broad_omit_service, True):
                products = _search_products_for_spec(spec, shop_id=shop_id, omit_service_from_api=omit_service)
                products = _filter_products_by_spec_services(products, spec)
                if products:
                    break
        return products

    for idx, spec in enumerate(specs):
        q = spec.get("q", "")
        score_q = original_query or q

        products = fetch_products(idx, spec)
        best = _select_best_product(products, q or score_q, prefer_cheaper=True, exclude_ids=used_ids)

        # store state
        product_lists[idx] = products
        used_lists[idx] = set(used_ids)
        best_lists[idx] = best

        # fallback if no best found
        if not best:
            best = _select_best_product(products, q or score_q, prefer_cheaper=True, exclude_ids=set())

            if idx > 0 and best:
                # backtracking to previous selections
                for prev in range(idx):
                    prev_spec = specs[prev]
                    prev_q = prev_spec.get("q", "")
                    prev_score_q = original_query or prev_q
                    prev_products = product_lists[prev]

                    prev_best = best_lists.get(prev)
                    if prev_best and str(best.get("product_id")) == str(prev_best.get("product_id")):
                        pid = str(best.get("product_id", ""))
                        used_lists[prev].add(pid)

                        new_prev_best = _select_best_product(
                            prev_products,
                            prev_q or prev_score_q,
                            prefer_cheaper=True,
                            exclude_ids=used_lists[prev]
                        )

                        if new_prev_best:
                            selected[prev] = new_prev_best
                            best_lists[prev] = new_prev_best

                            new_id = str(new_prev_best.get("product_id", ""))
                            used_ids.discard(pid)
                            used_ids.add(new_id)

                            # retry current best
                            best = _select_best_product(
                                products,
                                q or score_q,
                                prefer_cheaper=True,
                                exclude_ids=used_ids
                            )
                            if best:
                                break
            else:
                return None

        selected.append(best)
        pid = str(best.get("product_id", ""))
        if pid:
            used_ids.add(pid)

    return selected


def _score_product_relevance(product: Product, query_text: str,
                             detail: Optional[Product] = None) -> float:
    title = product.get("title", "").lower()
    t_words = set(re.findall(r"\b\w+\b", title))
    q_words = list(dict.fromkeys(
        w for w in re.findall(r"\b\w+\b", query_text.lower())
        if w not in _SCORING_STOPWORDS and len(w) > 1
    ))
    score = 0
    for qw in q_words:
        if qw in t_words:
            score += 2
        elif qw.endswith("s") and qw[:-1] in t_words:
            score += 2
        elif not qw.endswith("s") and f"{qw}s" in t_words:
            score += 2
        elif len(qw) >= 3 and any(tw.startswith(qw) for tw in t_words if len(tw) > len(qw)):
            score += 2
        elif any(qw.startswith(tw) or tw.startswith(qw) for tw in t_words if len(tw) > 2):
            score += 1
        if any(c.isdigit() for c in qw) and qw in title:
            score += 2
    if detail:
        exact_values: set[str] = set()
        detail_words: set[str] = set()
        for k, vs in (detail.get("attributes") or {}).items():
            detail_words.update(re.findall(r"\b\w+\b", k.replace("_", " ").lower()))
            for v in vs if isinstance(vs, list) else [vs]:
                v_str = str(v).strip().lower()
                exact_values.add(v_str)
                detail_words.update(re.findall(r"\b\w+\b", v_str))
        for opts in (detail.get("sku_options") or {}).values():
            if isinstance(opts, dict):
                for k, v in opts.items():
                    v_str = str(v).strip().lower()
                    detail_words.update(re.findall(r"\b\w+\b", k.replace("_", " ").lower()))
                    detail_words.update(re.findall(r"\b\w+\b", v_str))
                    exact_values.add(v_str)
        for qw in q_words:
            if qw in exact_values:
                score += 3
            elif f"{qw}#" in exact_values:
                score += 5
            elif qw in detail_words:
                score += 2
    return score


def _score_shop_coverage(shop_id: str, shop_coverage: Dict[str, Dict[int, List[Product]]],
                         specs: Sequence[SearchSpec], original_query: str) -> tuple[int, float]:
    coverage = shop_coverage[shop_id]
    total_score = 0.0
    for idx, products in coverage.items():
        q = original_query or specs[idx].get("q", "")
        filtered = _filter_products_by_spec_services(products, specs[idx])
        pool = filtered or products
        total_score += max((_score_product_relevance(p, q) for p in pool), default=0)
    return len(coverage), total_score


def _pick_products_for_voucher(shop_id: str, shop_coverage: Dict[str, Dict[int, List[Product]]],
                            specs: Sequence[SearchSpec], original_query: str, *,
                            broad_omit_service: bool) -> Optional[List[Product]]:
    selected: List[Product] = []
    used_ids: Set[str] = set()
    coverage = shop_coverage.get(shop_id, {})
    for idx, spec in enumerate(specs):
        q = spec.get("q", "")
        score_q = original_query or q
        products = list(coverage.get(idx) or [])
        if spec.get("service"):
            products = _filter_products_by_spec_services(products, spec)
        if not products:
            products = _search_products_for_spec(spec, shop_id=shop_id, omit_service_from_api=broad_omit_service)
            products = _filter_products_by_spec_services(products, spec)
        if not products:
            products = _search_products_for_spec(spec, shop_id=shop_id, omit_service_from_api=True)
            products = _filter_products_by_spec_services(products, spec)
        best = _select_best_product(products, q if idx else score_q, prefer_cheaper=True, exclude_ids=used_ids)
        if not best:
            return None
        selected.append(best)
        pid = str(best.get("product_id", ""))
        if pid:
            used_ids.add(pid)
    return selected

def _collect_broad_shop_results(
    specs: Sequence[SearchSpec], *, omit_service_from_api: bool = False
) -> List[List[Product]]:
    return [
        _search_products_for_spec(spec, omit_service_from_api=omit_service_from_api) for spec in specs
    ]


def _parse_product_queries(product_queries: Any) -> tuple[Optional[List[SearchSpec]], str, Optional[str]]:
    try:
        specs = json.loads(product_queries) if isinstance(product_queries, str) else product_queries
    except json.JSONDecodeError:
        return None, "", "Invalid JSON"

    if not specs or not isinstance(specs, list):
        return None, "", "Need non-empty list"

    original_query = ""
    if isinstance(specs[-1], dict) and specs[-1].get("_original_query"):
        original_query = specs.pop()["_original_query"]

    return specs, original_query, None


def _serialize_products(products: Sequence[Product]) -> List[Dict[str, Any]]:
    return [
        {
            "product_id": product.get("product_id"),
            "title": product.get("title", ""),
            "price": product.get("price"),
            "shop_id": product.get("shop_id"),
        }
        for product in products
    ]


@Tool
def find_products_in_same_shop(product_queries: str) -> Dict[str, Any]:
    """Find multiple products from the SAME shop."""
    specs, original_query, error = _parse_product_queries(product_queries)
    if error:
        return {"found": False, "error": error}
    if specs is None:
        return {"found": False, "error": "Need non-empty list"}

    max_shops = (
        MAX_SHOPS_FOR_TWO_OR_FEWER_SPECS if len(specs) <= 2 else MAX_SHOPS_WIDE_QUERY
    )
    shops_tried_total = 0

    for broad_omit_service in (False, True):
        broad_results = _collect_broad_shop_results(specs, omit_service_from_api=broad_omit_service)
        if not any(broad_results):
            continue

        shop_coverage = _group_products_by_shop(broad_results)
        candidate_shop_ids = sorted(
            shop_coverage,
            key=lambda sid: _score_shop_coverage(sid, shop_coverage, specs, original_query),
            reverse=True,
        )

        for shops_tried, shop_id in enumerate(candidate_shop_ids[:max_shops], start=1):
            shops_tried_total = shops_tried
            picked = _pick_products_for_shop(
                shop_id,
                shop_coverage,
                specs,
                original_query,
                broad_omit_service=broad_omit_service,
            )
            if picked is not None and len(picked) == len(specs):
                return {
                    "found": True,
                    "shop_id": shop_id,
                    "products": _serialize_products(picked),
                    "shops_tried": shops_tried,
                }

    return {
        "found": False,
        "error": f"No shop has all {len(specs)} products",
        "shops_tried": min(shops_tried_total, max_shops),
    }


@Tool
def find_voucher_products_in_same_shop(product_queries: str) -> Dict[str, Any]:
    """Find multiple products from the SAME shop."""
    try:
        specs = json.loads(product_queries) if isinstance(product_queries, str) else product_queries
    except json.JSONDecodeError:
        return {"found": False, "error": "Invalid JSON"}
    if not specs or not isinstance(specs, list):
        return {"found": False, "error": "Need non-empty list"}
    original_query = ""
    if isinstance(specs[-1], dict) and specs[-1].get("_original_query"):
        original_query = specs.pop()["_original_query"]
    n_specs = len(specs)
    max_shops = MAX_SHOPS_FOR_TWO_OR_FEWER_SPECS if n_specs <= 2 else MAX_SHOPS_WIDE_QUERY
    shops_tried_total = 0
    for broad_omit_service in (False, True):
        broad_results = [_search_products_for_spec(s, omit_service_from_api=broad_omit_service) for s in specs]
        if not any(broad_results):
            continue
        shop_coverage = _group_products_by_shop(broad_results)
        candidates = sorted(shop_coverage, key=lambda sid: _score_shop_coverage(sid, shop_coverage, specs, original_query), reverse=True)
        for tried, sid in enumerate(candidates[:max_shops], 1):
            shops_tried_total = tried
            picked = _pick_products_for_voucher(sid, shop_coverage, specs, original_query, broad_omit_service=broad_omit_service)
            if picked and len(picked) == n_specs:
                return {
                    "found": True, "shop_id": sid, "shops_tried": tried,
                    "products": [{"product_id": p.get("product_id"), "title": p.get("title", ""),
                                  "price": p.get("price"), "shop_id": p.get("shop_id")} for p in picked],
                }
    return {"found": False, "error": f"No shop has all {n_specs} products", "shops_tried": min(shops_tried_total, max_shops)}


@Tool
def calculate_voucher(product_prices: str, voucher_type: str, discount_value: float,
                      threshold: float, budget: float, cap: float = 0) -> dict:
    """Calculate the final price after applying a voucher discount."""
    try:
        prices = [float(p.strip()) for p in str(product_prices).split(",")]
    except ValueError:
        return {"error": "Invalid product_prices format."}
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
    return {"prices": prices, "total_before": round(total, 2), "discount_amount": round(discount, 2),
            "total_after": round(total_after, 2), "within_budget": total_after <= budget,
            "voucher_applied": voucher_applied, "budget": budget}


@Tool
def recommend_product(product_ids: str) -> str:
    """Recommend products to the user."""
    return f"Having recommended the products to the user: {product_ids}."


@Tool
def terminate(status: str = "success") -> str:
    """End the dialogue."""
    return f"The interaction has been completed with status: {status}"


def _fetch_product_details(product_ids: list[str]) -> dict[str, dict]:
    if not product_ids:
        return {}
    uncached = [pid for pid in product_ids if pid not in _product_detail_cache]
    for i in range(0, len(uncached), 10):
        batch = uncached[i:i + 10]
        result = _oro_get("/search/view_product_information", {"product_ids": ",".join(batch)})
        if result and isinstance(result, list):
            for p in result:
                _product_detail_cache[str(p.get("product_id", ""))] = p
    return {pid: _product_detail_cache[pid] for pid in product_ids if pid in _product_detail_cache}


def _score_product(product: dict, query_text: str, detail: dict = None) -> int:
    title = product.get("title", "").lower()
    t_words = set(re.findall(r"\b\w+\b", title))
    q_words = list(dict.fromkeys(
        w for w in re.findall(r"\b\w+\b", query_text.lower())
        if w not in _SCORING_STOPWORDS and len(w) > 1
    ))
    score = 0
    for qw in q_words:
        if (qw in t_words or qw.endswith("s") and qw[:-1] in t_words
                or not qw.endswith("s") and (qw + "s") in t_words
                or len(qw) >= 3 and any(tw.startswith(qw) for tw in t_words if len(tw) > len(qw))):
            score += 2
        elif any(qw.startswith(tw) or tw.startswith(qw) for tw in t_words if len(tw) > 2):
            score += 1
        if any(c.isdigit() for c in qw) and qw in title:
            score += 2
    if detail:
        exact_values = set()
        attr_words = set()
        for k, vs in (detail.get("attributes") or {}).items():
            attr_words.update(re.findall(r"\b\w+\b", k.replace("_", " ").lower()))
            for v in vs if isinstance(vs, list) else [vs]:
                v_str = str(v).strip().lower()
                exact_values.add(v_str)
                attr_words.update(re.findall(r"\b\w+\b", v_str))
        for _sid, opts in (detail.get("sku_options") or {}).items():
            if isinstance(opts, dict):
                for k, v in opts.items():
                    v_str = str(v).strip().lower()
                    exact_values.add(v_str)
                    attr_words.update(re.findall(r"\b\w+\b", v_str))
                    attr_words.update(re.findall(r"\b\w+\b", k.replace("_", " ").lower()))
        for qw in q_words:
            if qw in exact_values: score += 3
            elif (qw + "#") in exact_values: score += 5
            elif qw in attr_words: score += 2
    return score


def _parse_price_range_str(price_range: str) -> tuple:
    if not price_range or not isinstance(price_range, str):
        return None, None
    parts = price_range.split("-", 1)
    try: lo = float(parts[0]) if parts[0].strip() else None
    except ValueError: lo = None
    try: hi = float(parts[1]) if len(parts) > 1 and parts[1].strip() else None
    except ValueError: hi = None
    return lo, hi


def _score_product_for_product_case(product: dict, query_text: str,
                                    detail: dict = None, parsed_spec: dict = None) -> float:
    title = product.get("title", "").lower()
    t_words = set(re.findall(r"\b\w+\b", title))
    q_words = list(dict.fromkeys(
        w for w in re.findall(r"\b\w+\b", query_text.lower())
        if w not in _SCORING_STOPWORDS and len(w) > 1
    ))
    spec = parsed_spec or {}
    score = 0.0
    for qw in q_words:
        if qw in t_words: score += 2
        elif qw.endswith("s") and qw[:-1] in t_words: score += 2
        elif not qw.endswith("s") and (qw + "s") in t_words: score += 2
        elif len(qw) >= 3 and any(tw.startswith(qw) for tw in t_words if len(tw) > len(qw)): score += 2
        elif any(qw.startswith(tw) or tw.startswith(qw) for tw in t_words if len(tw) > 2): score += 1
        if any(c.isdigit() for c in qw) and qw in title: score += 2
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
            score += 5 if svc in product_services else -15
    else:
        if product_services:
            for svc in product_services:
                if svc not in ["COD", "official"]:
                    score -= 4
    
    if detail:
        exact_values: set[str] = set()
        attr_words: set[str] = set()
        brand_values: set[str] = set()
        for k, vs in (detail.get("attributes") or {}).items():
            k_lower = k.lower()
            attr_words.update(re.findall(r"\b\w+\b", k_lower.replace("_", " ")))
            for v in vs if isinstance(vs, list) else [vs]:
                v_str = str(v).strip().lower()
                exact_values.add(v_str)
                attr_words.update(re.findall(r"\b\w+\b", v_str))
        for _sid, opts in (detail.get("sku_options") or {}).items():
            if isinstance(opts, dict):
                for k, v in opts.items():
                    v_str = str(v).strip().lower()
                    exact_values.add(v_str)
                    attr_words.update(re.findall(r"\b\w+\b", v_str))
                    attr_words.update(re.findall(r"\b\w+\b", k.lower().replace("_", " ")))
        for qw in q_words:
            if qw in brand_values: score += 8
            elif qw in exact_values: score += 5
            elif (qw + "#") in exact_values: score += 5
            elif qw in attr_words: score += 2

    print("product id: ", product.get("product_id"))
    print("score: ", score)
    return score


def _select_best_product(products: Sequence[Product], query_text: str, *,
                         prefer_cheaper: bool = False,
                         exclude_ids: Optional[Set[str]] = None) -> Optional[Product]:
    if not products:
        return None
    if exclude_ids:
        products = [p for p in products if str(p.get("product_id", "")) not in exclude_ids]
    if not products:
        return None
    top = sorted(products, key=lambda p: _score_product_relevance(p, query_text), reverse=True)[:TOP_RELEVANCE_CANDIDATES]
    details = _fetch_product_details([str(p.get("product_id", "")) for p in top if p.get("product_id")])
    def final_score(p: Product) -> float:
        s = _score_product_relevance(p, query_text, details.get(str(p.get("product_id", ""))))
        if prefer_cheaper:
            s -= (p.get("price", 0) or 0) / 100_000
        return s
    return max(top, key=final_score)


def _select_best_product_for_voucher_case(products: list, query_text: str,
                                          top_count: int = 10, prefer_cheaper: bool = False,
                                          sel_count: int = 1) -> dict | None:
    if not products:
        return None
    top = sorted(products, key=lambda p: _score_product(p, query_text), reverse=True)[:top_count]
    pids = [str(p.get("product_id", "")) for p in top if p.get("product_id")]
    details = _fetch_product_details(pids)
    def _final_score(p: dict) -> float:
        s = _score_product(p, query_text, details.get(str(p.get("product_id", ""))))
        if prefer_cheaper:
            s -= (p.get("price", 0) or 0) / 100_000
        return s
    if sel_count == 1:
        return max(top, key=_final_score)
    return sorted(top, key=_final_score, reverse=True)[:sel_count]


def _iter_inference_attempts(
    *,
    messages: Sequence[Dict[str, str]],
    temperature: float = 0,
    model_candidates: Optional[Sequence[str]] = None,
    rounds: Optional[int] = None,
    log_tag: str = "LLM",
) -> Iterator[Tuple[int, int, str, Optional[Dict[str, Any]]]]:
    models = [model for model in (model_candidates or LLM_MODEL_CANDIDATES) if model]
    total_rounds = rounds if rounds is not None else LLM_MODEL_ROUNDS
    total_rounds = max(1, int(total_rounds))
    total_attempts = len(models) * total_rounds
    attempt = 0
    for round_index in range(1, total_rounds + 1):
        for model in models:
            attempt += 1
            logger.info(
                "[%s] attempt %d/%d round=%d/%d model=%s",
                log_tag,
                attempt,
                total_attempts,
                round_index,
                total_rounds,
                model,
            )
            result = _inference_client.post(
                "/inference/chat/completions",
                json_data={
                    "model": model,
                    "temperature": temperature,
                    "stream": False,
                    "messages": list(messages),
                },
            )
            if not (result and result.get("choices")):
                logger.warning(
                    "[%s] failed attempt %d/%d round=%d/%d model=%s",
                    log_tag,
                    attempt,
                    total_attempts,
                    round_index,
                    total_rounds,
                    model,
                )
            yield attempt, total_attempts, model, result


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


def _product_judge_payload(product: dict, detail: dict | None, score: float, query_text: str) -> dict:
    sku_options = (detail or {}).get("sku_options", {}) or {}
    q_words = set(w for w in re.findall(r"\b\w+\b", query_text.lower()) if w not in _SCORING_STOPWORDS and len(w) > 1)
    ranked = []
    for opt in sku_options.values():
        if not isinstance(opt, dict): continue
        opt_words = set(w for w in re.findall(r"\b\w+\b", " ".join(str(v).lower() for v in opt.values())) if len(w) > 1)
        ranked.append((len(q_words & opt_words), opt))
    sku_preview, seen = [], set()
    for _, opt in sorted(ranked, key=lambda x: x[0], reverse=True):
        key = json.dumps(opt, sort_keys=True, ensure_ascii=False)
        if key not in seen:
            seen.add(key)
            sku_preview.append(opt)
    return {"product_id": str(product.get("product_id", "")).strip(), "title": product.get("title", ""),
            "price": product.get("price"), "service": product.get("service", []),
            "attributes": (detail or {}).get("attributes", {}),
            "sku_options_preview": sku_preview[:8], "heuristic_score": score}


def _llm_choose_tied_product(
    query_text: str,
    tied_candidates: list,
    details: dict[str, dict],
    *,
    model_candidates: Optional[List[str]] = None,
    model_rounds: Optional[int] = None,
) -> dict | None:
    if len(tied_candidates) < 2:
        return tied_candidates[0][0] if tied_candidates else None
    payload = {
        "request": query_text,
        "candidates": [_product_judge_payload(p, details.get(str(p.get("product_id", ""))), s, query_text)
                       for p, s in tied_candidates[:5]],
    }
    user_content = json.dumps(payload, ensure_ascii=False)
    judge_models = model_candidates or PRODUCT_JUDGE_MODEL_CANDIDATES or LLM_MODEL_CANDIDATES
    judge_rounds = model_rounds if model_rounds is not None else PRODUCT_JUDGE_MODEL_ROUNDS
    for attempt, total_attempts, _model, result in _iter_inference_attempts(
        messages=[
            {"role": "system", "content": _PRODUCT_JUDGE_PROMPT},
            {"role": "user", "content": user_content},
        ],
        temperature=0,
        model_candidates=judge_models,
        rounds=judge_rounds,
        log_tag="ProductJudgeLLM",
    ):
        if not (result and result.get("choices")):
            logger.warning(
                "[ProductJudge] no LLM response on attempt %d/%d",
                attempt, total_attempts,
            )
            continue
        content = result["choices"][0].get("message", {}).get("content", "")
        parsed = _parse_json_object_from_llm(content)
        if not isinstance(parsed, dict):
            logger.warning(
                "[ProductJudge] invalid JSON response on attempt %d/%d",
                attempt, total_attempts,
            )
            continue
        best_pid = str(parsed.get("best_product_id", "")).strip()
        for product, _ in tied_candidates[:5]:
            if str(product.get("product_id", "")).strip() == best_pid:
                return product
    return None


def _select_best_product_for_product_case(products: list, query_text: str, top_count: int = 10,
                                          prefer_cheaper: bool = False, parsed_spec: dict = None,
                                          judge_model_candidates: Optional[List[str]] = None,
                                          judge_model_rounds: Optional[int] = None) -> dict | None:
    if not products:
        return None
    top = sorted(products, key=lambda p: _score_product_for_product_case(p, query_text, parsed_spec=parsed_spec), reverse=True)[:top_count]
    pids = [str(p.get("product_id", "")) for p in top if p.get("product_id")]
    details = _fetch_product_details(pids)
    def _final_score(p: dict) -> float:
        s = _score_product_for_product_case(p, query_text, details.get(str(p.get("product_id", ""))), parsed_spec=parsed_spec)
        if prefer_cheaper:
            s -= (p.get("price", 0) or 0) / 100_000
        return s
    scored = [(p, _final_score(p)) for p in top]
    if not scored:
        return None
    best_score = max(s for _, s in scored)
    tied = [(p, s) for p, s in scored if s == best_score]

    print("tied: ", tied)

    if len(tied) >= 2:
        llm_choice = _llm_choose_tied_product(
            query_text, tied, details,
            model_candidates=judge_model_candidates,
            model_rounds=judge_model_rounds,
        )
        if llm_choice is not None:
            return llm_choice
    return tied[0][0]


def _infer_task_type(query: str) -> str:
    q = query.lower()
    if "voucher" in q or "budget" in q or "discount" in q:
        return "voucher"
    if "shop" in q and any(w in q for w in ("both", "these", "offering", "sells", "same")):
        return "shop"
    return "product"


def _sanitize_keyword_text(text: str | None) -> str:
    if not text:
        return "product"
    filtered = [w for w in text.lower().split() if w not in _SCORING_STOPWORDS]
    return " ".join(dict.fromkeys(filtered)) if filtered else "product"


def _sanitize_product_search_params(params: dict) -> dict:
    sanitized = dict(params)
    products = []
    for p in sanitized.get("products", []) or []:
        if not isinstance(p, dict): continue
        cleaned = dict(p)
        if "keywords" in cleaned:
            cleaned["keywords"] = _sanitize_keyword_text(cleaned.get("keywords"))
        if "q" in cleaned:
            cleaned["q"] = _sanitize_keyword_text(cleaned.get("q"))
        products.append(cleaned)
    if products:
        sanitized["products"] = products
    return sanitized


def _extract_query_params_llm(
    query: str,
    kw_task: str,
    *,
    model_candidates: Optional[List[str]] = None,
    model_rounds: Optional[int] = None,
) -> dict:
    system_prompt = _TASK_EXTRACTION_PROMPTS.get(kw_task, _PRODUCT_PROMPT)
    for attempt, total_attempts, model, result in _iter_inference_attempts(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query},
        ],
        temperature=0,
        model_candidates=model_candidates,
        rounds=model_rounds,
        log_tag="ExtractParams",
    ):
        if not (result and result.get("choices")):
            logger.warning(
                "[ExtractParams] no LLM response on attempt %d/%d model=%s",
                attempt, total_attempts, model,
            )
            continue
        content = result["choices"][0].get("message", {}).get("content", "")
        parsed = _parse_json_object_from_llm(content)
        if parsed is not None:
            if kw_task == "product":
                return _sanitize_product_search_params(parsed)
            if kw_task in ("shop", "voucher"):
                for p in parsed.get("products", []):
                    if p.get("keywords"):
                        p["keywords"] = " ".join(w for w in p["keywords"].split() if w.lower() not in _SCORING_STOPWORDS)
            return parsed
    return _extract_query_params_regex(query)

def _extract_query_params_regex(query: str) -> dict:
    task_type = _infer_task_type(query)

    def _extract_product_spec(text: str) -> dict:
        alpha_words = [w for w in re.findall(r"\b[a-zA-Z]{2,}\b", text.lower()) if w not in _REGEX_STOPWORDS]
        alnum_tokens = re.findall(r"\b\d+[a-zA-Z]+\b|\b[a-zA-Z]+\d+[a-zA-Z]*\b", text.lower())
        words = alpha_words[:6]
        for t in alnum_tokens[:2]:
            if t not in words: words.append(t)
        for s in re.findall(r"(\d+)#", text)[:2]:
            if s not in words: words.append(s)
        keywords = " ".join(words) or "product"
        price_range = None
        m = re.search(r"(?:greater|more|over|above|>|cost[s]?\s+more)\s*(?:than\s*)?(\d+)", text, re.I)
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
        if "lazmall" in tl or "official" in tl: service = "official"
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
    parts = [p.strip() for p in _MULTI_PRODUCT_SPLIT_RE.split(product_text) if p and len(p.strip()) > 10]
    if not parts:
        parts = [query]
    products = [_extract_product_spec(p) for p in parts]
    products = [p for p in products if len(p["keywords"].split()) >= 2] or products
    is_shop = task_type == "shop" or (task_type == "voucher" and "same shop" in query.lower())
    return {"task_type": task_type, "products": products, "is_shop_voucher": is_shop}



def _spec_to_find_product_params(product: dict, *, include_price: bool = True) -> dict[str, Any]:
    params: dict[str, Any] = {"q": product.get("keywords", "product")}
    if include_price and product.get("price_range"):
        params["price"] = product["price_range"]
    if product.get("service"):
        params["service"] = product["service"]
    return params


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
    seen, out = set(), []
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
    rec = safe_tool_call("recommend_product", {"product_ids": _format_product_ids(product_ids)})
    term = safe_tool_call("terminate", {"status": status})
    _append_step("Done.", [rec, term], "Done.", query, steps)


def _run_multi_voucher_same_shop_search(params: dict, query: str, steps: list, voucher: dict = None) -> None:
    queries = [_spec_to_find_product_params(p) for p in params.get("products", [])]
    if not queries:
        queries = [{"q": "product"}]
    queries.append({"_original_query": query})
    result = safe_tool_call("find_voucher_products_in_same_shop", {"product_queries": json.dumps(queries)})
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
                logger.exception("_run_multi_voucher_same_shop_search: fallback failed for spec %s", p)
    _finish_session(pids if pids else [FALLBACKID], "success" if pids else "failure", query, steps)


def _run_multi_voucher_search(products, query, steps, discount_type, discount_value, threshold, budget, cap):
    cand_products = []
    for p in products:
        sp = _spec_to_find_product_params(p, include_price=False)
        result = safe_tool_call("find_product", sp)
        found = result["result"] or []
        if not found:
            break
        cand_products.append(found)
    if len(cand_products) == len(products):
        top_prices = [str(p[0]["price"]) for p in cand_products]
        vr = safe_tool_call("calculate_voucher", {"product_prices": ",".join(top_prices),
            "voucher_type": discount_type, "discount_value": discount_value,
            "threshold": threshold, "budget": budget, "cap": cap})
        if vr["result"]["within_budget"]:
            _finish_session([str(p[0]["product_id"]) for p in cand_products], "success", query, steps)
            return
    pids = []
    for p in products:
        sp = _spec_to_find_product_params(p, include_price=False)
        result = safe_tool_call("find_product", sp)
        found = result["result"] or []
        for page in range(1, 2):
            sp["page"] = page
            result = safe_tool_call("find_product", sp)
            found.extend(result["result"] or [])
        found = _deduplicate_products(found)
        if found:
            kw = p.get("keywords", "product")
            score_q = kw if len(products) > 1 else query
            best = _select_best_product_for_voucher_case(found, score_q, prefer_cheaper=True, sel_count=1)
            if best:
                pids.append(str(best["product_id"]))
    _finish_session(pids if pids else [FALLBACKID], "success" if pids else "failure", query, steps)


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


def _finalize_recommendation_with_status(
    product_ids: Iterable[Any],
    status: str,
    query: str,
    steps: List[Dict[str, Any]],
) -> None:
    recommendation = execute_tool_call(
        "recommend_product",
        {"product_ids": _format_product_ids(product_ids)},
    )
    termination = execute_tool_call("terminate", {"status": status})
    _add_dialogue_step("Done.", [recommendation, termination], "Done.", query, steps)


def _run_couple_voucher_search(products, discount_type, discount_value, threshold, budget, cap, query, steps):
    
    product_ids: List[str] = []
    voucher_budget = CHEAPER_PRICE_TIEBREAK_DIVISOR
    try:
        voucher_budget = max_total_price(budget, threshold, discount_value, cap, discount_type)
    except:
        pass

    def search_product(product, price_range = None):
        payload = {"q": product.get("keywords", DEFAULT_PRODUCT_QUERY)}
        if product.get("service"):
            payload["service"] = product["service"]
        if price_range:
            payload["price"] = price_range

        result = _execute_and_record("find_product", payload, query, steps)
        found_products = result.get("result") or []

        if found_products:
            for prod in found_products:
                price = prod.get('price', 0)
                if price <= voucher_budget:
                    return str(prod["product_id"]), price, voucher_budget - price

        return None, None, voucher_budget
    
    prices = []
    original_voucher_budget = voucher_budget
    for product in products:
        product_id, price, voucher_budget = search_product(product)
        if not product_id:
            product_id, price, voucher_budget = search_product(product, f"0-{voucher_budget}")

        if product_id:
            product_ids.append(product_id)
            prices.append(price)

    if len(prices) > 1 and prices[0] < prices[1]:
        products = list(reversed(products))
        voucher_budget = original_voucher_budget
        product_ids = []
        for idx, product in enumerate(products):
            product_id = None
            if idx == 0:
                product_id, _, voucher_budget = search_product(product, f"{budget}-{voucher_budget}")
            if not product_id:
                product_id, _, voucher_budget = search_product(product)

            if product_id:
                product_ids.append(product_id)
        product_ids = list(reversed(product_ids))

    if product_ids:
        _finalize_recommendation_with_status(product_ids, "success", query, steps)
    else:
        _finalize_recommendation_with_status([FALLBACKID], "failure", query, steps)


def _run_single_product_search(params: dict, query: str, steps: list) -> None:
    prods = params.get("products", [{}])
    p = prods[0] if prods else {}
    search_params = _spec_to_find_product_params(p)
    print("search_params: ", search_params)
    all_results = []
    tool_results: list = []
    for page in range(1, 2):
        search_params["page"] = page
        result = safe_tool_call("find_product", search_params)
        all_results.extend(result["result"] or [])
        tool_results.append(result)
    _append_step("Processing.", tool_results, "", query, steps)
    unique = _deduplicate_products(all_results)
    best = _select_best_product_for_product_case(
        unique, query, top_count=50, parsed_spec=p,
        judge_model_candidates=SINGLE_JUDGE_MODEL_CANDIDATES,
        judge_model_rounds=SINGLE_JUDGE_MODEL_ROUNDS,
    ) if unique else None
    if best:
        _finish_session([str(best["product_id"])], "success", query, steps)
    else:
        _finish_session([FALLBACKID], "failure", query, steps)

def _build_tool_search_payload(product: Dict[str, Any], *, include_price: bool = True) -> Dict[str, Any]:
    payload = {"q": product.get("keywords", DEFAULT_PRODUCT_QUERY)}
    if include_price and product.get("price_range"):
        payload["price"] = product["price_range"]
    if product.get("service"):
        payload["service"] = product["service"]
    return payload



def _search_products_individually(
    params: Dict[str, Any],
    query: str,
    steps: List[Dict[str, Any]],
    *,
    include_price: bool,
    prefer_cheaper: bool,
    use_keyword_query_for_multi: bool = False,
) -> List[str]:
    product_ids: List[str] = []
    products = params.get("products", [])
    for product in products:
        try:
            result = _execute_and_record(
                "find_product_for_shop",
                _build_tool_search_payload(product, include_price=include_price),
                query,
                steps,
            )
            found_products = result.get("result") or []
            if not found_products:
                continue

            score_query = (
                product.get("keywords", DEFAULT_PRODUCT_QUERY)
                if use_keyword_query_for_multi and len(products) > 1
                else query
            )
            best_product = _select_best_product(
                found_products,
                score_query,
                prefer_cheaper=prefer_cheaper,
            )
            if best_product:
                product_ids.append(str(best_product["product_id"]))
        except Exception:
            logger.exception("Fallback individual product search failed.")
    return product_ids


def _add_dialogue_step(
    think: str,
    tool_results: Sequence[Dict[str, Any]],
    response: str,
    query: str,
    steps: List[Dict[str, Any]],
) -> None:
    steps.append(create_dialogue_step(think, list(tool_results), response, query, len(steps) + 1))


def _finalize_recommendation(product_ids: Iterable[Any], query: str, steps: List[Dict[str, Any]]) -> None:
    recommendation = execute_tool_call(
        "recommend_product",
        {"product_ids": _format_product_ids(product_ids)},
    )
    termination = execute_tool_call("terminate", {"status": "success"})
    _add_dialogue_step("Done.", [recommendation, termination], "Done.", query, steps)


def _execute_and_record(
    tool_name: str,
    payload: Dict[str, Any],
    query: str,
    steps: List[Dict[str, Any]],
    *,
    think: str = "Processing.",
    response: str = "",
) -> Dict[str, Any]:
    result = execute_tool_call(tool_name, payload)
    _add_dialogue_step(think, [result], response, query, steps)
    return result

def _run_same_shop_search(params: dict, query: str, steps: list, is_voucher: bool = False) -> None:
    print("params: ", params)
    print("query: ", query)
    product_queries = [
        _build_tool_search_payload(product, include_price=not is_voucher)
        for product in params.get("products", [])
    ] or [{"q": DEFAULT_PRODUCT_QUERY}]
    product_queries.append({"_original_query": query})

    result = _execute_and_record(
        "find_products_in_same_shop",
        {"product_queries": json.dumps(product_queries)},
        query,
        steps,
    )

    print("result: ", result)

    same_shop_result = result.get("result")
    if isinstance(same_shop_result, dict) and same_shop_result.get("found"):
        product_ids = [str(product["product_id"]) for product in same_shop_result.get("products", [])]
    else:
        product_ids = _search_products_individually(
            params,
            query,
            steps,
            include_price=not is_voucher,
            prefer_cheaper=True,
        )

    _finalize_recommendation(product_ids, query, steps)


def _run_voucher_search(params: dict, query: str, steps: list) -> None:
    is_shop = params.get("is_shop_voucher", False) or "same shop" in query.lower()
    products = params.get("products", [])
    voucher = params.get("voucher", {})
    discount_type = voucher.get("discount_type", "percentage")
    discount_value = float(voucher.get("discount_value", 0))
    threshold = float(voucher.get("threshold", 0))
    cap = float(voucher.get("cap", 0))
    budget = float(voucher.get("budget", 0))
    if is_shop and len(products) > 2:
        _run_multi_voucher_same_shop_search(params, query, steps, {"discount_type": discount_type,
            "discount_value": discount_value, "threshold": threshold, "budget": budget, "cap": cap})
        return

    elif not is_shop and len(products) > 2:
        _run_multi_voucher_search(products, query, steps, discount_type, discount_value, threshold, budget, cap)
        return
    else:
        _run_couple_voucher_search(products, discount_type, discount_value, threshold, budget, cap, query, steps)
    

def agent_main(problem_data: dict) -> list[dict]:
    _product_detail_cache.clear()
    steps: list = []
    query: str = problem_data.get("query", "")
    try:
        kw_task = _infer_task_type(query)
        extract_overrides = {}
        if kw_task == "product":
            extract_overrides = dict(
                model_candidates=SINGLE_EXTRACT_MODEL_CANDIDATES,
                model_rounds=SINGLE_EXTRACT_MODEL_ROUNDS,
            )
        params = _extract_query_params_llm(query, kw_task, **extract_overrides)
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
    return steps
