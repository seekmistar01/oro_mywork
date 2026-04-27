import json
import logging
import re
from typing import Dict, List, Any
from urllib.parse import quote_plus
from itertools import product as cartesian_product
from os import getenv

from src.agent.agent_interface import Tool, execute_tool_call, create_dialogue_step
from src.agent.proxy_client import ProxyClient


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("oro_prompt_agent")


# ============================================================
# CLIENTS / CONSTANTS
# ============================================================

_search_client = ProxyClient(timeout=30, max_retries=5)
_inference_client = ProxyClient(timeout=90, max_retries=3)

FALLBACK_PRODUCT_ID = "0"

DEFAULT_PARSE_MODEL = getenv("SANDBOX_MODEL", "deepseek-ai/DeepSeek-V3.2-TEE")
DEFAULT_JUDGE_MODEL = getenv("SANDBOX_MODEL", "openai/gpt-oss-120b-TEE")
FALLBACK_MODEL = "deepseek-ai/DeepSeek-V3.1-TEE"

SEARCH_PAGES = [1, 2]
MAX_DETAIL_PRODUCTS = 12
MAX_JUDGE_PRODUCTS = 10
MAX_SHOP_CANDIDATE_SHOPS = 25
MAX_VOUCHER_CANDIDATES_PER_SPEC = 8


# ============================================================
# PROMPTS
# ============================================================

PRODUCT_PARSE_PROMPT = """You are parsing an e-commerce shopping request.

Return JSON only. No markdown.

Schema:
{
  "products": [
    {
      "query": "original part of the user request for this product",
      "keywords": "short search query",
      "price_range": "min-max or min- or 0-max or null",
      "service": "official|freeShipping|COD|flashsale|null",
      "only_product_type": true/false
    }
  ],
  "is_shop_voucher": false
}

Rules:
- keywords must include product type, brand, model, color, size, material, quantity, capacity, compatibility, category, and important attributes.
- Keep 2-10 useful words.
- Do NOT include budget/voucher words in keywords.
- Convert service words:
  LazMall / official / guaranteed authenticity / easy returns / fast shipping -> official
  LazFlash / flash deal -> flashsale
  free shipping / free delivery -> freeShipping
  cash on delivery / pay on delivery / COD -> COD
- price_range examples:
  "from 180 to 505" -> "180-505"
  "between 1191 and 2233" -> "1191-2233"
  "above 85" -> "85-"
  "under 100" -> "0-100"
- only_product_type is true only when the request is just a bare product type without qualifiers.
JSON only.
"""

SHOP_PARSE_PROMPT = """You are parsing a same-shop e-commerce request.

Return JSON only. No markdown.

Schema:
{
  "products": [
    {
      "query": "original part of the user request for this product",
      "keywords": "short search query",
      "price_range": "min-max or min- or 0-max or null",
      "service": "official|freeShipping|COD|flashsale|null",
      "only_product_type": true/false
    }
  ],
  "is_shop_voucher": false
}

Rules:
- Split the request into one product object per required product.
- Preserve order: first product, second product, third product.
- keywords must include product type, brand, model, color, size, material, quantity, capacity, compatibility, category, and important attributes.
- Do NOT include generic words like shop/store/offering/sells.
- Convert service words:
  LazMall / official / guaranteed authenticity / easy returns / fast shipping -> official
  LazFlash / flash deal -> flashsale
  free shipping / free delivery -> freeShipping
  cash on delivery / pay on delivery / COD -> COD
- price_range examples:
  "priced above 410" -> "410-"
  "from 79 to 1126" -> "79-1126"
  "between 17 and 49" -> "17-49"
JSON only.
"""

VOUCHER_PARSE_PROMPT = """You are parsing an e-commerce voucher/budget request.

Return JSON only. No markdown.

Schema:
{
  "products": [
    {
      "query": "original part of the user request for this product",
      "keywords": "short search query",
      "price_range": "min-max or min- or 0-max or null",
      "service": "official|freeShipping|COD|flashsale|null",
      "only_product_type": true/false
    }
  ],
  "voucher": {
    "same_shop": true/false,
    "discount_type": "fixed|percentage",
    "discount_value": number,
    "threshold": number,
    "cap": number,
    "budget": number
  },
  "is_shop_voucher": true/false
}

Rules:
- Split the request into one product object per required product.
- Do NOT create product entries for budget or voucher text.
- keywords must include product type, brand, model, color, size, material, quantity, capacity, compatibility, category, and important attributes.
- Convert service words:
  LazMall / official / guaranteed authenticity / easy returns / fast shipping -> official
  LazFlash / flash deal -> flashsale
  free shipping / free delivery -> freeShipping
  cash on delivery / pay on delivery / COD -> COD
- Voucher:
  "applies to all products" -> same_shop false
  "only applies to products from the same shop" -> same_shop true
  fixed discount of 49 -> discount_type fixed, discount_value 49
  percentage discount of 34% -> discount_type percentage, discount_value 34
  cap of 28 -> cap 28
  budget is only 82 -> budget 82
  total price exceeds 63 -> threshold 63
JSON only.
"""

PRODUCT_JUDGE_PROMPT = """You are choosing the best e-commerce product candidate.

Return JSON only. No markdown.

Schema:
{
  "best_product_id": "product id",
  "reason": "short reason",
  "relevance_score": 0-10
}

Choose the ONE candidate that best satisfies the request.

Priorities:
1. Exact product type and function.
2. Brand/model/compatibility.
3. Color/size/material/quantity/capacity.
4. Service requirements such as official, flashsale, freeShipping, COD.
5. Price constraints.
6. Attributes and SKU options are stronger evidence than title alone.
7. Prefer cheaper product only when candidates match equally well.
8. Penalize products with conflicting variants or missing requested specs.

JSON only.
"""

CANDIDATE_SCORE_PROMPT = """You are scoring e-commerce product candidates.

Return JSON array only. No markdown.

Schema:
[
  {"product_id": "id", "score": 0-10, "reason": "short reason"}
]

Score every candidate against the request.
10 = perfect match.
0 = not relevant.

Consider:
- product type
- brand
- model / compatibility
- color
- size
- material
- quantity / pack count
- capacity / volume
- category
- price range
- service requirements
- attributes and SKU options

JSON only.
"""


# ============================================================
# TOOLS
# ============================================================

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
    params = {k: v for k, v in params.items() if v not in [None, "", "default"]}

    logger.info("[STEP 4 SEARCH] find_product params=%s", params)
    return _search_client.get("/search/find_product", params) or []


@Tool
def view_product_information(product_ids: str) -> list[dict]:
    logger.info("[STEP 5 DETAILS] view_product_information product_ids=%s", product_ids)
    return _search_client.get(
        "/search/view_product_information",
        {"product_ids": product_ids},
    ) or []


@Tool
def recommend_product(product_ids: str) -> str:
    logger.info("[STEP 9 RECOMMEND] product_ids=%s", product_ids)
    return f"Having recommended the products to the user: {product_ids}."


@Tool
def terminate(status: str = "success") -> str:
    logger.info("[STEP 10 TERMINATE] status=%s", status)
    return f"The interaction has been completed with status: {status}"


# ============================================================
# BASIC FALLBACK HELPERS
# ============================================================

STOPWORDS = {
    "find", "show", "me", "looking", "for", "a", "an", "the", "and", "also",
    "with", "from", "that", "is", "are", "priced", "price", "cost", "costing",
    "php", "pesos", "between", "above", "under", "over", "below", "more",
    "than", "less", "shop", "shops", "offering", "offers", "sells", "both",
    "first", "second", "third", "product", "products", "item", "items",
    "budget", "voucher", "discount", "rules", "only", "please", "help",
}


def detect_task_type(query: str) -> str:
    logger.info("[STEP 1 GET TASK] query=%s", query)

    q = query.lower()
    if "voucher" in q or "budget" in q or "discount" in q:
        task_type = "voucher"
    elif "shop" in q or "store" in q or "both" in q or "offering" in q or "sells" in q:
        task_type = "shop"
    else:
        task_type = "product"

    logger.info("[STEP 2 UNDERSTAND] detected_task_type=%s", task_type)
    return task_type


def clean_keywords(text: str) -> str:
    words = re.findall(r"[a-zA-Z0-9#\.]+", text.lower())
    words = [w for w in words if w not in STOPWORDS and len(w) > 1]
    return " ".join(words[:10]) or "product"


def extract_price_range(text: str) -> str | None:
    q = text.lower()

    m = re.search(r"(?:between|from)\s+(\d+)\s+(?:and|to)\s+(\d+)", q)
    if m:
        return f"{m.group(1)}-{m.group(2)}"

    m = re.search(r"(?:above|over|more than|greater than|costs more than)\s+(\d+)", q)
    if m:
        return f"{m.group(1)}-"

    m = re.search(r"(?:under|below|less than)\s+(\d+)", q)
    if m:
        return f"0-{m.group(1)}"

    return None


def extract_service(text: str) -> str | None:
    q = text.lower()
    services = []

    if "lazmall" in q or "official" in q or "guaranteed authenticity" in q:
        services.append("official")
    if "lazflash" in q or "flash deal" in q or "flashsale" in q:
        services.append("flashsale")
    if "free shipping" in q or "free delivery" in q:
        services.append("freeShipping")
    if "cash on delivery" in q or "pay on delivery" in q or "cod" in q:
        services.append("COD")

    return ",".join(services) if services else None


def split_products_regex(query: str) -> list[str]:
    query = re.split(r"My budget|budget is|I have a voucher", query, flags=re.I)[0]

    parts = re.split(
        r"\bFirst,|\bSecond,|\bThird,|\balso\b|\band also\b|"
        r"\bFor the first\b|\bFor the second\b|\bFor the third\b|"
        r"\(\d+\)|\d+\.",
        query,
        flags=re.I,
    )
    parts = [p.strip(" .,:;") for p in parts if len(p.strip()) > 15]

    if len(parts) <= 1:
        parts = re.split(r"\s+and\s+", query, flags=re.I)
        parts = [p.strip(" .,:;") for p in parts if len(p.strip()) > 15]

    return parts or [query]


def parse_spec_regex(text: str) -> dict:
    return {
        "query": text,
        "keywords": clean_keywords(text),
        "price_range": extract_price_range(text),
        "service": extract_service(text),
        "only_product_type": False,
    }


def parse_voucher_regex(query: str) -> dict:
    q = query.lower()

    budget = extract_number_after(query, r"budget is only\s*`?(\d+)`?")
    threshold = extract_number_after(query, r"exceeds\s*`?(\d+)`?")
    fixed = extract_number_after(query, r"fixed discount of\s*`?(\d+)`?")
    percentage = extract_number_after(query, r"percentage discount of\s*`?(\d+)%?`?")
    cap = extract_number_after(query, r"cap of\s*`?(\d+)`?")

    if fixed is not None:
        discount_type = "fixed"
        discount_value = fixed
    else:
        discount_type = "percentage"
        discount_value = percentage or 0

    return {
        "same_shop": "same shop" in q or "only applies to the products from the same shop" in q,
        "budget": float(budget or 0),
        "threshold": float(threshold or 0),
        "discount_type": discount_type,
        "discount_value": float(discount_value or 0),
        "cap": float(cap or 0),
    }


def extract_number_after(text: str, pattern: str) -> float | None:
    m = re.search(pattern, text, flags=re.I)
    if not m:
        return None
    return float(m.group(1))


def parse_json_from_text(text: str) -> dict | list | None:
    cleaned = re.sub(r"```json?\s*", "", text or "")
    cleaned = re.sub(r"```\s*$", "", cleaned).strip()

    try:
        return json.loads(cleaned)
    except Exception:
        pass

    obj_match = re.search(r"\{.*\}", cleaned, flags=re.S)
    if obj_match:
        try:
            return json.loads(obj_match.group(0))
        except Exception:
            pass

    arr_match = re.search(r"\[.*\]", cleaned, flags=re.S)
    if arr_match:
        try:
            return json.loads(arr_match.group(0))
        except Exception:
            pass

    return None


# ============================================================
# AI PARSING / SCORING / JUDGING
# ============================================================

def call_llm_json(system_prompt: str, user_content: str, model: str) -> dict | list | None:
    try:
        result = _inference_client.post(
            "/inference/chat/completions",
            json_data={
                "model": model,
                "temperature": 0,
                "stream": False,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
            },
        )

        if not result or not result.get("choices"):
            logger.warning("[LLM] Empty response from model=%s", model)
            return None

        content = result["choices"][0].get("message", {}).get("content", "")
        parsed = parse_json_from_text(content)
        if parsed is None:
            logger.warning("[LLM] Could not parse JSON. Raw content=%s", content[:500])
        return parsed

    except Exception as exc:
        logger.exception("[LLM] Failed model=%s error=%s", model, exc)
        return None


def parse_query_with_ai(query: str, task_type: str) -> dict:
    logger.info("[STEP 2 UNDERSTAND] AI parsing started task_type=%s", task_type)

    if task_type == "product":
        prompt = PRODUCT_PARSE_PROMPT
    elif task_type == "shop":
        prompt = SHOP_PARSE_PROMPT
    else:
        prompt = VOUCHER_PARSE_PROMPT

    parsed = call_llm_json(prompt, query, DEFAULT_PARSE_MODEL)

    if isinstance(parsed, dict) and parsed.get("products"):
        logger.info("[STEP 2 UNDERSTAND] AI parsed JSON=%s", json.dumps(parsed, ensure_ascii=False)[:1200])
        return normalize_parsed_query(parsed, query, task_type)

    logger.warning("[STEP 2 UNDERSTAND] AI parse failed. Using regex fallback.")
    return parse_query_fallback(query, task_type)


def normalize_parsed_query(parsed: dict, query: str, task_type: str) -> dict:
    products = []
    for raw in parsed.get("products", []):
        if not isinstance(raw, dict):
            continue

        keywords = raw.get("keywords") or clean_keywords(raw.get("query") or query)
        spec = {
            "query": raw.get("query") or keywords,
            "keywords": clean_keywords(keywords) if len(str(keywords).split()) > 12 else str(keywords).strip(),
            "price_range": raw.get("price_range"),
            "service": normalize_service(raw.get("service")),
            "only_product_type": bool(raw.get("only_product_type", False)),
        }
        if not spec["keywords"]:
            spec["keywords"] = "product"
        products.append(spec)

    if not products:
        products = [parse_spec_regex(query)]

    voucher = parsed.get("voucher") if isinstance(parsed.get("voucher"), dict) else None
    if task_type == "voucher":
        fallback_voucher = parse_voucher_regex(query)
        voucher = {
            "same_shop": bool(
                (voucher or {}).get("same_shop", fallback_voucher["same_shop"])
                or parsed.get("is_shop_voucher", False)
            ),
            "budget": float((voucher or {}).get("budget", fallback_voucher["budget"]) or 0),
            "threshold": float((voucher or {}).get("threshold", fallback_voucher["threshold"]) or 0),
            "discount_type": (voucher or {}).get("discount_type", fallback_voucher["discount_type"]),
            "discount_value": float((voucher or {}).get("discount_value", fallback_voucher["discount_value"]) or 0),
            "cap": float((voucher or {}).get("cap", fallback_voucher["cap"]) or 0),
        }
    else:
        voucher = None

    return {
        "task_type": task_type,
        "products": products,
        "voucher": voucher,
        "is_shop_voucher": bool(parsed.get("is_shop_voucher", False)),
    }


def parse_query_fallback(query: str, task_type: str) -> dict:
    parts = split_products_regex(query) if task_type in ["shop", "voucher"] else [query]
    products = [parse_spec_regex(p) for p in parts]
    return {
        "task_type": task_type,
        "products": products,
        "voucher": parse_voucher_regex(query) if task_type == "voucher" else None,
        "is_shop_voucher": "same shop" in query.lower(),
    }


def normalize_service(service: Any) -> str | None:
    if not service:
        return None
    if isinstance(service, list):
        values = service
    else:
        values = [s.strip() for s in str(service).split(",") if s.strip()]

    allowed = {"official", "freeShipping", "COD", "flashsale"}
    values = [v for v in values if v in allowed]
    return ",".join(dict.fromkeys(values)) if values else None


def build_search_queries(spec: dict) -> list[str]:
    logger.info("[STEP 3 BUILD SEARCH QUERIES] spec=%s", spec)

    base = str(spec.get("keywords") or "product").strip()
    original = str(spec.get("query") or "").strip()

    queries = [base]

    # Add a slightly broader query by dropping very specific numeric tokens if possible.
    words = base.split()
    broad_words = [w for w in words if not re.search(r"\d", w)]
    if 2 <= len(broad_words) < len(words):
        queries.append(" ".join(broad_words))

    # Add product-only style query for bare product type.
    if spec.get("only_product_type"):
        queries.insert(0, base + " only")

    # Add fallback regex-cleaned original query.
    fallback = clean_keywords(original)
    if fallback and fallback not in queries:
        queries.append(fallback)

    # Deduplicate and limit.
    out = []
    seen = set()
    for q in queries:
        q = q.strip()
        if q and q not in seen:
            seen.add(q)
            out.append(q)

    logger.info("[STEP 3 BUILD SEARCH QUERIES] queries=%s", out[:3])
    return out[:3]


def compact_candidate_payload(product: dict, detail: dict | None = None) -> dict:
    title = str(product.get("title", ""))[:220]
    attrs = {}
    if isinstance(detail, dict):
        raw_attrs = detail.get("attributes") or {}
        if isinstance(raw_attrs, dict):
            attrs = {str(k)[:40]: str(v)[:160] for k, v in list(raw_attrs.items())[:10]}

    sku_preview = {}
    if isinstance(detail, dict):
        raw_sku = detail.get("sku_options") or {}
        if isinstance(raw_sku, dict):
            for k, v in list(raw_sku.items())[:5]:
                sku_preview[str(k)[:40]] = str(v)[:180]

    return {
        "product_id": str(product.get("product_id", "")),
        "title": title,
        "price": product.get("price"),
        "shop_id": product.get("shop_id"),
        "service": product.get("service") or [],
        "attributes": attrs,
        "sku_options_preview": sku_preview,
    }


def score_candidates_with_ai(spec: dict, candidates: list[dict], details: dict[str, dict]) -> dict[str, float]:
    logger.info("[STEP 7 SCORE] AI scoring candidates=%d keywords=%s", len(candidates), spec.get("keywords"))

    if not candidates:
        return {}

    payload = {
        "request": spec.get("query") or spec.get("keywords"),
        "keywords": spec.get("keywords"),
        "price_range": spec.get("price_range"),
        "service": spec.get("service"),
        "candidates": [
            compact_candidate_payload(p, details.get(str(p.get("product_id"))))
            for p in candidates[:MAX_JUDGE_PRODUCTS]
        ],
    }

    parsed = call_llm_json(CANDIDATE_SCORE_PROMPT, json.dumps(payload, ensure_ascii=False), DEFAULT_JUDGE_MODEL)
    if not isinstance(parsed, list):
        parsed = call_llm_json(CANDIDATE_SCORE_PROMPT, json.dumps(payload, ensure_ascii=False), FALLBACK_MODEL)

    scores: dict[str, float] = {}
    if isinstance(parsed, list):
        for item in parsed:
            if not isinstance(item, dict):
                continue
            pid = str(item.get("product_id", "")).strip()
            try:
                score = float(item.get("score", 0))
            except Exception:
                score = 0.0
            if pid:
                scores[pid] = score

    logger.info("[STEP 7 SCORE] AI scores=%s", scores)
    return scores


def choose_best_with_ai(spec: dict, candidates: list[dict], details: dict[str, dict]) -> dict | None:
    logger.info("[STEP 8 CHOOSE] AI choosing among candidates=%d", len(candidates))

    if not candidates:
        return None

    payload = {
        "request": spec.get("query") or spec.get("keywords"),
        "keywords": spec.get("keywords"),
        "price_range": spec.get("price_range"),
        "service": spec.get("service"),
        "candidates": [
            compact_candidate_payload(p, details.get(str(p.get("product_id"))))
            for p in candidates[:MAX_JUDGE_PRODUCTS]
        ],
    }

    parsed = call_llm_json(PRODUCT_JUDGE_PROMPT, json.dumps(payload, ensure_ascii=False), DEFAULT_JUDGE_MODEL)
    if not isinstance(parsed, dict):
        parsed = call_llm_json(PRODUCT_JUDGE_PROMPT, json.dumps(payload, ensure_ascii=False), FALLBACK_MODEL)

    if isinstance(parsed, dict):
        best_pid = str(parsed.get("best_product_id", "")).strip()
        logger.info("[STEP 8 CHOOSE] AI choice product_id=%s reason=%s", best_pid, parsed.get("reason"))
        for p in candidates:
            if str(p.get("product_id", "")) == best_pid:
                p = dict(p)
                p["_ai_reason"] = parsed.get("reason", "")
                p["_ai_relevance_score"] = parsed.get("relevance_score", 0)
                return p

    logger.warning("[STEP 8 CHOOSE] AI choose failed. Falling back to rule ranking.")
    return None


# ============================================================
# PRICE / FILTER / SCORE
# ============================================================

def price_after_voucher(total: float, voucher: dict) -> float:
    if total < voucher["threshold"]:
        return total

    if voucher["discount_type"] == "fixed":
        discount = voucher["discount_value"]
    else:
        discount = total * voucher["discount_value"] / 100.0
        if voucher["cap"] > 0:
            discount = min(discount, voucher["cap"])

    return total - discount


def is_within_price(product: dict, price_range: str | None) -> bool:
    if not price_range:
        return True

    price = product.get("price")
    if not isinstance(price, (int, float)):
        return True

    if "-" not in str(price_range):
        return True

    a, b = str(price_range).split("-", 1)
    lo = float(a) if a.strip() else None
    hi = float(b) if b.strip() else None

    if lo is not None and price < lo:
        return False
    if hi is not None and price > hi:
        return False
    return True


def matches_service(product: dict, service: str | None) -> bool:
    if not service:
        return True

    offered = set(product.get("service") or [])
    required = [s.strip() for s in service.split(",") if s.strip()]
    return all(s in offered for s in required)


def filter_candidates(candidates: list[dict], spec: dict) -> list[dict]:
    logger.info("[STEP 6 FILTER] before=%d spec=%s", len(candidates), spec)

    filtered = []
    for p in candidates:
        if not is_within_price(p, spec.get("price_range")):
            continue
        if not matches_service(p, spec.get("service")):
            continue
        filtered.append(p)

    # If strict filter removes everything, keep original candidates for fallback.
    if not filtered and candidates:
        logger.warning("[STEP 6 FILTER] strict filter removed all candidates; using unfiltered fallback.")
        filtered = candidates

    logger.info("[STEP 6 FILTER] after=%d", len(filtered))
    return filtered


def product_score(product: dict, spec: dict, detail: dict | None = None, ai_score: float | None = None) -> float:
    title = str(product.get("title", "")).lower()
    query_text = (str(spec.get("query", "")) + " " + str(spec.get("keywords", ""))).lower()
    words = [w for w in re.findall(r"[a-zA-Z0-9#\.]+", query_text) if w not in STOPWORDS and len(w) > 1]

    score = 0.0

    for word in dict.fromkeys(words):
        if word in title:
            score += 2

    if is_within_price(product, spec.get("price_range")):
        score += 5
    else:
        score -= 50

    if matches_service(product, spec.get("service")):
        if spec.get("service"):
            score += 8
    else:
        score -= 30

    if detail:
        detail_text = json.dumps(detail, ensure_ascii=False).lower()
        for word in dict.fromkeys(words):
            if word in detail_text:
                score += 2

    if ai_score is not None:
        score += ai_score * 6

    price = product.get("price")
    if isinstance(price, (int, float)):
        score -= price / 100000.0

    return score


# ============================================================
# SEARCH / DETAILS / CHOOSE
# ============================================================

def search_candidates(spec: dict, shop_id: str | None = None) -> list[dict]:
    logger.info("[STEP 4 SEARCH] start spec=%s shop_id=%s", spec, shop_id)

    all_products: list[dict] = []
    queries = build_search_queries(spec)

    for search_query in queries:
        for page in SEARCH_PAGES:
            result = execute_tool_call(
                "find_product",
                {
                    "q": search_query,
                    "page": page,
                    "price": spec.get("price_range"),
                    "service": spec.get("service"),
                    "shop_id": shop_id,
                },
            )
            products = result.get("result", result) if isinstance(result, dict) else result
            if isinstance(products, list):
                logger.info(
                    "[STEP 4 SEARCH] query='%s' page=%s found=%d",
                    search_query, page, len(products),
                )
                all_products.extend(products)

    # Retry without service if needed.
    if not all_products and spec.get("service"):
        logger.warning("[STEP 4 SEARCH] no results with service=%s; retrying without service", spec.get("service"))
        loose = dict(spec)
        loose["service"] = None
        return search_candidates(loose, shop_id=shop_id)

    # Deduplicate.
    seen = set()
    unique = []
    for p in all_products:
        pid = str(p.get("product_id", ""))
        if pid and pid not in seen:
            seen.add(pid)
            unique.append(p)

    logger.info("[STEP 5 COLLECT CANDIDATES] unique_candidates=%d", len(unique))
    return unique


def fetch_details(products: list[dict]) -> dict[str, dict]:
    ids = [str(p.get("product_id")) for p in products[:MAX_DETAIL_PRODUCTS] if p.get("product_id")]
    if not ids:
        return {}

    result = execute_tool_call(
        "view_product_information",
        {"product_ids": ",".join(ids)},
    )

    rows = result.get("result", result) if isinstance(result, dict) else result
    if not isinstance(rows, list):
        logger.warning("[STEP 5 DETAILS] invalid details response")
        return {}

    details = {str(r.get("product_id")): r for r in rows if r.get("product_id")}
    logger.info("[STEP 5 DETAILS] fetched_details=%d", len(details))
    return details


def choose_best(products: list[dict], spec: dict) -> dict | None:
    logger.info("[STEP 8 CHOOSE] start candidates=%d", len(products))

    if not products:
        return None

    filtered = filter_candidates(products, spec)

    # Pre-rank with rule score to limit expensive detail + LLM work.
    pre_top = sorted(filtered, key=lambda p: product_score(p, spec), reverse=True)[:MAX_JUDGE_PRODUCTS]
    details = fetch_details(pre_top)

    ai_scores = score_candidates_with_ai(spec, pre_top, details)
    ai_choice = choose_best_with_ai(spec, pre_top, details)

    if ai_choice is not None:
        pid = str(ai_choice.get("product_id", ""))
        if pid and product_score(ai_choice, spec, details.get(pid), ai_scores.get(pid)) > -10:
            logger.info("[STEP 8 CHOOSE] final_choice=AI product_id=%s", pid)
            return ai_choice

    best = max(
        pre_top,
        key=lambda p: product_score(
            p,
            spec,
            details.get(str(p.get("product_id"))),
            ai_scores.get(str(p.get("product_id"))),
        ),
    )

    logger.info(
        "[STEP 8 CHOOSE] final_choice=RULE product_id=%s title=%s",
        best.get("product_id"), best.get("title"),
    )

    if product_score(best, spec, details.get(str(best.get("product_id"))), ai_scores.get(str(best.get("product_id")))) < -10:
        return None

    return best


# ============================================================
# SOLVERS
# ============================================================

def solve_product(parsed: dict) -> list[str]:
    logger.info("[SOLVER PRODUCT] start")

    spec = parsed["products"][0]
    products = search_candidates(spec)
    best = choose_best(products, spec)

    if not best:
        logger.warning("[SOLVER PRODUCT] failed")
        return [FALLBACK_PRODUCT_ID]

    logger.info("[SOLVER PRODUCT] selected=%s", best.get("product_id"))
    return [str(best["product_id"])]


def solve_shop(parsed: dict) -> list[str]:
    logger.info("[SOLVER SHOP] start")

    specs = parsed["products"]
    results_per_spec = [search_candidates(spec) for spec in specs]

    shop_map: dict[str, dict[int, list[dict]]] = {}
    for idx, products in enumerate(results_per_spec):
        for p in products:
            shop_id = str(p.get("shop_id", ""))
            if not shop_id:
                continue
            shop_map.setdefault(shop_id, {}).setdefault(idx, []).append(p)

    full_shops = [
        shop_id for shop_id, coverage in shop_map.items()
        if all(i in coverage for i in range(len(specs)))
    ]

    logger.info("[SOLVER SHOP] full_coverage_shops=%d", len(full_shops))

    best_solution = None
    best_score = -999999.0

    for shop_id in full_shops[:MAX_SHOP_CANDIDATE_SHOPS]:
        selected = []
        total_score = 0.0

        for idx, spec in enumerate(specs):
            best = choose_best(shop_map[shop_id][idx], spec)
            if not best:
                break
            selected.append(best)
            total_score += product_score(best, spec)

        if len(selected) == len(specs) and total_score > best_score:
            best_score = total_score
            best_solution = selected

    # Anchor fallback: choose best product first, then search remaining specs in same shop.
    if not best_solution:
        logger.warning("[SOLVER SHOP] no full shop solution; trying anchor fallback")
        for idx, spec in enumerate(specs):
            anchor = choose_best(results_per_spec[idx], spec)
            if not anchor:
                continue

            shop_id = str(anchor.get("shop_id", ""))
            solution: list[dict | None] = [None] * len(specs)
            solution[idx] = anchor

            ok = True
            for j, other_spec in enumerate(specs):
                if j == idx:
                    continue
                candidates = search_candidates(other_spec, shop_id=shop_id)
                best = choose_best(candidates, other_spec)
                if not best:
                    ok = False
                    break
                solution[j] = best

            if ok:
                best_solution = solution
                break

    if not best_solution:
        logger.warning("[SOLVER SHOP] failed")
        return [FALLBACK_PRODUCT_ID]

    ids = [str(p["product_id"]) for p in best_solution if p and p.get("product_id")]
    logger.info("[SOLVER SHOP] selected_ids=%s", ids)
    return ids or [FALLBACK_PRODUCT_ID]


def solve_voucher(parsed: dict) -> list[str]:
    logger.info("[SOLVER VOUCHER] start")

    voucher = parsed.get("voucher") or {}
    if voucher.get("same_shop"):
        return solve_voucher_same_shop(parsed)

    specs = parsed["products"]
    selected_lists = []

    for spec in specs:
        candidates = search_candidates(spec)
        candidates = filter_candidates(candidates, spec)
        candidates = sorted(candidates, key=lambda p: product_score(p, spec), reverse=True)[:MAX_VOUCHER_CANDIDATES_PER_SPEC]
        if not candidates:
            logger.warning("[SOLVER VOUCHER] no candidates for spec=%s", spec)
            return [FALLBACK_PRODUCT_ID]
        selected_lists.append(candidates)

    best_combo = None
    best_score = -999999.0

    for combo in cartesian_product(*selected_lists):
        prices = [p.get("price") for p in combo]
        if not all(isinstance(x, (int, float)) for x in prices):
            continue

        total = float(sum(prices))
        final = price_after_voucher(total, voucher)

        if final <= float(voucher.get("budget", 0)):
            score = sum(product_score(p, specs[i]) for i, p in enumerate(combo))
            if total >= float(voucher.get("threshold", 0)):
                score += 10
            if score > best_score:
                best_score = score
                best_combo = combo

    if not best_combo:
        logger.warning("[SOLVER VOUCHER] no budget-valid combo; fallback to best per spec")
        fallback = []
        for i, spec in enumerate(specs):
            best = choose_best(selected_lists[i], spec)
            if best:
                fallback.append(best)
        return [str(p["product_id"]) for p in fallback] or [FALLBACK_PRODUCT_ID]

    ids = [str(p["product_id"]) for p in best_combo]
    logger.info("[SOLVER VOUCHER] selected_ids=%s", ids)
    return ids


def solve_voucher_same_shop(parsed: dict) -> list[str]:
    logger.info("[SOLVER VOUCHER SAME SHOP] start")

    voucher = parsed.get("voucher") or {}
    specs = parsed["products"]

    results_per_spec = [search_candidates(spec) for spec in specs]

    shop_map: dict[str, dict[int, list[dict]]] = {}
    for idx, products in enumerate(results_per_spec):
        for p in products:
            shop_id = str(p.get("shop_id", ""))
            if not shop_id:
                continue
            shop_map.setdefault(shop_id, {}).setdefault(idx, []).append(p)

    full_shops = [
        shop_id for shop_id, coverage in shop_map.items()
        if all(i in coverage for i in range(len(specs)))
    ]

    logger.info("[SOLVER VOUCHER SAME SHOP] full_coverage_shops=%d", len(full_shops))

    best_combo = None
    best_score = -999999.0

    for shop_id in full_shops[:MAX_SHOP_CANDIDATE_SHOPS]:
        candidate_lists = []
        for idx, spec in enumerate(specs):
            filtered = filter_candidates(shop_map[shop_id][idx], spec)
            top = sorted(filtered, key=lambda p: product_score(p, spec), reverse=True)[:5]
            candidate_lists.append(top)

        for combo in cartesian_product(*candidate_lists):
            prices = [p.get("price") for p in combo]
            if not all(isinstance(x, (int, float)) for x in prices):
                continue

            total = float(sum(prices))
            final = price_after_voucher(total, voucher)

            if final <= float(voucher.get("budget", 0)):
                score = sum(product_score(p, specs[i]) for i, p in enumerate(combo))
                if total >= float(voucher.get("threshold", 0)):
                    score += 10
                if score > best_score:
                    best_score = score
                    best_combo = combo

    if not best_combo:
        logger.warning("[SOLVER VOUCHER SAME SHOP] no valid combo; fallback to shop solver")
        fallback_parsed = dict(parsed)
        fallback_parsed["task_type"] = "shop"
        return solve_shop(fallback_parsed)

    ids = [str(p["product_id"]) for p in best_combo]
    logger.info("[SOLVER VOUCHER SAME SHOP] selected_ids=%s", ids)
    return ids


# ============================================================
# MAIN
# ============================================================

def agent_main(problem_data: Dict) -> List[Dict[str, Any]]:
    query = problem_data.get("query", "")
    steps: list[dict] = []

    task_type = detect_task_type(query)
    parsed = parse_query_with_ai(query, task_type)

    steps.append(create_dialogue_step(
        think=(
            f"I received the task and identified it as a {task_type} task. "
            f"I parsed the request into structured product specs: "
            f"{json.dumps(parsed.get('products', []), ensure_ascii=False)}"
        ),
        tool_results=[],
        response="",
        query=query,
        step=1,
    ))

    try:
        if task_type == "product":
            product_ids = solve_product(parsed)
        elif task_type == "shop":
            product_ids = solve_shop(parsed)
        else:
            product_ids = solve_voucher(parsed)

        product_ids_text = ",".join(product_ids) if product_ids else FALLBACK_PRODUCT_ID
        status = "success" if product_ids_text != FALLBACK_PRODUCT_ID else "failure"

    except Exception as exc:
        logger.exception("[MAIN] Agent failed: %s", exc)
        product_ids_text = FALLBACK_PRODUCT_ID
        status = "failure"

    rec = execute_tool_call(
        "recommend_product",
        {"product_ids": product_ids_text},
    )

    term = execute_tool_call(
        "terminate",
        {"status": status},
    )

    steps.append(create_dialogue_step(
        think=(
            f"I completed the workflow: parsed the query, searched candidates, "
            f"filtered hard constraints, scored candidates, chose the best result, "
            f"and now recommend product IDs: {product_ids_text}. Status: {status}."
        ),
        tool_results=[rec, term],
        response=f"I recommend: {product_ids_text}",
        query=query,
        step=2,
    ))

    return steps
