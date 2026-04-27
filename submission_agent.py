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
    - When a query has "any" immediately before a noun, keep BOTH in keywords as a phrase (e.g. "any season", "any weather")-never drop "any" as filler; include the full "any <word>" pair.
- price_range: "100-500", "100-", "0-500". null if none.
- only_product_type: true if the keywords are only nouns - even if it is a multi-word compound noun (the words together name the product, not describe it). Set false only when at least one additional word is a separate qualifier such as a brand name, color, material, numeric spec, or descriptive adjective added on top of the core product name.
- service: "LazMall / guaranteed authenticity / quick returns"->"official", "free shipping / free delivery" -> "freeShipping", "cash on delivery / COD / payment on delivery" -> "COD", "LazFlash / flash deal / limited-time deal" -> "flashsale"; null if none. Multiple options available, combine them with ",".
JSON only:"""

_SHOP_PROMPT = """Extract search params as JSON. No markdown. Find multi-products.
{"products":[{"query":"the part of the raw query describing this product","keywords":"search query","price_range":"min-max"|null,"service":"official"|"freeShipping"|"COD"|"flashsale"|null,"only_product_type":bool}]}
- keywords: product type + brand + material + color + size + quantity/units + weight/volume + dimensions + packaging/logistics + product/misc + sharp + fit + style + length + selling unit + use. 2-8 words, include ALL qualifying terms. Keep full color descriptors including any qualifier. Drop opening/fastening mechanism terms. IMPORTANT RULE:
    - Must preserve the left-to-right order of terms exactly as they appear in the query.
    - MUST never include service related terms in keywords.
    - Compact any number+unit pair: remove the space and use the standard abbreviation.
    - When a query has "any" immediately before a noun, keep BOTH in keywords as a phrase (e.g. "any season", "any weather")-never drop "any" as filler; include the full "any <word>" pair.
- price_range: "100-500", "100-", "0-500". null if none.
- service: "LazMall"->"official", "free shipping / free delivery" -> "freeShipping", "cash on delivery / COD / payment on delivery" -> "COD", "LazFlash / flash deal / limited-time deal" -> "flashsale"; null if none. Multiple options available, combine them with ",".
- only_product_type: true if the keywords are the product type name alone - even if it is a multi-word compound noun (the words together name the product, not describe it). Set false only when at least one additional word is a separate qualifier such as a brand name, color, material, numeric spec, or descriptive adjective added on top of the core product name.
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
    - When a query has "any" immediately before a noun, keep BOTH in keywords as a phrase (e.g. "any season", "any weather")-never drop "any" as filler; include the full "any <word>" pair.
- price_range: "100-500", "100-", "0-500". null if none.
- service: "LazMall / guaranteed authenticity / quick returns"->"official", "free shipping / free delivery" -> "freeShipping", "cash on delivery / COD / payment on delivery" -> "COD", "LazFlash / flash deal / limited-time deal" -> "flashsale"; null if none. Multiple options available, combine them with ",".
- only_product_type: true if the keywords are the product type name alone - even if it is a multi-word compound noun (the words together name the product, not describe it). Set false only when at least one additional word is a separate qualifier such as a brand name, color, material, numeric spec, or descriptive adjective added on top of the core product name.
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
1. Prefer explicit structured evidence in attributes and sku_options. In the user_content.request (it is the requirement query), "any" does not refer to special terms or items; it means the same as "all."
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

_THINK_NARRATOR_PROMPT = """You are an AI shopping assistant. Write 2-4 sentences of internal, first-person reasoning explaining what you are doing at this step.

You receive a JSON object with a "query" field and additional context. Identify which case applies from the keys present and write accordingly:

CASE 1 - "keywords" + "price_constraints" + "service_filters" present (query analysis / planning step):
You are analysing the user's request before searching. State what the user wants to buy, list the exact search keywords you will use, mention any price range and service type constraints. If "only_product_type" is true, explain that the query is a bare product type with no extra qualifiers so you will append "only" to the search to avoid unrelated products - quote the "only_product_type_reason" value if present. If "budget_constraint" is present, note the voucher discount type, threshold, and budget.

CASE 2 - "search_query" + "top_candidates" present (search results step):
You just ran a product search. State the exact search query and any price/service filters applied. Report how many results came back ("total_results"). Name the most relevant top candidates by their title and price from "top_candidates". State what you will evaluate next.

CASE 3 - "selected" + "constraints" present (product selection step):
You are choosing the best product. Name the selected product by its product_id and title. Explain which specific attributes, SKU options, or specs from "selected.attributes" and "selected.sku_options_sample" satisfy the constraints (price, service, keywords). Quote the "llm_reason" value if it is non-empty and explain why it is the best match.

CASE 4 - "product_count" + "products" present (multi-product shop planning step):
You are about to search for multiple products from the same shop. State how many products are needed and name each one using its keywords value. Mention the price range and service constraint for each product.

CASE 5 - "shop_id" + "selected_products" present (shop found step):
You found a shop carrying all required products. State the shop ID. For each entry in "selected_products", name its title and price. Confirm they collectively satisfy the query. If "llm_reasoning" is also present, reference the relevance scores that led to this shop being chosen.

CASE 6 - "budget_constraint" + "candidates_per_product" present (voucher candidate evaluation step):
You are checking which products fit within the voucher budget. State the voucher discount type, threshold, and the max allowed total from "max_allowed_total". For each entry in "candidates_per_product", name the keywords and the top product candidate's title and price.

CASE 7 - "selected_products" + "budget_constraint" present, no "candidates_per_product" (voucher selection confirmed step):
You selected products that fit the voucher budget. Name each product from "selected_products" by title and price. State the total price before discount from "total_before_discount" and confirm it is within the allowed budget. Quote "llm_reason" if present.

CASE 8 - "selected_products" + "total_spent" + "allowed_total" present (fixed-budget selection step):
You finalised products within a fixed spending limit. Name each product, state the exact total spent and the allowed maximum, and confirm the selection is within budget.

CASE 9 - "scoring_summary" + "score_threshold" present (LLM scoring / shop-coverage step):
You just LLM-scored all candidate products against the query. State the score threshold from "score_threshold". For each entry in "scoring_summary", report how many products were collected ("total_collected") and how many passed the threshold ("passed_threshold"), naming the top-scoring candidates by title and score from "top_candidates". State how many full-coverage shops were found using "full_coverage_shops_found" and what you will do next.

CASE 10 - "case_c_resolution" present (anchor-product fallback step):
No single shop covered all required products after score filtering. Describe the sub-case strategy. If "sub_case" is 4, explain how you evaluated partial-coverage shops and filled the missing spec by searching inside the winner shop. Otherwise, name the anchor product using its spec index, keywords, product_id, and shop_id, and explain that you searched the remaining specs within that shop to maximise coverage.

CASE 11 - "recommended_product_ids" + "status" present (final recommendation step):
You are finalising the session. State the product IDs you are recommending from "recommended_product_ids". Confirm the outcome using "status" (success or failure). If "llm_reason" is present, quote it to justify the choice. If "note" is present, mention it.

Rules:
- Always write in first person ("I searched...", "I selected...", "I found...", "I am planning to...").
- Reference actual values from the context: IDs, titles, prices, keywords, shop IDs, attributes, scores.
- Be specific and concrete - never vague or generic.
- Do NOT output JSON or markdown. Plain text only.
- 2-4 sentences maximum."""

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

# NOTE: Delegate to the full top-miner reference implementation in `refer_top/1.py`.
# This keeps behavior aligned with the strong baseline while exposing `agent_main`
# from this submission file.
import importlib.util
from pathlib import Path

_TOP_IMPL_PATH = Path(__file__).resolve().parent / "refer_top" / "1.py"
_TOP_IMPL_SPEC = importlib.util.spec_from_file_location("top_miner_impl", _TOP_IMPL_PATH)
if _TOP_IMPL_SPEC is None or _TOP_IMPL_SPEC.loader is None:
    raise RuntimeError(f"Unable to load top miner agent from {_TOP_IMPL_PATH}")
_TOP_IMPL_MODULE = importlib.util.module_from_spec(_TOP_IMPL_SPEC)
_TOP_IMPL_SPEC.loader.exec_module(_TOP_IMPL_MODULE)

agent_main = _TOP_IMPL_MODULE.agent_main
