import json
import re
from typing import Any, Dict, List
from urllib.parse import quote_plus

from src.agent.agent_interface import Tool, create_dialogue_step, execute_tool_call
from src.agent.proxy_client import ProxyClient

_search = ProxyClient(timeout=40, max_retries=4)
_llm = ProxyClient(timeout=60, max_retries=2)
_detail_cache: Dict[str, Dict[str, Any]] = {}
FALLBACK_PRODUCT_ID = "0"

ALLOWED_MODELS = [
    "deepseek-ai/DeepSeek-V3.2-TEE",
    "deepseek-ai/DeepSeek-V3.1-TEE",
    "Qwen/Qwen3-235B-A22B-Instruct-2507-TEE",
    "moonshotai/Kimi-K2.5-TEE",
    "openai/gpt-oss-120b-TEE",
]

PRODUCT_PARSE_PROMPT = """Extract JSON only:
{"products":[{"keywords":"2-8 words","price_range":"min-max"|null,"service":"official|freeShipping|COD|flashsale|comma-separated|null"}],"is_shop_voucher":bool}
Rules:
- Keep only product words in keywords. Remove service words.
- For single-product query: return one product.
- For multi-product query: return one object per requested product.
- price_range examples: "100-500", "200-", "0-300", or null.
"""

VOUCHER_PARSE_PROMPT = """Extract JSON only:
{"products":[{"keywords":"2-8 words","price_range":"min-max"|null,"service":"official|freeShipping|COD|flashsale|comma-separated|null"}],"voucher":{"discount_type":"fixed|percentage","discount_value":number,"threshold":number,"cap":number,"budget":number},"is_shop_voucher":bool}
Rules:
- Keep voucher fields numeric.
- If cap absent, set cap=0.
"""


@Tool
def find_product(
    q: str,
    page: int = 1,
    shop_id: str | None = None,
    price: str | None = None,
    sort: str | None = None,
    service: str | None = None,
) -> List[Dict]:
    params = {
        "q": quote_plus(q),
        "page": page,
        "shop_id": shop_id,
        "price": price,
        "sort": sort,
        "service": service,
    }
    if not sort:
        params.pop("sort")
    if not shop_id:
        params.pop("shop_id")
    if not price:
        params.pop("price")
    if not service:
        params.pop("service")
    result = _search.get("/search/find_product", params)
    return result if result else []


@Tool
def view_product_information(product_ids: str) -> List[Dict]:
    result = _search.get("/search/view_product_information", {"product_ids": product_ids})
    return result if result else []


@Tool
def recommend_product(product_ids: str) -> str:
    return f"Having recommended the products to the user: {product_ids}."


@Tool
def terminate(status: str = "success") -> str:
    return f"The interaction has been completed with status: {status}"


def _append_step(
    steps: List[Dict[str, Any]],
    query: str,
    think: str,
    tool_results: List[Dict[str, Any]],
    response: str = "",
) -> None:
    steps.append(
        create_dialogue_step(
            think=think,
            tool_results=tool_results,
            response=response,
            query=query,
            step=len(steps) + 1,
        )
    )


def _finish(steps: List[Dict[str, Any]], query: str, product_ids: List[str], status: str) -> None:
    pid_csv = ",".join([p for p in product_ids if p]) if product_ids else FALLBACK_PRODUCT_ID
    rec = execute_tool_call("recommend_product", {"product_ids": pid_csv})
    term = execute_tool_call("terminate", {"status": status})
    _append_step(steps, query, f"Finalizing with ids={pid_csv}, status={status}.", [rec, term], "Done.")


def _json_object(text: str) -> Dict[str, Any] | None:
    cleaned = re.sub(r"```json?\s*", "", text).replace("```", "").strip()
    try:
        out = json.loads(cleaned)
        return out if isinstance(out, dict) else None
    except Exception:
        m = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not m:
            return None
        try:
            out = json.loads(m.group(0))
            return out if isinstance(out, dict) else None
        except Exception:
            return None


def _llm_extract(query: str, with_voucher: bool) -> Dict[str, Any]:
    prompt = VOUCHER_PARSE_PROMPT if with_voucher else PRODUCT_PARSE_PROMPT
    for model in ALLOWED_MODELS:
        result = _llm.post(
            "/inference/chat/completions",
            json_data={
                "model": model,
                "temperature": 0,
                "stream": False,
                "messages": [
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": query},
                ],
            },
        )
        if not (result and result.get("choices")):
            continue
        content = result["choices"][0].get("message", {}).get("content", "")
        parsed = _json_object(content)
        if parsed:
            return parsed
    return {"products": [{"keywords": query, "price_range": None, "service": None}]}


def _get_details(product_ids: List[str]) -> Dict[str, Dict[str, Any]]:
    ids = [pid for pid in product_ids if pid and pid not in _detail_cache]
    if ids:
        result = _search.get("/search/view_product_information", {"product_ids": ",".join(ids)})
        if isinstance(result, list):
            for item in result:
                _detail_cache[str(item.get("product_id", ""))] = item
    return {pid: _detail_cache[pid] for pid in product_ids if pid in _detail_cache}


def _score(product: Dict[str, Any], query: str, detail: Dict[str, Any] | None = None) -> float:
    words = [w for w in re.findall(r"\w+", query.lower()) if len(w) > 1]
    title = str(product.get("title", "")).lower()
    score = 0.0
    for w in words:
        if w in title:
            score += 2.0
    if detail:
        attrs = json.dumps(detail.get("attributes", {})).lower()
        sku = json.dumps(detail.get("sku_options", {})).lower()
        for w in words:
            if w in attrs:
                score += 2.0
            if w in sku:
                score += 2.0
    price = product.get("price")
    if isinstance(price, (int, float)):
        score -= float(price) / 100000.0
    return score


def _best_from_results(query: str, products: List[Dict[str, Any]]) -> Dict[str, Any] | None:
    if not products:
        return None
    top = products[:12]
    pids = [str(p.get("product_id", "")) for p in top if p.get("product_id")]
    details = _get_details(pids)
    ranked = sorted(top, key=lambda p: _score(p, query, details.get(str(p.get("product_id", "")))), reverse=True)
    return ranked[0] if ranked else None


def _run_single(query: str, spec: Dict[str, Any], steps: List[Dict[str, Any]]) -> List[str]:
    q = spec.get("keywords") or query
    payload = {"q": q, "page": 1}
    if spec.get("price_range"):
        payload["price"] = spec["price_range"]
    if spec.get("service"):
        payload["service"] = spec["service"]
    r = execute_tool_call("find_product", payload)
    _append_step(steps, query, f"Searched for '{q}'.", [r], "")
    products = r.get("result") or []
    best = _best_from_results(query, products)
    return [str(best.get("product_id"))] if best else []


def _run_multi(query: str, specs: List[Dict[str, Any]], same_shop: bool) -> List[str]:
    selected: List[str] = []
    selected_shop: str | None = None
    for spec in specs:
        q = spec.get("keywords") or query
        payload: Dict[str, Any] = {"q": q, "page": 1}
        if spec.get("price_range"):
            payload["price"] = spec["price_range"]
        if spec.get("service"):
            payload["service"] = spec["service"]
        if same_shop and selected_shop:
            payload["shop_id"] = selected_shop
        r = execute_tool_call("find_product", payload)
        products = r.get("result") or []
        best = _best_from_results(q, products)
        if not best:
            continue
        selected.append(str(best.get("product_id")))
        if same_shop and not selected_shop:
            selected_shop = str(best.get("shop_id", "")) or None
    return selected


def agent_main(problem_data: Dict) -> List[Dict]:
    _detail_cache.clear()
    query = problem_data.get("query", "")
    steps: List[Dict[str, Any]] = []
    try:
        ql = query.lower()
        is_voucher = any(k in ql for k in ("voucher", "budget", "discount"))
        parsed = _llm_extract(query, with_voucher=is_voucher)
        specs = parsed.get("products") or [{"keywords": query, "price_range": None, "service": None}]
        same_shop = ("same shop" in ql) or (is_voucher and bool(parsed.get("is_shop_voucher")))

        _append_step(
            steps,
            query,
            f"Parsed request into {len(specs)} product spec(s); same_shop={same_shop}, voucher={is_voucher}.",
            [],
            "",
        )

        if len(specs) == 1:
            picks = _run_single(query, specs[0], steps)
        else:
            picks = _run_multi(query, specs, same_shop)

        _finish(steps, query, picks, "success" if picks else "failure")
    except Exception:
        _finish(steps, query, [FALLBACK_PRODUCT_ID], "failure")
    return steps