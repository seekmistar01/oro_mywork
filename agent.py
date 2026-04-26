from typing import Dict, List
from src.agent.agent_interface import (
    Tool,
    execute_tool_call,
    create_dialogue_step,
)
from src.agent.proxy_client import ProxyClient
from urllib.parse import quote_plus

_proxy = ProxyClient(timeout=120, max_retries=2)


@Tool
def find_product(q: str, page: int = 1) -> List[Dict]:
    q_encoded = quote_plus(q)
    result = _proxy.get("/search/find_product", {"q": q_encoded, "page": page})
    return result if result else []


@Tool
def view_product_information(product_ids: str) -> List[Dict]:
    result = _proxy.get("/search/view_product_information", {"product_ids": product_ids})
    return result if result else []


@Tool
def recommend_product(product_ids: str) -> str:
    return f"Having recommended the products to the user: {product_ids}."


@Tool
def terminate(status: str = "success") -> str:
    return f"The interaction has been completed with status: {status}"


def agent_main(problem_data: Dict) -> List[Dict]:
    query = problem_data.get("query", "")
    steps = []

    # Step 1: Search
    search_result = execute_tool_call("find_product", {"q": query})
    steps.append(create_dialogue_step(
        think=f"Searching for: {query}",
        tool_results=[search_result],
        response="",
        query=query,
        step=1,
    ))

    products = search_result["result"]
    if not products:
        term = execute_tool_call("terminate", {"status": "failure"})
        steps.append(create_dialogue_step(
            think="No products found.",
            tool_results=[term],
            response="No matching products found.",
            query=query,
            step=2,
        ))
        return steps

    # Step 2: View top result
    top_id = str(products[0]["product_id"])
    view_result = execute_tool_call("view_product_information", {"product_ids": top_id})
    steps.append(create_dialogue_step(
        think=f"Viewing product {top_id}",
        tool_results=[view_result],
        response="",
        query=query,
        step=2,
    ))

    # Step 3: Recommend and terminate
    rec = execute_tool_call("recommend_product", {"product_ids": top_id})
    term = execute_tool_call("terminate", {"status": "success"})
    steps.append(create_dialogue_step(
        think="Recommending the best match.",
        tool_results=[rec, term],
        response=f"I recommend product {top_id}.",
        query=query,
        step=3,
    ))

    return steps