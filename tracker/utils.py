def calc_cost_from_completion(resp, pricing) -> float:
    usage = None
    for attr in ("usage", "usage_metadata"):
        usage = getattr(resp, attr, None)
        if usage:
            break
    if not usage:
        return 0.0

    prompt_keys = ("prompt_tokens", "input_tokens", "prompt_token_count")
    completion_keys = ("completion_tokens", "output_tokens", "candidates_token_count")

    pt = next((getattr(usage, k) for k in prompt_keys if hasattr(usage, k)), 0)
    ct = next((getattr(usage, k) for k in completion_keys if hasattr(usage, k)), 0)
    cost = round((pt * pricing.get("prompt", 0) + ct * pricing.get("completion", 0)), 6)
    return pt, ct, cost

def calc_cost_from_aimessages(class_name, resp):
    """
    Get everythinkg in Langchain calss
    """
    usage = getattr(resp, "response_metadata", None)
    if not usage:
        raise ValueError("Can't get attr 'response_metadata' in your response!")

    model_meta_keys = ("model_name", "model")
    model_name = next((usage[k] for k in model_meta_keys if k in usage), 0)
    pricing = check_and_set_price_detail(class_name, model_name)

    meta_keys = ("token_usage", "usage", "usage_metadata")
    prompt_keys = ("prompt_tokens", "input_tokens", "prompt_token_count")
    completion_keys = ("completion_tokens", "output_tokens", "candidates_token_count")

    token_usage = next((usage[k] for k in meta_keys if k in usage), 0)

    pt = next((token_usage[k] for k in prompt_keys if k in token_usage), 0)
    ct = next((token_usage[k] for k in completion_keys if k in token_usage), 0)
    cost = round((pt * pricing[model_name].get("prompt", 0) + ct * pricing[model_name].get("completion", 0)), 6)
    return pt, ct, cost, model_name

def is_ai_message(obj) -> bool:
    """
    Checks if the variable obj is an instance of langchain_core.messages.ai.AIMessage.
    (no library imports, just judged by module name and class name)
    """
    cls = getattr(obj, "__class__", None)
    if cls is None:
        return False

    module_name = getattr(cls, "__module__", "")
    class_name  = getattr(cls, "__name__",  "")

    return (module_name == "langchain_core.messages.ai"
            and class_name == "AIMessage")

def check_and_set_price_detail(target, model_name: str):
    """
    target.pricing 에서 model_name 에 맞는 가격 상세를 꺼내서
    target.price_detail 속성으로 설정해 줍니다.
    """
    if model_name is None:
        raise ValueError("Model name is required for pricing lookup.")
    lower = model_name.lower()
    pricing = getattr(target, "pricing", {})

    if any(key in lower for key in ("gpt", "o1", "o3", "o4")):
        detail = pricing["openai"]
    elif "claude" in lower:
        detail = pricing.get("antrophic", {})
    elif "gemini" in lower:
        detail = pricing.get("google", {})
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    setattr(target, "price_detail", detail)
    return detail