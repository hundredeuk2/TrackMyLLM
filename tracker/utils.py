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