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

    return round((pt * pricing.get("prompt", 0) + ct * pricing.get("completion", 0)), 6)