def _get_attr_any(obj, keys, default=0):
    """키 리스트 중 obj에 있는 첫 번째 attr 값을 반환, 없으면 default."""
    for k in keys:
        val = getattr(obj, k, None)
        if val is not None:
            return val
    return default

def _get_nested_attr(obj, parent_key, child_key, default=0):
    parent = getattr(obj, parent_key, None)
    if parent is None:
        if isinstance(obj, dict):
            parent = obj.get(parent_key)
    if parent is not None:
        if isinstance(parent, dict):
            return parent.get(child_key, default)
        return getattr(parent, child_key, default)
    return default

def calc_cost_from_completion(resp, pricing) -> tuple[int,int,int,int,float]:
    """
    resp 의 usage, usage_metadata, usage_tokens 등에서
      - prompt_tokens
      - completion_tokens
      - cache_tokens (cache_creation_input + cache_read_input + cached_tokens)
      - thinking_tokens (reasoning_tokens or thoughts_token_count)
    를 뽑아내어, cost까지 계산해 반환합니다.
    """
    # 1) usage 객체 찾기
    usage = None
    for attr in ("usage", "usage_metadata", "usage_tokens", "response_metadata"):
        usage = getattr(resp, attr, None)
        if usage is not None:
            break
    if not usage:
        return 0, 0, 0, 0, 0.0

    # 2) 토큰 키 추출
    pt = _get_attr_any(usage, ("prompt_tokens", "input_tokens", "prompt_token_count"))
    ct = _get_attr_any(usage, ("completion_tokens", "output_tokens", "candidates_token_count"))
    
    # cache tokens (Claude: cache_creation_input_tokens/cache_read_input_tokens, OpenAI: prompt_tokens_details.cached_tokens)
    cache_created = _get_attr_any(usage, ("cache_creation_input_tokens",))
    cache_read    = _get_attr_any(usage, ("cache_read_input_tokens",))
    
    # OpenAI nested details
    openai_cached = _get_nested_attr(usage, "prompt_tokens_details", "cached_tokens", 0)
    
    cache_tokens  = cache_created + cache_read + openai_cached
    
    # thinking tokens (OpenAI “reasoning_tokens” or Google “thoughts_token_count”)
    thinking = _get_attr_any(usage, ("reasoning_tokens", "thoughts_token_count"))
    
    # OpenAI nested reasoning details
    openai_reasoning = _get_nested_attr(usage, "completion_tokens_details", "reasoning_tokens", 0)
    thinking = max(thinking, openai_reasoning)

    # For OpenAI specifically, pt includes cached_tokens, but pricing is different
    # If openai_cached > 0, we should subtract it from standard prompt tokens for accurate split billing
    pt_billable = pt
    if openai_cached > 0:
        pt_billable = max(0, pt - openai_cached)

    cost = round(
        pt_billable * pricing.get("prompt", 0)
      + ct * pricing.get("completion", 0)
      + openai_cached * pricing.get("cache_read_input_tokens", pricing.get("cache", 0))
      + cache_created * pricing.get("cache_creation_input_tokens", pricing.get("cache", 0))
      + cache_read * pricing.get("cache_read_input_tokens", pricing.get("cache", 0))
      + thinking * pricing.get("thinking", 0)
    , 6)
    # Extract tool calls if available
    tc_count = 0
    choices = getattr(resp, "choices", None)
    if choices and isinstance(choices, list) and len(choices) > 0:
        first_choice = choices[0]
        message = getattr(first_choice, "message", None)
        if message:
            tool_calls = getattr(message, "tool_calls", None)
            if tool_calls and isinstance(tool_calls, list):
                tc_count = len(tool_calls)

    return pt, ct, cache_tokens, thinking, cost, tc_count

def calc_cost_from_aimessages(class_name, resp):
    usage = getattr(resp, "response_metadata", None)
    if not usage:
        raise ValueError("Can't get attr 'response_metadata' in your response!")

    # 1) 모델 이름 뽑기
    model_meta_keys = ("model_name", "model")
    model_name = next((usage[k] for k in model_meta_keys if k in usage), None)
    if model_name is None:
        raise ValueError("No model_name found in response_metadata")

    # 2) 해당 모델의 요율(딕셔너리) 가져오기
    detail = check_and_set_price_detail(class_name, model_name)

    # 3) 토큰 사용량 뽑기
    token_usage = next((usage[k] for k in ("token_usage","usage","usage_metadata") if k in usage), {})
    pt = token_usage.get("prompt_tokens",
         token_usage.get("input_tokens",
         token_usage.get("prompt_token_count", 0)))
    ct = token_usage.get("completion_tokens",
         token_usage.get("output_tokens",
         token_usage.get("candidates_token_count", 0)))

    # 4) 캐시 생성·읽기 토큰, thinking 토큰(예: reasoning_tokens)
    cache_created = token_usage.get("cache_creation_input_tokens", 0)
    cache_read    = token_usage.get("cache_read_input_tokens", 0)
    
    # OpenAI nested structures inside token_usage
    openai_cached = _get_nested_attr(token_usage, "prompt_tokens_details", "cached_tokens", 0)
    openai_reasoning = _get_nested_attr(token_usage, "completion_tokens_details", "reasoning_tokens", 0)
    
    thinking      = token_usage.get("reasoning_tokens",
                   token_usage.get("thoughts_token_count", 0))
    thinking = max(thinking, openai_reasoning)
    
    total_cache = cache_created + cache_read + openai_cached
    
    pt_billable = pt
    if openai_cached > 0:
        pt_billable = max(0, pt - openai_cached)

    # 5) 비용 계산 — detail 에서 바로 get
    cost = round(
        pt_billable * detail.get("prompt", 0)
      + ct * detail.get("completion", 0)
      + openai_cached * detail.get("cache_read_input_tokens", 0)
      + cache_created * detail.get("cache_creation_input_tokens", 0)
      + cache_read    * detail.get("cache_read_input_tokens", 0)
      + thinking      * detail.get("thinking", 0)
    , 6)

    # 6) Extract tool calls (Langchain AIMessage)
    tc_count = 0
    if hasattr(resp, "tool_calls") and isinstance(resp.tool_calls, list):
        tc_count = len(resp.tool_calls)
    elif hasattr(resp, "additional_kwargs") and isinstance(resp.additional_kwargs, dict):
        tc_add = resp.additional_kwargs.get("tool_calls")
        if tc_add and isinstance(tc_add, list):
            tc_count = len(tc_add)

    return pt, ct, total_cache, thinking, cost, tc_count, model_name

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
    all_pricing = getattr(target, "pricing", {})
    # 1) 먼저 어떤 카테고리(openai, antrophic, google) 에 속하는지 골라낸다
    if any(key in lower for key in ("gpt", "o1", "o3", "o4")):
        category = "openai"
    elif "claude" in lower:
        category = "antrophic"
    elif "gemini" in lower:
        category = "google"
    elif "deepseek" in lower:
        category = "deepseek"
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    # 2) 그 카테고리 딕셔너리에서 실제 model_name 키를 꺼낸다
    category_dict = all_pricing.get(category, {})
    detail = category_dict.get(model_name)
    if detail is None:
        raise ValueError(f"No pricing entry for model '{model_name}' in category '{category}'")
    return detail