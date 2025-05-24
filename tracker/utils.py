def _get_attr_any(obj, keys, default=0):
    """키 리스트 중 obj에 있는 첫 번째 attr 값을 반환, 없으면 default."""
    for k in keys:
        val = getattr(obj, k, None)
        if val is not None:
            return val
    return default

def calc_cost_from_completion(resp, pricing) -> tuple[int,int,int,int,float]:
    """
    resp 의 usage, usage_metadata, usage_tokens 등에서
      - prompt_tokens
      - completion_tokens
      - cache_tokens (cache_creation_input + cache_read_input)
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
    # cache tokens (Claude)
    cache_created = _get_attr_any(usage, ("cache_creation_input_tokens",))
    cache_read    = _get_attr_any(usage, ("cache_read_input_tokens",))
    cache_tokens  = cache_created + cache_read
    # thinking tokens (OpenAI “reasoning_tokens” or Google “thoughts_token_count”)
    thinking = _get_attr_any(usage, ("reasoning_tokens", "thoughts_token_count"))

    cost = round(
        pt * pricing.get("prompt", 0)
      + ct * pricing.get("completion", 0)
      + cache_tokens * pricing.get("cache", 0)
      + thinking   * pricing.get("thinking", 0)
    , 6)
    return pt, ct, cache_tokens, thinking, cost

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
    # detail 예: {"prompt":0.0000025, "completion":0.0000100, 
    #          "cache_creation_input_tokens":0.00001875, ...}

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
    thinking      = token_usage.get("reasoning_tokens",
                   token_usage.get("thoughts_token_count", 0))

    # 5) 비용 계산 — detail 에서 바로 get
    cost = round(
        pt * detail.get("prompt", 0)
      + ct * detail.get("completion", 0)
      + cache_created * detail.get("cache_creation_input_tokens", 0)
      + cache_read    * detail.get("cache_read_input_tokens", 0)
      + thinking      * detail.get("thinking", 0)
    , 6)

    # (원하시면 cache_created+cache_read 를 합쳐서 내보내셔도 됩니다)
    return pt, ct, cache_created + cache_read, thinking, cost, model_name

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
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    # 2) 그 카테고리 딕셔너리에서 실제 model_name 키를 꺼낸다
    category_dict = all_pricing.get(category, {})
    detail = category_dict.get(model_name)
    if detail is None:
        raise ValueError(f"No pricing entry for model '{model_name}' in category '{category}'")
    return detail