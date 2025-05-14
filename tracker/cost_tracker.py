import inspect
from functools import wraps
from collections import defaultdict
from typing import Any, Callable

GPT_PRICING = {
    "gpt-4.1":                   {"prompt": 2.00   / 1000000, "completion":  8.00   / 1000000},
    "gpt-4.1-mini":              {"prompt": 0.40   / 1000000, "completion":  1.60   / 1000000},
    "gpt-4.1-nano":              {"prompt": 0.10   / 1000000, "completion":  0.40   / 1000000},
    "gpt-4.5-preview":           {"prompt":75.00   / 1000000, "completion":150.00  / 1000000},
    "gpt-4o":                    {"prompt": 2.50   / 1000000, "completion": 10.00   / 1000000},
    "gpt-4o-audio-preview":      {"prompt": 2.50   / 1000000, "completion": 10.00   / 1000000},
    "gpt-4o-realtime-preview":   {"prompt": 5.00   / 1000000, "completion": 20.00   / 1000000},
    "gpt-4o-mini":               {"prompt": 0.15   / 1000000, "completion":  0.60   / 1000000},
    "gpt-4o-mini-audio-preview": {"prompt": 0.15   / 1000000, "completion":  0.60   / 1000000},
    "gpt-4o-mini-realtime-preview":{"prompt":0.60  / 1000000, "completion":  2.40   / 1000000},
    "o1":                        {"prompt":15.00   / 1000000, "completion": 60.00   / 1000000},
    "o1-pro":                    {"prompt":150.00  / 1000000, "completion":600.00   / 1000000},
    "o3":                        {"prompt":10.00   / 1000000, "completion": 40.00   / 1000000},
    "o4-mini":                   {"prompt": 1.10   / 1000000, "completion":  4.40   / 1000000},
    "o3-mini":                   {"prompt": 1.10   / 1000000, "completion":  4.40   / 1000000},
    "o1-mini":                   {"prompt": 1.10   / 1000000, "completion":  4.40   / 1000000},
    "gpt-4o-mini-search-preview":{"prompt":0.15   / 1000000, "completion":  0.60   / 1000000},
    "gpt-4o-search-preview":     {"prompt": 2.50   / 1000000, "completion": 10.00   / 1000000},
    "computer-use-preview":      {"prompt": 3.00   / 1000000, "completion": 12.00   / 1000000},
    "gpt-image-1":                          {"prompt": 5.00   / 1000000}
}

class CostTracker:
    def __init__(self):
        self.standalone_costs: dict[str, list[float]] = defaultdict(list)

    def total_cost(self, instance: Any = None) -> float:
        if instance is not None and hasattr(instance, "costs"):
            data = instance.costs.values()
        else:
            data = self.standalone_costs.values()
        return round(sum(sum(lst) for lst in data), 6)

    def track_cost(self, response_index: int = 0):
        """
        - response_index: usage inform index from completion
        - neccessary to get usage information from the response
            • class method → self.model_name
            • def method   → args[0] => it's must be model_name 
        """
        def decorator(fn: Callable):
            is_async = inspect.iscoroutinefunction(fn)

            def _calc_cost(resp, pricing):
                usage = getattr(resp, "usage", None)
                pt = getattr(usage, "prompt_tokens", 0)
                ct = getattr(usage, "completion_tokens", 0)
                return pt * pricing["prompt"] + ct * pricing["completion"]

            if is_async:
                @wraps(fn)
                async def async_wrapper(*args, **kwargs):
                    result = await fn(*args, **kwargs)
                    resp = (result[response_index]
                            if isinstance(result, (tuple, list)) else result)
                    inst = args[0] if args else None
                    if hasattr(inst, "model_name"):
                        model_name = inst.model_name
                    elif args:
                        model_name = args[0]
                    else:
                        model_name = None
                    if model_name not in GPT_PRICING:
                        raise ValueError(f"No pricing for model: {model_name}")
                    cost = _calc_cost(resp, GPT_PRICING[model_name])
                    if hasattr(inst, "costs"):
                        inst.costs.setdefault(model_name, []).append(cost)
                    else:
                        self.standalone_costs.setdefault(model_name, []).append(cost)
                    return result
                return async_wrapper

            else:
                @wraps(fn)
                def sync_wrapper(*args, **kwargs):
                    result = fn(*args, **kwargs)
                    resp = (result[response_index]
                            if isinstance(result, (tuple, list)) else result)
                    inst = args[0] if args else None
                    if hasattr(inst, "model_name"):
                        model_name = inst.model_name
                    elif args:
                        model_name = args[0]
                    else:
                        model_name = None
                    if model_name not in GPT_PRICING:
                        raise ValueError(f"No pricing for model: {model_name}")
                    cost = _calc_cost(resp, GPT_PRICING[model_name])
                    if hasattr(inst, "costs"):
                        inst.costs.setdefault(model_name, []).append(cost)
                    else:
                        self.standalone_costs.setdefault(model_name, []).append(cost)
                    return result
                return sync_wrapper

        return decorator

cost_tracker = CostTracker()