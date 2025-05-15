import inspect
from functools import wraps
from collections import defaultdict
from typing import Any, Callable
from .pricing_loader import load_pricing_yaml
from .utils import calc_cost_from_completion

class CostTracker:
    def __init__(self, 
                pricing: dict[str, dict[str, float]] = None, 
                pricing_path: str = "pricing.yaml"):
        self.pricing = pricing or load_pricing_yaml(pricing_path)
        self.costs: dict[str, list[float]] = defaultdict(list)

    def total_cost(self, instance: Any = None) -> float:
        if instance is not None and hasattr(instance, "costs"):
            data = instance.costs.values()
        else:
            data = self.costs.values()
        return round(sum(sum(lst) for lst in data), 6)

    def track_cost(self, response_index: int = 0):
        def decorator(fn: Callable):
            is_async = inspect.iscoroutinefunction(fn)
            
            if is_async:
                @wraps(fn)
                async def async_wrapper(*args, **kwargs):
                    result = await fn(*args, **kwargs)
                    resp = (result[response_index]
                            if isinstance(result, (tuple, list)) else result)
                    inst = args[0] if args else None

                    # 모델 이름 추출 로직은 기존과 동일
                    if hasattr(inst, "model_name"):
                        model_name = inst.model_name
                    elif args:
                        model_name = args[0]
                    else:
                        model_name = None

                    self.check_company(model_name)

                    cost = calc_cost_from_completion(resp, self.price_detail[model_name])
                    if hasattr(inst, "costs"):
                        inst.costs.setdefault(model_name, []).append(cost)
                    else:
                        self.costs.setdefault(model_name, []).append(cost)
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

                    self.check_company(model_name)

                    cost = calc_cost_from_completion(resp, self.price_detail[model_name])
                    if hasattr(inst, "costs"):
                        inst.costs.setdefault(model_name, []).append(cost)
                    else:
                        self.costs.setdefault(model_name, []).append(cost)
                    return result
                return sync_wrapper

        return decorator

    def check_company(self, model_name):
        if "gpt" in model_name or "o1" in model_name or "o3" in model_name or "o4" in model_name :
            self.price_detail = self.pricing["openai"]
        
        elif "claude" in model_name:
            self.price_detail = self.pricing["antrophic"]

        elif "gemini" in model_name:
            self.price_detail = self.pricing["google"]
            
        else:
            raise ValueError(f"Unsurppot Model: {model_name}")

cost_tracker = CostTracker()