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
        self.token_logs: dict[str, dict[str, list[int]]] = defaultdict(lambda: {"prompt_tokens": [], "completion_tokens": []})

    def total_cost(self, instance: Any = None) -> float:
        if instance is not None and hasattr(instance, "costs"):
            data = instance.costs.values()
        else:
            data = self.costs.values()
        return round(sum(sum(lst) for lst in data), 6)

    def track_cost(self, response_index: int = 0):
        def decorator(fn: Callable):
            is_async = inspect.iscoroutinefunction(fn)
            
            async def _async_wrapper(*args, **kwargs):
                result = await fn(*args, **kwargs)
                resp = (result[response_index] if isinstance(result, (tuple, list)) else result)
                inst = args[0] if args else None

                model_name = self._extract_model_name(inst, args)
                self.check_company(model_name)

                usage = getattr(resp, 'usage', None) or getattr(resp, 'usage_metadata', None)
                pt = getattr(usage, 'prompt_tokens', getattr(usage, 'input_tokens', getattr(usage, 'prompt_token_count', 0)))
                ct = getattr(usage, 'completion_tokens', getattr(usage, 'output_tokens', getattr(usage, 'candidates_token_count', 0)))
                cost = calc_cost_from_completion(resp, self.price_detail[model_name])

                target_costs = (inst.costs if hasattr(inst, 'costs') else self.costs)
                target_tokens = (inst.token_logs if hasattr(inst, 'token_logs') else self.token_logs)

                target_costs.setdefault(model_name, []).append(cost)
                target_tokens.setdefault(model_name, {"prompt_tokens": [], "completion_tokens": []})
                target_tokens[model_name]["prompt_tokens"].append(pt)
                target_tokens[model_name]["completion_tokens"].append(ct)

                return result

            def _sync_wrapper(*args, **kwargs):
                result = fn(*args, **kwargs)
                resp = (result[response_index] if isinstance(result, (tuple, list)) else result)
                inst = args[0] if args else None

                model_name = self._extract_model_name(inst, args)
                self.check_company(model_name)

                usage = getattr(resp, 'usage', None) or getattr(resp, 'usage_metadata', None)
                pt = getattr(usage, 'prompt_tokens', getattr(usage, 'input_tokens', getattr(usage, 'prompt_token_count', 0)))
                ct = getattr(usage, 'completion_tokens', getattr(usage, 'output_tokens', getattr(usage, 'candidates_token_count', 0)))
                cost = calc_cost_from_completion(resp, self.price_detail[model_name])

                target_costs = (inst.costs if hasattr(inst, 'costs') else self.costs)
                target_tokens = (inst.token_logs if hasattr(inst, 'token_logs') else self.token_logs)

                target_costs.setdefault(model_name, []).append(cost)
                target_tokens.setdefault(model_name, {"prompt_tokens": [], "completion_tokens": []})
                target_tokens[model_name]["prompt_tokens"].append(pt)
                target_tokens[model_name]["completion_tokens"].append(ct)

                return result

            return _async_wrapper if is_async else _sync_wrapper

        return decorator

    def _extract_model_name(self, inst, args):
        if hasattr(inst, "model_name"):
            return inst.model_name
        elif args:
            return args[0]
        else:
            return None

    def check_company(self, model_name: str):
        if model_name is None:
            raise ValueError("Model name is required for pricing lookup.")
        lower = model_name.lower()
        if "gpt" in lower or "o1" in lower or "o3" in lower or "o4" in lower:
            self.price_detail = self.pricing["openai"]
        elif "claude" in lower:
            self.price_detail = self.pricing.get("antrophic", {})
        elif "gemini" in lower:
            self.price_detail = self.pricing.get("google", {})
        else:
            raise ValueError(f"Unsupported model: {model_name}")

cost_tracker = CostTracker()