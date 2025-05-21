import inspect
from functools import wraps
from collections import defaultdict
from typing import Any, Callable
from .pricing_loader import load_pricing_yaml
from .utils import check_and_set_price_detail, calc_cost_from_completion, calc_cost_from_aimessages, is_ai_message
from tabulate import tabulate

class CostTracker:
    def __init__(self, 
                 pricing: dict[str, dict[str, float]] = None, 
                 pricing_path: str = "pricing.yaml"):
        self.pricing = pricing or load_pricing_yaml(pricing_path)
        self.costs: dict[str, list[float]] = defaultdict(list)
        self.token_logs: dict[str, dict[str, list[int]]] = defaultdict(lambda: {
            "prompt_tokens": [], "completion_tokens": []
        })

    def total_cost(self, instance: Any = None) -> float:
        if instance is not None and hasattr(instance, "costs"):
            data = instance.costs.values()
        else:
            data = self.costs.values()
        return round(sum(sum(lst) for lst in data), 6)

    def track_cost(self, *d_args, **d_kwargs):
        def wrapper(fn: Callable):
            response_index = d_kwargs.get("response_index", 0)
            model_name = d_kwargs.get("model_name", None)
            is_async = inspect.iscoroutinefunction(fn)

            if is_async:
                @wraps(fn)
                async def async_wrapper(*args, **kwargs):
                    result = await fn(*args, **kwargs)
                    resp = result[response_index] if isinstance(result, (tuple, list)) else result
                    inst = args[0] if args else None
                    if is_ai_message(resp):
                        pt, ct, cost, extract_model_name = calc_cost_from_aimessages(self, resp)
                        model_used = model_name or extract_model_name
                    else:
                        model_used = model_name or self._extract_model_name(inst, args, kwargs, fn)
                        check_and_set_price_detail(self, model_used)
                        pt, ct, cost = calc_cost_from_completion(resp, self.price_detail[model_used])
                    self._log_cost(inst, model_used, pt, ct, cost)
                    return result
                return async_wrapper

            else:
                @wraps(fn)
                def sync_wrapper(*args, **kwargs):
                    result = fn(*args, **kwargs)
                    resp = result[response_index] if isinstance(result, (tuple, list)) else result
                    inst = args[0] if args else None
                    if is_ai_message(resp):
                        pt, ct, cost, extract_model_name = calc_cost_from_aimessages(self, resp)
                        model_used = model_name or extract_model_name
                    else:
                        model_used = model_name or self._extract_model_name(inst, args, kwargs, fn)
                        check_and_set_price_detail(self, model_used)
                        pt, ct, cost = calc_cost_from_completion(resp, self.price_detail[model_used])
                    self._log_cost(inst, model_used, pt, ct, cost)
                    return result
                return sync_wrapper

        # Handle: @track_cost OR @track_cost(...)
        if len(d_args) == 1 and callable(d_args[0]):
            return wrapper(d_args[0])  # @track_cost
        else:
            return wrapper               # @track_cost(...)

    def _log_cost(self, inst, model_name: str, pt: int, ct: int, cost: float):
        target_costs = getattr(inst, "costs", self.costs)
        target_tokens = getattr(inst, "token_logs", self.token_logs)

        target_costs.setdefault(model_name, []).append(cost)
        target_tokens.setdefault(model_name, {"prompt_tokens": [], "completion_tokens": []})

        # record tokens
        target_tokens[model_name]["prompt_tokens"].append(pt)
        target_tokens[model_name]["completion_tokens"].append(ct)

        # summary
        prompt_list = target_tokens[model_name]["prompt_tokens"]
        completion_list = target_tokens[model_name]["completion_tokens"]
        calls = len(prompt_list)
        total_prompt = sum(prompt_list)
        total_completion = sum(completion_list)

        target_tokens[model_name]["summary"] = {
            "calls": calls,
            "total_prompt_tokens": total_prompt,
            "total_completion_tokens": total_completion,
            "avg_prompt_tokens": round(total_prompt / calls, 2) if calls else 0,
            "avg_completion_tokens": round(total_completion / calls, 2) if calls else 0,
        }

    def _extract_model_name(self, inst, args, kwargs, fn):
        try:
            sig = inspect.signature(fn)
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()
            return (
                bound.arguments.get("model") or
                bound.arguments.get("model_name") or
                getattr(bound.arguments.get("self", None), "model_name", None) or
                getattr(inst, "model_name", None)
            )
        except Exception as e:
            print(f"[ModelName Extract Fail] {e}")
            return getattr(inst, "model_name", None)

    def report(self, instance: Any = None, include_detail: bool = False) -> str:

        target_costs = getattr(instance, "costs", self.costs)
        target_tokens = getattr(instance, "token_logs", self.token_logs)

        report_data = []
        for model, token_info in target_tokens.items():
            summary = token_info.get("summary", {})
            total_cost = round(sum(target_costs.get(model, [])), 6)

            report_data.append([
                model,
                summary.get("calls", 0),
                summary.get("total_prompt_tokens", 0),
                summary.get("total_completion_tokens", 0),
                summary.get("avg_prompt_tokens", 0),
                summary.get("avg_completion_tokens", 0),
                total_cost
            ])

        table = tabulate(
            report_data,
            headers=["Model", "Calls", "Prompt (sum)", "Completion (sum)", "Prompt (avg)", "Completion (avg)", "Total Cost ($)"],
            tablefmt="pretty"
        )

        if include_detail:
            from pprint import pformat
            detail_str = "\n\n[Detailed Token Logs]\n" + pformat(target_tokens, indent=2, width=100)
            return table + detail_str

        return table
    
cost_tracker = CostTracker()
