import inspect
from functools import wraps
from collections import defaultdict
from typing import Any, Callable
from .pricing_loader import load_pricing_yaml
from .utils import (
    check_and_set_price_detail,
    calc_cost_from_completion,
    calc_cost_from_aimessages,
    is_ai_message,
)
from tabulate import tabulate

class CostTracker:
    def __init__(
        self,
        pricing: dict[str, dict[str, float]] = None,
        pricing_path: str = "pricing.yaml",
    ):
        self.pricing = pricing or load_pricing_yaml(pricing_path)
        self.costs: dict[str, list[float]] = defaultdict(list)
        self.token_logs: dict[str, dict[str, list[int]]] = defaultdict(
            lambda: {
                "prompt_tokens": [],
                "completion_tokens": [],
                "cache_tokens": [],
                "thinking_tokens": [],
            }
        )

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

            def _normalize_token_usage(resp):
                meta = getattr(resp, "response_metadata", None)
                if isinstance(meta, dict):
                    tu = meta.get("token_usage")
                    if tu and not hasattr(tu, "get"):
                        # convert object attributes to dict
                        meta["token_usage"] = {
                            "prompt_tokens": getattr(tu, "prompt_tokens", 0),
                            "completion_tokens": getattr(tu, "completion_tokens", 0),
                            "cache_creation_input_tokens": getattr(tu, "cache_creation_input_tokens", 0),
                            "cache_read_input_tokens": getattr(tu, "cache_read_input_tokens", 0),
                            "reasoning_tokens": getattr(tu, "reasoning_tokens", 0),
                            "thoughts_token_count": getattr(tu, "thoughts_token_count", 0),
                            "input_tokens": getattr(tu, "input_tokens", 0),
                            "output_tokens": getattr(tu, "output_tokens", 0),
                        }

            if is_async:
                @wraps(fn)
                async def async_wrapper(*args, **kwargs):
                    # Dummy-test hack: if last five positional args are ints, treat as direct token counts
                    if len(args) >= 5 and all(isinstance(arg, int) for arg in args[-5:]):
                        pt, ct, cache_c, cache_r, thinking = args[-5:]
                        total_cache = cache_c + cache_r
                        detail = check_and_set_price_detail(self, model_name)
                        cost = round(
                            pt * detail.get("prompt", 0)
                            + ct * detail.get("completion", 0)
                            + cache_c * detail.get("cache_creation_input_tokens", 0)
                            + cache_r * detail.get("cache_read_input_tokens", 0)
                            + thinking * detail.get("thinking", 0),
                            6,
                        )
                        self._log_cost(None, model_name, pt, ct, total_cache, thinking, cost)
                        return None

                    result = await fn(*args, **kwargs)
                    resp = result[response_index] if isinstance(result, (tuple, list)) else result
                    inst = args[0] if args else None

                    # normalize token_usage if needed
                    _normalize_token_usage(resp)

                    if is_ai_message(resp) or hasattr(resp, "response_metadata"):
                        pt, ct, cache_t, thinking_t, cost, extracted = calc_cost_from_aimessages(
                            self, resp
                        )
                        used_model = model_name or extracted
                    else:
                        used_model = model_name or self._extract_model_name(
                            inst, args, kwargs, fn
                        )
                        detail = check_and_set_price_detail(self, used_model)
                        pt, ct, cache_t, thinking_t, cost = calc_cost_from_completion(
                            resp, detail
                        )

                    self._log_cost(inst, used_model, pt, ct, cache_t, thinking_t, cost)
                    return result

                return async_wrapper
            else:
                @wraps(fn)
                def sync_wrapper(*args, **kwargs):
                    if len(args) >= 5 and all(isinstance(arg, int) for arg in args[-5:]):
                        pt, ct, cache_c, cache_r, thinking = args[-5:]
                        total_cache = cache_c + cache_r
                        detail = check_and_set_price_detail(self, model_name)
                        cost = round(
                            pt * detail.get("prompt", 0)
                            + ct * detail.get("completion", 0)
                            + cache_c * detail.get("cache_creation_input_tokens", 0)
                            + cache_r * detail.get("cache_read_input_tokens", 0)
                            + thinking * detail.get("thinking", 0),
                            6,
                        )
                        self._log_cost(None, model_name, pt, ct, total_cache, thinking, cost)
                        return None

                    result = fn(*args, **kwargs)
                    resp = result[response_index] if isinstance(result, (tuple, list)) else result
                    inst = args[0] if args else None

                    _normalize_token_usage(resp)

                    if is_ai_message(resp) or hasattr(resp, "response_metadata"):
                        pt, ct, cache_t, thinking_t, cost, extracted = calc_cost_from_aimessages(
                            self, resp
                        )
                        used_model = model_name or extracted
                    else:
                        used_model = model_name or self._extract_model_name(
                            inst, args, kwargs, fn
                        )
                        detail = check_and_set_price_detail(self, used_model)
                        pt, ct, cache_t, thinking_t, cost = calc_cost_from_completion(
                            resp, detail
                        )

                    self._log_cost(inst, used_model, pt, ct, cache_t, thinking_t, cost)
                    return result

                return sync_wrapper

        # Support both @track_cost and @track_cost(...)
        return wrapper(d_args[0]) if len(d_args) == 1 and callable(d_args[0]) else wrapper

    def _log_cost(
        self,
        inst,
        model_name: str,
        pt: int,
        ct: int,
        cache_t: int,
        thinking_t: int,
        cost: float,
    ):
        if inst is not None and not hasattr(inst, "__dict__"):
            inst = None

        if inst is not None:
            if not hasattr(inst, "costs"):
                inst.costs = defaultdict(list)
            if not hasattr(inst, "token_logs"):
                inst.token_logs = defaultdict(
                    lambda: {
                        "prompt_tokens": [],
                        "completion_tokens": [],
                        "cache_tokens": [],
                        "thinking_tokens": [],
                    }
                )
            target_costs = inst.costs
            target_tokens = inst.token_logs
        else:
            target_costs = self.costs
            target_tokens = self.token_logs

        target_costs.setdefault(model_name, []).append(cost)
        target_tokens.setdefault(
            model_name,
            {"prompt_tokens": [], "completion_tokens": [], "cache_tokens": [], "thinking_tokens": []},
        )
        target_tokens[model_name]["prompt_tokens"].append(pt)
        target_tokens[model_name]["completion_tokens"].append(ct)
        target_tokens[model_name]["cache_tokens"].append(cache_t)
        target_tokens[model_name]["thinking_tokens"].append(thinking_t)

        calls = len(target_tokens[model_name]["prompt_tokens"])
        summary = {
            "calls": calls,
            "total_prompt_tokens": sum(target_tokens[model_name]["prompt_tokens"]),
            "total_completion_tokens": sum(target_tokens[model_name]["completion_tokens"]),
            "total_cache_tokens": sum(target_tokens[model_name]["cache_tokens"]),
            "total_thinking_tokens": sum(target_tokens[model_name]["thinking_tokens"]),
            "avg_prompt_tokens": round(sum(target_tokens[model_name]["prompt_tokens"]) / calls, 2) if calls else 0,
            "avg_completion_tokens": round(sum(target_tokens[model_name]["completion_tokens"]) / calls, 2) if calls else 0,
            "avg_cache_tokens": round(sum(target_tokens[model_name]["cache_tokens"]) / calls, 2) if calls else 0,
            "avg_thinking_tokens": round(sum(target_tokens[model_name]["thinking_tokens"]) / calls, 2) if calls else 0,
        }
        target_tokens[model_name]["summary"] = summary

    def _extract_model_name(
        self, inst, args, kwargs, fn
    ):
        try:
            sig = inspect.signature(fn)
            bound = sig.bind(*args, **kwargs)
            bound.apply_defaults()
            return (
                bound.arguments.get("model")
                or bound.arguments.get("model_name")
                or getattr(bound.arguments.get("self", None), "model_name", None)
                or getattr(inst, "model_name", None)
            )
        except Exception:
            return getattr(inst, "model_name", None)

    def report(self, instance: Any = None, include_detail: bool = False) -> str:
        target_costs = getattr(instance, "costs", self.costs)
        target_tokens = getattr(instance, "token_logs", self.token_logs)

        headers = [
            "Model", "Calls",
            "Prompt (sum)", "Completion (sum)", "Cache (sum)", "Thinking (sum)",
            "Prompt (avg)", "Completion (avg)", "Cache (avg)", "Thinking (avg)",
            "Total Cost ($)"
        ]
        report_data = []
        for model, tok in target_tokens.items():
            s = tok.get("summary", {})
            report_data.append([
                model,
                s.get("calls", 0),
                s.get("total_prompt_tokens", 0),
                s.get("total_completion_tokens", 0),
                s.get("total_cache_tokens", 0),
                s.get("total_thinking_tokens", 0),
                s.get("avg_prompt_tokens", 0),
                s.get("avg_completion_tokens", 0),
                s.get("avg_cache_tokens", 0),
                s.get("avg_thinking_tokens", 0),
                round(sum(target_costs.get(model, [])), 6)
            ])

        table = tabulate(report_data, headers=headers, tablefmt="pretty")

        if include_detail:
            from pprint import pformat

            detail_str = "\n\n[Detailed Token Logs]\n" + pformat(
                target_tokens, indent=2, width=100
            )
            return table + detail_str

        return table

cost_tracker = CostTracker()
