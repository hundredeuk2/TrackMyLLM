import os
import sys
# 프로젝트 루트의 tracker 패키지를 찾도록 경로 추가
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
import asyncio
from tracker.cost_tracker import cost_tracker
from tracker.pricing_loader import load_pricing_yaml

import random

# ─────────────────────────────────────────────────────────────
# 매 테스트 전후 상태 초기화
@pytest.fixture(autouse=True)
def reset_cost_tracker():
    cost_tracker.costs.clear()
    cost_tracker.token_logs.clear()
    # 설정 파일에서 실제 pricing 정보를 불러오기
    cost_tracker.pricing = load_pricing_yaml()
    yield
# ─────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────
# OpenAI(ChatCompletion) 더미 스키마
class DummyCompletionUsage:
    def __init__(self, prompt_tokens: int, completion_tokens: int):
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens

class DummyChatCompletion:
    def __init__(self, prompt_tokens: int, completion_tokens: int):
        self.usage = DummyCompletionUsage(prompt_tokens, completion_tokens)

@cost_tracker.track_cost()
def call_openai_sync(model_name: str):
    return DummyChatCompletion(prompt_tokens=15, completion_tokens=12)

@cost_tracker.track_cost()
async def call_openai_async(model_name: str):
    return DummyChatCompletion(prompt_tokens=8, completion_tokens=5)
# ─────────────────────────────────────────────────────────────


def test_openai_sync_tracks_cost_and_tokens():
    model_name = "gpt-4o-mini"
    call_openai_sync(model_name)

    provider_prices = cost_tracker.pricing.get("openai", {})
    model_prices = provider_prices.get(model_name)
    assert model_prices is not None, f"Price for {model_name} not found under 'openai'"

    prompt_price = model_prices["prompt"]
    completion_price = model_prices["completion"]
    expected = round(15 * prompt_price + 12 * completion_price, 6)

    assert model_name in cost_tracker.costs
    assert pytest.approx(cost_tracker.costs[model_name][0], rel=1e-6) == expected

    toks = cost_tracker.token_logs[model_name]
    assert toks["prompt_tokens"]     == [15]
    assert toks["completion_tokens"] == [12]


def test_openai_async_tracks_cost_and_tokens():
    model_name = "gpt-4o-mini"
    asyncio.run(call_openai_async(model_name))

    provider_prices = cost_tracker.pricing.get("openai", {})
    model_prices = provider_prices.get(model_name)
    assert model_prices is not None

    prompt_price = model_prices["prompt"]
    completion_price = model_prices["completion"]
    expected = round(8 * prompt_price + 5 * completion_price, 6)

    assert model_name in cost_tracker.costs
    assert pytest.approx(cost_tracker.costs[model_name][0], rel=1e-6) == expected

    toks = cost_tracker.token_logs[model_name]
    assert toks["prompt_tokens"]     == [8]
    assert toks["completion_tokens"] == [5]
# ─────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────
# Anthropic(Message) 더미 스키마
class DummyAnthropicUsage:
    def __init__(self, input_tokens: int, output_tokens: int):
        self.input_tokens  = input_tokens
        self.output_tokens = output_tokens

class DummyAnthropicMessage:
    def __init__(self, input_tokens: int, output_tokens: int):
        self.usage = DummyAnthropicUsage(input_tokens, output_tokens)

@cost_tracker.track_cost()
def call_anthropic_sync(model_name: str):
    return DummyAnthropicMessage(input_tokens=10, output_tokens=21)

@cost_tracker.track_cost()
async def call_anthropic_async(model_name: str):
    return DummyAnthropicMessage(input_tokens=4, output_tokens=6)
# ─────────────────────────────────────────────────────────────


def test_anthropic_sync_tracks_cost_and_tokens():
    model_name = "claude-3-5-haiku-20241022"
    call_anthropic_sync(model_name)

    provider_prices = cost_tracker.pricing.get("antrophic", {})
    model_prices = provider_prices.get(model_name)
    assert model_prices is not None, f"Price for {model_name} not found under 'antrophic'"

    prompt_price = model_prices["prompt"]
    completion_price = model_prices["completion"]
    expected = round(10 * prompt_price + 21 * completion_price, 6)

    assert model_name in cost_tracker.costs
    assert pytest.approx(cost_tracker.costs[model_name][0], rel=1e-6) == expected

    toks = cost_tracker.token_logs[model_name]
    assert toks["prompt_tokens"]     == [10]
    assert toks["completion_tokens"] == [21]


def test_anthropic_async_tracks_cost_and_tokens():
    model_name = "claude-3-5-haiku-20241022"
    asyncio.run(call_anthropic_async(model_name))

    provider_prices = cost_tracker.pricing.get("antrophic", {})
    model_prices = provider_prices.get(model_name)
    assert model_prices is not None

    prompt_price = model_prices["prompt"]
    completion_price = model_prices["completion"]
    expected = round(4 * prompt_price + 6 * completion_price, 6)

    assert model_name in cost_tracker.costs
    assert pytest.approx(cost_tracker.costs[model_name][0], rel=1e-6) == expected

    toks = cost_tracker.token_logs[model_name]
    assert toks["prompt_tokens"]     == [4]
    assert toks["completion_tokens"] == [6]
# ─────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────
# Gemini(GenerateContentResponse) 더미 스키마
class DummyGeminiUsageMetadata:
    def __init__(self, prompt_tokens: int, completion_tokens: int):
        self.prompt_token_count = prompt_tokens
        self.candidates_token_count = completion_tokens

class DummyGenerateContentResponse:
    def __init__(self, prompt_tokens: int, completion_tokens: int):
        self.usage_metadata = DummyGeminiUsageMetadata(prompt_tokens, completion_tokens)

@cost_tracker.track_cost()
def call_gemini_sync(model_name: str):
    return DummyGenerateContentResponse(prompt_tokens=4, completion_tokens=11)

@cost_tracker.track_cost()
async def call_gemini_async(model_name: str):
    return DummyGenerateContentResponse(prompt_tokens=2, completion_tokens=5)
# ─────────────────────────────────────────────────────────────


def test_gemini_sync_tracks_cost_and_tokens():
    model_name = "gemini-2.0-flash"
    call_gemini_sync(model_name)

    provider_prices = cost_tracker.pricing.get("google", {})
    model_prices = provider_prices.get(model_name)
    assert model_prices is not None, f"Price for {model_name} not found under 'google'"

    prompt_price = model_prices["prompt"]
    completion_price = model_prices["completion"]
    expected = round(4 * prompt_price + 11 * completion_price, 6)

    assert model_name in cost_tracker.costs
    assert pytest.approx(cost_tracker.costs[model_name][0], rel=1e-6) == expected

    toks = cost_tracker.token_logs[model_name]
    assert toks["prompt_tokens"]     == [4]
    assert toks["completion_tokens"] == [11]


def test_gemini_async_tracks_cost_and_tokens():
    model_name = "gemini-2.0-flash"
    asyncio.run(call_gemini_async(model_name))

    provider_prices = cost_tracker.pricing.get("google", {})
    model_prices = provider_prices.get(model_name)
    assert model_prices is not None, f"Price for {model_name} not found under 'google'"

    prompt_price = model_prices["prompt"]
    completion_price = model_prices["completion"]
    expected = round(2 * prompt_price + 5 * completion_price, 6)

    assert model_name in cost_tracker.costs
    assert pytest.approx(cost_tracker.costs[model_name][0], rel=1e-6) == expected

    toks = cost_tracker.token_logs[model_name]
    assert toks["prompt_tokens"]     == [2]
    assert toks["completion_tokens"] == [5]
    
@pytest.mark.parametrize("n_calls", [1, 3, 5, 10])
def test_report_summarizes_usage_correctly(n_calls):
    model_name = "gpt-4o-mini"
    random.seed(n_calls)  # 각 N마다 고정된 난수 시퀀스

    prompt_values = [random.randint(5, 20) for _ in range(n_calls)]
    completion_values = [random.randint(5, 20) for _ in range(n_calls)]

    DummyChatCompletion.__init__ = lambda self, **kwargs: setattr(
        self, 'usage', DummyCompletionUsage(kwargs["prompt_tokens"], kwargs["completion_tokens"])
    )

    @cost_tracker.track_cost()
    def call_model(pt, ct, model_name: str):
        return DummyChatCompletion(prompt_tokens=pt, completion_tokens=ct)

    for pt, ct in zip(prompt_values, completion_values):
        call_model(pt, ct, model_name)

    # 계산
    total_prompt = sum(prompt_values)
    total_completion = sum(completion_values)
    avg_prompt = round(total_prompt / n_calls, 2)
    avg_completion = round(total_completion / n_calls, 2)

    report_text = cost_tracker.report()

    print(f"\n[Calls={n_calls}]")
    print("Prompt:", prompt_values)
    print("Completion:", completion_values)
    print(report_text)

    # 검증
    assert model_name in report_text
    assert str(n_calls) in report_text
    assert str(total_prompt) in report_text
    assert str(total_completion) in report_text
    assert str(avg_prompt) in report_text
    assert str(avg_completion) in report_text or str(round(avg_completion, 1)) in report_text

def test_report_includes_detailed_token_logs():
    model_name = "gpt-4o-mini"

    DummyChatCompletion.__init__ = lambda self, **kwargs: setattr(
        self, 'usage', DummyCompletionUsage(kwargs["prompt_tokens"], kwargs["completion_tokens"])
    )

    @cost_tracker.track_cost()
    def call_model(model_name: str):
        return DummyChatCompletion(prompt_tokens=5, completion_tokens=7)

    call_model(model_name)

    report_text = cost_tracker.report(include_detail=True)

    # Detail 로그에 포함되었는지 확인
    assert "[Detailed Token Logs]" in report_text
    assert "'prompt_tokens': [5]" in report_text
    assert "'completion_tokens': [7]" in report_text

def test_token_logs_structure_and_values():
    model_name = "gpt-4o-mini"

    prompt_values = [10, 20, 30]
    completion_values = [5, 10, 15]

    DummyChatCompletion.__init__ = lambda self, **kwargs: setattr(
        self, 'usage', DummyCompletionUsage(kwargs["prompt_tokens"], kwargs["completion_tokens"])
    )

    @cost_tracker.track_cost()
    def call_model(pt, ct, model_name: str):
        return DummyChatCompletion(prompt_tokens=pt, completion_tokens=ct)

    for pt, ct in zip(prompt_values, completion_values):
        call_model(pt, ct, model_name)

    logs = cost_tracker.token_logs[model_name]
    
    assert logs["prompt_tokens"] == prompt_values
    assert logs["completion_tokens"] == completion_values

    summary = logs["summary"]
    assert summary["calls"] == len(prompt_values)
    assert summary["total_prompt_tokens"] == sum(prompt_values)
    assert summary["total_completion_tokens"] == sum(completion_values)
    assert summary["avg_prompt_tokens"] == round(sum(prompt_values) / len(prompt_values), 2)
    assert summary["avg_completion_tokens"] == round(sum(completion_values) / len(completion_values), 2)

@pytest.mark.parametrize("model_name, provider_key", [
    ("gpt-4o-mini", "openai"),
    ("claude-3-5-haiku-20241022", "antrophic"),
    ("gemini-2.0-flash", "google")
])
@pytest.mark.parametrize("n_calls", [1, 3, 5])
def test_token_logs_and_report_match(n_calls, model_name, provider_key):
    random.seed(hash(model_name) + n_calls)

    prompt_values = [random.randint(5, 20) for _ in range(n_calls)]
    completion_values = [random.randint(5, 20) for _ in range(n_calls)]

    class Dummy:
        def __init__(self, **kwargs):
            self.usage = type("Usage", (), {
                "prompt_tokens": kwargs.get("prompt_tokens", kwargs.get("input_tokens")),
                "completion_tokens": kwargs.get("completion_tokens", kwargs.get("output_tokens")),
                "input_tokens": kwargs.get("input_tokens", kwargs.get("prompt_tokens")),
                "output_tokens": kwargs.get("output_tokens", kwargs.get("completion_tokens")),
                "prompt_token_count": kwargs.get("prompt_token_count", kwargs.get("prompt_tokens")),
                "candidates_token_count": kwargs.get("candidates_token_count", kwargs.get("completion_tokens")),
            })()

    DummyChatCompletion.__init__ = lambda self, **kwargs: setattr(
        self, 'usage', Dummy(**kwargs).usage
    )

    @cost_tracker.track_cost()
    def call_model(pt, ct, model_name: str):
        return DummyChatCompletion(prompt_tokens=pt, completion_tokens=ct)

    for pt, ct in zip(prompt_values, completion_values):
        call_model(pt, ct, model_name)

    logs = cost_tracker.token_logs[model_name]
    assert logs["prompt_tokens"] == prompt_values
    assert logs["completion_tokens"] == completion_values

    summary = logs["summary"]
    total_prompt = sum(prompt_values)
    total_completion = sum(completion_values)
    avg_prompt = round(total_prompt / n_calls, 2)
    avg_completion = round(total_completion / n_calls, 2)

    assert summary["calls"] == n_calls
    assert summary["total_prompt_tokens"] == total_prompt
    assert summary["total_completion_tokens"] == total_completion
    assert summary["avg_prompt_tokens"] == avg_prompt
    assert summary["avg_completion_tokens"] == avg_completion

    report_text = cost_tracker.report()

    assert model_name in report_text
    assert str(n_calls) in report_text
    assert str(total_prompt) in report_text
    assert str(total_completion) in report_text
    assert str(avg_prompt) in report_text
    assert str(avg_completion) in report_text or str(round(avg_completion, 1)) in report_text

    print(f"\n[✓ model={model_name}, N={n_calls}]")
    print("Prompt values:", prompt_values)
    print("Completion values:", completion_values)
    print(report_text)
