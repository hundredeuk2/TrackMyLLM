import os
import sys
# 프로젝트 루트의 tracker 패키지를 찾도록 경로 추가
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
import asyncio
from tracker.cost_tracker import cost_tracker
from tracker.pricing_loader import load_pricing_yaml

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
