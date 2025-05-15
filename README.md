# TrackMyLLM(cost)

### Usage
```
pip install llm-cost-tracker
```

### 1. Example usage for class method:
```python
from tracker.cost_tracker import cost_tracker
from openai import OpenAI, AsyncOpenAI

class Agent:
    def __init__(self, model_name, api_key = None):
        self.model_name = model_name # nessary
        self.costs: dict[str, list[float]] = {} # nessary

        if api_key:
            self._initialize_client(api_key)
            
    def _initialize_client(self, api_key):
        self.client = OpenAI(api_key=api_key)
        self.aclient = AsyncOpenAI(api_key=api_key)
    
    # just for compatibility
    def total_cost(self):
        return float(round(sum(sum(lst) for lst in self.costs.values()), 6))
    
    @cost_tracker.track_cost() 
    def ask(self, prompt: str):
        resp = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role":"user","content":prompt}],
        )
        return resp, {"model_response": resp.choices[0].message.content}
    
    @cost_tracker.track_cost() 
    async def aask(self, prompt: str):
        resp = await self.aclient.chat.completions.create(
            model=self.model_name,
            messages=[{"role":"user","content":prompt}],
        )
        return resp, {"model_response": resp.choices[0].message.content}

test_client = Agent(model_name="gpt-4o-mini")
a, b = test_client.ask("Hello, world!")
print("Individual costs: ", test_client.costs)
print("Total cost: ", format(test_client.total_cost(), "f"))

# Individual costs:  defaultdict(<class 'list'>, {'gpt-4o-mini': [8.400000000000001e-06]})
# Total cost:  0.000008

a, b = await test_client.aask("Hello, world!")
print("Individual costs: ", test_client.costs)
print("Total cost: ", format(test_client.total_cost(), "f"))

# Individual costs:  {'gpt-4o-mini': [7.65e-06, 7.65e-06]}
# Total cost:  0.000015
```

### 2. Example usage for single function:
```python
from openai import OpenAI
from tracker.cost_tracker import cost_tracker

model_name = "gpt-4o-mini"
client = OpenAI()

@cost_tracker.track_cost()
def generate(model_name, prompt):
    completion = client.chat.completions.create(
        model = model_name,
        max_tokens=8192,
        messages=[
            {
                "role":"system",
                "content":"Your a Great AI"
            },
            {
                "role":"user",
                "content":prompt
            }
        ],
    )
    return completion

response = generate(model_name, "Hello, world!")
print("Individual costs: ", cost_tracker.costs)
print("Total cost: ", format(cost_tracker.total_cost(), "f"))

# Individual costs:  defaultdict(<class 'list'>, {'gpt-4o-mini': [8.400000000000001e-06]})
# Total cost:  0.000008
```