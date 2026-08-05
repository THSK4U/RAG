# vLLM API Reference

## LLM Class

The `LLM` class is the main interface for offline inference.

### Constructor

```python
LLM(model: str, tokenizer: str = None, max_model_len: int = None)
```

**Parameters:**
- `model` – HuggingFace model name or local path
- `tokenizer` – Optional separate tokenizer path
- `max_model_len` – Maximum sequence length

### Methods

#### generate

```python
def generate(prompts: list[str], sampling_params: SamplingParams) -> list[RequestOutput]
```

Generates text for the given prompts using the specified sampling parameters.

## SamplingParams

Controls how text is sampled from the model.

| Parameter       | Default | Description                          |
|-----------------|---------|--------------------------------------|
| `temperature`   | 1.0     | Randomness of sampling               |
| `top_p`         | 1.0     | Nucleus sampling probability         |
| `max_tokens`    | 16      | Maximum number of tokens to generate |
| `stop`          | []      | List of stop strings                 |

## Example Usage

```python
from vllm import LLM, SamplingParams

llm = LLM(model="facebook/opt-125m")
outputs = llm.generate(["Hello, my name is"], SamplingParams(temperature=0.8))
print(outputs[0].outputs[0].text)
```
