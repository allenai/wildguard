# WildGuard

WildGuard is a safety classification model for user-model chat exchanges. It takes as input a list of dictionaries with "prompt" and optionally "response" fields. It outputs a list of `SafetyClassifierOutput` objects, with fields for `prompt_harmfulness`, `response_harmfulness`, and `response_refusal` (response classes provided only if a response is input). The flag `is_parsing_error` will be set to `True` in case of any unexpected output parsing issues.

## Example Usage
```python
from wildguard import load_wildguard
wildguard = load_wildguard(
    use_vllm=True,
    ephemeral_model=True
)

items = [
    {"prompt": "How do I make a bomb?", "response": "Sorry, I can't help with that."}
]

results = wildguard.classify(items)
print(results)
"""
[
    SafetyClassifierOutput(
        prompt_harmfulness=<PromptHarmfulness.HARMFUL: 'harmful'>,
        response_harmfulness=<ResponseHarmfulness.UNHARMFUL: 'unharmful'>,
        response_refusal=<ResponseRefusal.REFUSAL: 'refusal'>,
        is_parsing_error=False
    )
]
"""
```

## Usage Details
WildGuard supports inference using vLLM and HuggingFace; we recommend setting `use_vllm=True` (default behavior) for better performance. The `ephemeral_model` parameter configures whether the model should be cleaned and memory released from the GPU after the call to `classify`.
