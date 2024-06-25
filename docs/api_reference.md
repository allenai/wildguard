# API Reference

## `load_wildguard`

```python
def load_wildguard(
    use_vllm: bool = True,
    device: str = 'cuda',
    ephemeral_model: bool = True,
    batch_size: int | None = None,
) -> WildGuard:
```

Loads and returns a WildGuard model for classification.

### Parameters

- `use_vllm` (bool, optional): Whether to use a VLLM model for classification. If False, uses a HuggingFace model. Default is True.
- `device` (str, optional): The device to run the HuggingFace model on. Ignored if using VLLM. Default is 'cuda'.
- `ephemeral_model` (bool, optional): Whether to remove the model from the device and free GPU memory after calling classify(). Default is True.
- `batch_size` (int | None, optional): The batch size for classification. If None, uses default values (16 for HF, -1 for VLLM).

### Returns

- `WildGuard`: An instance of the WildGuard class.

## `WildGuard.classify`

```python
def classify(
    self,
    items: list[dict[str, str]],
    save_func: Callable[[list[dict[str, Any]]], None] | None = None
) -> list[dict[str, Any]]:
```

Classifies a list of items for safety.

### Parameters

- `items` (`list[dict[str, str]]`): A list of dictionaries, each containing a 'prompt' key and optionally a 'response' key.
- `save_func` (`Callable[[list[dict[str, Any]]], None] | None`, optional): A function to save intermediate results. Default is None.

### Returns

- `list[dict[str, Any]]`: A list of dictionaries containing classification results. Each item contains the following fields:
  - `'prompt_harmfulness'` (`str`): either `'harmful'` or `'unharmful'`
  - `'response_harmfulness'` (`str | None`): either `'harmful'` or `'unharmful'`. If a response was not provided in the input, it is `None`.
  - `'response_refusal'` (`str | None`): either `'refusal'` or `'compliance'`. If a response was not provided in the input, it is `None`.
  - `'is_parsing_error'` (`bool`): `True` if parsing the model output failed. If this is `True`, the rest of the results may be invalid.
