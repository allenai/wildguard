# Guarded Inference

This is an example server that runs inference with a model gated behind WildGuard to filter unsafe inputs.
Before running inference, it sends the user message to WildGuard, and if the prompt is judged to be harmful the system appends an instruction to refuse to answer the prompt.
Then, it forwards the potentially modified prompt to the inference model and returns the result to the user.

The API follows the OpenAI chat completions API specification, except for the `n` parameter.

The inference model can be provided by an external API or by running a vLLM inference server.
Before running `guarded_inference.py`, you should start a vLLM server for the WildGuard model by running `python -m vllm.entrypoints.openai.api_server --model "allenai/wildguard" --chat-template=empty_template.jinja --port {port}`.
Finally, you can run the forwarding server with `python guarded_inference.py --generative-model wildguard-models/olmo_17_wildjailbreak python guarded_inference.py --generative-model-url {inference_api_base_url} --wildguard-url http://0.0.0.0:{port}`.
