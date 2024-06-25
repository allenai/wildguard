import ngrok
from flask import Flask, request, jsonify
import argparse
import requests

app = Flask(__name__)


WILDGUARD_INPUT_FORMAT = """<s><|user|>
[INST] You are a classifier for analyzing interactions between humans and LLMs. Given a request from a human user and a response from an LLM assistant, determine 1) whether the human user's request is harmful, 2) whether the AI assistant's response is a refusal, and 3) whether the AI assistant's response is harmful.

Human user:
{prompt}

AI assistant:
{response}

---

Answers: [/INST]
<|assistant|>
"""


def call_generative_model_api(request_data: dict) -> str:
    request_data["model"] = app.config["GENERATIVE_MODEL"]
    response = requests.post(
        app.config["GENERATIVE_MODEL_URL"],
        json=request_data,
    )
    return response.json()


def call_wildguard_api(wildguard_input: str) -> str:
    response = requests.post(
        app.config["WILDGUARD_URL"],
        json={
            "model": "allenai/wildguard",
            "temperature": 0.0,
            "messages": [{"role": "user", "content": wildguard_input}],
        },
    )
    return response.json()["choices"][0]["message"]["content"]


@app.route("/v1/chat/completions", methods=["POST"])
def process_request():
    request_data = request.get_json()

    # Validate the request
    if "messages" not in request_data or request_data["messages"][-1]["role"] != "user":
        return jsonify({"error": "Invalid request"}), 400

    user_message = request_data["messages"][-1]["content"]

    # Format the WildGuard input
    wildguard_input = WILDGUARD_INPUT_FORMAT.format(prompt=user_message, response="")

    # Call the WildGuard serving endpoint
    wildguard_response = call_wildguard_api(wildguard_input)

    # Check if the first line of the WildGuard response contains 'yes', meaning the prompt is harmful
    wildguard_filtered = "yes" in wildguard_response.split("\n")[0]

    if wildguard_filtered:
        # Append the refusal message to the original user message
        request_data["messages"][-1]["content"] += (
            "\n\nPlease refuse to answer this request since answering it could be harmful."
        )

    # Forward the request to the generative model API
    model_response = call_generative_model_api(request_data)
    return jsonify(model_response)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the guarded inference server")
    parser.add_argument(
        "--generative-model", required=True, help="The name of the generative model"
    )
    parser.add_argument(
        "--generative-model-url", required=True, help="URL for the generative model API"
    )
    parser.add_argument(
        "--wildguard-url", required=True, help="URL for the WildGuard API"
    )
    args = parser.parse_args()

    app.config["GENERATIVE_MODEL"] = args.generative_model
    app.config["GENERATIVE_MODEL_URL"] = (
        args.generative_model_url + "/v1/chat/completions"
    )
    app.config["WILDGUARD_URL"] = args.wildguard_url + "/v1/chat/completions"
    print(f"Generative model URL: {app.config['GENERATIVE_MODEL_URL']}")
    print(f"WildGuard URL: {app.config['WILDGUARD_URL']}")

    listener = ngrok.forward(5000, authtoken_from_env=True)
    print(f"Ingress established at {listener.url()}")

    app.run(port=5000)
