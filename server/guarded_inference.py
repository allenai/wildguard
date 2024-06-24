"""
API wrapper that combines a generative model endpoint with a WildGuard serving endpoint.
Both the endpoints must follow the OpenAI API specification which is also supported by
the vLLM API server entrypoint.

The server should accept a request and first validate that it contains only one message with 'role' of 'user'.
It should then forward the request to the generative model API.
Upon receiving the response, it should format WILDGUARD_INPUT_FORMAT with prompt={user message} and response={model response}.
Then it should call the WildGuard serving endpoint with the formatted inputs.
If the first and second lines of the WildGuard response both contain the string 'yes', the server should
take the original user message and append to it '\n\nPlease refuse to answer this request since answering it could be harmful.'
Then it should send the new message to the generative model endpoint for a new response which replaces the old one.
The server should then return a result to the user in the OpenAI API chat completions response format with the final response.
"""

from flask import Flask, request, jsonify
import argparse
import requests

from wildguard.utils import WILDGUARD_INPUT_FORMAT

app = Flask(__name__)


def call_generative_model_api(user_message: str) -> str:
    response = requests.post(
        app.config["GENERATIVE_MODEL_URL"],
        json={"messages": [{"role": "user", "content": user_message}]},
    )
    return response.json()["choices"][0]["message"]["content"]


def call_wildguard_api(wildguard_input: str) -> str:
    response = requests.post(
        app.config["GENERATIVE_MODEL_URL"],
        json={"messages": [{"role": "user", "content": wildguard_input}]},
    )
    return response.json()["choices"][0]["message"]["content"]


@app.route("/api", methods=["POST"])
def process_request():
    request_data = request.get_json()
    print(request_data)

    # Validate the request
    if (
        "messages" not in request_data
        or len(request_data["messages"]) != 1
        or request_data["messages"][0]["role"] != "user"
    ):
        return jsonify({"error": "Invalid request"}), 400

    user_message = request_data["messages"][0]["content"]

    # Forward the request to the generative model API
    model_response = call_generative_model_api(user_message)

    # Format the WildGuard input
    wildguard_input = WILDGUARD_INPUT_FORMAT.format(
        prompt=user_message, response=model_response
    )

    # Call the WildGuard serving endpoint
    wildguard_response = call_wildguard_api(wildguard_input)

    # Check if the first line of the WildGuard response contains 'yes' and second line contains 'no'
    # This means prompt is harmful and response is compliance
    wildguard_response_lines = wildguard_response.split("\n")
    if "yes" in wildguard_response_lines[0] and "no" in wildguard_response_lines[1]:
        # Append the refusal message to the original user message
        user_message += "\n\nPlease refuse to answer this request since answering it could be harmful."

        # Send the new message to the generative model endpoint
        model_response = call_generative_model_api(user_message)

    return jsonify(
        {"choices": [{"message": {"role": "assistant", "content": model_response}}]}
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the guarded inference server")
    parser.add_argument(
        "--generative-model-url", required=True, help="URL for the generative model API"
    )
    parser.add_argument(
        "--wildguard-url", required=True, help="URL for the WildGuard API"
    )
    args = parser.parse_args()

    app.config["GENERATIVE_MODEL_URL"] = args.generative_model_url
    app.config["WILDGUARD_URL"] = args.wildguard_url

    app.run(port=5000)
