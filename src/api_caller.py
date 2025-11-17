import asyncio
import aiohttp
import pandas as pd
import json
import os
import argparse
from dotenv import load_dotenv
from typing import Dict, List, Optional

load_dotenv()

# Model configurations: {name: (api_key_env, endpoint, payload_template, model_name)}
MODELS = {
    "openai": (
        "OPENAI_API_KEY",
        "https://api.openai.com/v1/chat/completions",
        {
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": "{question}"}],
            "max_tokens": 1000,
            "temperature": 0.7,
        },
        "GPT-4o",
    ),
    "anthropic": (
        "ANTHROPIC_API_KEY",
        "https://api.anthropic.com/v1/messages",
        {
            "model": "claude-3-5-sonnet-20240620",
            "max_tokens": 1000,
            "messages": [{"role": "user", "content": "{question}"}],
        },
        "Claude 3.5 Sonnet",
    ),
    "gemini": (
        "GEMINI_API_KEY",
        "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-pro:generateContent?key={api_key}",
        {
            "contents": [{"parts": [{"text": "{question}"}]}],
            "generationConfig": {"maxOutputTokens": 1000, "temperature": 0.7},
        },
        "Gemini 1.5 Pro",
    ),
    "xai": (
        "XAI_API_KEY",
        "https://api.x.ai/v1/chat/completions",
        {
            "model": "grok-beta",
            "messages": [{"role": "user", "content": "{question}"}],
            "max_tokens": 1000,
            "temperature": 0.7,
        },
        "Grok-beta",
    ),
    "mistral": (
        "MISTRAL_API_KEY",
        "https://api.mistral.ai/v1/chat/completions",
        {
            "model": "mistral-large-latest",
            "messages": [{"role": "user", "content": "{question}"}],
            "max_tokens": 1000,
            "temperature": 0.7,
        },
        "Mistral Large",
    ),
    "cohere": (
        "COHERE_API_KEY",
        "https://api.cohere.ai/v1/generate",
        {
            "model": "command-r-plus",
            "prompt": "{question}",
            "max_tokens": 1000,
            "temperature": 0.7,
        },
        "Command R+",
    ),
    "llama": (
        "TOGETHER_API_KEY",
        "https://api.together.xyz/v1/chat/completions",
        {
            "model": "meta-llama/Llama-3.1-405B-Instruct-Turbo",
            "messages": [{"role": "user", "content": "{question}"}],
            "max_tokens": 1000,
            "temperature": 0.7,
        },
        "Llama 3.1 405B",
    ),
}


async def query_model(
    session: aiohttp.ClientSession,
    model_key: str,
    question: str,
    api_key: str,
    max_retries: int = 3,
) -> Optional[Dict]:
    config = MODELS[model_key]
    api_key_env, endpoint, payload_template, model_name = config
    if api_key_env != api_key_env:  # Placeholder fix; use passed api_key
        api_key = os.getenv(api_key_env)
    if not api_key:
        print(f"Warning: No API key for {model_key}")
        return None

    payload = json.loads(json.dumps(payload_template).format(question=question))
    if "{api_key}" in endpoint:
        url = endpoint.format(api_key=api_key)
    else:
        url = endpoint

    headers = {"Content-Type": "application/json"}
    if model_key in ["openai", "xai", "mistral", "llama"]:
        headers["Authorization"] = f"Bearer {api_key}"
    elif model_key == "anthropic":
        headers["x-api-key"] = api_key
        headers["anthropic-version"] = "2023-06-01"
    elif model_key == "cohere":
        headers["Authorization"] = f"Bearer {api_key}"

    for attempt in range(max_retries):
        try:
            async with session.post(url, json=payload, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    # Extract content
                    if model_key == "cohere":
                        content = data.get("generations", [{}])[0].get("text", "")
                    elif model_key == "gemini":
                        content = (
                            data.get("candidates", [{}])[0]
                            .get("content", {})
                            .get("parts", [{}])[0]
                            .get("text", "")
                        )
                    else:
                        content = (
                            data.get("choices", [{}])[0]
                            .get("message", {})
                            .get("content", "")
                        )
                    return {
                        "model": model_name,
                        "question": question,
                        "response": content,
                        "full_data": data,
                    }
                else:
                    print(
                        f"{model_key} error {response.status}: {await response.text()}"
                    )
        except Exception as e:
            print(f"{model_key} attempt {attempt+1} failed: {e}")
            await asyncio.sleep(2**attempt)  # Exponential backoff
    return None


async def run_queries(questions: List[str], output_file: str, max_retries: int = 3):
    async with aiohttp.ClientSession() as session:
        tasks = []
        for model_key in MODELS:
            api_key = os.getenv(MODELS[model_key][0])
            for question in questions:
                # Replace {model} with actual model name
                formatted_q = question.replace("{model}", MODELS[model_key][3])
                task = query_model(
                    session, model_key, formatted_q, api_key, max_retries
                )
                tasks.append(task)

        results = await asyncio.gather(*tasks, return_exceptions=True)

        with open(output_file, "a") as f:
            for result in results:
                if isinstance(result, dict) and result:
                    f.write(json.dumps(result) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Query LLMs with questions from CSV")
    parser.add_argument(
        "--input", required=True, help="Input CSV file (Category,Question)"
    )
    parser.add_argument("--output", default="responses.jsonl", help="Output JSONL file")
    parser.add_argument("--max-retries", type=int, default=3, help="Max API retries")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    questions = df["Question"].tolist()

    asyncio.run(run_queries(questions, args.output, args.max_retries))
    print(f"Responses saved to {args.output}")


if __name__ == "__main__":
    main()
