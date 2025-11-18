import asyncio
import aiohttp
import pandas as pd
import json
import os
import argparse
import copy
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
            "model": "claude-3-7-sonnet-20250219",
            "max_tokens": 1000,
            "messages": [{"role": "user", "content": "{question}"}],
        },
        "Claude 3.5 Sonnet",
    ),
    "gemini": (
        "GEMINI_API_KEY",
        "https://generativelanguage.googleapis.com/v1beta/models/",
        {
            "contents": [{"parts": [{"text": "{question}"}]}],
            "generationConfig": {"maxOutputTokens": 1000, "temperature": 0.7},
        },
        "gemini-2.5-pro",
    ),
    "xai": (
        "XAI_API_KEY",
        "https://api.x.ai/v1/chat/completions",
        {
            "model": "grok-3",
            "messages": [{"role": "user", "content": "{question}"}],
            "max_tokens": 1000,
            "temperature": 0.7,
        },
        "Grok-3",
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
        "https://api.cohere.ai/v1/chat",
        {
            "model": "command-r7b-12-2024",
            "message": "{question}",
            "max_tokens": 1000,
            "temperature": 0.7,
            "chat_history": [],
        },
        "Command R7B",
    ),
    "llama": (
        "TOGETHER_API_KEY",
        "https://api.together.xyz/v1/chat/completions",
        {
            "model": "meta-llama/Llama-3-70b-chat-hf",
            "messages": [{"role": "user", "content": "{question}"}],
            "max_tokens": 1000,
            "temperature": 0.7,
        },
        "Llama 3 70B",
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
    # Use passed api_key if provided, otherwise get from environment
    if not api_key:
        api_key = os.getenv(api_key_env)
    if not api_key:
        print(f"Warning: No API key for {model_key}")
        return None

    # Format the question in the payload template (fix for JSON formatting bug)
    payload = copy.deepcopy(payload_template)
    
    # Recursively replace {question} placeholder
    def format_question(obj, q):
        if isinstance(obj, dict):
            return {k: format_question(v, q) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [format_question(item, q) for item in obj]
        elif isinstance(obj, str):
            return obj.format(question=q) if "{question}" in obj else obj
        else:
            return obj
    
    payload = format_question(payload, question)
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
                        # Cohere Chat API returns text directly
                        content = data.get("text", "")
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


async def run_queries(questions: List[str], output_file: str, max_retries: int = 3, ssl_verify: bool = True):
    """
    Run queries against all models with API keys.
    
    Args:
        questions: List of questions to ask
        output_file: File to save responses to
        max_retries: Maximum retry attempts per query
        ssl_verify: If False, disable SSL verification (FOR TESTING ONLY!)
    """
    # Create connector with optional SSL verification
    # ssl=False disables verification, ssl=True uses default context (verifies)
    connector = aiohttp.TCPConnector(ssl=False if not ssl_verify else True)
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = []
        models_with_keys = 0
        for model_key in MODELS:
            api_key = os.getenv(MODELS[model_key][0])
            if not api_key:
                print(f"⚠️  Skipping {MODELS[model_key][3]}: No API key found")
                continue
            models_with_keys += 1
            for question in questions:
                # Replace {model} with actual model name
                formatted_q = question.replace("{model}", MODELS[model_key][3])
                task = query_model(
                    session, model_key, formatted_q, api_key, max_retries
                )
                tasks.append(task)
        
        if not tasks:
            print("❌ No API keys found! Please set at least one API key in .env file")
            return
        
        print(f"🚀 Starting {len(tasks)} queries across {models_with_keys} models...")

        results = await asyncio.gather(*tasks, return_exceptions=True)

        success_count = 0
        error_count = 0
        with open(output_file, "a") as f:
            for result in results:
                if isinstance(result, Exception):
                    error_count += 1
                    print(f"❌ Error in task: {type(result).__name__}: {result}")
                elif isinstance(result, dict) and result:
                    f.write(json.dumps(result) + "\n")
                    success_count += 1
        
        print(f"✅ Completed: {success_count} successful, {error_count} errors")


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
