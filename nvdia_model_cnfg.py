import os
import sys
from dotenv import load_dotenv
from openai import OpenAI

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

load_dotenv()

api_key = os.getenv("NVIDIA_API_KEY") or os.getenv("NEMOTRON_API_KEY")

if not api_key:
    raise ValueError("NVIDIA_API_KEY or NEMOTRON_API_KEY not found in environment variables.")

client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=api_key
)

print(f"Testing nvidia/nemotron-3-ultra-550b-a55b with API Key...")

completion = client.chat.completions.create(
    model="nvidia/nemotron-3-ultra-550b-a55b",
    messages=[{"role": "user", "content": "Hello! What model are you and what are your capabilities?"}],
    temperature=0.7,
    top_p=0.95,
    max_tokens=1024,
    stream=True
)

for chunk in completion:
    if not chunk.choices:
        continue
    reasoning = getattr(chunk.choices[0].delta, "reasoning_content", None)
    if reasoning:
        print(reasoning, end="")
    if chunk.choices[0].delta.content is not None:
        print(chunk.choices[0].delta.content, end="")
print()