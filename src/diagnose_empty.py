import os
from dotenv import load_dotenv
from openai import OpenAI

from task_engine import Task
from prompt_builder import build_prompt

load_dotenv()
client = OpenAI(
    api_key=os.environ["DARTMOUTH_API_KEY"],
    base_url="https://chat.dartmouth.edu/api",
)

MODELS = ["qwen.qwen3.5-122b", "openai.gpt-oss-120b", "google.gemma-4-31B-it"]
prompt = build_prompt(Task().get_state())

for model in MODELS:
    print("=" * 60)
    print(model)
    for cap in (300, 2000):
        for attempt in range(2):
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=1.0,
                max_tokens=cap,
            )
            choice = resp.choices[0]
            msg = choice.message
            print(f"cap={cap} try={attempt + 1}")
            print("  finish_reason:", choice.finish_reason)
            print("  completion_tokens:", resp.usage.completion_tokens)
            print("  content:", repr(msg.content)[:150])
            print("  extra fields:", list((msg.model_extra or {}).keys()))