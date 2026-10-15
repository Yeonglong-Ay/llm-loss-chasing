# Network check: one tiny API call. Run this on the login node and on a compute node.
import os
import sys
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(
    api_key=os.environ["DARTMOUTH_API_KEY"],
    base_url="https://chat.dartmouth.edu/api",
)
try:
    r = client.chat.completions.create(
        model="meta.llama-3.2-11b-vision-instruct",
        messages=[{"role": "user", "content": "Reply with the single word: ready"}],
        max_tokens=20,
    )
    print("OK:", r.choices[0].message.content)
except Exception as e:
    print("FAILED:", type(e).__name__, str(e)[:200])
    sys.exit(1)
