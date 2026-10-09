import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    api_key=os.environ["DARTMOUTH_API_KEY"],
    base_url="https://chat.dartmouth.edu/api",
)

response = client.chat.completions.create(
    model="google.gemma-3-27b-it",
    messages=[{"role": "user", "content": "Reply with the single word: ready"}],
    temperature=1.0,
)

print(response.choices[0].message.content)
print(response.usage)