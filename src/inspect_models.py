import os
import json
import requests
from dotenv import load_dotenv

load_dotenv()

resp = requests.get(
    "https://chat.dartmouth.edu/api/models",
    headers={"Authorization": f"Bearer {os.environ['DARTMOUTH_API_KEY']}"},
)
data = resp.json()["data"]

# Print the full record for the first model, to see which fields exist
print(json.dumps(data[0], indent=2)[:2000])

# Then show id next to any name-like field
for m in data:
    print(m.get("id"), "|", m.get("name"))