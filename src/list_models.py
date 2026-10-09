import os
import requests
from dotenv import load_dotenv
from pprint import pprint

load_dotenv()

resp = requests.get(
    "https://chat.dartmouth.edu/api/models",
    headers={"Authorization": f"Bearer {os.environ['DARTMOUTH_API_KEY']}"},
)
pprint([m["id"] for m in resp.json()["data"]])