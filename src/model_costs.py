import os
import requests
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

resp = requests.get(
    "https://chat.dartmouth.edu/api/models",
    headers={"Authorization": f"Bearer {os.environ['DARTMOUTH_API_KEY']}"},
)
models = resp.json()["data"]

rows = []
for m in models:
    info = m.get("upstream_model_info") or {}
    params = info.get("litellm_params") or {}
    minfo = info.get("model_info") or {}
    rows.append({
        "id": m.get("id"),
        "name": m.get("name"),
        "input_cost": minfo.get("input_cost_per_token", params.get("input_cost_per_token")),
        "output_cost": minfo.get("output_cost_per_token", params.get("output_cost_per_token")),
        "api_base": params.get("api_base"),
    })

df = pd.DataFrame(rows)
df = df.sort_values("output_cost", na_position="last")
df.to_csv("results/model_costs.csv", index=False)
print(df[["id", "input_cost", "output_cost"]].to_string(index=False))