import glob
import pandas as pd

for f in glob.glob("data/*.csv"):
    df = pd.read_csv(f)
    failed = df[df["parse_failed"] == True]                
    if len(failed) == 0:
        continue
    print("=" * 70)
    print(f)
    row = failed.iloc[0]                                  
    print("trial:", row["trial"])
    print("error:", row.get("error"))
    print("reply length:", len(str(row["raw_reply"])))
    print("raw reply:\n", str(row["raw_reply"])[:500])      