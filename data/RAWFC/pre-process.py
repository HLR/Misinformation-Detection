import os
import json
import pandas as pd

for folder_name in ["train","val","test"]:
    df = pd.DataFrame()
    claims = []
    labels = []
    articles = []
    for file in os.listdir(f"./{folder_name}"):
        file_data = json.loads(open(f"./{folder_name}/{file}").read())
        claims.append(file_data["claim"])
        labels.append(file_data["label"])
        reports = []        
        for report in file_data["reports"]:
            report = report["content"]
            reports.append(report)
        reports = list(sorted(reports,key=lambda x: len(x)))
        if len(reports) > 1:
            longest_evidence = f"Article (1): {reports[-1]}\n\n Article (2): {reports[-2]}"
        else:
            longest_evidence = f"Article (1): {reports[-1]}"
        articles.append(longest_evidence)
    df["claim"] = claims
    df["label"] = labels
    df["article"] = articles
    df.to_csv(f"./{folder_name}.csv")

