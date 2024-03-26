import os
import json
import pandas as pd
from sklearn.metrics import f1_score, classification_report
from tqdm import tqdm
import time
import openai
openai.api_key = os.environ["OPENAI_API_KEY"]

def request_chat(messages):
    time.sleep(10)    
    while True:
        try:
            response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=messages,
            temperature=0,
            top_p=0.0,
            )
            return response["choices"][0]["message"]
        except Exception as e:
            print(e)
            print("Error")
            time.sleep(120)
samples = []




in_context = "\n".join(samples)


merged = pd.read_csv("./data/RAWFC/test.csv")

merged[merged["label"] == "half"] = "half-true"

all_labels = [f'\"{x}\"' for x in list(set(merged["label"].to_list()))]

predictions = []
responses = []

for idx in tqdm(range(len(merged["claim"]))):
    claim, article, label = merged["claim"][idx],merged["article"][idx],merged["label"][idx]
    messages = [
        {"role": "user", "content": f"Article: {article} \n\n Claim: {claim} \n. Which of the {', '.join(all_labels)} as possible veracity labels, best describes the veracity of this claim. You must choose one."}
    ]
    response = request_chat(messages)
    pred = response["content"]
    if pred.lower().strip().strip("\"").strip("'").startswith("true") :
        predictions.append("true")
    elif pred.lower().strip().strip("\"").strip("'").startswith("false"):
        predictions.append("false")
    elif pred.lower().strip().strip("\"").strip("'").startswith("half-true"):
        predictions.append("half-true")
    else:
        predictions.append(pred)
    responses.append(pred)
merged["in-context"] = predictions
merged["responses"] = responses
merged.to_csv("./data/rawfc-test-results.csv")


labels = merged["label"].to_list()  
predictions = [x if x in ["true","false"] else "half-true" for x in merged["in-context"]]
print(predictions)

print(classification_report(labels,predictions , digits=3))

macro_f1 = f1_score(y_true=labels,
                                    y_pred=predictions,
                                    average='macro',
                                    zero_division="warn")
micro_f1 = f1_score(y_true=labels,
                                    y_pred=predictions,
                                    average='micro',
                                    zero_division="warn")
print(f"Micro F1: {micro_f1:0.3f} Macro F1: {macro_f1:0.3f}")