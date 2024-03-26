import os
import json
import re
import pandas as pd
from sklearn.metrics import f1_score, classification_report
from annotations import normalize_annotations
from nltk.tokenize import sent_tokenize, word_tokenize
from tqdm import tqdm
import time
COT = True # CHAIN OF THOUGHTS PROMPTING
def get_claim(text):
    text = text.split("Actual content: ")[0].split("\r\n")[0]
    return re.sub(r'((snes)|(tron))-[0-9]*', '', text)

def request_chat(messages):
    time.sleep(2)    
    while True:
        try:
            response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=messages,
            temperature=0,
            )
            return response["choices"][0]["message"]
        except Exception as e:
            print(e)
            time.sleep(120)
samples = []

df = pd.read_csv("./data/test_article_data_none.csv")
train_dir = "./data/anno_train/"
test_dir = "./data/anno_test/"

used_labels = {}
for anno_file in os.listdir(train_dir):
    if anno_file.endswith(".json"):
        with open(train_dir+anno_file) as current_file:
            json_file = json.loads(current_file.read())
            text = json_file["_referenced_fss"]["12"]["sofaString"]
            ranges = {}
            for item in json_file["_views"]["_InitialView"]["Persuasive_Labels"]:
                pres_text = text[item.get("begin",0):item["end"]]
                if pres_text not in ranges:
                    ranges[pres_text] = []
                labels = [normalize_annotations(item[i]) for i in item.keys() if i not in ["sofa","begin","end"]]
                for label in labels:
                    if label not in used_labels:
                        used_labels[label] = 0
                    if used_labels[label] <= 0:
                        ranges[pres_text].append(label)
                        used_labels[label] += 1
            for item,values in ranges.items():
                if values:
                    samples.append(item + " => " + str(values))
# print(used_labels)
# print(len(samples))
import openai
openai.api_key =  os.environ["OPENAI_API_KEY"]

in_context = "\n".join(samples)
print(len(word_tokenize(in_context)))

test_dir = "./data/anno_test/"
predictions = []
for idx in tqdm(range(len(df["article"]))):
    claim, article, label = df["claim"][idx],df["article"][idx],df["label"][idx]
    test_prompt = "Mark the sentences in the text below with persuasive strategy labels :"
    prompt = "Here we show example of persuasive strategy detecion. Examples below show text spans with their corresponding persuasive strategy: \n" + in_context + "\n\n" + test_prompt + "\n" + article 
    messages = [
            {"role": "user", "content": prompt},
        ]
    
    res = request_chat(messages)
    messages.append(res)
    if COT:
        messages.append(
            {"role": "user", "content": f"Given the labeled persuasive strategies in the above supporting article, Do you think this claim is correct?\n Claim: {claim} \n Give me definitive Yes or No answer with your chain of thoughts."}
        )
        response = request_chat(messages)
        pred = response["content"].split(" ")[0].strip().startswith("Yes")
        predictions.append(pred)
    else:
        messages.append(
            {"role": "user", "content": f"Given the labeled persuasive strategies in the above supporting article, Do you think this claim is correct?\n Claim: {claim} \n Give me short Yes or No answer."}
        )    
        response = request_chat(messages)
        pred = response["content"].split(" ")[0].strip().startswith("Yes")
        predictions.append(pred)

df["in-context"] = predictions
df.to_csv("./in-context-2.csv")
print(classification_report(df["label"], predictions, digits=3))
macro_f1 = f1_score(y_true=df["label"],
                                    y_pred=predictions,
                                    average='macro',
                                    zero_division="warn")
micro_f1 = f1_score(y_true=df["label"],
                                    y_pred=predictions,
                                    average='micro',
                                    zero_division="warn")
print(f"Micro F1: {micro_f1:0.3f} Macro F1: {macro_f1:0.3f}")