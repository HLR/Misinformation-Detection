import os
import json
import re
import pandas as pd
from sklearn.metrics import f1_score, classification_report
from annotations import normalize_annotations
from nltk.tokenize import sent_tokenize, word_tokenize
from tqdm import tqdm
import time
import re

# Extract only characters from a string
def extract_characters(string):
  return re.sub('[^\w]', '', string)

def truncate_text(text, max_tokens=16000):
    """
    Truncate the text to a specified number of tokens while trying 
    to preserve whole sentences.
    
    Parameters:
        text (str): The text to be truncated.
        max_tokens (int): The maximum number of tokens allowed.
        
    Returns:
        str: The truncated text.
    """
    # Tokenize the text into words while preserving their original indices
    words_with_indices = [(m.group(0), m.start()) for m in re.finditer(r'\S+|\n', text)]
    
    # Ensure text is not already below the max token count
    if len(words_with_indices) <= max_tokens:
        return text
    
    # Truncate words and recombine into a string, trying to preserve whole sentences
    truncated_text = text[:words_with_indices[max_tokens-1][1]]
    
    # Find the last full stop that occurs in the text and truncate there
    last_full_stop = truncated_text.rfind('.')
    
    # If a full stop was found, truncate text at that point
    if last_full_stop != -1:
        return truncated_text[:last_full_stop+1]
    else:
        # Otherwise, return the truncated text as is
        return truncated_text


def request_completion(prompt):
    while True:
        try:
            response = openai.Completion.create(
                engine="text-davinci-003",  # choose an appropriate engine
                prompt=prompt,
                max_tokens=100,  # adjust as needed
                temperature=0,
                top_p=1,
                n=1
            )
            return response.choices[0].text.strip()
        except Exception as e:
            print(e)
            print("Error")
            time.sleep(120)

samples = []

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
                pres_text = text[item["begin"]:item["end"]]
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

import openai
openai.api_key = os.environ["OPENAI_API_KEY"]

in_context = "\n".join(samples)

merged = pd.read_csv("./data/RAWFC/test.csv")
result = pd.read_csv("./data/rawfc-gpt-3-in-context-test-results.csv")
merged["label"] = merged["label"].replace("half","half-true")

all_labels = [f'\"{x}\"' for x in list(set(merged["label"].to_list()))]

predictions = []
responses = []
for idx in tqdm(range(len(merged["claim"]))):

    if result["in-context"][idx] in ["true","false","half-true"]:
        predictions.append(result["in-context"][idx].strip('"'))
        continue
    claim, article, label = merged["claim"][idx],merged["article"][idx],merged["label"][idx]
    test_prompt = "Mark the supporting article below with persuasive strategy labels. You should show a text span of the article and the corresponding labels"
    prompt = "Here we show example of persuasive strategy detecion. Examples below show text spans with their corresponding persuasive strategy: \n" + in_context + "\n\n" + test_prompt + "\n Article: " + article + "\n\n Don't mark a sentence with one strategy more than once."
    res = request_completion(prompt)

    res = truncate_text(res, max_tokens=8_000)

    prompt_2 = f"Claim: {claim} \n. Based on the known facts and given the labeled persuasive strategies in the above supporting article, among of the {', '.join(all_labels)} as possible veracity labels, the best label that describes the veracity of this claim is"
    prompt_2 = res + "\n\n" + prompt_2
    pred = request_completion(prompt_2)
    stripped_pred = extract_characters(pred)
    if stripped_pred.lower().strip().strip("\"").strip("'").startswith("true") :
        predictions.append("true")
    elif stripped_pred.lower().strip().strip("\"").strip("'").startswith("false"):
        predictions.append("false")
    elif stripped_pred.lower().strip().strip("\"").strip("'").startswith("halftrue"):
        predictions.append("half-true")
    else:
        predictions.append(pred)
    responses.append(pred)
merged["in-context"] = predictions
if len(responses) == len(predictions):
    merged["responses"] = responses
# merged.to_csv("./data/rawfc-gpt-3-in-context-test-results.csv")


labels = merged["label"].to_list()  
predictions = [x if x in ["true","false"] else "half-true" for x in merged["in-context"]]
print(predictions)
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
precision = precision_score(y_true=labels,
                                    y_pred=predictions,
                                    average='macro',
                                    zero_division="warn")
recall = recall_score(y_true=labels,
                                    y_pred=predictions,
                                    average='macro',
                                    zero_division="warn")

print(classification_report(labels,predictions , digits=3))

macro_f1 = f1_score(y_true=labels,
                                    y_pred=predictions,
                                    average='macro',
                                    zero_division="warn")
micro_f1 = f1_score(y_true=labels,
                                    y_pred=predictions,
                                    average='micro',
                                    zero_division="warn")
print(f"Micro F1: {micro_f1:0.3f} Macro F1: {macro_f1:0.3f} Precision: {precision:0.3f} Recall: {recall:0.3f}")