import os
import json
import re
import pandas as pd
from sklearn.metrics import f1_score, classification_report
from annotations import normalize_annotations
from nltk.tokenize import sent_tokenize, word_tokenize
from tqdm import tqdm
import time
import openai
import argparse

openai.api_key = os.environ["OPENAI_API_KEY"]


def get_total(messages):
    """ Get the total tokens in a prompt """
    total = 0
    for message in messages:
        total += len(word_tokenize(message["content"]))
    return total


def extract_characters(string):
  return re.sub('[^\w]', '', string)


def get_context_limit(model,total_length):
    if model == 'gpt-4':
        if total_length < 8_000:
            return 'gpt-4', 8_000
        else:
            return 'gpt-4-32k', 32_000
        
    elif model == 'gpt-3':
        return 'gpt-3.5-turbo-16k', 16_000


def request_chat(messages, model, wait_time=10, top_p=1):
    time.sleep(wait_time)    
    while True:
        try:
            response = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            top_p=top_p,
            )
            return response["choices"][0]["message"]
        except Exception as e:
            print(e)
            print("Error")
            time.sleep(120)


def get_samples():
    samples = []

    train_dir = "./data/anno_train/"
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
                        if used_labels[label] <= 1:
                            ranges[pres_text].append(label)
                            used_labels[label] += 1
                for item,values in ranges.items():
                    if values:
                        samples.append(item + " => " + str(values))
    return samples


def predict_veracity_label(claim, article, all_labels, args):
    messages = [
        {"role": "user", "content": f"Act as a news fact-checker.\n Claim: {claim} \n\n Article :{article} \n\n  Based on the known facts and given the labeled persuasive strategies in the above supporting article, which of the {', '.join(all_labels)} as possible veracity labels, best describes the veracity of this claim. You must choose one."}
    ]
    model, max_tokens = get_context_limit(args.model, get_total(messages))
    res = request_chat(messages, model, args.wait_time,top_p=0.0)
    pred = res["content"]
    print(pred)
    return pred, pred


def evaluate(labels,predictions):

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

def fix_preds(args):
    results = pd.read_csv(f"./data/rawfc-{args.model}-sanity-test-results.csv")
    predictions = []
    i = 0
    for indx, pred in enumerate(results["prediction"]):
        res = extract_characters(pred.lower())
        if res.endswith("halftrue") or "ishalftrue" in res or "\"half-true.\"" in pred or "\"half-true\"" in pred:
            predictions.append("half-true")
        elif res.endswith("true") or "istrue" in res or "betrue" in res or "astrue" in res  or "\"true.\"" in pred:
            predictions.append("true")
        elif res.endswith("false") or "isfalse" in res or "\"false.\"" in pred:
            predictions.append("false")
        else:
            predictions.append("half-true")
            print(i,indx,"ERROR",pred)
            i+=1 

    evaluate(results["label"].to_list(),predictions)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--model', type=str, default="gpt-4",
                        help='model to use')
    parser.add_argument('--force_call', type=int, default=0,
                        help='force call openai')
    parser.add_argument('--wait_time', type=int, default=0,
                        help='wait time for call openai')
    args = parser.parse_args()

    samples = get_samples()
    in_context = "\n".join(samples)

    dataset = pd.read_csv("./data/RAWFC/test.csv")
    dataset["label"] = dataset["label"].replace("half", "half-true")

    results_address = f"./data/rawfc-{args.model}-sanity-test-results.csv"
    if not os.path.exists(results_address) or args.force_call == 1:
        print("Creating new results file")
        results = dataset.copy()
        results["prediction"] = ["none" for _ in range(len(dataset["claim"]))]
        results["response"] = ["none" for _ in range(len(dataset["claim"]))]
    else:
        print("Loading results file")
        results = pd.read_csv(results_address)
        results["prediction"] = results["prediction"].astype('str').map(lambda x: x.lower()) 
        results["response"] = results["response"].astype('str') 
    all_labels = [f'\"{x}\"' for x in list(set(dataset["label"].to_list()))]
    in_context = "\n".join(samples)


    predictions = []
    responses = []
    total_items = len(dataset["claim"])
    
    for idx in tqdm(range(total_items)):
        print(idx)
        prediction = results["prediction"][idx]
        response = results["response"][idx]
        if response != "none" and prediction != "none" and args.force_call == 0:
            predictions.append(prediction)
            responses.append(response)
            print(idx, "Already predicted", prediction)
            continue

        response, answer  = predict_veracity_label(dataset["claim"][idx], dataset["article"][idx], all_labels, args)
        res = extract_characters(response.lower())
        if res.startswith("true"):
            predictions.append("true")
        elif res.startswith("false"):
            predictions.append("false")
        elif res.startswith("half-true"):
            predictions.append("half-true")
        else:
            predictions.append(response)
        responses.append(response)

        results["prediction"] = predictions + ["none"] * (total_items - len(predictions))
        results["response"] = responses + ["none"] * (total_items - len(predictions))
        results.to_csv(results_address)


    labels = dataset["label"].to_list()  

    # predictions = [x if x in ["true","false"] else "half-true" for x in dataset["prediction"]]
    fix_preds(args)    
