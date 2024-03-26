import os
import openai
import time
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import f1_score, classification_report
openai.api_key = os.environ["OPENAI_API_KEY"]

def request_chat(prompt):
    
    response = openai.ChatCompletion.create(
    model="gpt-4",#gpt-3.5-turbo",
    messages=[
            # {"role": "system", "content": "I am a journalism and communication expert."},
            {"role": "user", "content": prompt},
        ],
    temperature=0,
    )
    res = response["choices"][0]["message"]
    return res.content


def request(prompt):
    response = openai.Completion.create(
    model="text-davinci-003",
    prompt=prompt,
    temperature=0,
    max_tokens=2000,
    top_p=1.0,
    frequency_penalty=0.0,
    presence_penalty=0.0
    )
    res = response["choices"][0]["text"]
    return res


def check_without_labels():
    PROMPT_WITH_NO_LABEL = """
Assuming you are a journalism and communication expert. Is this claim correct? 
\n\n
{}
You must answer to the best of your knowledge. Give me one word answer "Yes" or "No"?
"""

    data = pd.read_csv("./data/test_article_data_none.csv")
    responses = []
    for claim in tqdm(data["claim"]):
        res = request_chat(PROMPT_WITH_NO_LABEL.format(claim))
        # print(res)
        responses.append(res)
        time.sleep(10)
    data["gpt"] = responses
    data.to_csv("./data/test_article_data_none.csv")
    # data = pd.read_csv("./data/test_article_data_none_gpt.csv")
    predictions = []
    for x in data["gpt"]:
        # print(x.split(" ")[0].strip())
        predictions.append(x.split(" ")[0].strip().startswith("Yes"))

    labels = data["label"]
    print(classification_report(labels, predictions, digits=3))
    macro_f1 = f1_score(y_true=labels,
                                        y_pred=predictions,
                                        average='macro',
                                        zero_division="warn")
    micro_f1 = f1_score(y_true=labels,
                                        y_pred=predictions,
                                        average='micro',
                                        zero_division="warn")
    print(f"Micro F1: {micro_f1:0.3f} Macro F1: {macro_f1:0.3f}")
    # print(list(zip(labels,predictions)))

def check_with_labels():
    PROMPT_WITH_LABEL = """
Assuming you are a journalism and communication expert. Is this claim correct? 
claim: {claim}
\n
our communication expert reported that that the article supporting this claim follows the below persuasive strategies:
{labels}\n\n
To the best of your knowledge give me one word answer "Yes" or "No"?
    """

    data = pd.read_csv("./data/test_article_data_none.csv")
    responses = []
    # for gt_strategy, claim in tqdm(zip(data["pred_strategy"],data["claim"])):
    for gt_strategy, claim in tqdm(zip(data["gt_strategy"], data["claim"])):
        gt_strategy = ". ".join(list(gt_strategy.split(" </s> "))).strip()
        res = request_chat(PROMPT_WITH_LABEL.format(claim=claim,labels=gt_strategy))
        responses.append(res)
        time.sleep(10)
    data["gpt_gt"] = responses
    data.to_csv("./data/test_article_data_none.csv")
    predictions = []
    for x in data["gpt_gt"]:
        # print(x.split(" ")[0].strip())
        predictions.append(x.split(" ")[0].strip().startswith("Yes"))

    labels = data["label"]
    print(classification_report(labels, predictions, digits=3))
    macro_f1 = f1_score(y_true=labels,
                                        y_pred=predictions,
                                        average='macro',
                                        zero_division="warn")
    micro_f1 = f1_score(y_true=labels,
                                        y_pred=predictions,
                                        average='micro',
                                        zero_division="warn")
    print(f"Micro F1: {micro_f1:0.3f} Macro F1: {macro_f1:0.3f}")

print("NO LABELS")
check_without_labels()
"""
NO LABELS
              precision    recall  f1-score   support

       False      0.933     0.875     0.903        32
        True      0.750     0.857     0.800        14

    accuracy                          0.870        46
   macro avg      0.842     0.866     0.852        46
weighted avg      0.878     0.870     0.872        46

Micro F1: 0.870 Macro F1: 0.852
NO LABELS
100%|██████████████████████████████████████████████████████████████████████████████████████████████| 46/46 [08:28<00:00, 11.05s/it]
              precision    recall  f1-score   support

       False      0.903     0.875     0.889        32
        True      0.733     0.786     0.759        14

    accuracy                          0.848        46
   macro avg      0.818     0.830     0.824        46
weighted avg      0.852     0.848     0.849        46

Micro F1: 0.848 Macro F1: 0.824
"""
# print("WITH LABELS")
# check_with_labels()