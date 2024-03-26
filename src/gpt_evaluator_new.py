import os
import openai
import time
import argparse
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import f1_score, classification_report


parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str,default="gpt3",choices=["gpt-3.5-turbo","gpt-4"])
parser.add_argument('--context', type=str,choices="none,low,high".split(","),default="none")
parser.add_argument('--source', type=str,choices=["claim","article", "claim_article","claim_article_gt" ,"claim_gt","gt_strategy","pred_strategy","claim_pred","claim_article_pred"],default="claim")
args = parser.parse_args()
source_file_name = f"./data/test_article_data_{args.context}.csv"
print(args.model, args.source)
def request_chat(prompt):
    while True:
        try:
            time.sleep(2)
            response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                    # {"role": "system", "content": "I am a journalism and communication expert."},
                    {"role": "user", "content": prompt},
                ],
            temperature=0,
            top_p=0.0,
            )
            res = response["choices"][0]["message"]
            return res.content
        except:
            time.sleep(60)

def request(prompt):
    response = openai.Completion.create(
    model="gpt-3.5-turbo-instruct",
    # model="text-davinci-003",
    prompt=prompt,
    temperature=0,
    top_p=0.0,
    )
    res = response["choices"][0]["text"]
    return res




def get_prompt(source):
    if source == "claim":
        return """
Assuming you are a journalism and communication expert. Is this claim correct? 
\n\n
claim: {claim}
\n\n
You must answer to the best of your knowledge. Give me one word answer "Yes" or "No"?
"""
    elif source == "article":
        return """
Assuming you are a journalism and communication expert. Is this article correct? 
\n\n
article: {article}
\n\n
You must answer to the best of your knowledge. Give me one word answer "Yes" or "No"?
        """
    elif source == "claim_article":
        return """
Assuming you are a journalism and communication expert. Is this claim correct? 
\n\n
claim: {claim}
\n\n
We have found an article supporting the claim: \n\n
article: {article}
\n\n
You must answer to the best of your knowledge. Give me one word answer "Yes" or "No"?
"""
    elif source in ["claim_article_gt","claim_article_pred"]:
        return """
Assuming you are a journalism and communication expert. Is this claim correct? 
claim: {claim}
\n\n
We have found this article supporting the claim: \n\n
article: {article}
\n\n
our communication expert reported that the article supporting this claim follows the below persuasive strategies:
{labels}\n\n
To the best of your knowledge give me one word answer "Yes" or "No"?
"""
    elif source in ["claim_gt","claim_pred"]:
        return """
Assuming you are a journalism and communication expert. Is this claim correct? 
claim: {claim}
\n\n
our communication expert reported that the article supporting this claim follows the below persuasive strategies:
{labels}
\n\n
To the best of your knowledge give me one word answer "Yes" or "No"?
"""

    elif "strategy" in source:
        return """
Assuming you are a journalism and communication expert.  Is a claim correct if our communication expert reported that an article supporting that claim follows the below persuasive strategies.
{labels}
\n\n
To the best of your knowledge give me one word answer "Yes" or "No"?
    """




def get_prompts(source, prompt_template, data):
    prompts = []
    if source == "claim":
        for claim in data["claim"]:
            prompts.append(prompt_template.format(claim=claim))
    elif source == "article":
        for article in data["article"]:
            prompts.append(prompt_template.format(article=article))
    elif source == "claim_article":
        for claim, article in zip(data["claim"],data["article"]):
            prompts.append(prompt_template.format(claim=claim,article=article))
    elif source == "claim_article_gt":
        for claim, article, gt_labels in zip(data["claim"],data["article"],data["gt_strategy"]):
            gt_labels = ". ".join(list(gt_labels.split(" </s> "))).strip()
            prompts.append(prompt_template.format(claim=claim,article=article,labels=gt_labels))
    elif source == "claim_gt":
        for claim, gt_labels in zip(data["claim"],data["gt_strategy"]):
            gt_labels = ". ".join(list(gt_labels.split(" </s> "))).strip()
            prompts.append(prompt_template.format(claim=claim,labels=gt_labels))
    elif source == "claim_article_pred":
        for claim, article, pred_labels in zip(data["claim"],data["article"],data["pred_strategy"]):
            if not pred_labels or not isinstance(pred_labels,str):
                pred_labels = ""
            pred_labels = ". ".join(list(set(pred_labels.split(" </s> ")))).strip()
            prompts.append(prompt_template.format(claim=claim,article=article,labels=pred_labels))
    elif source == "claim_pred":
        for claim, pred_labels in zip(data["claim"],data["pred_strategy"]):
            if not pred_labels or not isinstance(pred_labels,str):
                pred_labels = ""
            pred_labels = ". ".join(list(set(pred_labels.split(" </s> ")))).strip()
            prompts.append(prompt_template.format(claim=claim,labels=pred_labels))
    elif "strategy" in source:
        for temp_labels in data[source]:
            if not temp_labels or not isinstance(temp_labels,str):
                temp_labels = ""
            temp_labels = ". ".join(list(temp_labels.split(" </s> "))).strip()
            prompts.append(prompt_template.format(labels=temp_labels))
    
    return prompts

openai.api_key = os.environ["OPENAI_API_KEY"]
if args.model == "gpt-3.5-turbo":
    validator_fn = request
elif args.model == "gpt-4":
    validator_fn = request_chat

data = pd.read_csv(source_file_name)
prompt_template = get_prompt(args.source)
prompts = get_prompts(args.source, prompt_template, data)

predictions = []
for prompt in tqdm(prompts):
    response = validator_fn(prompt)

    pred = response.split(" ")[0].strip().startswith("Yes")
    predictions.append(pred)

print(classification_report(data["label"], predictions, digits=3))
macro_f1 = f1_score(y_true=data["label"],
                                    y_pred=predictions,
                                    average='macro',
                                    zero_division="warn")
micro_f1 = f1_score(y_true=data["label"],
                                    y_pred=predictions,
                                    average='micro',
                                    zero_division="warn")
print(f"Micro F1: {micro_f1:0.3f} Macro F1: {macro_f1:0.3f}")

# def check_without_labels():

#     data = pd.read_csv("./data/test_article_data_none.csv")
#     responses = []
#     for claim in tqdm(data["claim"]):
#         res = request_chat(PROMPT_WITH_NO_LABEL.format(claim))
#         # print(res)
#         responses.append(res)
#         # 
#     data["gpt"] = responses
#     data.to_csv("./data/test_article_data_none.csv")
#     # data = pd.read_csv("./data/test_article_data_none_gpt.csv")
#     predictions = []
#     for x in data["gpt"]:
#         # print(x.split(" ")[0].strip())
#         predictions.append(x.split(" ")[0].strip().startswith("Yes"))

    
#     print(classification_report(labels, predictions, digits=3))
#     macro_f1 = f1_score(y_true=labels,
#                                         y_pred=predictions,
#                                         average='macro',
#                                         zero_division="warn")
#     micro_f1 = f1_score(y_true=labels,
#                                         y_pred=predictions,
#                                         average='micro',
#                                         zero_division="warn")
#     print(f"Micro F1: {micro_f1:0.3f} Macro F1: {macro_f1:0.3f}")
#     # print(list(zip(labels,predictions)))

# def check_with_labels():
#     PROMPT_WITH_LABEL = """
# Assuming you are a journalism and communication expert. Is this claim correct? 
# claim: {claim}
# \n
# our communication expert reported that the article supporting this claim follows the below persuasive strategies:
# {labels}\n\n
# To the best of your knowledge give me one word answer "Yes" or "No"?
#     """

#     data = pd.read_csv("./data/test_article_data_none.csv")
#     responses = []
#     # for gt_strategy, claim in tqdm(zip(data["pred_strategy"],data["claim"])):
#     for gt_strategy, claim in tqdm(zip(data["gt_strategy"], data["claim"])):
#         gt_strategy = ". ".join(list(gt_strategy.split(" </s> "))).strip()
#         res = request_chat(PROMPT_WITH_LABEL.format(claim=claim,labels=gt_strategy))
#         responses.append(res)
#         # time.sleep(10)
#     data["gpt_gt"] = responses
#     data.to_csv("./data/test_article_data_none.csv")
#     predictions = []
#     for x in data["gpt_gt"]:
#         # print(x.split(" ")[0].strip())
#         predictions.append(x.split(" ")[0].strip().startswith("Yes"))

#     labels = data["label"]
#     print(classification_report(labels, predictions, digits=3))
#     macro_f1 = f1_score(y_true=labels,
#                                         y_pred=predictions,
#                                         average='macro',
#                                         zero_division="warn")
#     micro_f1 = f1_score(y_true=labels,
#                                         y_pred=predictions,
#                                         average='micro',
#                                         zero_division="warn")
#     print(f"Micro F1: {micro_f1:0.3f} Macro F1: {macro_f1:0.3f}")

# print("NO LABELS")
# check_without_labels()
# """
# NO LABELS
#               precision    recall  f1-score   support

#        False      0.933     0.875     0.903        32
#         True      0.750     0.857     0.800        14

#     accuracy                          0.870        46
#    macro avg      0.842     0.866     0.852        46
# weighted avg      0.878     0.870     0.872        46

# Micro F1: 0.870 Macro F1: 0.852
# NO LABELS
# 100%|██████████████████████████████████████████████████████████████████████████████████████████████| 46/46 [08:28<00:00, 11.05s/it]
#               precision    recall  f1-score   support

#        False      0.903     0.875     0.889        32
#         True      0.733     0.786     0.759        14

#     accuracy                          0.848        46
#    macro avg      0.818     0.830     0.824        46
# weighted avg      0.852     0.848     0.849        46

# Micro F1: 0.848 Macro F1: 0.824
# """
# print("WITH LABELS")
# check_with_labels()