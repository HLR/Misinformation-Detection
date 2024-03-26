import os
import json
import re
import numpy as np
import pandas as pd
from gensim.summarization import summarize
import nltk
from nltk import tokenize
import torch
from tqdm import tqdm
from utils import strat_pred
nltk.download('punkt')


# Construct a dataframe of articles, along with a list of persuasive strategies
# that they were annotated with.
def construct_article_df(directory):
  ids, claims, articles, strategies = [], [], [], []

  # Loop through the annotation folder
  for filename in os.listdir(directory):
    f = os.path.join(directory, filename)

    # Open the json file with utf encoding
    with open(f, encoding="UTF-8") as read_file:
      file = json.load(read_file)

    # Extract the article text
    article_and_content = file['_referenced_fss']['12']['sofaString']
    splited = re.split(r"(?P<url>https?://[^\s]+)",article_and_content)
    if len(splited) == 3:
      splited1 = re.split(r"(?P<url>www\.[^\s]+)",article_and_content)
      claim_and_link, article = splited1[0],splited1[2].strip()
    else:
      claim_and_link, article = splited[0], " ".join(splited[4:])
    article = " ".join(re.split("Actual content:",article)).strip()
    claim_with_id, *_ = re.split("Source", claim_and_link, flags=re.IGNORECASE)
    id, claim = claim_with_id.split(" ")[0]," ".join(claim_with_id.split(" ")[1:])
    ids.append(id.strip())
    articles.append(article.strip())
    claims.append(claim.strip())

    # Extract the annotations
    strategy_location = file['_views']['_InitialView']
    singular_strats = []

    # If this article was annotated with any strategies:
    if "Persuasive_Labels" in strategy_location:
      persuasive_labels = strategy_location["Persuasive_Labels"]
      for strat in persuasive_labels:
        # Extract the annotation name
        try:
          annotation = list(strat.items())[3][1]
          singular_strats.append(annotation)
        except:
          continue

    strategies.append(singular_strats)

  data = pd.DataFrame(list(zip(ids ,claims, articles, strategies)), columns=["id", "claim", "article", "gt_strategy"])
  return data


# Remove duplicates from a list of lists
def remove_duplicates(list1):
  list2 = []
  for inner_list in list1:
    list2.append(list(set(inner_list)))
  return list2


# Convert a list of lists into a list of strings, separated by the given token
def list_to_str(list1, sep=' '):
  list2 = []
  for inner_list in list1:
    list2.append(sep.join(map(lambda x:x.lower(),inner_list)))
  return list2


# Given a string, return its length in tokens
def token_length(new_str, config):
  from utils import get_tokenizer
  tokenizer = get_tokenizer(config.longformer)
  tokens = tokenizer(new_str, padding=True, max_length=config.max_length*2,
                     truncation=True, return_tensors='pt')

  # Token arrays are numpy ones arrays, so add up all numbers
  # in the list that do not equal one
  lengths = (tokens['input_ids'] > 2).sum()
  return int(lengths)


# Prepare MultiFC files to be used
def prepare_multiFC(file):
  data = pd.read_csv(file, sep="\t")

  data.columns = ["claimID", "claim", "label", "claimURL", "reason", "category", "speaker",
                  "checker", "tags", "title", "publishDate", "claimDate", "entities"]

  data = data.reset_index(drop=True)

  return data


# Add labels to our dataframe by cross referencing the multiFC dataset
def add_labels(ids, data, norm):
  labels = []

  for id in ids:
    id = id.replace("\ufeff", "")

    # Match the claim ID to a row in MultiFC and extract the label
    row = data.loc[data["claimID"] == id]
    try:
      labels.append(row["label"].tolist()[0])
    except:
      labels.append("none")

  norm_labels = {
    'mostly true': 'true',
    'truth!': 'true',
    'true': 'true',
    'in-the-green': 'true',
    'mostly truth!': 'true',
    'disputed!': 'mixed',
    'mostly false': 'false',
    'mostly fiction!': 'false',
    'in-the-red': 'false',
    'fiction!': 'false',
    'false': 'false',
    'none': 'none',
    True: 'true',
    "True": 'true',
    "False": "false",
    False: 'false',
    'pants on fire!':'false'
  }
  if norm:
    labels = pd.Series(labels).apply(lambda x: norm_labels[x])
  return labels


# Cut irrelevant pre-text from the article texts
def cut_pre_text(articles):
  new_articles = []
  for article in articles:
    content = re.split("actual content", article, flags=re.IGNORECASE)
    if len(content) > 1:
      new_article = content[1][2:]
      new_articles.append(new_article)
    else:
      new_articles.append(article)
  return new_articles


# Summarize the article text so that it can be combined with the persuasive
# strategies and inputted into Roberta
def correct_length_inputs(claims,articles, strategies, config):
  combined = []

  # For all articles and the strategies they are annotated with
  for claim, article, strategy in tqdm(zip(claims, articles, strategies)):

    # The maximum token length the article can be is 512 - (strategy token length)
    strat_token_len = token_length(strategy,config) + token_length(claim,config)
    max_article_len = config.max_length - strat_token_len

    combined_str = ""
    summary = article
    for i in range(10, 0, -1):
      token_len = token_length(summary, config)
      if token_len < max_article_len:
        break
      # summary = summarize(summary, ,ratio=i / 10)
      try:
        summary = summarize(summary, word_count=max_article_len)
      except ValueError:
        summary = " ".join(summary.split(" ")[:max_article_len])
    combined_str = claim + ' </s></s> ' + summary + ' </s></s> ' + strategy
    combined.append(combined_str)
  return combined


# Summarize the article text so that it can be combined with the persuasive
# strategies and inputted into Roberta
def correct_length_inputs_claim(claims, strategies, config):
  combined = []

  # For all articles and the strategies they are annotated with
  for claim, strategy in tqdm(zip(claims, strategies)):

    # The maximum token length the article can be is 512 - (strategy token length)
    strat_token_len = token_length(strategy,config) + token_length(claim,config)
    max_article_len = config.max_length - strat_token_len

    combined_str = ""
    summary = claim
    for i in range(10, 0, -1):
      token_len = token_length(summary, config)
      if token_len < max_article_len:
        break
      try:
        summary = summarize(summary, word_count=max_article_len)
      except ValueError:
        summary = " ".join(summary.split(" ")[:max_article_len])
    combined_str = summary + ' </s></s> ' + strategy
    combined.append(combined_str)
  return combined


def split_article(article, context):
  # Using the nltk "punkt" corpus, tokenize the article into sentences
  sentences = tokenize.sent_tokenize(article)

  # A list of all segments in the current file
  segments = []

  # Loop through the sentences of the article, making segments along the way.
  #
  # A segment is the focus sentence, followed by a separating character,
  # followed by two sentences to the left of the focus sentence, followed by
  # another separating character, followed by the two sentences to the right
  for idx, sentence in enumerate(sentences):

    # If the left context is out of range, leave it blank
    try:
      left1 = sentences[idx - 2] if (idx - 2) >= 0 else ""
    except:
      left1 = ""
    try:
      left2 = sentences[idx - 1] if (idx - 1) >= 0 else ""
    except:
      left2 = ""
    # If the right context is out of range, leave it blank
    try:
      right1 = sentences[idx + 1]
    except:
      right1 = ""
    try:
      right2 = sentences[idx + 2]
    except:
      right2 = ""

    if context == "none":
      segment = sentence
    elif context == "low":
      segment = sentence + " </s> " + left2 + " </s> " + right1
    elif context == "high":
      segment = sentence + " </s> " + left1 + left2 + " </s> " + right1 + right2
    else:
      raise "Invalid Context given (must be 'none', 'low', or 'high')"
    segments.append(segment)
  return segments


# Predict the persuasive strategies in a list of articles
def predict_strategies(articles, models, context, config):
  from annotations import get_annotation_layer_and_indexes
  
  l2, l3, l4 = get_annotation_layer_and_indexes()

  pred = []
  # Loop through the list of articles
  for article in tqdm(articles):

    article_strats = []

    # Split the article into sequences
    text_sequences = split_article(article, context)

    # Loop through the sequences
    for seq in text_sequences:
      # Predict the persuasive strategies at the three layers
      # of classification
      l2_output = strat_pred(models[1], seq, context, config)
      l2_output = np.array(l2_output > 0.5, dtype=float)
      l2_idx = np.where(l2_output == 1)[0]
      l2_output = [l2[idx].lower() for idx in l2_idx]

      l3_output = strat_pred(models[2], seq, context,config)
      l3_output = np.array(l3_output > 0.5, dtype=float)
      l3_idx = np.where(l3_output == 1)[0]
      l3_output = [l3[idx].lower() for idx in l3_idx]

      l4_output = strat_pred(models[3], seq, context,config)
      l4_output = np.array(l4_output > 0.5, dtype=float)
      l4_idx = np.where(l4_output == 1)[0]
      l4_output = [l4[idx].lower() for idx in l4_idx]

      # Concatenate teh strategies
      seq_strats = l2_output + l3_output + l4_output
      # Add the sequence strategies to the article strategies
      article_strats += seq_strats
    pred.append(article_strats)

  return pred


def combine_claim_article(claims_articles,config):
  combined = []
  print(config.max_length)
  # For all articles and the strategies they are annotated with
  for claim, article in tqdm(claims_articles):

    # The maximum token length the article can be is 512 - (strategy token length)
    strat_token_len = token_length(claim,config)
    max_article_len = config.max_length - strat_token_len

    combined_str = ""
    summary = article
    for i in range(10, 0, -1):
      try:
        token_len = token_length(summary,config)
        if token_len < max_article_len:
          break
        summary = summarize(article, ratio=i / 10)
      except:
        import traceback
        print(traceback.format_exc())
        break
    combined_str =  claim + ' </s></s> ' + summary
    combined.append(combined_str)

  return combined

def summarize_article(articles,config):
  combined = []
  print(config.max_length)
  # For all articles and the strategies they are annotated with
  for article in tqdm(articles):

    # The maximum token length the article can be is 512 - (strategy token length)
    summary = article
    token_len = token_length(summary,config)
    if token_len > config.max_length:
        summary = summarize(article, word_count=config.max_length)
    combined.append(summary)

  return combined

# Build a dataset from the annotated articles
def build_complete_dataset(file, models, context,config):
  # Construct the basic dataframe with article texts and ground truth strategies
  data = construct_article_df(file)
  # Prepare the multiFC files for labelling
  file = "data/all.tsv"
  multiFC_data = prepare_multiFC(file)
  # Add the normalized labels
  data["label"] = add_labels(data["id"], multiFC_data, norm=True)
  data = data[data["label"] != "mixed"]
  data = data[data["label"] != "none"]


  # Remove duplicates and convert the strategies into a token separated list
  data["claim_article"] = combine_claim_article(zip(data["claim"],data["article"]),config)
  

  # data["gt_strategy"] = remove_duplicates(data["gt_strategy"])
  data["gt_strategy"] = list_to_str(data["gt_strategy"], sep=' </s> ')

  # Predict the article strategies at the given context
  data["pred_strategy"] = predict_strategies(data["article"], models, context, config)
  # data["pred_strategy"] = remove_duplicates(data["pred_strategy"])
  data["pred_strategy"] = list_to_str(data["pred_strategy"], sep=' </s> ')


  # Create the column with the article text and target strategies
  data["claim_gt"] = correct_length_inputs_claim(data["claim"], data["gt_strategy"], config)
  data["claim_article_gt"] = correct_length_inputs(data["claim"], data["article"], data["gt_strategy"], config)

  # Create the column with the article text and predicted strategies
  data["claim_pred"] = correct_length_inputs_claim(data["claim"], data["pred_strategy"], config)
  data["claim_article_pred"] = correct_length_inputs(data["claim"], data["article"], data["pred_strategy"], config)

  data["article"] = summarize_article(data["article"],config)
  return data


# Article dataset used in training and testing
class ArticleDataset(torch.utils.data.Dataset):

  def __init__(self, data, column):
    labels = \
      {
        True:0,
        False:1,
        "true": 0,
        "false": 1
      }
    self.labels = [labels[label] for label in data['label']]
    self.texts = tuple(data[column])

  def classes(self):
    return self.labels

  def __len__(self):
    return len(self.labels)

  def get_batch_labels(self, idx):
    return np.array(self.labels[idx])

  def get_batch_texts(self, idx):
    return self.texts[idx]

  def __getitem__(self, idx):
    batch_texts = self.get_batch_texts(idx)
    batch_y = self.get_batch_labels(idx)
    return batch_texts, batch_y