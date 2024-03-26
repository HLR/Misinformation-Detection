import os
import json
import re
import traceback
import torch
import sys
import numpy as np
import pandas as pd
from nltk import tokenize

from annotations import get_annotations,get_annotation_layer_keys

annotation_to_key = get_annotations()

# Construct a dictionary from the annotation files, The keys
# being text segments, and the values being their respective annotations
# The context determines how much surrounding text will be included.
def construct_segment_dict(directory, context):
  # This dictionary will contain every segment as keys, and their
  # respective annotations as values
  segment_dict = {}

  # Loop through the annotation folder
  for filename in os.listdir(directory):
    f = os.path.join(directory, filename)

    with open(f, encoding="UTF-8") as read_file:
      file = json.load(read_file, )

    # Extract the article text
    article = file['_referenced_fss']['12']['sofaString']

    # Extract the annotations
    if "Persuasive_Labels" in file['_views']['_InitialView']:
      persuasive_labels = file['_views']['_InitialView']["Persuasive_Labels"]
      sentences_ids = file['_views']['_InitialView']["Sentence"]
    else:
      persuasive_labels = []

    begin_to_end = {}
    end_to_begin = {}
    for index, sentence_item in enumerate(sentences_ids):
      if index == 0:
        begin = 0
        end = sentence_item.get("end")
      else:
        begin = sentence_item.get("begin")
        end = sentence_item.get("end")
      begin_to_end[begin] = end
      end_to_begin[end] = begin
        
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
        raise Exception("Invalid Context given (must be 'none', 'low', or 'high')")

      # Add the segment to the list, as well as the global dictionary
      segments.append(segment)
      segment_dict[segment] = []

      # This dictionary contains all annotation signs as keys. Its values will be
      # lists of pairs, representing starting and ending indices where the article
      # is annotated by the corresponding label
      anno_dict = {}
      for anno_key in annotation_to_key.values():
        anno_dict[anno_key] = []

      # Loop through the persuasive labels of this article
      # An example label is:
      # {'sofa': 12, 'begin': 3779, 'end': 3851, 'EmotionalAppeals': 'anger'}
      for label in persuasive_labels:
        # Define the significant values
        if label.get('begin'):
          start_idx = label.get('begin')
        else:
          start_idx = 0
        
        if label.get('end'):
          end_idx = label.get('end')
        else:
          end_idx = begin_to_end[start_idx]
        
        try:
          annotations = list(filter(lambda x:x[0] not in ['end','sofa','begin'],label.items()))
          for key, annotation in annotations:
            # annotation = normalize_annotations(annotation)
            anno_sign = annotation_to_key[annotation]
            anno_dict[anno_sign].append((start_idx, end_idx))

            # If a segment is annotated with a child tag, it must also
            # be annotated with the associated parent tag
            if anno_sign in ["1a", "1b", "1c"] and \
                    (start_idx, end_idx) not in anno_dict["1"]:
              anno_dict["1"].append((start_idx, end_idx))
            if anno_sign in ["3a"] and \
                    (start_idx, end_idx) not in anno_dict["3"]:
              anno_dict["3"].append((start_idx, end_idx))
            if anno_sign in ["4a", "4b", "4c", "4d"] and \
                    (start_idx, end_idx) not in anno_dict["4"]:
              anno_dict["4"].append((start_idx, end_idx))
            if anno_sign in ["7a", "7b", "7c"] and \
                    (start_idx, end_idx) not in anno_dict["7"]:
              anno_dict["7"].append((start_idx, end_idx))
            if anno_sign in ["8a"] and \
                    (start_idx, end_idx) not in anno_dict["8"]:
              anno_dict["8"].append((start_idx, end_idx))
            if anno_sign in ["9a", "9b", "9c", "9d", "9e"] and \
                    (start_idx, end_idx) not in anno_dict["9"]:
              anno_dict["9"].append((start_idx, end_idx))
            if anno_sign in ["10a", "10b", "10c", "10d"] and \
                    (start_idx, end_idx) not in anno_dict["10"]:
              anno_dict["10"].append((start_idx, end_idx))
            if anno_sign in ["11a", "11b", "11c", "11d", "11e"] and \
                    (start_idx, end_idx) not in anno_dict["11"]:
              anno_dict["11"].append((start_idx, end_idx))
            if anno_sign in ["12a", "12b", "12c", "12d"] and \
                    (start_idx, end_idx) not in anno_dict["12"]:
              anno_dict["12"].append((start_idx, end_idx))

            # If a segment is annotated with a sub tag, it must also
            # be annotated with the associated child tag
            if anno_sign in ["12ai", "12aii", "12aiii", "12aiv"] and \
                    (start_idx, end_idx) not in anno_dict["12a"]:
              anno_dict["12a"].append((start_idx, end_idx))
            if anno_sign in ["12bi", "12biii"] and \
                    (start_idx, end_idx) not in anno_dict["12b"]:
              anno_dict["12b"].append((start_idx, end_idx))
            if anno_sign in ["12ci", "12cii", "12civ"] and \
                    (start_idx, end_idx) not in anno_dict["12c"]:
              anno_dict["12c"].append((start_idx, end_idx))
        except Exception as e:
          print(e)
          print(traceback.format_exc())
          continue

      # Loop through all segments in the list,
      for segment in segments:

        # Extract the original sentence by splitting
        # at the first separating token
        og_sentence = re.split(" </s> ", segment)[0]

        # Define a left index, right index, and range for the sentence
        left_sent = article.find(og_sentence)
        right_sent = left_sent + len(og_sentence)
        sent_range = range(left_sent, right_sent)

        # Loop through all annotation ranges, checking which annotations
        # The current sentence is classified by
        for anno in anno_dict:
          for value in anno_dict[anno]:

            # Define a left index, right index and range for the annotation
            left_anno = value[0]
            right_anno = value[1]
            anno_range = range(left_anno, right_anno)

            # If the annotation and sentence overlap each other, add the
            # annotation to the segments list of labels
            if (left_anno in sent_range or right_anno in sent_range or
                left_sent in anno_range or right_sent in anno_range) \
                    and anno not in segment_dict[segment]:
              segment_dict[segment].append(anno)

  # If a sentence has no annotations, classify it as "0"
  for segment in segment_dict:
    if len(segment_dict[segment]) == 0:
      segment_dict[segment].append("0")

  for i in segment_dict:
    for j in segment_dict[i]:
      num = ""
      for k in j:
        if k.isdigit():
          num += str(k)
      if num not in segment_dict[i]:
        segment_dict[i].append(num)
  return segment_dict


def construct_segment_dict_v2(file_address, context, layer):
  all_labels = get_annotation_layer_keys(2) + get_annotation_layer_keys(3) + get_annotation_layer_keys(4) 

  data = pd.read_excel(file_address)

  hasAnno = []
  segments = []
  for index, row in data.iterrows():
    sentence = row["sentence"]
    prefix_sentence_1 = row["prefix_sentence_1"]
    if prefix_sentence_1 == "nan" or pd.isna(prefix_sentence_1):
      prefix_sentence_1 = ""
    suffix_sentence_1 = row["suffix_sentence_1"]
    if suffix_sentence_1 == "nan" or pd.isna(suffix_sentence_1):
      suffix_sentence_1 = ""
    prefix_sentence_2 = row["prefix_sentence_2"]
    if prefix_sentence_2 == "nan" or pd.isna(prefix_sentence_2):
      prefix_sentence_2 = ""
    suffix_sentence_2 = row["suffix_sentence_2"]
    if suffix_sentence_2 == "nan" or pd.isna(suffix_sentence_2):
      suffix_sentence_2 = ""
    
    if context == "none":
      segment = sentence
    elif context == "low":
      segment =  prefix_sentence_1 + " </s> " +  sentence + " </s> " + suffix_sentence_1
      # segment =  sentence + " </s> " + prefix_sentence_1 + " </s> " +   suffix_sentence_1
      
    elif context == "high":
      segment = prefix_sentence_2 + prefix_sentence_1 + " </s> " +  sentence +  " </s> " + suffix_sentence_1 + suffix_sentence_2
      # segment = sentence +  " </s> " + prefix_sentence_2 + prefix_sentence_1 + " </s> " +  suffix_sentence_1 + suffix_sentence_2
    else:
      raise Exception("Invalid Context given (must be 'none', 'low', or 'high')")
    segments.append(segment)
    hasAnno.append(int(len([1 for label in all_labels if data.iloc[index][label] == 1]) > 0))
  data["hasAnno"] = hasAnno
  data["hasAnno"] = data["hasAnno"].astype(float)
  data["sentence"] = segments
  if layer == "1":
    data = data[["claim_id", "sentence", "hasAnno"]]
  elif layer in ["2","3","4"]:
    labels = get_annotation_layer_keys(layer)
    data = data[["claim_id", "sentence"]+labels]
    total_labels = 0
    for label in labels:
      data[label] = data[label].astype(float)
      total_labels+= data[label].sum()
  else:
    raise "Invalid Layer Given (Must be '1', '2', '3', or '4')"
  
  data = data[data["sentence"].notna()]
  data = data[data["sentence"] != ""]
  
  return data

# Convert a dictionary to a pandas dataframe, so it can be used by
# standard machine learning models
def convert_to_dataframe(segment_dict, layer):
  length = len(segment_dict)
  texts = []
  hasAnno = []
  label_dict = {}
  idx = 0

  # For every segment in our dictionary, test whether it is annotated
  # with any persuasive strategy.
  for i in segment_dict:
    texts.append(i)
    if "0" in segment_dict[i]:
      hasAnno.append(0)
    else:
      hasAnno.append(1)

    # For particular persuasive strategies, create a zeros array and
    # edit it accordingly
    for anno in segment_dict[i]:
      if anno not in label_dict:
        label_dict[anno] = np.zeros(length)
      label_dict[anno][idx] = 1
    idx += 1

  # Create a dataframe with the binary labels
  data_dict = {"Texts": texts}
  data = pd.DataFrame(data_dict)

  # Depending on the layer, add all relevant labels to the dataframe
  if layer == "1":
    data["hasAnno"] = hasAnno
    labels = []
  elif layer in ["2","3","4"]:
    labels = get_annotation_layer_keys(layer)
  else:
    raise "Invalid Layer Given (Must be '1', '2', '3', or '4')"

  for label in labels:
    if label not in label_dict:
      label_dict[label] = np.zeros(length)
    data[label] = label_dict[label]

  data.reset_index(drop=True, inplace=True)
  return data


# Segment dataset used in training and testing
class SegmentDataset(torch.utils.data.Dataset):

  def __init__(self, data, config):

    self.dataframe = data
    self.texts = tuple(data["sentence"])

  def __len__(self):
    return len(self.texts)

  def get_batch_texts(self, idx):
    return self.texts[idx]

  def __getitem__(self, idx):
    batch_texts = self.get_batch_texts(idx)
    d = self.dataframe.iloc[idx]
    batch_y = torch.tensor(d[2:].tolist(), dtype=torch.float32)
    return batch_texts, batch_y
