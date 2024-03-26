import os
import random 
import sklearn
import pandas as pd
import argparse
import json
import re
import shutil
import sys
from zipfile import ZipFile
from tqdm import tqdm

from annotations import get_low_freq, normalize_annotations, get_annotation_keys, get_annotations

def remove_file_id(text):
  return re.sub(r'((snes)|(tron))-[0-9]*', '', text)

def normalize_claim_label(old_label):
  return {
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
      "True": 'true',
      "False": 'false',
      "pants on fire!": 'false',
      True: 'true',
      False: 'false'
    }[old_label]

def prepare_multiFC_files(data_address):
  items_dict = {
      'tron-01817': False,
      'tron-01837': False,
      'snes-04381': False,
      'snes-03912': False,
      'tron-02206': True,
      'snes-02036': False,
      'tron-02277': False,
      'tron-01822': True,
      'tron-01423': True,
      'snes-02821': False,
      'snes-05888': True,
      'snes-02739': True,
      'snes-02381': True,
      'snes-01222': False,
      'snes-05800': False,
      'tron-01369': False,
      'tron-01432': False,
      'snes-02554': False,
      'tron-01436': False,
      'snes-01402': True,
      'tron-02254': False,
      'snes-02061': False,
      'tron-02202': True,
      'snes-01976': False,
      'snes-04853': False,
      'tron-01402': True
  }
  
  train_data = pd.read_csv(os.path.join(data_address,"multi-fc/train.tsv"),sep="\t")
  
  train_data.columns = ["claimID", "claim", "label", "claimURL", "reason", "category", "speaker",
                  "checker", "tags", "title", "publishDate", "claimDate", "entities"]
  train_data = train_data.dropna()
  train_data = train_data.reset_index(drop=True)
  
  
  dev_data = pd.read_csv(os.path.join(data_address,"multi-fc/dev.tsv"),sep="\t")
  
  dev_data.columns = ["claimID", "claim", "label", "claimURL", "reason", "category", "speaker",
                  "checker", "tags", "title", "publishDate", "claimDate", "entities"]
  dev_data = dev_data.dropna()
  dev_data = dev_data.reset_index(drop=True)
  
  test_data = pd.read_csv(os.path.join(data_address,"multi-fc/test.tsv"),sep="\t")
  test_data.columns = ["claimID", "claim", "claimURL", "reason", "category", "speaker",
                  "checker", "tags", "title", "publishDate", "claimDate", "entities"]
  test_data["label"] = [None for _ in range(len(test_data))]
  test_data = test_data.reset_index(drop=True)
  
  merged_data = pd.concat([train_data,dev_data,test_data])
  
  for claimID, label in items_dict.items():
      merged_data.loc[merged_data['claimID'] == claimID, 'label'] = label
  
  merged_data.to_csv(os.path.join(data_address,"all.tsv"),sep="\t",index=False)
  
def article_name_from_file(file_address):
  open_file = open(file_address)
  json_data = json.load(open_file)
  article = json_data['_referenced_fss']['12']['sofaString'].replace("\ufeff", "")
  article_id = article.split(" ")[0]
  if re.search("[0-9]+",article_id):
    article_id = re.search("[a-z]+-[0-9]+",article_id)[0]
    return article_id
  return None


def normalize_file(file_address):
  with open(file_address,) as open_file:
    json_data = json.load(open_file)
    open_file.close()
  if "Persuasive_Labels" in json_data['_views']['_InitialView']:
    persuasive_labels = json_data['_views']['_InitialView']["Persuasive_Labels"]
  else:
    persuasive_labels = []
  for label in persuasive_labels:
    for key in label:
      if key not in ["sofa","begin","end"]:
        label[key] = normalize_annotations(label[key])
  json_data['_views']['_InitialView']["Persuasive_Labels"] = persuasive_labels
  os.remove(file_address)

  if "tron-01336" in file_address:
    article_id = "snes-02739"
    file_address = file_address.replace("tron-01336", article_id)
    json_data['_referenced_fss']['12']['sofaString'] = json_data['_referenced_fss']['12']['sofaString'].replace("tron-01336", article_id)
  if "snes-05888" in file_address and "Ricans" in json_data['_referenced_fss']['12']['sofaString']:
    article_id = "tron-02206" 
    file_address = file_address.replace("snes-05888", article_id)
    json_data['_referenced_fss']['12']['sofaString'] = json_data['_referenced_fss']['12']['sofaString'].replace("snes-05888", article_id)
  if "snes-02739" in file_address and "Tuna" in json_data['_referenced_fss']['12']['sofaString']:
    article_id = "pomt-12304" 
    file_address = file_address.replace("snes-02739", article_id)
    json_data['_referenced_fss']['12']['sofaString'] = json_data['_referenced_fss']['12']['sofaString'].replace("snes-02739", article_id)
  if "snes-00862" in file_address and "Jean" in json_data['_referenced_fss']['12']['sofaString']:
      article_id = "snes-04210"
      file_address = file_address.replace("snes-00862", article_id)
      json_data['_referenced_fss']['12']['sofaString'] = json_data['_referenced_fss']['12']['sofaString'].replace("snes-00862", article_id)
    
  
  with open(file_address, 'w', encoding='utf-8') as write_file:
      json.dump(json_data, write_file, indent=4)
      write_file.close()
    

def add_to_dir(filename, index, outer_file, output_dir):

  # For all inner files in the outer file:
  assert len(os.listdir(filename + '/' + outer_file)) == 1
  inner_file =  os.listdir(filename + '/' + outer_file)[0]
  # Unzip the inner file
  archive = ZipFile(filename + '/' + outer_file + '/' + inner_file, 'r')
  # For all files in the archive:
  assert len(list(filter(lambda x:x.filename.endswith('.json'), archive.filelist))) == 1
  file_name = list(filter(lambda x:x.filename.endswith('.json'), archive.filelist))[0].filename
  # Extract the relevant json file into the final folder, renaming it
  # with the current index to prevent duplicate names.
  archive.extract(file_name, 'data/'+output_dir)

  article_id = article_name_from_file('data/'+output_dir+'/admin.json')
  file_article_id = outer_file.split(" ")[1]
  if article_id is None:
    article_id = file_article_id
  final_address = f"data/{output_dir}/{article_id}-{index}.json"
  os.rename(f"data/{output_dir}/admin.json",final_address)
  return final_address
      

def clean_folder(address):
  address = os.path.join(os.getcwd(),address) 
  files = os.listdir(address)
  for f in files:
    file_address = os.path.join(address,f) 
    os.remove(file_address)


def prepare_files(filename):
  base_file_address = "/".join(filename.split("/")[:-1])
  clean_folder(base_file_address+"/anno_all/")
  for idx, outer_file in enumerate(tqdm(os.listdir(filename))):
    final_file_address = add_to_dir(filename, idx, outer_file, "anno_all")
    normalize_file(final_file_address)
 

def extract_data(directory):
  folder = "/".join(directory.split("/")[:-1])
  # This dictionary will contain every segment as keys, and their
  # respective annotations as values
  all_labels = {x:0 for x in get_annotation_keys() if x not in get_low_freq()}
  segment_dict = []
  article_dict = []
  file_label_dict = {}
    
  multifc_data = pd.read_csv(f"{folder}/all.tsv", sep="\t")

  multifc_data.columns = ["claimID", "claim", "label", "claimURL", "reason", "category", "speaker",
                  "checker", "tags", "title", "publishDate", "claimDate", "entities"]

  multifc_data = multifc_data.reset_index(drop=True)
  # Loop through the annotation folder
  for filename in tqdm(os.listdir(directory)):
    f = os.path.join(directory, filename)
    file_id = "-".join(filename.split("-")[:2])
    with open(f, encoding="UTF-8") as read_file:
      file = json.load(read_file, )

    # Extract the article text
    article = file['_referenced_fss']['12']['sofaString']
    article_cleaned = " ".join(re.split("Actual content:",article)).strip()
    article_cleaned = remove_file_id(article_cleaned)

    if "Persuasive_Labels" in file['_views']['_InitialView']:
      persuasive_labels = file['_views']['_InitialView']["Persuasive_Labels"]
    else:
      persuasive_labels = []
    
    claim = None
    search_result = multifc_data[multifc_data["claimID"].str.contains(file_id)]
    if len(search_result) > 0:
      claim = search_result["claim"].to_list()[0]
      claim_label = search_result["label"].to_list()[0]
      claim_label = normalize_claim_label(claim_label)
    else:
      print(f"{file_id}")
    

    file_label_dict[file_id] = 0
  

    # This dictionary contains all annotation signs as keys. Its values will be
    # lists of pairs, representing starting and ending indices where the article
    # is annotated by the corresponding label
    anno_dict = {}
    for anno_key in annotation_to_key.values():
      anno_dict[anno_key] = set()
    # Loop through the persuasive labels of this article
    # An example label is:
    # {'sofa': 12, 'begin': 3779, 'end': 3851, 'EmotionalAppeals': 'anger'}
    
    begin_to_end = {}
    end_to_begin = {}
    sentences = []
    # begin = 0
    # for sentence in nltk.tokenize.sent_tokenize(article):
    #   end = begin + len(sentence) + 1
    #   begin_to_end[begin] = end
    #   end_to_begin[end] = begin
    #   if "\r\nActual content:" in sentence:
    #     sentence = sentence.split("\r\nActual content:")[1]
    #     sentences = []
    #   if file_id in sentence:
    #     sentence = sentence.replace(file_id,"")
    #   sentences.append((begin,end,sentence))
    #   assert sentence in article[begin:end]
    #   begin = end
    # sentence_infos = file["_views"]["_InitialView"]["Sentence"]
    # for sentnce_info in sentence_infos:
    #     begin = sentnce_info.get("begin",0)
    #     end = sentnce_info.get("end",len(article))
    #     begin_to_end[begin] = end
    #     end_to_begin[end] = begin
    #     sentence = article[begin:end]
    #     if "\r\nActual content:" in sentence:
    #       sentence = sentence.split("\r\nActual content:")[1]
    #       sentences = []
    #     if file_id in sentence:
    #       sentence = sentence.replace(file_id,"")
    #     sentences.append((begin,end,sentence))
    import spacy
    tokenizer = spacy.load('en_core_web_sm')
    for span_obj in tokenizer(article).sents:
      sentence = str(span_obj).replace("\ufeff", "")
      begin = span_obj.start_char
      end = span_obj.end_char
      
      begin_to_end[begin] = end
      end_to_begin[end] = begin
      if "\r\nActual content:" in sentence:
        sentence = sentence.split("\r\nActual content:")[1]
        sentences = []
      if file_id in sentence:
        sentence = sentence.replace(file_id,"")
      sentences.append((begin,end,sentence))
      begin = end
    
    for label in persuasive_labels:
      # Define the significant values
      if label.get('begin'):
        start_idx = label.get('begin')
      else:
        label['begin'] = 0
        start_idx = 0
      
      if label.get('end'):
        end_idx = label.get('end')
      else:
        label['end'] = begin_to_end[start_idx]
        end_idx = begin_to_end[start_idx]
      
      annotations = list(filter(lambda x:x[0] not in ['end','sofa','begin'],label.items()))
      for key, annotation in annotations:
        anno_sign = annotation_to_key[annotation]
        anno_range = (start_idx, end_idx)
        anno_dict[anno_sign].add(anno_range)
        # If a segment is annotated with a child tag, it must also
        # be annotated with the associated parent tag
        if re.match("[0-9]+",anno_sign):
          new_sign = re.match("[0-9]+",anno_sign)[0]
          if new_sign not in get_low_freq():
            anno_dict[new_sign].add(anno_range)
        if re.match("[0-9]+[a-f]",anno_sign):
          new_sign = re.match("[0-9]+[a-f]",anno_sign)[0]
          if new_sign not in get_low_freq():
            anno_dict[new_sign].add(anno_range)
        if re.match("[0-9]+[a-f][iv]+",anno_sign):
          new_sign = re.match("[0-9]+[a-f][iv]+",anno_sign)[0]
          if new_sign not in get_low_freq():
            anno_dict[new_sign].add(anno_range)
        

    # senteces = nltk.sent_tokenize()
    for idx, (begin,end,sentence) in enumerate(sentences):
      # If the left context is out of range, leave it blank
      try:
        prefix_sentence_2 = sentences[idx - 2][2].strip() if (idx - 2) >= 0 else ""
      except:
        prefix_sentence_2 = ""
      try:
        prefix_sentence_1 = sentences[idx - 1][2].strip() if (idx - 1) >= 0 else ""
      except:
        prefix_sentence_1 = ""
      # If the right context is out of range, leave it blank
      try:
        suffix_sentence_1 = sentences[idx + 1][2].strip()
      except:
        suffix_sentence_1 = ""
      try:
        suffix_sentence_2 = sentences[idx + 2][2].strip()
      except:
        suffix_sentence_2 = ""
      segment = sentence.strip()

      labels = {key:0 for key in get_annotation_keys()}

      for label, ranges in anno_dict.items():
        if label not in get_low_freq():
          for range_values in ranges:
            if len({begin,end}.intersection(set(range(*range_values)))) > 0 or len({*range_values}.intersection(set(range(begin,end)))) > 0 :
              labels[label] = 1
              all_labels[label] += 1
              file_label_dict[file_id] += 1

      # Add the segment to the list, as well as the global dictionary
      segment_dict.append({
        "labels":labels,
        "sentence_id":idx,
        "claim_id":file_id,
        "claim":claim,
        "claim_label":claim_label,
        "sentence":segment,
        "prefix_sentence_2": prefix_sentence_2,
        "prefix_sentence_1": prefix_sentence_1,
        "suffix_sentence_1": suffix_sentence_1,
        "suffix_sentence_2": suffix_sentence_2,
        })
    article_dict.append({
      "claim_id":file_id,
      "claim":claim,
      "claim_label":claim_label,
      "article":article_cleaned,
      })
    
  # print(directory,all_labels)
  return segment_dict, article_dict


def create_xlsx_file(source_folder,target_file):
  segment_dict, article_dict = extract_data(source_folder)
  
  df = pd.DataFrame()
  df["claim_id"] = list(map(lambda x:x["claim_id"],segment_dict))
  df["claim"] = list(map(lambda x:x["claim"],segment_dict))
  df["claim_label"] = list(map(lambda x:x["claim_label"],segment_dict))
  df["sentence_id"] = list(map(lambda x:x["sentence_id"],segment_dict))
  df["sentence"] = list(map(lambda x:x["sentence"].strip(),segment_dict))
  df["prefix_sentence_2"] = list(map(lambda x:x["prefix_sentence_2"].strip(),segment_dict))
  df["prefix_sentence_1"] = list(map(lambda x:x["prefix_sentence_1"].strip(),segment_dict))
  df["suffix_sentence_1"] = list(map(lambda x:x["suffix_sentence_1"].strip(),segment_dict))
  df["suffix_sentence_2"] = list(map(lambda x:x["suffix_sentence_2"].strip(),segment_dict))

  for key in get_annotation_keys():
    df[key] = list(map(lambda x:x["labels"][key],segment_dict))
    
  df = df[df["sentence"].notna()]
  df = df[df["sentence"] != ""]
  df.to_excel(target_file)
  
  print(len(df))

  df = pd.DataFrame()
  df["claim_id"] = list(map(lambda x:x["claim_id"],article_dict))
  df["claim"] = list(map(lambda x:x["claim"],article_dict))
  df["claim_label"] = list(map(lambda x:x["claim_label"],article_dict))
  df["article"] = list(map(lambda x:x["article"],article_dict))
  
  df.to_excel(target_file.replace(".xlsx","_article.xlsx"))


def folder_check(folder):
  anno_dict = {x:0 for x in get_annotation_keys() if x not in get_low_freq()}
  for idx, file in enumerate(os.listdir(f"./data/{folder}/")):
    file_address = f"./data/{folder}/"+file
    with open(file_address, encoding="UTF-8") as read_file:
      file = json.load(read_file)
      # Extract the annotations
      # print(file['_views']['_InitialView'])
      persuasive_labels = file['_views']['_InitialView'].get("Persuasive_Labels",[])
      
      for label in persuasive_labels:
        annotations = [normalize_annotations(x[1]) for x in filter(lambda x:x[0] not in ['end','sofa','begin'],label.items())]
        for anno in annotations:
          anno_sign = annotation_to_key[anno]
          if re.match("[0-9]+",anno_sign):
            
            new_sign = re.match("[0-9]+",anno_sign)[0]
            if new_sign not in get_low_freq():
              anno_dict[new_sign] += 1
          
          if re.match("[0-9]+[a-f]",anno_sign):
            new_sign = re.match("[0-9]+[a-f]",anno_sign)[0]

            if new_sign not in get_low_freq():
              anno_dict[new_sign] += 1
          
          if re.match("[0-9]+[a-f][iv]+",anno_sign):
            new_sign = re.match("[0-9]+[a-f][iv]+",anno_sign)[0]
            if new_sign not in get_low_freq():
              anno_dict[new_sign] += 1
  # print(folder ,anno_dict)
  return min(anno_dict.values())
  

def find_proper_split(filename, split=0.2):
  base_file_address = "/".join(filename.split("/")[:-1])
  test_data_files=  ['tron-01430',
    'snes-05761',
    'tron-02208',
    'snes-05624',
    'snes-04217',
    'snes-03960',
    'tron-02266',
    'snes-01752',
    'tron-01803',
    'tron-01822',
    'snes-00512',
    'snes-02502',
    'snes-00321',
    'snes-06263',
    'snes-06122',
    'tron-01402',
    'snes-03533',
    'snes-05674',
    'snes-06100',
    'snes-01953',
    'snes-03207',
    'tron-01837',
    'tron-01390',
    'snes-02126',
    'tron-02250',
    'tron-01447',
    'tron-01811',
    'snes-02985',
    'snes-05709',
    'tron-02207',
    'tron-01438',
    'snes-01190',
    'snes-05953',
    'tron-02211',
    'snes-01085',
    'snes-01017',
    'snes-03921',
    'snes-00102',
    'snes-01467',
    'snes-01033',
    'tron-01437',
    'tron-02252',
    'snes-05837',
    'snes-01968',
    'snes-03632',
    'snes-01874',
    'tron-01784',
    'snes-05648',
    'snes-01236']
  clean_folder(base_file_address+"/anno_train/")
  clean_folder(base_file_address+"/anno_test/")
  for outer_file in os.listdir(filename):
      file_id = "-".join(outer_file.split("-")[:-1])
      if file_id in test_data_files:
        split = "test"
      else:
        split = "train"
      shutil.copyfile(f"{base_file_address}/anno_all/{outer_file}", f"{base_file_address}/anno_{split}/{outer_file}")
  if folder_check("anno_train") >= 3 and folder_check("anno_test") >= 3:
    print("Passed Fail Test")
    


def find_proper_split_old(filename, seed, split=0.2):
  base_file_address = "/".join(filename.split("/")[:-1])
  # all_min = folder_check("anno_all")   
  # print("all_min", all_min)

  while True:
    train_data, test_data = sklearn.model_selection.train_test_split(os.listdir(filename),test_size=split,random_state=seed)
    # print("Train",train_data)
    # print("Test", test_data)
    test_data_files=  ['tron-01430',
     'snes-05761',
     'tron-02208',
     'snes-05624',
     'snes-04217',
     'snes-03960',
     'tron-02266',
     'snes-01752',
     'tron-01803',
     'tron-01822',
     'snes-00512',
     'snes-02502',
     'snes-00321',
     'snes-06263',
     'snes-06122',
     'tron-01402',
     'snes-03533',
     'snes-05674',
     'snes-06100',
     'snes-01953',
     'snes-03207',
     'tron-01837',
     'tron-01390',
     'snes-02126',
     'tron-02250',
     'tron-01447',
     'tron-01811',
     'snes-02985',
     'snes-05709',
     'tron-02207',
     'tron-01438',
     'snes-01190',
     'snes-05953',
     'tron-02211',
     'snes-01085',
     'snes-01017',
     'snes-03921',
     'snes-00102',
     'snes-01467',
     'snes-01033',
     'tron-01437',
     'tron-02252',
     'snes-05837',
     'snes-01968',
     'snes-03632',
     'snes-01874',
     'tron-01784',
     'snes-05648',
     'snes-01236']
    
    sys.exit()
    print(test_data)
    break
    # For all files in the train split:
    # clean_folder(base_file_address+"/anno_train/")
    # for idx, outer_file in enumerate(train_data):
    #   shutil.copyfile(f"{base_file_address}/anno_all/{outer_file}", f"{base_file_address}/anno_train/{outer_file}")
    # # For all files in the annotation folder:
    # clean_folder(base_file_address+"/anno_test/")
    # for idx, outer_file in enumerate(test_data):
    #   shutil.copyfile(f"{base_file_address}/anno_all/{outer_file}", f"{base_file_address}/anno_test/{outer_file}")
    # if folder_check("anno_train") >= 3 and folder_check("anno_test") >= 3:
    #   break
    # else:
    #   seed = random.randint(0,999999999)
    #   print(seed)


if __name__ == "__main__":
  
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_address', type=str,default="./data/")
  parser.add_argument('--seed', type=int,default=165441453)#default=573439060)
  args = parser.parse_args()

  
  annotation_to_key = get_annotations()
  prepare_multiFC_files(args.data_address)
  prepare_files(os.path.join(args.data_address,"annotation"))
  
  # find_proper_split(os.path.join(args.data_address,"anno_all"), seed=args.seed)
  find_proper_split(os.path.join(args.data_address,"anno_all"))

  create_xlsx_file(os.path.join(args.data_address,"anno_all"),os.path.join(args.data_address,"all.xlsx"))
  create_xlsx_file(os.path.join(args.data_address,"anno_train"),os.path.join(args.data_address,"train.xlsx"))
  create_xlsx_file(os.path.join(args.data_address,"anno_test"),os.path.join(args.data_address,"test.xlsx"))


