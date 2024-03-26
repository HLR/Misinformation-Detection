import os
from typing import *
import sklearn
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from copy import deepcopy
from torch import nn
from models import RobertaClassifier
from utils import set_seed, get_collate_fn


# The Roberta classifiers use the Adam
# Optimizer
from torch.optim import Adam

# All models use a linear scheduler with warmup
from transformers import get_linear_schedule_with_warmup

# Used for output readability
from tqdm import tqdm

from sklearn.metrics import f1_score, classification_report
import random
SEED = random.randint(0,2**32 - 1)
set_seed(SEED)


class MultiFC(Dataset):
    def __init__(self, file_address:str="./data/multi-fc", mode:str="train"):
        # mode train,test,dev
        self.mode = mode
        self.file_address = file_address
        self.data = pd.read_csv(f"{file_address}/{mode}.tsv",sep="\t")
        if mode == "test":
            self.data.columns = ["claimID", "claim", "claimURL", "reason", "category", "speaker",
                  "checker", "tags", "title", "publishDate", "claimDate", "entities"]
        else:
            self.data.columns = ["claimID", "claim", "label", "claimURL", "reason", "category", "speaker",
                  "checker", "tags", "title", "publishDate", "claimDate", "entities"]
            self.data = self.data[self.data['label'].str.contains("None") == False]
        
        self.data = self.data[self.data['claimID'].str.contains("pomt")]
        self.labels = ['true', 'mostly false', 'false', 'full flop', 'half-true', 'mostly true','pants on fire!', 'half flip', 'no flip']
        self.data = self.data.reset_index(drop=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        claim_id = self.data["claimID"][idx]
        claim =  self.data["claim"][idx]
        try:
          snippets_file = pd.read_csv(f"{self.file_address}/snippets/{claim_id}",sep="\t",engine="python",encoding="utf8",quoting=3,error_bad_lines=False)
          snippets_file.columns = ["rank_position", "title", "snippet", "snippet_url"]
          snippets = " </s></s> ".join(map(str,snippets_file["snippet"]))
          claim =  claim  + " </s></s> " + snippets
        except FileNotFoundError:
          pass
        except pd.errors.EmptyDataError:
          pass

        if self.mode == "test":
            return claim
        label = self.data["label"][idx]
        label = self.labels.index(label)

        return claim, label
    
    def get_weights(self):
        result = self.data["label"].value_counts(normalize=True)
        weights = []
        for label in self.labels:
            weights.append(result[label])
        return 1 - torch.tensor(np.array(weights))



# Training loop for the multifc modelling
def multifc_train(model, train_dataset, test_dataset, config, weights):
  # Define the multifc dataset for training
  path = f"models/multifc_{SEED}.pt"
  if os.path.exists(path):
    checkpoint = torch.load(path)
    best_model_weights = checkpoint["best_model_weights"]
    max_val_acc = checkpoint["max_val_acc"]
  else:
    best_model_weights = model.state_dict()
    max_val_acc = -10e10

  model.load_state_dict(best_model_weights)
  train_dataset,dev_dataset = sklearn.model_selection.train_test_split(train_dataset,test_size=0.1,random_state=SEED)

  train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=config.batch_size, collate_fn=get_collate_fn(),shuffle=True)

  # Use GPU if available
  use_cuda = torch.cuda.is_available()
  device = torch.device("cuda" if use_cuda else "cpu")

  # Define the loss function as weighted cross entropy
  criterion = nn.CrossEntropyLoss(weight=weights.float())

  # Define the optimizer as Adam
  optimizer = Adam(model.parameters(), lr=config.learning_rate)

  # Define the scheduler as linear with warmup
  scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=len(train_dataloader),
    num_training_steps=len(train_dataloader) * 100
  )

  if use_cuda:
    model = model.cuda()
    criterion = criterion.cuda()

  for epoch_num in range(config.epochs):
    model.train()
    train_output = []
    train_targets = []
    train_loss = 0

    for train_input, train_label in tqdm(train_dataloader):
      train_label = train_label.to(device)
      mask = train_input['attention_mask'].to(device)
      input_id = train_input['input_ids'].squeeze(1).to(device)
      model_batch_result = model(input_id, mask)

      batch_loss = criterion(model_batch_result, train_label.long())
      train_output.extend(np.argmax(model_batch_result.detach().cpu().numpy(), axis=1))
      train_targets.extend(train_label.detach().cpu().numpy())

      train_loss += batch_loss

      model.zero_grad()
      batch_loss.backward()
      optimizer.step()
      scheduler.step()

    print("epoch:{:2d} training: "
          "micro f1: {:.3f} "
          "macro f1: {:.3f} "
          "loss: {:.3f} ".format(epoch_num + 1,
                                 f1_score(y_true=train_targets,
                                          y_pred=train_output,
                                          average='micro',
                                          zero_division="warn"),
                                 f1_score(y_true=train_targets,
                                          y_pred=train_output,
                                          average='macro',
                                          zero_division="warn"),
                                 train_loss / len(train_dataloader)))

    val_acc, _ = multifc_evaluate(model, dev_dataset, config)
    if max_val_acc < val_acc:
      print("SAVED")
      max_val_acc = val_acc
      torch.save({
        "max_val_acc": max_val_acc,
        "best_model_weights": deepcopy(model.state_dict())
        },
        path)
def multifc_evaluate(model, dev_dataset, config ,verbose=False):
  model.eval()
  test_dataloader = torch.utils.data.DataLoader(dev_dataset,collate_fn=get_collate_fn(), batch_size=config.batch_size)

  use_cuda = torch.cuda.is_available()
  device = torch.device("cuda" if use_cuda else "cpu")
  criterion = nn.CrossEntropyLoss()

  if use_cuda:
    model = model.cuda()
    criterion = criterion.cuda()
  test_loss = 0
  with torch.no_grad():
    test_output = []
    test_targets = []

    for test_input, test_label in test_dataloader:
      test_label = test_label.to(device)
      mask = test_input['attention_mask'].to(device)
      input_id = test_input['input_ids'].squeeze(1).to(device)

      model_batch_result = model(input_id, mask)
      test_output.extend(np.argmax(model_batch_result.cpu().numpy(), axis=1))
      test_targets.extend(test_label.cpu().numpy())
      batch_loss = criterion(model_batch_result.float(), test_label)
      test_loss += batch_loss

  macro_f1 = f1_score(y_true=test_targets,
                                      y_pred=test_output,
                                      average='macro',
                                      zero_division="warn")
  micro_f1 = f1_score(y_true=test_targets,
                                      y_pred=test_output,
                                      average='micro',
                                      zero_division="warn")
  if verbose:
    print(classification_report(test_targets, test_output, target_names=dev_dataset.labels, digits=3))
  print("Validate micro f1: {:.3f} "
      "macro f1: {:.3f} "
      "loss: {:.4f} "
      .format(micro_f1,macro_f1,test_loss/len(test_dataloader)))
  return macro_f1 ,micro_f1


def generate_data(model, dev_dataset, config):
    model.eval()
    test_dataloader = torch.utils.data.DataLoader(dev_dataset, collate_fn=get_collate_fn(),batch_size=config.batch_size)
    labels = ['true', 'mostly false', 'false', 'full flop', 'half-true', 'mostly true','pants on fire!', 'half flip', 'no flip']

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    if use_cuda:
        model = model.cuda()
    with torch.no_grad():
        test_output = []
        for test_input in test_dataloader:
            mask = test_input['attention_mask'].to(device)
            input_id = test_input['input_ids'].squeeze(1).to(device)
            model_batch_result = model(input_id, mask)
            test_output.extend(np.argmax(model_batch_result.cpu().numpy(), axis=1))
    labels_str = list(map(lambda x:labels[x],test_output))
    open("test.predict","w").write("\n".join(labels_str))


class Config:
    dropout = 0.5
    num_labels = 9
    freeze_layers = None #133#149#165#181#197 #181
    learning_rate = 5e-5
    batch_size = 10
    epochs = 12
    longformer = False
    classifier_second_layer = None #128
if __name__ == "__main__":
  config = Config()
  train_dataset = MultiFC(mode="train")
  dev_dataset = MultiFC(mode="dev")
  test_dataset = MultiFC(mode="test")

  model = RobertaClassifier(config)
  weights = train_dataset.get_weights()
  print(len(dev_dataset))
  # multifc_train(model, train_dataset, dev_dataset, config, weights)
  # best_model_weights = torch.load(f"models/multifc_{SEED}.pt")["best_model_weights"]
  # model.load_state_dict(best_model_weights)
  # multifc_evaluate(model, dev_dataset, config,verbose=True)
  # # generate_data(model, test_dataset, config)