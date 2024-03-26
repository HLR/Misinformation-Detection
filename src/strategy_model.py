import os
import numpy as np
import sklearn
import random
import torch
from sklearn.metrics import classification_report, f1_score
from torch import nn
from torch.optim import Adam
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup


from annotations import get_annotation_layer_names
from segment_data import SegmentDataset
from utils import get_collate_fn

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

g = torch.Generator()
g.manual_seed(0)


# Find the values used in weighted loss. The formula used is:
# ((1 - # of instances per class) / Total Instances)
def calculate_weights(data):
  counts = []
  if "hasAnno" in data.columns:
    for i in data["hasAnno"].value_counts()[::-1]:
      counts.append(i)
  else:
    for i in data.columns[2:]:
      counts.append(sum(data[i].values))

  num_instances = sum(counts)
  class_weights = 1 - (torch.tensor(counts, dtype=torch.float64) / num_instances)

  return class_weights


# Edit the predicted output into a workable format
def calculate_metrics(pred, layer, threshold=0.5):

  # If layer 1, change the prediction output to be the argmax
  if layer == "1":
    new_pred = []
    for prediction in pred:
      new_pred.append(np.argmax(prediction))
    pred = new_pred

  # If any other layer, binarize the output using the threshold.
  elif layer in ["2", "3", "4"]:
    pred = np.array(pred > threshold, dtype=float)

  return pred


def strategy_train(model, train_data, learning_rate, epochs,
                  batch_size, context, layer, class_weights,config,test_data,path):
  max_val_acc = -10e10
  best_model_weights = model.state_dict()

  # Define the segment dataset for training
  train_data, dev_data = sklearn.model_selection.train_test_split(train_data,test_size=0.1,)
  print("train_data_length", len(train_data))
  print("dev_data_length", len(dev_data))

  train = SegmentDataset(train_data, config)

  # Define the train dataloader
  train_dataloader = torch.utils.data.DataLoader(train, batch_size=batch_size,collate_fn=get_collate_fn(), shuffle=True,
                                                    worker_init_fn=seed_worker,generator=g)

  # Use GPU if available
  use_cuda = torch.cuda.is_available()
  device = torch.device("cuda" if use_cuda else "cpu")

  # Define the loss function as weighted cross entropy, or binary cross entropy
  if layer == "1":
    criterion = nn.CrossEntropyLoss(weight=class_weights.float())
  elif layer in ["2", "3", "4"]:
    criterion = nn.BCEWithLogitsLoss(weight=class_weights.float())
  else:
    raise Exception("Invalid Layer given (must be '1', '2', '3' or '4')")

  # Define the optimizer as Adam
  optimizer = Adam(model.parameters(), lr=learning_rate)

  # Define the scheduler as linear with warmup
  scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=len(train_dataloader),
    num_training_steps=len(train_dataloader) * epochs
  )
  if os.path.exists(path):
    loaded = torch.load(path)
    weights = loaded["weights"]
    max_val_acc = loaded["max_val_acc"]
    model.load_state_dict(weights)

  if use_cuda:
    model = model.cuda()
    criterion = criterion.cuda()

  for epoch_num in range(epochs):

    train_output = []
    train_targets = []
    train_loss = 0
    model.train()

    # for train_input, train_label in tqdm(train_dataloader):
    for train_input, train_label in tqdm(train_dataloader):
      train_label = train_label.to(device)
      mask = train_input['attention_mask'].to(device)
      input_id = train_input['input_ids'].squeeze(1).to(device)
      model_batch_result = model(input_id, mask)

      if layer == "1":
        batch_loss = criterion(model_batch_result.float(), train_label.long().squeeze(1))
      else:
        batch_loss = criterion(model_batch_result.float(), train_label)

      train_output.extend(model_batch_result.detach().cpu().numpy())
      train_targets.extend(train_label.detach().cpu().numpy())

      train_loss += batch_loss

      model.zero_grad()
      batch_loss.backward()
      optimizer.step()
      scheduler.step()
    

    train_pred = calculate_metrics(np.array(train_output), layer)
    print("epoch:{:2d} training: "
          "micro f1: {:.3f} "
          "macro f1: {:.3f} "
          "loss: {:.5f} ".format(epoch_num + 1,
                                 f1_score(y_true=train_targets,
                                          y_pred=train_pred,
                                          average='micro',
                                          zero_division="warn"),
                                 f1_score(y_true=train_targets,
                                          y_pred=train_pred,
                                          average='macro',
                                          zero_division="warn"),
                                 train_loss / len(train_data)))
    print("Validation",layer,context, end="  ")
    val_acc, _ = strategy_evaluate(model, dev_data, context, layer, config)
    print("Test_VALID",layer,context, end="  ")
    # strategy_evaluate(model,test_data,context,layer,config)
    if max_val_acc <= val_acc:
      best_model_weights = model.state_dict()
      max_val_acc = val_acc
      print("SAVED")
      torch.save({
        "weights":best_model_weights,
        "max_val_acc":max_val_acc
      },path)


def strategy_evaluate(model, test_data, context, layer,config=None,verbose=False,return_predictions=False):
  model.eval()
  test = SegmentDataset(test_data, config)
  test_dataloader = torch.utils.data.DataLoader(test,collate_fn=get_collate_fn(), batch_size=2,worker_init_fn=seed_worker,generator=g)

  use_cuda = torch.cuda.is_available()
  device = torch.device("cuda" if use_cuda else "cpu")

  if layer == "1":
    criterion = nn.CrossEntropyLoss()
  elif layer in ["2", "3", "4"]:
    criterion = nn.BCEWithLogitsLoss()
  else:
    raise Exception("Invalid Layer given (must be '1', '2', '3' or '4')")
  
  if use_cuda:
    model = model.cuda()
  
  with torch.no_grad():
    test_output = []
    test_targets = []
    test_loss = 0
    for test_input, test_label in test_dataloader:
      test_label = test_label.to(device)
      mask = test_input['attention_mask'].to(device)
      input_id = test_input['input_ids'].squeeze(1).to(device)

      model_batch_result = model(input_id, mask)
      if criterion:
        if layer == "1":
          batch_loss = criterion(model_batch_result.float(), test_label.long().squeeze(1))
        else:
          batch_loss = criterion(model_batch_result.float(), test_label)
        test_loss += batch_loss
      test_output.extend(model_batch_result.cpu().numpy())
      test_targets.extend(test_label.long().squeeze(1).cpu().numpy())

  pred = calculate_metrics(np.array(test_output), layer)

  if layer == "1":
    key_to_label = {
      0: 'No Annotation',
      1: 'Has Annotation'
    }
  elif layer in ["2","3","4"]:
    key_to_label = get_annotation_layer_names(layer)
  

  labels = []
  label_set = set()
  if layer == "1":
    labels = ["NoAnno", "HasAnno"]
  else:
    for i in test_targets:
      winner = np.argwhere(i == np.amax(i))
      for j in winner.flatten().tolist():
        label_set.add(j)
    for i in pred:
      winner = np.argwhere(i == np.amax(i))
      for j in winner.flatten().tolist():
        label_set.add(j)
    
    for i in sorted(label_set):
      labels.append(key_to_label[i])
  if verbose:
    print(classification_report(test_targets, pred, target_names=labels, digits=3))
  mean_loss = test_loss / len(test)
  micro_f1 = f1_score(y_true=test_targets,
                                          y_pred=pred,
                                          average='micro',
                                          zero_division="warn")
  macro_f1 = f1_score(y_true=test_targets,
                                          y_pred=pred,
                                          average='macro',
                                          zero_division="warn")
  if return_predictions:
    return pred
  # print("Validation : "
  #         "micro f1: {:.3f} "
  #         "macro f1: {:.3f} "
  #         "loss: {:.3f} ".format(micro_f1, macro_f1, mean_loss))
  return macro_f1, micro_f1
  
