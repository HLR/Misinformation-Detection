import os
import pandas as pd
import torch
import argparse
import random
from detection_model import RobertaClassifier
from detection_model import article_evaluate
from utils import set_seed
import numpy as np
class Config:
    dropout = 0.5
    max_length = 512
    num_labels = 2
    freeze_layers = 181
    longformer = False
    classifier_second_layer = None
    batch_size = 10
    epochs = 50
    learning_rate = 5e-5
    seed = random.randint(0,99999999)
    # seed = 9877

parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str,default="train")
parser.add_argument('--context', type=str,choices="none,low,high".split(","),default="none")
parser.add_argument('--source', type=str,choices=["claim","article","gt_strategy", "claim_article","pred_strategy","claim_gt","claim_article_gt" ,"claim_pred","claim_article_pred"])
parser.add_argument('--norm', type=int,choices=[0,1])
args = parser.parse_args()
context = args.context

# Declare the learning rate and batch size for detection training
article_config = Config()
set_seed(article_config.seed)


test_data = pd.read_csv(f"data/test_article_data_{context}.csv")
corr_labels = {True: "true", False: "false"}
test_data["label"] = test_data["label"].apply(lambda x: corr_labels[x])
test_data = test_data[test_data["label"] != "none"]

# Define the layer to train and test on.
# "article" for base article text
# "target_combined" for article and ground truth labels
# "pred_combined" for article and predicted labels

models_address = "./models"
column = args.source
macro_f1s = []
micro_f1s = []
min_loss = 10e10
print(column,context)
import re
for model_address in filter(lambda x:re.match(f"{column}_{context}_[0-9]+", x),os.listdir(path=models_address)):
    path = os.path.join(models_address, model_address)
    print(path)
    strategy_model = RobertaClassifier(article_config)
    loaded = torch.load(path)
    best_model_weights = loaded["best_model_weights"]
    strategy_model.load_state_dict(best_model_weights)
    vaL_loss, val_micro_f1, val_macro_f1 = article_evaluate(model=strategy_model, test_data=test_data, batch_size=article_config.batch_size, column=column, verbose=False)
    micro_f1s.append(val_micro_f1)
    macro_f1s.append(val_macro_f1)
    if vaL_loss < min_loss:
        min_loss =  vaL_loss
        print("saved")
        torch.save(loaded, os.path.join(models_address, f"{column}_{context}_best.pt"))
micro_f1s = np.array(micro_f1s)
macro_f1s = np.array(macro_f1s)
print("############","Average RESULT", "micro", f"{micro_f1s.mean():.3f}", "macro",f"{ macro_f1s.mean():.3f}")