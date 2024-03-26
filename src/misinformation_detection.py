import os
import pandas as pd
import torch
import argparse
import random
from segment_data import construct_segment_dict, construct_segment_dict_v2
from segment_data import convert_to_dataframe
from models import RobertaClassifier
from detection_model import article_train
from detection_model import article_evaluate
from detection_model import calculate_article_weights
from article_data import build_complete_dataset
from configs import get_config
from utils import set_seed
class Config:
    dropout = 0.5
    max_length = 512
    num_labels = 2
    freeze_layers = 181
    longformer = False
    classifier_second_layer = None
    batch_size = 10
    epochs = 40
    learning_rate = 7e-5
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


layer_models = []
for layer in ["1", "2", "3", "4"]:
    path = f"models/{layer}_{context}_best.pt"
    config = get_config(layer,context)

    # Create the test annotation dictionary
    test = construct_segment_dict_v2("data/test.xlsx", context, layer)


    model = RobertaClassifier(config)


    # Declare the learning rate and batch size for strategy training
    # Train and evaluate the 4 layer strategy models
    if os.path.exists(path):
        loaded = torch.load(path)
        weights = loaded["weights"]
        max_val_acc = loaded["max_val_acc"]
        model.load_state_dict(weights)
        # strategy_evaluate(model=model, test_data=test, context=context, layer=layer,config=config,verbose=True)
    layer_models.append(model)

if not os.path.exists(f"data/train_article_data_{context}.csv"):
    # Create the train article dataset (This takes a very long time)
    train_article_data = build_complete_dataset("data/anno_train",
                                                layer_models,
                                                context,
                                                article_config)
    train_article_data.to_csv(f"data/train_article_data_{context}.csv")
if not os.path.exists(f"data/test_article_data_{context}.csv"):
    # Create the test article dataset (This takes a very long time)
    test_article_data = build_complete_dataset("data/anno_test",
                                            layer_models,
                                            context,
                                            article_config)
    test_article_data.to_csv(f"data/test_article_data_{context}.csv")

# # Read the train and test article datasets into dataframes
train_data = pd.read_csv(f"data/train_article_data_{context}.csv")
train_data = train_data[train_data["label"] != "none"]

test_data = pd.read_csv(f"data/test_article_data_{context}.csv")
corr_labels = {True: "true", False: "false"}
test_data["label"] = test_data["label"].apply(lambda x: corr_labels[x])
test_data = test_data[test_data["label"] != "none"]

# Calculate the weights based on the training set
print()
weights = calculate_article_weights(train_data)
print("train",weights)
weights = calculate_article_weights(test_data)
print("test",weights)

# Define the layer to train and test on.
# "article" for base article text
# "target_combined" for article and ground truth labels
# "pred_combined" for article and predicted labels

column = args.source
path = f"models/{column}_{context}_{article_config.seed}.pt"
print(path)
# Define the fake news detection model

strategy_model = RobertaClassifier(article_config)
# Train and test the detection model
if args.mode == "train":
    article_train(model=strategy_model, train_data=train_data, learning_rate=article_config.learning_rate,
                epochs=article_config.epochs, batch_size=article_config.batch_size, weights=weights, column=column,
                test_data=test_data, path=path)
best_model_weights = torch.load(path)["best_model_weights"]
strategy_model.load_state_dict(best_model_weights)
print(column)
article_evaluate(model=strategy_model, test_data=test_data, batch_size=article_config.batch_size, column=column, verbose=True)


