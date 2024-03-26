from segment_data import construct_segment_dict, construct_segment_dict_v2
from segment_data import convert_to_dataframe
from strategy_model import calculate_weights
from strategy_model import strategy_train
from strategy_model import strategy_evaluate
from models import RobertaClassifier
from configs import get_config

import torch
import argparse
from utils import set_seed

parser = argparse.ArgumentParser()
parser.add_argument('--test_article', type=int,default=0)
parser.add_argument('--mode', type=str,default="train")
parser.add_argument('--layer', type=str,choices="1,2,3,4".split(","),default="1")
parser.add_argument('--context', type=str,choices="none,low,high".split(","),default="none")
parser.add_argument('--seed', type=int,default=None)
args = parser.parse_args()
layer = args.layer
context = args.context
print(layer, context)


config = get_config(layer,context,seed=args.seed)
print(config.seed)
set_seed(config.seed)


path = f"models/{layer}_{context}_{config.seed}.pt"

# Create the train annotation dictionary
# train_dict = construct_segment_dict("data/anno_train", context)
# train = convert_to_dataframe(train_dict, layer=layer)

train = construct_segment_dict_v2("data/train.xlsx", context, layer)
weights = calculate_weights(train)
# Create the test annotation dictionary
# test_dict = construct_segment_dict("data/anno_test", context)

# Create the test dataframes
# test = convert_to_dataframe(test_dict, layer=layer)
test = construct_segment_dict_v2("data/test.xlsx", context, layer)

print("train sentences", len(train))
print("test sentences", len(test))

# Create the Roberta strategy models, the output length being the number
# of labels in that layer
model = RobertaClassifier(config)


# Declare the learning rate and batch size for strategy training
# Train and evaluate the 4 layer strategy models
strategy_train(model=model, train_data=train, learning_rate=config.learning_rate, epochs=config.epochs,
            batch_size=config.batch_size, context=context, layer=layer, class_weights=weights,config=config,test_data=test,path=path)
loaded = torch.load(path)
weights = loaded["weights"]
max_val_acc = loaded["max_val_acc"]
model.load_state_dict(weights)
print("test_data_length", len(test))
print("Test",layer,context, end="  ")
strategy_evaluate(model=model, test_data=test, context=context, layer=layer,config=config,verbose=True)
