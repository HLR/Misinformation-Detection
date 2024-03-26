import torch

from transformers import RobertaModel, LongformerModel

# Used to define the dataset and classifier
# objects
import torch
from torch import nn

class RobertaClassifier(nn.Module):

  def __init__(self, config):
    super(RobertaClassifier, self).__init__()
    if config.longformer:
      self.roberta = LongformerModel.from_pretrained("allenai/longformer-base-4096")
    else:
      self.roberta = RobertaModel.from_pretrained("roberta-base")

    if config.freeze_layers:
      for idx, (name,params) in enumerate(self.roberta.named_parameters()):
        if idx < config.freeze_layers:
          params.requires_grad = False
        # print(idx, name)
        # if name.startswith("encoder.layer") and int(name.split(".")[2]) > 7
    self.dropout = nn.Dropout(config.dropout)
    if config.classifier_second_layer:
      self.linear = nn.Linear(768, config.classifier_second_layer)
      torch.nn.init.xavier_uniform(self.linear.weight)  
      self.linear2 = nn.Linear(config.classifier_second_layer, config.num_labels)
      torch.nn.init.xavier_uniform(self.linear2.weight)
    else:
      self.linear = nn.Linear(768, config.num_labels)
      torch.nn.init.xavier_uniform(self.linear.weight)  
      self.linear2 = None

    self.relu = nn.ReLU()

  def forward(self, input_id, mask):
    output_1 = self.roberta(input_ids=input_id, attention_mask=mask)
    pooler = output_1.last_hidden_state[:, 0]
    dropout_output = self.dropout(pooler)
    linear_output = self.linear(dropout_output)
    if self.linear2:
      relu_layer = self.relu(linear_output)
      dropout_output2 = self.dropout(relu_layer)
      linear_output2 = self.linear2(dropout_output2)
      return linear_output2
    else:
      return linear_output

