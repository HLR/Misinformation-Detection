import torch
import numpy as np
import random 
import os


def set_seed(SEED):
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(SEED)
    os.environ["PYTHONHASHSEED"] = str(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_collate_fn(longformer=False):
    tokenizer = get_tokenizer(longformer)
    def collate_fn(batch):
        """
        data: is a list of tuples with (claim, label)
                where 'example' is a tensor of arbitrary shape
                and label/length are scalars
        """
        trans = lambda text: text if not isinstance(text,float) else ""
        batch = [(trans(item[0]),item[1]) for item in batch]
        texts,labels = torch.utils.data.default_collate(batch)
        texts = tokenizer(texts,
                            padding=True,
                            max_length=4096 if longformer else 512,
                            truncation=True,
                            return_tensors="pt")
        return texts, labels
    return collate_fn


def get_tokenizer(longformer=False):
    from transformers import LongformerTokenizer, RobertaTokenizer
    if longformer:
        tokenizer = LongformerTokenizer.from_pretrained("allenai/longformer-base-4096")
    else:
        tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    return tokenizer


def strat_pred(model, sequence, context, config):
  device = torch.device("cuda")
  model = model.to(device)
  model.eval()
  tokenizer = get_tokenizer(config.longformer)
  tokens = tokenizer(sequence, padding=True,
                     max_length=config.max_length,
                     truncation=True, return_tensors="pt")
  input = tokens["input_ids"].squeeze(1).to(device)
  mask = tokens["attention_mask"].to(device)
  result = model(input, mask)

  return result.detach().cpu().numpy()[0]