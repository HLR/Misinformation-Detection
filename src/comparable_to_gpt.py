
from transformers import LongformerTokenizerFast, LongformerForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import Dataset
import torch
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, classification_report
import pandas as pd
import sklearn.model_selection

import torch
import torch.nn as nn
from transformers import LongformerForSequenceClassification

def get_data(mode="train"):
    data = pd.read_excel(f"./data/{mode}.xlsx")
    article_data = pd.read_excel(f"./data/{mode}_article.xlsx")
    grouped = data.groupby("claim_id").sum().reset_index()
    grouped[grouped.columns[3:]] = grouped[grouped.columns[3:]].applymap(lambda x: 1 if x > 1 else x)
    grouped["article"] = grouped["claim_id"].map(lambda x: article_data[article_data["claim_id"] == x]["article"].values[0])
    columns = list(filter(lambda x: x.isnumeric() or x == "article",grouped.columns))
    
    return grouped[columns]
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions >= 0.5  # Using 0.5 as threshold
    report = classification_report(labels, preds, output_dict=True)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='macro')
    acc = accuracy_score(labels, preds)

    return {
        'accuracy': acc, 
        'f1': f1, 
        'precision': precision, 
        "report": report,
    }


class CustomDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        self.len = len(dataframe)
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __getitem__(self, index):
        article = str(self.data.article[index])
        inputs = self.tokenizer.encode_plus(
            article,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            return_token_type_ids=False,
            truncation=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']

        # Assuming labels are in columns '1' to '12'
        labels = self.data.iloc[index, 0:12].values
        labels = list(map(float, labels))  # Ensure labels are float

        return {
            'input_ids': torch.tensor(ids, dtype=torch.long),
            'attention_mask': torch.tensor(mask, dtype=torch.long),
            'labels': torch.tensor(labels, dtype=torch.float)
        }

    def __len__(self):
        return self.len


if __name__ == "__main__":
    
    train_df = get_data("train")
    train_df, val_df = sklearn.model_selection.train_test_split(train_df, test_size=0.2)

    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)

    test_df = get_data("test")
    # Assuming 'train_df' and 'test_df' are your Pandas dataframes
    # Assuming 'train_df' has label columns named '1' to '12'
    label_columns = [str(i) for i in range(1, 13)]
    class_counts = train_df[label_columns].sum()
    import numpy as np
    # Use max to ensure we don't divide by zero
    adjusted_counts = np.maximum(class_counts, 2)
    
    # Calculate inverse frequency for each class
    inverse_frequency = len(train_df) / adjusted_counts
    # Compute sample weights (simple example, customize as needed)
    sample_weights = train_df[label_columns].dot(inverse_frequency)
    # Convert to a tensor
    sample_weights_tensor = torch.from_numpy(sample_weights.to_numpy())
    from torch.utils.data import WeightedRandomSampler
    
    # Initialize the sampler
    sampler = WeightedRandomSampler(
        weights=sample_weights_tensor,
        num_samples=len(sample_weights_tensor),
        replacement=True
    )

    ##########
    
    tokenizer = LongformerTokenizerFast.from_pretrained('allenai/longformer-base-4096')
    MAX_LEN = 4096  # Adjust as needed
    train_dataset = CustomDataset(train_df, tokenizer, MAX_LEN)
    val_dataset = CustomDataset(val_df, tokenizer, MAX_LEN)
    test_dataset = CustomDataset(test_df, tokenizer, MAX_LEN)
    
    model = LongformerForSequenceClassification.from_pretrained(
        'allenai/longformer-base-4096',
        num_labels=12,  # Assuming 12 labels to predict
        problem_type="multi_label_classification",  # Important for multi-label classification
    )
    
    
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=10,
        per_device_train_batch_size=4,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        evaluation_strategy="epoch",
    )
    

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )
    
    trainer.train()
    validation_result = trainer.evaluate(val_dataset)
    print("Validation Result")
    print(validation_result)
    test_result = trainer.evaluate(test_dataset)
    print("Test Result")
    print(test_result)
    

