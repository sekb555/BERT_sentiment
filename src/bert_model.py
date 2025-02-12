import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.metrics import classification_report
import transformers
import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer, BertModel, BertConfig, AdamW
from custom_dataset import CustomDataset

device = torch.accelerator.current_accelerator(
).type if torch.accelerator.is_available() else "cpu"


class BERT(torch.nn.Module):

    def __init__(self):
        super(BERT, self).__init__()
        self.l1 = BertModel.from_pretrained(
            "bert-base-uncased", attn_implementation="sdpa")
        self.l2 = torch.nn.Dropout(0.3)
        self.l3 = torch.nn.Linear(768, 1)

    def forward(self, ids, mask, token_type_ids):
        outputs = self.l1(ids, attention_mask=mask, token_type_ids=token_type_ids)
        output_1 = outputs.last_hidden_state[:, 0, :]
        output_2 = self.l2(output_1)
        output = self.l3(output_2)
        return output


class BertTrain:

    def __init__(self):
        self.model = BERT().to(device)
        self.max_len = 100
        self.file = "data/processed_data_testing.csv"
        self.epochs = 1
        self.learning_rate = 1e-05
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def read_data(self):
        df = pd.read_csv(self.file, encoding="utf-8")
        df['Processed_Tweets'] = df['Processed_Tweets'].astype(str)
        return df

    def load_data(self):
        df = self.read_data()

        train_data = df.sample(frac=0.8, random_state=42)
        test_data = df.drop(train_data.index).reset_index(drop=True)
        train_data = train_data.reset_index(drop=True)

        train_set = CustomDataset(train_data, self.tokenizer, self.max_len)
        test_set = CustomDataset(test_data, self.tokenizer, self.max_len)

        train_params = {'batch_size': 32,
                        'shuffle': True,
                        'num_workers': 2
                        }

        test_params = {'batch_size': 32,
                       'shuffle': True,
                       'num_workers': 2
                       }

        self.train_dataloader = DataLoader(train_set, **train_params)
        self.test_dataloader = DataLoader(test_set, **test_params)

    def train(self):
        self.model.train()
        optimizer = torch.optim.AdamW(
            params=self.model.parameters(), lr=self.learning_rate)
        loss_fn = torch.nn.CrossEntropyLoss()
        
        for epoch in range(self.epochs):
            i = 0
            for batch in (self.train_dataloader):
                ids = batch["ids"].to(device)
                mask = batch["mask"].to(device)
                token_type_ids = batch["token_type_ids"].to(device)
                targets = batch["targets"].to(device)
                
                outputs = self.model(ids, mask, token_type_ids)
                
                optimizer.zero_grad()
                loss = loss_fn(outputs, targets)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                print("Batch: ", i, " Loss: ", loss.item())
            
            print(f"Epoch {epoch} completed")
        
        self.model.eval()    
                 
if __name__ == '__main__':
    bert = BertTrain()
    bert.load_data()
    bert.train()
