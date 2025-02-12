import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from custom_dataset import CustomDataset


class BERT(torch.nn.Module):

    def __init__(self):
        super(BERT, self).__init__()
        self.model = BertForSequenceClassification.from_pretrained(
            'bert-base-uncased')

    def forward(self, ids, mask, token_type_ids):
        output = self.model(ids, attention_mask=mask,
                          token_type_ids=token_type_ids)
        return output.logits


class BertTrain:

    def __init__(self):
        self.model = BERT().to(device)
        self.max_len = 64
        self.file = "data/processed_data.csv"
        self.epochs = 5
        self.learning_rate = 1e-3
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

                loss.backward()
                optimizer.step()

                if i % 500 == 0:
                    print("Batch ", i, " Loss: ", loss.item())
                i += 1

            print(f"Epoch {epoch} completed")
            
        self.model.eval()

        

    def save_model(self):
        torch.save(self.model.state_dict(), "data/bert_model.pth")
        print("Model saved!")
        

        
        
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(device)
    
    bert = BertTrain()
    bert.load_data()
    bert.train()
    bert.save_model()
