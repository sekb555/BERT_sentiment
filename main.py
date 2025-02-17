from src.bert_model import BertTrain
import src.preprocess_data
import torch


bert = BertTrain()
while(1):
    text = input("Enter the text to predict: ")
    if text == '':
        break
    else:
        ppd = src.preprocess_data.PreprocessData()
        text = ppd.preprocess_text(str(text))
        bert.input_predict(text, torch.load("data/bert_model.pth"))

