from bert_model import BertTrain
import preprocess_data


bert = BertTrain()
while(1):
    text = input("Enter the text to predict: ")
    if text == '':
        break
    else:
        ppd = preprocess_data.PreprocessData(str(text))
        bert.input_predict(text)

