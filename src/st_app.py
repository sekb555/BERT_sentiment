import streamlit as st
from preprocess_data import PreprocessData
from bert_model import BertTrain
import torch



ppd = PreprocessData()
bert = BertTrain()


st.title("Sentiment Analysis App")
st.subheader("Sentiment analysis using BERT model")

if 'sentiment' not in st.session_state:
    st.session_state.sentiment = {
        'Positive': 0, 'Negative': 0, 'Neutral': 0}

text = st.text_input("### Enter a text to analyze:")

prob_negi = 0
prob_posi = 0
if text.strip() == "":
    st.markdown(f"""
        <p style="font-size:18px;" > Nothing entered. Please enter some text.</p>
    """, unsafe_allow_html=True)
    
else:
    model = torch.load("../data/bert_model.pth")
    text = ppd.preprocess_text(text)
    prediction = bert.input_predict(text, model)
    st.markdown(f"""
    <p style="font-size:18px;" >The sentiment of the text is {prediction}.</p>
    """, unsafe_allow_html=True)