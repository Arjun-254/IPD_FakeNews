from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import streamlit as st

# Load the pre-trained model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    "hamzab/roberta-fake-news-classification")

model = AutoModelForSequenceClassification.from_pretrained(
    "hamzab/roberta-fake-news-classification")


def predict_fake(title, text):
    input_str = "<title>" + title + "<content>" + text + "<end>"
    input_ids = tokenizer.encode_plus(
        input_str, max_length=512, padding="max_length", truncation=True, return_tensors="pt")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    with torch.no_grad():
        output = model(input_ids["input_ids"].to(
            device), attention_mask=input_ids["attention_mask"].to(device))
    return dict(zip(["Fake", "Real"], [x.item() for x in list(torch.nn.Softmax()(output.logits)[0])]))


# Streamlit App
st.title("Binary Classification (BERT)")

# Input Fields
title_input = st.text_input("Enter Title")
text_input = st.text_area("Enter Text")

# ... (previous code)

# Predict Button
if st.button('Predict'):
    result = predict_fake(title_input, text_input)
    st.write("Prediction:")

    # Assuming result[0] and result[1] are probabilities between 0 and 1
    progress_fake = int(result['Fake'] * 100)
    progress_real = int(result['Real'] * 100)

    # Create progress bars
    st.subheader("Predictions")
    st.write("Real = "+str(result['Real']*100)+"%")
    st.progress(progress_real)
    st.write("Fake = "+str(result['Fake']*100)+"%")
    st.progress(progress_fake)
