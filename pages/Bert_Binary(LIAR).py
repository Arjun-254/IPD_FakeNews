from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import streamlit as st

# Load the pre-trained model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    "Arjun24420/BERT-FakeNews-BinaryClassification")
model = AutoModelForSequenceClassification.from_pretrained(
    "Arjun24420/BERT-FakeNews-BinaryClassification")


def predict(text):
    # Tokenize the input text and move tensors to the GPU if available
    inputs = tokenizer(text, padding=True, truncation=True,
                       max_length=512, return_tensors="pt")

    # Get model output (logits)
    outputs = model(**inputs)

    probs = outputs.logits.softmax(1)
    # Get the probabilities for each class
    class_probabilities = {class_mapping[i]: probs[0, i].item()
                           for i in range(probs.shape[1])}

    return class_probabilities


# Define class labels mapping
class_mapping = {
    0: 'reliable',
    1: 'unreliable',
}

# Streamlit App
st.title("Binary Classification (BERT - LIAR Dataset)")

# Input Fields
text_input = st.text_area("Enter Text")

# Predict Button
if st.button('Predict'):
    result_class_probabilities = predict(text_input)
    st.write("Prediction:")

    # Sort class probabilities in descending order
    sorted_probs = sorted(result_class_probabilities.items(),
                          key=lambda x: x[1], reverse=True)

    # Display the sorted probabilities for each class
    st.subheader("Class Probabilities:")
    for class_label, prob in sorted_probs:
        st.write(f"{class_label}: {prob * 100:.2f}%")
        st.progress(prob)
