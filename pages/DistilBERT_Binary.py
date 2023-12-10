from transformers import AutoTokenizer, AutoModelForSequenceClassification
import streamlit as st
from scrapingbsf import make_data
# Load model directly
tokenizer = AutoTokenizer.from_pretrained(
    "Arjun24420/DistilBERT-FakeOrReal-BinaryClassification")
model = AutoModelForSequenceClassification.from_pretrained(
    "Arjun24420/DistilBERT-FakeOrReal-BinaryClassification")


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
    1: 'Reliable',
    0: 'Unreliable',
}

# Streamlit App
st.title("Binary Classification (DistilBERT-FakeOrReal)")

title_input = st.text_input("Enter Title")
text_input = st.text_area("Enter Text")

if st.button('Scrape'):
    scraped_df = make_data(title_input)
    scraped_df.dropna(inplace=True)
    st.dataframe(scraped_df)
# Predict Button
if st.button('Predict'):
    text_input = "<title>" + title_input + "<content>" + text_input + "<end>"
    result_class_probabilities = predict(text_input)
    st.write("Prediction:")

    # Sort class probabilities in descending order
    sorted_probs = sorted(result_class_probabilities.items(),
                          key=lambda x: x[1], reverse=True)

    # Display the sorted probabilities for each class
    st.subheader("Class Probabilities:")
    for class_label, prob in sorted_probs:
        st.write(f"{class_label}: {prob * 100}%")
        st.progress(prob)
