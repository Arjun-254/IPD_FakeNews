from transformers import pipeline
import streamlit as st
from scrapingbsf import make_data
import pandas as pd

# Title
st.title("FactCC Demo")
text_input = st.text_area("Enter the News Headline")

# Initialize session_state if not already present
if 'scraped_df' not in st.session_state:
    st.session_state.scraped_df = None

if 'summary_input' not in st.session_state:
    st.session_state.summary_input = ""

# Function to generate the DataFrame
def dataframegen(text_input):
    scraped_df = make_data(text_input)
    scraped_df.dropna(inplace=True)
    st.session_state.scraped_df = scraped_df


if st.button("Scrape"):
    # Generate the DataFrame
    dataframegen(text_input)
    st.dataframe(st.session_state.scraped_df)
    st.session_state.summary_input = st.session_state.scraped_df['Content'][0]

if st.button("Classify"):
    # Load the FactCC model and increase max_length for auto-truncation
    pipe = pipeline(model="manueldeprada/FactCC", task="text-classification", max_length=512)
    #after 400 tokens slice the summary_input till the next fullstop
    if len(st.session_state.summary_input) > 400:
        st.session_state.summary_input = st.session_state.summary_input[:400] + st.session_state.summary_input[400:].split('.')[0] + '.'
    st.write(st.session_state.summary_input)
    # Perform text classification
    ans = pipe([[[text_input, st.session_state.summary_input]]], truncation='only_first', padding='max_length')

    # Display the result
    if ans[0]['label'] == 'INCORRECT':
        ans[0]['score'] = 1 - ans[0]['score']   
    st.subheader("Classification Result")
    st.text("Correctness: " + str(ans[0]['score']))
    st.progress(ans[0]['score'])
    st.text("Label: " + ans[0]['label'])
