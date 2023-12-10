from transformers import pipeline
import streamlit as st
from scrapingbsf import make_data
import pandas as pd

st.title("FactCC Text Classification")

# Create text and summary input areas
summary_input = ""
text_input = st.text_area("Enter the Headline:")

# Check if 'scraped_df' is already in the session state, otherwise initialize it
if 'scraped_df' not in st.session_state:
    st.session_state.scraped_df = pd.DataFrame()


def dataframegen(text_input):
    scraped_df = make_data(text_input)
    scraped_df.dropna(inplace=True)
    return scraped_df


if st.button('Scrape'):
    st.session_state.scraped_df = dataframegen(text_input)

if 'content' in st.session_state.scraped_df.columns:
    summary_input = st.session_state.scraped_df.iloc[2]['Content']


# Display the scraped_df
if not st.session_state.scraped_df.empty:
    st.dataframe(st.session_state.scraped_df)

# Check if text and summary are provided
if st.button("Classify"):
    # Load the FactCC model
    pipe = pipeline(model="manueldeprada/FactCC")
    # Perform text classification
    st.write(summary_input)
    ans = pipe([[[text_input, summary_input]]],
               truncation='only_first', padding='max_length')

    # Display the result
    if ans[0]['label'] == 'INCORRECT':
        ans[0]['score'] = 1 - ans[0]['score']
    st.subheader("Classification Result")
    st.text("Correctness: " + str(ans[0]['score']))
    st.progress(ans[0]['score'])
    st.text("Label: " + ans[0]['label'])
