from flair.data import Sentence
from flair.models import SequenceTagger
import streamlit as st


tagger = SequenceTagger.load("flair/ner-english-ontonotes-large")

# make example sentence
sentence = Sentence(
    "FPIs invest Rs 26,505 crore in Indian equities in just 6 trading sessions")

# predict NER tags
tagger.predict(sentence)

# print sentence
print(sentence)

# print predicted NER spans
print('The following NER tags are found:')
tags = sentence.get_spans('ner')
print(tags)
# # iterate over entities and print
# for entity in sentence.get_spans('ner'):
#     print(entity)
