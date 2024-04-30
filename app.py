import pandas as pd
import numpy as np
import streamlit as st 
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import plotly.express as px 
import os
import requests
import dotenv
from dotenv import load_dotenv

load_dotenv()

st.title('Sentiment Analysis Tool')
st.subheader('HELLO USER')

text = st.text_input('Enter a comment')
click = st.button('Compute')

def senti(text):
    obj = SentimentIntensityAnalyzer()
    senti_dict = obj.polarity_scores(text)
    
    if senti_dict['compound'] >= 0.05:
        st.write("😁 Positive")
    elif senti_dict['compound'] <= -0.05:
        st.write("😥 Negative")
    else:
        st.write("🙂 Neutral")


API_URL = "https://api-inference.huggingface.co/models/j-hartmann/emotion-english-distilroberta-base"
headers = {"Authorization": "Bearer " + os.getenv('HUGGINGFACE_API')}

def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()

def senti_class(text):
    output = query({"inputs": [text]})
    if output is None:
        st.error("Failed to fetch data from the API")
        return None
    elif "error" in output:
        st.error("API Error: {}".format(output["error"]))
        return None
    else:
        print("API Response:", output)  # Print API response for debugging
        st.write("Emotion Table")
        st.table(pd.DataFrame(output))
        return output

def viz(o, text):
    obj = SentimentIntensityAnalyzer()
    senti_dict = obj.polarity_scores(text)
    labels = [item['label'] for item in o[0]]
    scores = [item['score'] for item in o[0]]
    labels.append('Neutral')
    labels.append('positive')
    labels.append('negative')
    scores.append(senti_dict['neu'])
    scores.append(senti_dict['pos'])
    scores.append(senti_dict['neg'])
    fig = px.bar(x=labels, y=scores)
    fig2 = px.pie(names=labels, values=scores)
    fig.update_layout(
        title="Sentiment Analysis",
        xaxis_title="Emotion",
        yaxis_title="Score"
    )

    st.plotly_chart(fig)
    st.plotly_chart(fig2)

if click:
    if not text:
        st.warning("Please enter a comment.")
    else:
        senti(text)
        o = senti_class(text)
        if o:
            viz(o, text)
