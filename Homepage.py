import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

# Set page configuration
st.set_page_config(page_title="Home Page", page_icon="🏡")

# Apply tile styles globally
st.markdown("""
    <style>
        /* General tile styles */
        .tile {
            border: 1px solid black;
            border-radius: 10px;
            padding: 20px;
            margin: 10px 0;
            transition: all 0.3s ease-in-out;
            text-align: center;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
            height: 100%;
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
        }
        .tile:hover {
            transform: scale(1.05);
            box-shadow: 0 6px 12px rgba(0, 0, 0, 0.2);
        }
        .heading {
            font-size: 24px;
            font-weight: bold;
            margin-bottom: 10px;
        }
        .subheading {
            font-size: 16px;
            color: #666;
            line-height: 1.5;
        }
    </style>
""", unsafe_allow_html=True)

# Main heading of the page
st.markdown("<h1 class='heading' style='text-align: center;' >SMABOT</h1>", unsafe_allow_html=True)
# Button to navigate to chatbot page

# Tile 1: What is SMABOT?
st.markdown("""
    <div class='tile'>
        <div class='heading'>What is SMABOT?</div>
        <p>SMABOT (Sentiment. Market. Analysis. Robot) <br> Is a chatbot that performs sentiment analysis on finance and stock market-related news and tweets.</p>
    </div>
""", unsafe_allow_html=True)

# Tile 2: Technology Behind SMABOT
st.markdown("""
    <div class='tile'>
        <div class='heading'>Model Architechture</div>
        <p>Built using a pre-trained RoBERTa model for tweets, fine-tuned with additional datasets on companies and market news using Hugging Face's pipeline.</p>
    </div>
""", unsafe_allow_html=True)

# Tile 3: Additional Technologies Used
st.markdown("""
    <div class='tile'>
        <div class='heading'>Tech Stack</div>
        <p> <b>spaCy</b> for Named Entity Recognition (NER) and fuzzy matching.<br> <b>Hugging Face</b> for Natural Language Processing (NLP) models.</p>
    </div>
""", unsafe_allow_html=True)

# Tile 4: Model Performance
st.markdown("""
    <div class='tile'>
        <div class='heading'>Model Performance</div>
        <p>SMABOT achieves an accuracy of <b>64%</b> in sentiment classification.</p>
    </div>
""", unsafe_allow_html=True)

# Tile 5: Datasets Used
st.markdown("""
    <div class='tile'>
        <div class='heading'>Dataset</div>
        <p>SMABOT is trained on datasets including companies listed in<br>  <b>S&P 500</b> <b>NASDAQ</b> <b>Cryptocurrency</b></p>
    </div>
""", unsafe_allow_html=True)

# Tile 6: Sentiment Categories
st.markdown("""
    <div class='tile'>
        <div class='heading'>Sentiment Categories</div>
        <p>The model is trained to classify sentiments into three categories<br> <b>Positive</b> <b>Neutral</b> <b>Negative</b></p>
    </div>
""", unsafe_allow_html=True)



# Tile 8: Model Accuracy Over Time (Graph)
st.markdown("""
    <div class='tile'>
        <div class='heading'>Stats</div>
        <p>Graph demonstrating model statistics on sentiment performance</p>
    </div>
""", unsafe_allow_html=True)

import pandas as pd
import streamlit as st

# Data for graph
data = pd.DataFrame({
    "Sentiment": ["Negative", "Neutral", "Positive"],
    "Precision": [0.78, 0.64, 0.59],
    "Recall": [0.23, 0.87, 0.34],
    "F1-Score": [0.35, 0.74, 0.43]
})

# Set "Sentiment" as the index for better visualization
data.set_index("Sentiment", inplace=True)

# Plot bar chart using Streamlit's API
st.bar_chart(data)

if st.button("Try SMABOT Now ↗️"):
    st.switch_page("pages/SMABOT.py")

