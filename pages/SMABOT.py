import streamlit as st
import pandas as pd
import logging
from transformers import pipeline
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
import spacy
import re

st.set_page_config(page_title="SMABOT Page", page_icon="🤖")
# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
# Load sentiment analysis model (better for tweets)
sentiment_pipeline = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment")

# Load and preprocess datasets
def load_data():
    try:
        sentiment_data = pd.read_csv("all-data.csv", encoding="ISO-8859-1", header=None, names=["Sentiment", "Text"])
        company_data = pd.read_csv("company_names.csv", encoding="ISO-8859-1")  # Load company names dataset
        crypto_data = pd.read_csv("crypto.csv", encoding="ISO-8859-1")  # Load cryptocurrency dataset
        sp500_data = pd.read_csv("sp500_companies.csv", encoding="ISO-8859-1")  # Load S&P 500 companies dataset

        sentiment_data.dropna(inplace=True)
        company_data.dropna(inplace=True)
        crypto_data.dropna(inplace=True)
        sp500_data.dropna(inplace=True)

        logging.info("Datasets loaded successfully.")
        return sentiment_data, company_data, crypto_data, sp500_data
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        return None, None, None, None

# Load datasets
sentiment_data, company_data, crypto_data, sp500_data = load_data()

# Initialize session state
if "awaiting_text" not in st.session_state:
    st.session_state.awaiting_text = False

# Function to match sentiment from dataset
def match_sentiment(tweet):
    matched_sentiment = sentiment_data[sentiment_data["Text"] == tweet]
    if not matched_sentiment.empty:
        return matched_sentiment.iloc[0]["Sentiment"]
    return "Unknown"

# Load spaCy's language model
nlp = spacy.load("en_core_web_sm")

def extract_potential_entities(text):
    words = text.split()
    print(words)
    # Filter out words that are lowercase or stop words (you can adjust based on your needs)
    potential_entities = [word for word in words if word.istitle()]  # Only consider words that start with capital letters
    print(potential_entities)
    return potential_entities



def extract_entities(text):
    # Extract potential entities (capitalized words)
    potential_entities = extract_potential_entities(text)
    print("Potential Entities:", potential_entities)
    
    if not potential_entities:
        return ["Unknown"]
    
    # Prepare lists of company names from each dataset (already lowercased for comparison)
    crypto_companies = [name.lower() for name in crypto_data["Coin Name"].str.lower().tolist()]
    sp500_companies = [name.lower() for name in sp500_data["Shortname"].str.lower().tolist()]
    company_names_from_file = [name.lower() for name in company_data["Security Name"].str.lower().tolist()]

    # List to store matched company names
    matched_companies = []

    # Check for potential company names in the input text using LIKE-style check
    for entity in potential_entities:
        entity_lower = entity.lower()  # Lowercase the word for case-insensitive matching
        matched = False  # Flag to check if a match is found
        
        # Clean up the entity by removing non-alphanumeric characters (e.g., commas, periods)
        entity_clean = re.sub(r'\W+', '', entity_lower)  # \W+ matches non-alphanumeric characters

        # Create a regex pattern to match the full word (with boundaries to ensure full word match)
        pattern = r'\b' + re.escape(entity_clean) + r'\b'  # \b ensures full word match
        
        # First, check for exact full word matches (case-insensitive) in the company lists
        for company_list in [crypto_companies, sp500_companies, company_names_from_file]:
            for company in company_list:
                # Clean the company name by removing non-alphanumeric characters as well
                company_clean = re.sub(r'\W+', '', company)

                # Use 'in' for LIKE-style check, after cleaning the company and entity
                if entity_clean in company_clean:  # Match if the entity is found in the company name
                    print(f"Match found: '{entity_clean}' in '{company_clean}'")
                    matched_companies.append(company.capitalize())  # Capitalize to return the name properly
                    matched = True  # Mark as matched
                    break  # Stop checking further if match found

        # If no exact match, proceed with fuzzy matching (optional step if needed)
        if not matched:
            for company_list in [crypto_companies, sp500_companies, company_names_from_file]:
                for company in company_list:
                    print(f"Comparing '{entity_lower}' with '{company}'")
                    if len(entity) > 2 and len(company) > 2 and entity_lower in company:
                        matched_companies.append(company.capitalize())  # Capitalize to return the name properly
                        matched = True  # Mark as matched
                        break  # Stop checking further in this list if match found
                if matched:
                    break  # Stop checking further company lists if match found

    # Remove duplicates and return the result
    matched_companies = list(set(matched_companies))

    return matched_companies if matched_companies else ["Unknown"]

# Define intents for market sentiment analysis
intents = [
    {
        "patterns": ["what is market sentiment analysis", "what is sentiment analysis", "what"],
        "response": (
            "This is a chatbot to perform market sentiment analysis, a process of using natural language processing (NLP) and machine learning to analyze financial news, social media, and other text data to determine the overall sentiment (positive, negative, or neutral) toward a stock, market, or financial instrument."
        ),
    },
    {
        "patterns": ["how does market sentiment analysis work", "how does it work", "how"],
        "response": (
            "Market sentiment analysis works by collecting text data from financial news, stock forums, and social media. "
            "It then applies NLP techniques to classify the sentiment of the text using pre-trained models like Twitter-RoBERTa, BERT, or other transformers."
        ),
    },
    {
        "patterns": ["what data sources are used", "sources of data", "source"],
        "response": (
            "Common data sources for market sentiment analysis include:\n"
            "- Twitter and social media discussions\n"
            "- Financial news articles\n"
            "- Stock forum posts (e.g., Reddit, StockTwits)\n"
            "- Earnings reports and press releases\n"
            "- Analyst reports and market research"
        ),
    },
    {
        "patterns": ["applications of market sentiment analysis", "use cases"],
        "response": (
            "Market sentiment analysis is used in:\n"
            "- Stock price prediction\n"
            "- Algorithmic trading strategies\n"
            "- Risk assessment and market trends\n"
            "- Understanding investor behavior\n"
            "- Portfolio management and decision-making"
        ),
    },
    {
        "patterns": ["which model are you using", "what model is used"],
        "response": (
            "I'm using a sentiment analysis model based on `cardiffnlp/twitter-roberta-base-sentiment` and dataset related to the markets such as Crypto, NASDAQ and S&P500 index, which is optimized for analyzing financial and stock-related tweets and texts."
        ),
    },
    {
        "patterns": ["limitations", "challenges"],
        "response": (
            "Some limitations of market sentiment analysis include:\n"
            "- Difficulty in detecting sarcasm and market manipulation (e.g., pump-and-dump schemes)\n"
            "- Limited understanding of complex financial jargon and context\n"
            "- Market sentiment alone may not be sufficient for stock prediction\n"
            "- High noise in social media data"
        ),
    },
    {
        "patterns": ["can you analyze stock-related tweets", "can you analyze tweets"],
        "response": (
            "Yes! I can analyze stock-related tweets and determine their sentiment. Just provide a tweet, and I'll classify it as positive, neutral, or negative."
        ),
    },
    {
        "patterns": ["is sentiment analysis accurate for stock trading", "accuracy of sentiment analysis", "is it accurate"],
        "response": (
            "Sentiment analysis can be a useful tool in stock trading, but it should not be the sole factor in making trading decisions. "
            "Accuracy depends on the dataset quality, market conditions, and the model's ability to understand financial context."
        ),
    },
]



# Generate chatbot response
def chatbot_response(user_input):
    sentiment_input = user_input.lower()
    responses = []
    for intent in intents:
        if any(pattern in user_input for pattern in intent["patterns"]):
            responses.append(intent["response"])

    # Combine and return responses for all matching intents
    if responses:
        return " ".join(responses)
    if "analyze" in user_input or "sentiment" in user_input:
        st.session_state.awaiting_text = True
        return "Sure! Provide a tweet or text, and I'll analyze the sentiment."
    
    if "again" in user_input or "another" in user_input or "one more" in user_input:
        st.session_state.awaiting_text = True
        return "Sure! Provide me the text!"

    if "hi" in user_input or "hey" in user_input:
        return "Hello! How can I assist you today? You can ask me about market sentiment based on news or texts related to finance!"

    if "goodbye" in user_input or "bye" in user_input:
        return "Goodbye! Have a great day!"
    if st.session_state.awaiting_text:
        st.session_state.awaiting_text = False
        sentiment_label = match_sentiment(sentiment_input)
        entities = extract_entities(user_input)
        print(user_input)
        label_mapping = {
            "LABEL_0": "Negative",
            "LABEL_1": "Neutral",
            "LABEL_2": "Positive"
        }
        if sentiment_label != "Unknown":
            return f"The sentiment is: **{sentiment_label}** (from dataset). Entities mentioned: **{', '.join(entities)}**"

        sentiment = sentiment_pipeline(user_input)
        sentiment_type = label_mapping.get(sentiment[0]['label'], "Unknown")
        confidence = sentiment[0]['score']
        
        return f"The sentiment of your text is **'{sentiment_type}'** with a confidence of **{confidence:.2f}**. Entities mentioned: **{', '.join(entities)}**."

    if "hi" in user_input or "hello" in user_input:
        return "Hello! You can ask me about sentiment analysis or provide a stock-related tweet for analysis."

    return "I'm sorry, I didn't understand that. Please try again!"

# Streamlit App
st.title("Stock Market Sentiment Chatbot")
st.write("Analyze the sentiment of stock-related texts!")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    if message.get("content"):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

if prompt := st.chat_input("Enter your message here..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response = chatbot_response(prompt)
        st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})
