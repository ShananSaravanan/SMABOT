import streamlit as st
from transformers import pipeline

# Initialize the sentiment analysis model
sentiment_pipeline = pipeline("sentiment-analysis")

# Global variable to track the flow
if "awaiting_text" not in st.session_state:
    st.session_state.awaiting_text = False

# Define intents with multiple patterns
intents = [
    {
        "patterns": ["what is sentiment analysis","what is this"],
        "response": (
            "This is called sentiment analysis, a natural language processing technique used to determine whether a piece of text is positive, negative, or neutral. "
            "It is often used to analyze opinions, reviews, and social media content."
        ),
    },
    {
        "patterns": ["how does", "work"],
        "response": (
            "Sentiment analysis works by using machine learning or rule-based methods to analyze the emotions conveyed in a text. "
            "It involves tokenizing the text, extracting features, and classifying it using a pre-trained model like the one I'm using here."
        ),
    },
    {
        "patterns": ["applications of sentiment analysis", "application"],
        "response": (
            "Sentiment analysis is widely used in various fields, such as:\n"
            "- Social media monitoring\n"
            "- Customer feedback analysis\n"
            "- Market research\n"
            "- Brand reputation management\n"
            "- Political campaign analysis"
        ),
    },
    {
        "patterns": ["limitations"],
        "response": (
            "Some limitations of sentiment analysis include:\n"
            "- Difficulty in understanding sarcasm or irony\n"
            "- Challenges in handling ambiguous or context-dependent language\n"
            "- Inability to capture cultural nuances effectively\n"
            "- Limited accuracy on complex texts"
        ),
    },
    {
        "patterns": ["which model are you using", "model"],
        "response": (
            "I'm using a sentiment analysis model provided by the Transformers library, which is based on pre-trained models like BERT or DistilBERT."
        ),
    },
    {
        "patterns": ["long texts"],
        "response": (
            "Yes, I can analyze long texts, but the accuracy might decrease if the text contains multiple conflicting sentiments. "
            "In such cases, splitting the text into smaller chunks is recommended for better results."
        ),
    },
    {
        "patterns": ["is sentiment analysis always accurate", "always accurate"],
        "response": (
            "No, sentiment analysis is not always accurate. The accuracy depends on the quality of the dataset used to train the model and the complexity of the text. "
            "It can struggle with sarcasm, idioms, and cultural references."
        ),
    },
]

# Define the response generation logic
def chatbot_response(user_input):
    user_input = user_input.lower()

    # Check if the bot is awaiting text for analysis
    if st.session_state.awaiting_text:
        st.session_state.awaiting_text = False
        sentiment = sentiment_pipeline(user_input)
        sentiment_type = sentiment[0]['label']
        confidence = sentiment[0]['score']
        return f"The sentiment of your text is '{sentiment_type}' with a confidence of {confidence:.2f}."

    # Collect responses for all matching intents
    responses = []
    for intent in intents:
        if any(pattern in user_input for pattern in intent["patterns"]):
            responses.append(intent["response"])

    # Combine and return responses for all matching intents
    if responses:
        return " ".join(responses)

    # Default responses
    if "analyze" in user_input or "find" in user_input:
        st.session_state.awaiting_text = True
        return "Shall I analyze some text for you? Please provide me the text directly."

    if "again" in user_input or "another" in user_input or "one more" in user_input:
        st.session_state.awaiting_text = True
        return "Sure! Provide me the text!"

    if "hi" in user_input or "hey" in user_input:
        return "Hello! How can I assist you today? You can ask me about sentiment analysis or test it here on your texts!"

    if "goodbye" in user_input or "bye" in user_input:
        return "Goodbye! Have a great day!"

    return "I'm sorry, I didn't understand that. Could you please rephrase?"

# Streamed response emulator
def response_generator(user_input):
    return chatbot_response(user_input)

st.title("Sentiment Analysis Chatbot")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    if message.get("content"):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("What would you like to say?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response = response_generator(prompt)
        st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})
