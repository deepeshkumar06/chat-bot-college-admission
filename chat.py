import streamlit as st
import random
import json
import torch
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize

# Importing necessary libraries for styling
import streamlit.components.v1 as components

# Function to load custom CSS for styling
def local_css(file_name):
    try:
        with open(file_name) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    except FileNotFoundError:
        st.markdown('<style>body { background-color: #f0f2f6; }</style>', unsafe_allow_html=True)  # Default styling

local_css("style.css")  # Make sure you have a file named style.css in the same directory

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('intents.json', 'r') as json_data:
    intents = json.load(json_data)

FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "PSR"

def get_response(msg):
    sentence = tokenize(msg)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                return random.choice(intent['responses'])
    
    return "I do not understand..."

# Initialize chat history
chat_history = []

# Streamlit UI
st.title("P.S.R Engineering College Chatbot")
st.markdown("### Welcome! Type a message to chat with P.S.R.")

# Define CSS for message bubbles
st.markdown("""
<style>
    .user-bubble {
        background-color: #DCF8C6;
        padding: 10px;
        border-radius: 10px;
        margin: 10px 0;
        max-width: 70%;
        align-self: flex-start;
    }
    .bot-bubble {
        background-color: #E5E5EA;
        padding: 10px;
        border-radius: 10px;
        margin: 10px 0;
        max-width: 70%;
        align-self: flex-end;
    }
</style>
""", unsafe_allow_html=True)

input_message = st.text_input("You:")
if input_message:
    if input_message.lower() == "quit":
        st.stop()
    else:
        # Add user message to chat history
        chat_history.append(("You", input_message))

        # Get bot response
        response = get_response(input_message)
        
        # Add bot response to chat history
        chat_history.append((bot_name, response))
        
        # Display chat history with message bubbles
        for sender, message in chat_history:
            if sender == "You":
                st.markdown(f'<div class="user-bubble">{sender}: {message}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="bot-bubble">{sender}: {message}</div>', unsafe_allow_html=True)
