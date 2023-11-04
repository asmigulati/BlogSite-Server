import streamlit as st
import numpy as np
import tensorflow_hub as hub
import pandas as pd

# Initialize the model
@st.cache(allow_output_mutation=True)
def load_model():
    module_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
    model = hub.load(module_url)
    print(f"Module {module_url} loaded")
    return model

model = load_model()

# Function to embed input text using USE
def embed(input_text):
    return model([input_text]).numpy()[0]

# Function to calculate cosine similarity
def cosine_similarity(vec_a, vec_b):
    dot_product = np.dot(vec_a, vec_b)
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    return dot_product / (norm_a * norm_b)
data = { 
    'Timestamp': [ 
        '2023-03-13 08:00:00', '2023-03-13 08:30:00', '2023-03-13 09:00:00', 
        '2023-03-13 09:30:00', '2023-03-13 10:00:00', '2023-03-13 10:30:00', 
        '2023-03-13 11:00:00', '2023-03-13 11:30:00', '2023-03-13 12:00:00', 
        '2023-03-13 12:30:00', '2023-03-13 13:00:00', '2023-03-13 13:30:00', 
        '2023-03-13 14:00:00', '2023-03-13 14:30:00', '2023-03-13 15:00:00', 
        '2023-03-13 15:30:00', '2023-03-13 16:00:00', '2023-03-13 16:30:00', 
        '2023-03-13 17:00:00', '2023-03-13 17:30:00' 
    ], 
    'Activity': [ 
        'Alex wakes up and stretches', 'Alex enjoys a hearty breakfast', 'Alex starts working on the computer', 
        'Alex takes a short coffee break', 'Alex attends a team meeting', 'Alex reads industry news', 
        'Alex prepares a light lunch', 'Alex reviews project timelines', 'Alex joins a client call', 
        'Alex responds to emails', 'Alex goes for a quick walk', 'Alex brainstorms with colleagues', 
        'Alex works on a presentation', 'Alex has a snack', 'Alex wraps up work', 
        'Alex heads out for an evening jog', 'Alex cooks dinner', 'Alex relaxes with a book', 
        'Alex plans the next day', 'Alex goes to sleep' 
    ] 
} 

df = pd.DataFrame(data)
df['Embed'] = df['Activity'].apply(lambda x: embed(x))

# Streamlit interface
st.title('Vector Database Query')

# User input text
user_input = st.text_input('Enter a sentence to find similar sentences in the database:')

# Number of results to return
num_results = st.number_input('Number of results to show:', min_value=1, max_value=len(df), value=min(5, len(df)), step=1)

# Button to perform query
if st.button('Query'):
    if user_input:
        # Calculate the similarity of user input to each item in the DataFrame
        input_embedding = embed(user_input)
        df['Similarity'] = df['Embed'].apply(lambda x: cosine_similarity(x, input_embedding))

        # Display the most similar entries
        results = df.nlargest(num_results, 'Similarity')
        st.write(results[['Timestamp', 'Activity']])
    else:
        st.write('Please enter a sentence to query.')

# To run the Streamlit app, save the code in a file (e.g., app.py) and run `streamlit run app.py` in your terminal.
