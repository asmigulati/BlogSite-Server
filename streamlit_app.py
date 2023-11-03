import streamlit as st
import numpy as np
import tensorflow_hub as hub
import pandas as pd

# Initialize the model
@st.experimental_singleton
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
    'Timestamp': ['2023-02-13 22:48:20', '2023-02-13 22:48:20', '2023-02-13 22:48:10', '2023-02-13 22:48:10', 
                  '2023-02-13 22:48:10', '2023-02-13 22:33:30', '2023-02-13 22:33:10', '2023-02-13 22:18:10', 
                  '2023-02-13 22:49:00', '2023-02-13 21:48:50', '2023-02-13 21:48:50', '2023-02-13 21:48:50', 
                  '2023-02-13 21:48:10', '2023-02-13 21:48:10', '2023-02-13 21:19:10', '2023-02-13 21:18:10', 
                  '2023-02-13 21:18:10', '2023-02-13 21:03:40', '2023-02-13 21:03:30', '2023-02-13 21:03:30'],
    'Activity': ['desk is idle', 'bed is idle', 'closet is idle', 'refrigerator is idle', 
                 'Isabella Rodriguez is stretching', 'shelf is idle', 'desk is neat and organized', 
                 'Isabella Rodriguez is writing in her journal', 'desk is idle', 
                 'Isabella Rodriguez is taking a break', 'bed is idle', 
                 'Isabella Rodriguez is cleaning up the kitchen', 'refrigerator is idle', 'bed is being used', 
                 'shelf is idle', 'Isabella Rodriguez is watching a movie', 'shelf is organized and tidy', 
                 'desk is idle', 'Isabella Rodriguez is reading a book', 'bed is idle']
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
        st.write(results[['Time', 'String']])
    else:
        st.write('Please enter a sentence to query.')

# To run the Streamlit app, save the code in a file (e.g., app.py) and run `streamlit run app.py` in your terminal.
