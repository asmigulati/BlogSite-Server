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

# Sample DataFrame to store embeddings
df = pd.DataFrame({
    'Time': ['2023-01-01', '2023-01-02'],
    'String': ['Hello world', 'Another sentence'],
    'Embed': [embed('Hello world'), embed('Another sentence')]
})

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
