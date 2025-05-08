import os
import pickle
from transformers import AutoTokenizer, AutoModel
import torch

# Get the current user's name from the environment variables
user_name = os.environ.get('USER')

# Define paths
directory_path = '/farmshare/learning/data/emerson'
embeddings_dir = f'/scratch/users/{user_name}/embeddings'

# Create directory if it does not exist
os.makedirs(embeddings_dir, exist_ok=True)

# Load pre-trained model and tokenizer
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Function to get embeddings for a word
def get_word_embedding(word):
    inputs = tokenizer(word, return_tensors="pt")
    outputs = model(**inputs)
    last_hidden_states = outputs.last_hidden_state
    return last_hidden_states.mean(dim=1).squeeze()

# Function to read and process text files
def process_text_files(directory_path):
    word_embeddings = {}
    
    for filename in os.listdir(directory_path):
        if filename.endswith('.txt'):
            file_path = os.path.join(directory_path, filename)
            with open(file_path, 'r', encoding='utf-8') as file:
                text = file.read()
                
                # Tokenize the text and get unique words
                words = list(set(tokenizer.tokenize(text)))
                
                # Get embeddings for all words
                for word in words:
                    if word not in word_embeddings:
                        word_embeddings[word] = get_word_embedding(word)
    
    return word_embeddings

# Process text files and get word embeddings
word_embeddings = process_text_files(directory_path)

# Save word embeddings to a file within the embeddings directory
output_model_file = os.path.join(embeddings_dir, 'emerson_word_embeddings.pkl')
with open(output_model_file, 'wb') as f:
    pickle.dump(word_embeddings, f)

print(f"Word embeddings have been saved to {output_model_file}")
