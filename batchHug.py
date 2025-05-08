import os
import pickle
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer
import nltk

# Set custom NLTK data path
nltk_data_path = os.environ.get('NLTK_DATA', '/scratch/users/{}/nltk_data'.format(os.environ['USER']))
nltk.data.path.append(nltk_data_path)

# Ensure NLTK resources are downloaded
nltk.download('punkt_tab', download_dir=nltk_data_path)
nltk.download('stopwords', download_dir=nltk_data_path)

# Debugging: Check if the necessary resources are present
print("Checking NLTK data path:", nltk_data_path)
print("Contents of tokenizer directory:", os.listdir(os.path.join(nltk_data_path, 'tokenizers')))
print("Contents of corpora directory:", os.listdir(os.path.join(nltk_data_path, 'corpora')))

# Load pre-trained model
model_name = "sentence-transformers/all-MiniLM-L6-v2"
model = SentenceTransformer(model_name)

# Define paths
directory_path = '/farmshare/learning/data/emerson'
embeddings_dir = f'/scratch/users/{os.environ["USER"]}/embeddings'

# Create the directory if it does not exist
os.makedirs(embeddings_dir, exist_ok=True)

# Custom cleaning function
def clean(text):
    # split into words
    tokens = word_tokenize(text)
    # convert to lower case
    tokens = [w.lower() for w in tokens]
    # remove punctuation from each word
    table = str.maketrans('', '', string.punctuation)
    stripped = [w.translate(table) for w in tokens]
    # remove remaining tokens that are not alphabetic
    words = [word for word in stripped if word.isalpha()]
    # filter out stop words
    stop_words = set(stopwords.words('english'))
    words = [w for w in words if not w in stop_words]
    return words

# Function to get embeddings for a word
def get_word_embedding(word, model):
    return model.encode(word, convert_to_tensor=True)

# Function to read and process text files and get embeddings for unique words
def process_text_files(directory_path, model):
    word_embeddings = {}

    for filename in os.listdir(directory_path):
        if filename.endswith('.txt'):
            file_path = os.path.join(directory_path, filename)
            with open(file_path, 'r', encoding='utf-8') as file:
                text = file.read()
                
                # Clean text
                words = clean(text)
                
                for word in words:
                    if word not in word_embeddings:
                        word_embeddings[word] = get_word_embedding(word, model)
    
    return word_embeddings

# Process text files and get word embeddings
word_embeddings = process_text_files(directory_path, model)

# Save word embeddings to a file within the embeddings directory
output_model_file = os.path.join(embeddings_dir, 'emerson_word_embeddings.pkl')
with open(output_model_file, 'wb') as f:
    pickle.dump(word_embeddings, f)

print(f"Word embeddings have been saved to {output_model_file}")
