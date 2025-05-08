import os
from transformers import AutoTokenizer, AutoModel
import torch

# Load pre-trained model and tokenizer
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Path to directory containing text files
directory_path = '/farmshare/learning/data/emerson'

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

# Function to calculate cosine similarity
def cosine_similarity(embedding1, embedding2):
    cos = torch.nn.CosineSimilarity(dim=0)
    return cos(embedding1, embedding2).item()

# Function to find and rank closest words
def find_closest_words(word_embeddings, target_word, n=5):
    if target_word not in word_embeddings:
        raise ValueError(f"'{target_word}' not found in word embeddings.")
    
    target_embedding = word_embeddings[target_word]
    similarities = {word: cosine_similarity(target_embedding, embedding) for word, embedding in word_embeddings.items() if word != target_word}
    closest_words = sorted(similarities.items(), key=lambda item: item[1], reverse=True)
    return closest_words[:n]

# Function to find words between two words based on cosine similarity
def words_between(word_embeddings, word1, word2):
    if word1 not in word_embeddings or word2 not in word_embeddings:
        raise ValueError(f"'{word1}' or '{word2}' not found in word embeddings.")
    
    embedding1 = word_embeddings[word1]
    embedding2 = word_embeddings[word2]
    similarities = {word: (cosine_similarity(embedding1, embedding) + cosine_similarity(embedding2, embedding)) / 2 for word, embedding in word_embeddings.items() if word != word1 and word != word2}
    closest_words = sorted(similarities.items(), key=lambda item: item[1], reverse=True)
    return closest_words

# Process text files and get word embeddings
word_embeddings = process_text_files(directory_path)

while True:
    print("\nSelect Mode:")
    print("1. Most similar to target word")
    print("2. N most similar to target word")
    print("3. Words between two words")
    print("4. Exit")
    choice = input("Enter your choice (1, 2, 3, 4): ").strip()

    if choice == '4':
        break
    elif choice == '1':
        target_word = input("Enter the target word: ").strip()
        closest_word = find_closest_words(word_embeddings, target_word, n=1)
        print(f"Most similar word to '{target_word}': {closest_word[0][0]} (Similarity: {closest_word[0][1]:.4f})")
    elif choice == '2':
        target_word = input("Enter the target word: ").strip()
        n = int(input("Enter the number of similar words (N): ").strip())
        closest_words = find_closest_words(word_embeddings, target_word, n=n)
        print(f"{n} most similar words to '{target_word}':")
        for word, similarity in closest_words:
            print(f"{word}: {similarity:.4f}")
    elif choice == '3':
        word1 = input("Enter the first word: ").strip()
        word2 = input("Enter the second word: ").strip()
        words_between_list = words_between(word_embeddings, word1, word2)
        print(f"Words between '{word1}' and '{word2}':")
        for word, similarity in words_between_list:
            print(f"{word}: {similarity:.4f}")
    else:
        print("Invalid choice. Please try again.")
