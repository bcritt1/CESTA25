import os
import pickle
import re
import torch
from sentence_transformers import util, SentenceTransformer

# Load pre-trained model
model_name = "sentence-transformers/all-MiniLM-L6-v2"
model = SentenceTransformer(model_name)

# Define paths
embeddings_dir = f'/scratch/users/{os.environ["USER"]}/embeddings'
embedding_file = os.path.join(embeddings_dir, 'emerson_word_embeddings.pkl')

# Function to calculate cosine similarity
def cosine_similarity(embedding1, embedding2):
    return util.pytorch_cos_sim(embedding1, embedding2).item()

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

# Load word embeddings from file
with open(embedding_file, 'rb') as f:
    word_embeddings = pickle.load(f)

while True:
    print("\nSelect Mode:")
    print("1. Most similar to target word")
    print("2. N most similar to target word")
    print("3. Words between two words")
    print("4. View embedding vector")
    print("5. Exit")
    choice = input("Enter your choice (1, 2, 3, 4, 5): ").strip()

    if choice == '5':
        break
    elif choice == '1':
        target_word = input("Enter the target word: ").strip().lower()
        try:
            closest_word = find_closest_words(word_embeddings, target_word, n=1)
            print(f"Most similar word to '{target_word}': {closest_word[0][0]} (Similarity: {closest_word[0][1]:.4f})")
        except ValueError as e:
            print(e)
    elif choice == '2':
        target_word = input("Enter the target word: ").strip().lower()
        n = int(input("Enter the number of similar words (N): ").strip())
        try:
            closest_words = find_closest_words(word_embeddings, target_word, n=n)
            print(f"{n} most similar words to '{target_word}':")
            for word, similarity in closest_words:
                print(f"{word}: {similarity:.4f}")
        except ValueError as e:
            print(e)
    elif choice == '3':
        word1 = input("Enter the first word: ").strip().lower()
        word2 = input("Enter the second word: ").strip().lower()
        try:
            words_between_list = words_between(word_embeddings, word1, word2)
            print(f"Words between '{word1}' and '{word2}':")
            for word, similarity in words_between_list:
                print(f"{word}: {similarity:.4f}")
        except ValueError as e:
            print(e)
    elif choice == '4':
        target_word = input("Enter the word to view its embedding: ").strip().lower()
        if target_word in word_embeddings:
            print(f"Embedding vector for '{target_word}': {word_embeddings[target_word]}")
        else:
            print(f"'{target_word}' not found in word embeddings.")
    else:
        print("Invalid choice. Please try again.")
