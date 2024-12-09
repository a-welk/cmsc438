# Alex Welk - CMSC 438 - Machine Learning - Homework 5
# Description: in this program we are using a pretrained model to take in list of strings and a query and outputs the similarity score of each string based on the given query.
from sentence_transformers import SentenceTransformer
import numpy as np
import torch

def find_most_relevant(query, text_list):
    # Initialize model
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Embed 
    query_embedding = model.encode(query, convert_to_tensor=True)
    text_embeddings = model.encode(text_list, convert_to_tensor=True)
    
    # Calculate cosine similarities between query and text_list embeddings
    similarities = torch.cosine_similarity(query_embedding, text_embeddings)
    
    scores = similarities.cpu().numpy()

    # Round scores to get them in the same form as the example in the slides
    scores = np.round(scores, 4)

    # Find the index of the highest similarity score
    idx = np.argmax(scores)

    # Cast to ints for gradescope
    idx = int(idx)
    
    return idx, scores

if __name__ == "__main__":
    query = "city"
    text_list = [
    'undergraduate curriculum committee',
    'townhall planning group',
    'ski resort management',
    ]

    idx, scores = find_most_relevant(query, text_list)

    print(f"Index of most relevant text: {idx}")
    print(f"Similarity scores: {scores}")