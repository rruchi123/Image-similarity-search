import numpy as np
from numpy.linalg import norm

def cosine_similarity(vec1, vec2):
    """
    Compute cosine similarity between two feature vectors
    """
    vec1 = vec1.flatten()
    vec2 = vec2.flatten()
    
    return np.dot(vec1, vec2) / (norm(vec1) * norm(vec2))