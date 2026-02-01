from src.feature_extractor import FeatureExtractor
from src.similarity import cosine_similarity

# Initialize feature extractor
extractor = FeatureExtractor()

# Image paths
img1_path = "data/images/img1.jpg"
img2_path = "data/images/img2.jpg"

# Extract features
features1 = extractor.extract(img1_path)
features2 = extractor.extract(img2_path)

# Compute similarity
similarity_score = cosine_similarity(features1, features2)

print("Cosine Similarity Score:", similarity_score)