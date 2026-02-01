🖼️ Image Similarity Search using Deep Learning
📌 Overview
This project implements an Image Similarity Search system using Deep Learning.
Given two images, the system extracts deep feature embeddings using a pretrained ResNet-50 CNN and computes their cosine similarity to quantify how visually similar the images are.
This approach is widely used in visual search engines, duplicate image detection, recommendation systems, and content moderation.
🚀 Key Features
Uses ResNet-50 (CNN) pretrained on ImageNet
Extracts deep embeddings from images
Computes similarity using cosine similarity
Modular and clean project structure
Runs locally with minimal setup
Easy to extend to large image datasets
🧠 Deep Learning Approach
Input images are passed through a pretrained ResNet-50
The final classification layer is removed
Intermediate CNN features are extracted as embeddings
Cosine similarity is computed between embeddings
This allows comparison of images based on semantic visual features, not raw pixels.
📂 Project Structure
image-similarity-search/
│
├── data/
│   └── images/
│       ├── img1.jpg
│       ├── img2.jpg
│
├── src/
│   ├── feature_extractor.py   # ResNet50 feature extraction
│   ├── similarity.py          # Cosine similarity computation
│   └── utils.py               # Helper functions
│
├── main.py                    # Entry point
├── requirements.txt
├── README.md
├── .gitignore
🛠 Tech Stack
Python
PyTorch
Torchvision
ResNet-50
NumPy
PIL (Python Imaging Library)
💡 Applications
Image search engines
Duplicate image detection
Product similarity in e-commerce
Content moderation
Recommendation systems
## How to Run the Project

Follow the steps below to run the project locally.

### Step 1: Clone the repository
```bash
git clone <repository-url>
cd image-similarity-search
Step 2: Install required dependencies
pip install -r requirements.txt
Step 3: Run the program
python main.py
Step 4: View the output
The program computes and prints a cosine similarity score that represents how similar two images are based on deep learning–based feature extraction.