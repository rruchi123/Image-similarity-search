import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np


class FeatureExtractor:
    def __init__(self):
        # Load pretrained ResNet50
        self.model = models.resnet50(pretrained=True)
        
        # Remove the classification layer (FC)
        self.model = torch.nn.Sequential(
            *list(self.model.children())[:-1]
        )
        
        self.model.eval()

        # Image preprocessing pipeline
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def extract(self, image_path: str) -> np.ndarray:
        """
        Extract feature vector from an image.
        Returns a 2048-dimensional numpy vector.
        """
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image).unsqueeze(0)

        with torch.no_grad():
            features = self.model(image)

        # Flatten from (1, 2048, 1, 1) -> (2048,)
        return features.squeeze().numpy()