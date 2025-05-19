import cv2
import numpy as np
import torch
from torchvision import transforms

def preprocess_image(image_bytes):
    # Decode image
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)  # Load as grayscale
    if img is None:
        raise ValueError("Invalid image file")
    
    # Convert grayscale to RGB (duplicate channels)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    
    # Resize and convert to tensor
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((299, 299)),  # InceptionV3 expects 299x299
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # InceptionV3 normalization
    ])
    
    img_tensor = transform(img_rgb).unsqueeze(0)  # Add batch dimension
    return img_tensor