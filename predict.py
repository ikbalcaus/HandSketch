import torch
import torch.nn as nn
import numpy as np
import cv2
import os
from torchvision import transforms
from PIL import Image

class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def predict_multiple(model, image_path):
    model.eval()
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    image = Image.open(image_path).convert("L")
    image_np = np.array(image)
    
    _, thresh = cv2.threshold(image_np, 127, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    results = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w > 5 and h > 5:
            roi = thresh[y:y+h, x:x+w]
            roi_pil = Image.fromarray(roi)
            roi_pil = transform(roi_pil).unsqueeze(0)
            
            with torch.no_grad():
                output = model(roi_pil)
                _, predicted = torch.max(output, 1)
                results.append(predicted.item())

    return results

def main(image_path):
    model = CNNModel()
    if os.path.exists("logs/model.pth"):
        model.load_state_dict(torch.load("logs/model.pth", weights_only=True))
    model.eval()

    characters = predict_multiple(model, image_path)
    return characters

if __name__ == "__main__":
    if os.path.exists("image.jpg"):
        characters = main("image.jpg")
        print(characters)
        main("image.jpg")
    else:
        print("Image 'image.jpg' not found in the current directory")
