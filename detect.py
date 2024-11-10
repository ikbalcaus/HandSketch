import torch
import torch.nn as nn
import numpy as np
import cv2
import os
from PIL import Image
from configuration import CNNModel, transform

def map_class_to_char(class_idx):
    if class_idx < 10:
        return str(class_idx)
    elif class_idx < 36:
        return chr(class_idx - 10 + ord("A"))
    else:
        return chr(class_idx - 36 + ord("a"))

def detect_characters(model, image_path):
    model.eval()
    image = Image.open(image_path).convert("L")
    image_np = np.array(image)
    _, thresh = cv2.threshold(image_np, 127, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=lambda contour: cv2.boundingRect(contour)[0])
    results = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w > 5 and h > 5:
            roi = thresh[y:y+h, x:x+w]
            roi_pil = Image.fromarray(roi)
            roi_pil = transform(roi_pil).unsqueeze(0)
            with torch.no_grad():
                output = model(roi_pil)
                _, detected = torch.max(output, 1)
                char = map_class_to_char(detected.item())
                results.append(char)
    return results

if __name__ == "__main__":
    results = {}
    if os.path.exists("images"):
        model = CNNModel()
        if os.path.exists("logs/model.pth"):
            model.load_state_dict(torch.load("logs/model.pth", weights_only=True))
        for filename in os.listdir("images"):
            if filename.endswith((".jpg", ".png")):
                characters = detect_characters(model, os.path.join("images", filename))
                results[filename] = characters
    for filename, characters in results.items():
        print(f"{filename}: {characters}")
    input("Press Enter to exit...")
