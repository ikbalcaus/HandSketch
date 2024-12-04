import cv2
import torch
from PIL import Image
from configuration import transform

def map_class_to_char(class_idx):
    if class_idx < 10:
        return str(class_idx)
    elif class_idx < 36:
        return chr(class_idx - 10 + ord("A"))
    else:
        return chr(class_idx - 36 + ord("a"))
    
def get_character_contours(canvas):
    gray_canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray_canvas, 127, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=lambda contour: cv2.boundingRect(contour)[0])
    return contours, thresh

def detect_characters(model, canvas):
    model.eval()
    results = []
    contours, thresh = get_character_contours(canvas)
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
    print("This file is not intended for direct use. Please run the 'main.py' file instead")
    input("Press Enter to exit...")
