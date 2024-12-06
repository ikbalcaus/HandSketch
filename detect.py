import cv2
import torch
import numpy as np
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

def get_characters_contours(canvas):
    gray_canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray_canvas, 127, 255, cv2.THRESH_BINARY_INV)
    dilated_thresh = cv2.dilate(thresh, np.ones((20, 3), np.uint8), iterations=1)
    contours, _ = cv2.findContours(dilated_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bounding_boxes = [cv2.boundingRect(contour) for contour in contours]
    bounding_boxes = sorted(bounding_boxes, key=lambda x: x[1])

    rows = []
    end_rows = []
    current_row = [bounding_boxes[0]]
    for i, box in enumerate(bounding_boxes[1:]):
        if abs(box[1] - current_row[-1][1]) <= np.mean([box[3] for box in bounding_boxes]) * 0.5:
            current_row.append(box)
        else:
            rows.append(current_row)
            end_rows.append(i)
            current_row = [box]
    rows.append(current_row)
    sorted_boxes = []
    for row in rows:
        sorted_boxes.extend(sorted(row, key=lambda x: x[0]))

    sorted_contours = []
    for box in sorted_boxes:
        for contour in contours:
            if cv2.boundingRect(contour) == box:
                sorted_contours.append(contour)
                break
    return sorted_contours, dilated_thresh, end_rows

def detect_characters(model, canvas):
    model.eval()
    results = []
    contours, thresh, end_rows = get_characters_contours(canvas)
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
    for i in range(len(results) - 1, -1, -1):
        if i in end_rows:
            results.insert(i + 1, " ")
    return results

if __name__ == "__main__":
    results = {}
    if os.path.exists("images"):
        model = CNNModel()
        if os.path.exists("logs/model.pth"):
            model.load_state_dict(torch.load("logs/model.pth", weights_only=True))
        for filename in os.listdir("images"):
            if filename.endswith((".jpg", ".png")):
                characters = detect_characters(model, cv2.imread(os.path.join("images", filename)))
                results[filename] = characters
    for filename, characters in results.items():
        print(f"{filename}: {characters}")
    if len(results) == 0:
        print("No images found inside 'images' folder with extension '.jpg' or '.png'")
    input("Press Enter to exit...")
