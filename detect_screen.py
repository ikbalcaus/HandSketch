import os
import cv2
import torch
from datetime import datetime
from PIL import Image, ImageTk
import pyperclip
import tkinter as tk
from detect import detect_characters, CNNModel

model = CNNModel()
if os.path.exists("logs/model.pth"):
    model.load_state_dict(torch.load("logs/model.pth", weights_only=True))
model.eval()

def validate_input(char):
    return len(char) <= 1

def copy_characters(characters):
    text = "".join(characters)
    pyperclip.copy(text)

def get_character_bounds(canvas):
    gray_canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, binary_canvas = cv2.threshold(gray_canvas, 240, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(binary_canvas, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bounds = [cv2.boundingRect(contour) for contour in contours]
    bounds = sorted(bounds, key=lambda x: x[0])
    return bounds

def export_char_images(detect_window, canvas, characters, bounds, entries):
    for i, (char, entry) in enumerate(zip(characters, entries)):
        if not entry.get():
            continue
        x, y, w, h = bounds[i]
        char_img = canvas[y:y+h, x:x+w]
        white_space = 7
        char_img_with_border = cv2.copyMakeBorder(
            char_img, 
            top=white_space, bottom=white_space, left=white_space, right=white_space, 
            borderType=cv2.BORDER_CONSTANT, 
            value=[255, 255, 255]
        )
        char_name = entry.get() or f"{char}_{i}"
        os.makedirs(f"dataset/{char_name}", exist_ok=True)
        char_image_path = f"dataset/{char_name}/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_{i}.jpg"
        Image.fromarray(char_img_with_border).save(char_image_path)
    canvas.fill(255)
    detect_window.destroy()

def detect_screen(canvas, root):
    os.makedirs("images/temp", exist_ok=True)
    temp_image_path = f"images/temp/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.jpg"
    img = Image.fromarray(canvas)
    img.save(temp_image_path)
    characters = detect_characters(model, temp_image_path)
    detect_window = tk.Toplevel(root)
    detect_window.title("Detected Characters")
    max_columns = 6
    detect_window_width = len(characters) * 150 if len(characters) <= max_columns else max_columns * 150
    detect_window_height = int((len(characters) / max_columns + 1) * 230)
    box_width = 120
    box_height = 180
    img_size = (80, 80)
    padding = 5
    detect_window.geometry(f"{detect_window_width}x{detect_window_height}")
    bounds = get_character_bounds(canvas)

    row_frame = None
    entries = []
    for i, char in enumerate(characters):
        if i % max_columns == 0:
            row_frame = tk.Frame(detect_window)
            row_frame.pack(fill=tk.X, padx=10, pady=5)
        
        x, y, w, h = bounds[i]
        x_padded = max(x - padding, 0)
        y_padded = max(y - padding, 0)
        w_padded = min(w + 2 * padding, canvas.shape[1] - x_padded)
        h_padded = min(h + 2 * padding, canvas.shape[0] - y_padded)
        char_img = canvas[y_padded:y_padded + h_padded, x_padded:x_padded + w_padded]

        original_height, original_width = char_img.shape[:2]
        aspect_ratio = original_width / original_height
        if aspect_ratio > 1:
            new_width = img_size[0]
            new_height = int(new_width / aspect_ratio)
        else:
            new_height = img_size[1]
            new_width = int(new_height * aspect_ratio)

        char_img_resized = cv2.resize(char_img, (new_width, new_height), interpolation=cv2.INTER_AREA)
        char_box = tk.Frame(row_frame, width=box_width, height=box_height, borderwidth=2, relief="groove")
        char_box.pack_propagate(False)
        char_box.pack(side=tk.LEFT, padx=10, pady=5)
        char_img_pil = Image.fromarray(char_img_resized)
        char_img_tk = ImageTk.PhotoImage(char_img_pil)
        img_label = tk.Label(char_box, image=char_img_tk)
        img_label.image = char_img_tk
        img_label.pack(pady=(10, 5))
        char_label = tk.Label(char_box, text=char, font=("Arial", 14), fg="black")
        char_label.pack(pady=(0, 5))
        validate_cmd = detect_window.register(validate_input)
        entry = tk.Entry(char_box, font=("Arial", 12), width=5, validate="key", validatecommand=(validate_cmd, "%P"))
        entry.pack(pady=5)
        entries.append(entry)

    menu_frame = tk.Frame(detect_window)
    menu_frame.pack(fill=tk.X)
    tk.Button(menu_frame, text="Copy Characters", command=lambda: copy_characters(characters), bg="lightgray", fg="black").pack(side=tk.BOTTOM, pady=2)
    tk.Button(menu_frame, text="Export Character Images", command=lambda: export_char_images(detect_window, canvas, characters, bounds, entries), bg="lightgray", fg="black").pack(side=tk.BOTTOM, pady=2)

if __name__ == "__main__":
    print("This screen is not intended for direct use. Please run the 'main.py' file instead")
    input("Press Enter to exit...")
