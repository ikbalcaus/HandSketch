import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import pyperclip
import tkinter as tk
import os
from datetime import datetime
from PIL import Image, ImageTk
from configuration import CNNModel
from train import train_new_images
from detect import detect_characters

model = CNNModel()
if os.path.exists("logs/model.pth"):
    model.load_state_dict(torch.load("logs/model.pth", weights_only=True))
model.eval()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

def validate_input(char):
    return char == "" or (len(char) == 1 and (char.isdigit() or char.isalpha() and char.isascii()))

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

def save_results(detect_window, canvas, characters, bounds, entries):
    for i, (char, entry) in enumerate(zip(characters, entries)):
        if not entry.get():
            continue
        x, y, w, h = bounds[i]
        char_img = canvas[y:y+h, x:x+w]
        char_img_with_border = cv2.copyMakeBorder(
            char_img, 
            top=7, bottom=7, left=7, right=7, 
            borderType=cv2.BORDER_CONSTANT, 
            value=[255, 255, 255]
        )
        char_name = entry.get() or f"{char}_{i}"
        os.makedirs(f"dataset/{char_name}", exist_ok=True)
        char_image_path = f"dataset/{char_name}/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_{i}.jpg"
        Image.fromarray(char_img_with_border).save(char_image_path)
        train_new_images(model, criterion, optimizer, char_image_path, char_name)
    detect_window.destroy()

def detect_screen(canvas, root):
    os.makedirs("images/temp", exist_ok=True)
    temp_image_path = f"images/temp/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.jpg"
    img = Image.fromarray(canvas)
    img.save(temp_image_path)
    characters = detect_characters(model, temp_image_path)
    detect_window = tk.Toplevel(root)
    detect_window.title("Detected Characters")
    if os.name == "nt":
        detect_window.iconbitmap("icon.ico")
    detect_window.geometry("880x430")
    detect_window.resizable(False, False)
    detect_window.grab_set()
    detect_window.update()
    detect_window.focus_set()
    bounds = get_character_bounds(canvas)
    canvas_widget = tk.Canvas(detect_window, highlightthickness=0, borderwidth=0)
    frame = tk.Frame(canvas_widget)
    canvas_widget.create_window((0, 0), window=frame, anchor="nw")
    canvas_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    row_frame = None

    entries = []
    for i, char in enumerate(characters):
        if i % 6 == 0:
            row_frame = tk.Frame(frame)
            row_frame.pack(fill=tk.X, padx=10, pady=5)

        x, y, w, h = bounds[i]
        x_padded = max(x - 5, 0)
        y_padded = max(y - 5, 0)
        w_padded = min(w + 2 * 5, canvas.shape[1] - x_padded)
        h_padded = min(h + 2 * 5, canvas.shape[0] - y_padded)
        char_img = canvas[y_padded:y_padded + h_padded, x_padded:x_padded + w_padded]

        original_height, original_width = char_img.shape[:2]
        aspect_ratio = original_width / original_height
        if aspect_ratio > 1:
            new_width = 80
            new_height = int(new_width / aspect_ratio)
        else:
            new_height = 80
            new_width = int(new_height * aspect_ratio)

        char_img_resized = cv2.resize(char_img, (new_width, new_height), interpolation=cv2.INTER_AREA)
        char_box = tk.Frame(row_frame, width=120, height=180, borderwidth=2, relief="groove")
        char_box.pack_propagate(False)
        char_box.pack(side=tk.LEFT, padx=10, pady=5)
        char_img_pil = Image.fromarray(char_img_resized)
        char_img_tk = ImageTk.PhotoImage(char_img_pil)
        img_label = tk.Label(char_box, image=char_img_tk)
        img_label.image = char_img_tk
        img_label.pack(pady=(10, 5))
        tk.Label(char_box, text=char, font=("Arial", 14), fg="black").pack(pady=(0, 5))
        validate_cmd = detect_window.register(validate_input)
        entry = tk.Entry(char_box, font=("Arial", 12), width=5, validate="key", validatecommand=(validate_cmd, "%P"))
        entry.pack(pady=5)
        entries.append(entry)
        entry.bind("<FocusIn>", lambda e: e.widget.config(highlightthickness=0))
        entry.bind("<FocusOut>", lambda e: e.widget.config(highlightthickness=0))

    frame.update_idletasks()
    frame.config(height=frame.winfo_height() + 25)
    canvas_widget.config(scrollregion=canvas_widget.bbox("all"))

    menu_frame = tk.Frame(detect_window)
    menu_frame.place(relx=0.5, rely=1.0, anchor="s", width=detect_window.winfo_width())
    tk.Button(menu_frame, text="Save Results", command=lambda: save_results(detect_window, canvas, characters, bounds, entries), bg="lightgray", fg="black").pack(side=tk.LEFT, padx=5, pady=2)
    tk.Button(menu_frame, text="Copy Characters", command=lambda: copy_characters(characters), bg="lightgray", fg="black").pack(side=tk.LEFT, padx=5, pady=2)

    scroll_y = tk.Scrollbar(detect_window, orient="vertical", command=canvas_widget.yview)
    scroll_y.place(relx=1.0, rely=0.0, relheight=1.0, anchor="ne")
    canvas_widget.config(yscrollcommand=scroll_y.set)
    detect_window.bind_all("<MouseWheel>", lambda event: canvas_widget.yview_scroll(-1 if event.delta > 0 else 1, "units"))

if __name__ == "__main__":
    print("This file is not intended for direct use. Please run the 'main.py' file instead")
    input("Press Enter to exit...")
