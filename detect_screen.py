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
from detect import detect_characters, get_character_contours

model = CNNModel()
if os.path.exists("logs/model.pth"):
    model.load_state_dict(torch.load("logs/model.pth", weights_only=True))
model.eval()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

def validate_input(char):
    return char == "" or (len(char) == 1 and (char.isdigit() or char.isalpha() and char.isascii()))

def focus_next_entry(event, entries):
    current_index = entries.index(event.widget)
    next_index = (current_index + 1) % len(entries)
    entries[next_index].focus_set()
    return "break"

def copy_characters(characters):
    text = "".join(characters)
    pyperclip.copy(text)

def delete_character(i, entries, char_boxes, characters, bounds):
    entries[i].delete(0, tk.END)
    char_boxes[i].destroy()
    entries[i] = None
    char_boxes[i] = None
    characters[i] = None
    bounds[i] = None

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
    detect_window.destroy()

def detect_screen(canvas, root):
    characters = detect_characters(model, canvas)
    detect_window = tk.Toplevel(root)
    detect_window.title("Detected Characters")
    if os.name == "nt":
        detect_window.iconbitmap("icon.ico")
    detect_window.geometry("880x260")
    detect_window.resizable(False, False)
    detect_window.update()
    detect_window.focus_set()
    contours, _ = get_character_contours(canvas)
    bounds = [cv2.boundingRect(contour) for contour in contours]
    frame = tk.Frame(detect_window)
    char_boxes = []
    entries = []
    
    for i, char in enumerate(characters):
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
        char_box = tk.Frame(detect_window, width=120, height=210, borderwidth=2, relief="groove")
        char_box.pack_propagate(False)
        char_box.pack(side=tk.LEFT, padx=5, pady=5, anchor="n")
        char_boxes.append(char_box)

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
        entry.bind("<Tab>", lambda event, entries=entries: focus_next_entry(event, entries))
        tk.Button(char_box, text="X", command=lambda i=i: delete_character(i, entries, char_boxes, characters, bounds), font=("Arial", 10)).pack(side=tk.BOTTOM, padx=5, pady=5)

    frame.update_idletasks()
    frame.config(height=frame.winfo_height() + 25)
    menu_frame = tk.Frame(detect_window)
    menu_frame.place(relx=0.5, rely=1.0, anchor="s", width=detect_window.winfo_width())
    tk.Button(menu_frame, text="Save Results", command=lambda: save_results(detect_window, canvas, characters, bounds, entries), bg="lightgray", fg="black").pack(side=tk.LEFT, padx=5, pady=2)
    tk.Button(menu_frame, text="Copy Characters", command=lambda: copy_characters(characters), bg="lightgray", fg="black").pack(side=tk.LEFT, padx=5, pady=2)

if __name__ == "__main__":
    print("This file is not intended for direct use. Please run the 'main.py' file instead")
    input("Press Enter to exit...")
