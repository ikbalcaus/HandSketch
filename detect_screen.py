import cv2
import torch
import tkinter as tk
import os
import pyperclip
from datetime import datetime
from PIL import Image, ImageTk
from configuration import CNNModel
from detect import detect_characters, get_characters_contours

detect_window_ref = None

def format_characters(characters):
    return "".join([character for character in characters if character is not None])

def focus_next_entry(event, entries):
    entries = [entry for entry in entries if entry is not None]
    current_index = entries.index(event.widget)
    next_index = (current_index + 1) % len(entries)
    entries[next_index].focus_set()
    return "break"

def copy_characters(characters):
    text = format_characters(characters)
    pyperclip.copy(text)

def delete_space(i, space_boxes, characters, characters_label):
    characters[i] = None
    space_boxes[i].destroy()
    space_boxes[i] = None
    characters_label.config(text=f"Characters: {format_characters(characters)}")

def delete_character(i, entries, char_boxes, space_boxes, characters, bounds, detect_window, characters_label):
    characters[i] = None
    bounds[i] = None
    entries[i].delete(0, tk.END)
    char_boxes[i].destroy()
    entries[i] = None
    char_boxes[i] = None
    temp_character = None
    if format_characters(characters).strip() == "":
        detect_window.destroy()
        return
    for character in characters:
        if character is not None:
            if character == " ":
                for first_space_box_index in range(len(space_boxes)):
                    if space_boxes[first_space_box_index] is not None:
                        delete_space(first_space_box_index, space_boxes, characters, characters_label)
                        break
            else:
                break
    for character in characters:
        if character is not None:
            if character == temp_character == " ":
                delete_space(i + 1, space_boxes, characters, characters_label)
                break
            temp_character = character
    characters_label.config(text=f"Characters: {format_characters(characters)}")

def save_results(detect_window, canvas, characters, bounds, entries):
    characters = format_characters(characters)
    entries = [entry for entry in entries if entry is not None]
    bounds = [bound for bound in bounds if bound is not None]
    for i, (character, entry) in enumerate(zip(characters, entries)):
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
        char_name = entry.get() or f"{character}_{i}"
        os.makedirs(f"dataset/{char_name}", exist_ok=True)
        char_image_path = f"dataset/{char_name}/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_{i}.jpg"
        Image.fromarray(char_img_with_border).save(char_image_path)
    detect_window.destroy()

def on_close():
    global detect_window_ref
    detect_window_ref.destroy()
    detect_window_ref = None

def detect_screen(root, canvas, reopen):
    global detect_window_ref

    if reopen and detect_window_ref is not None:
        detect_window_ref.destroy()
        detect_window_ref = None

    if detect_window_ref is not None and tk.Toplevel.winfo_exists(detect_window_ref):
        detect_window_ref.focus_set()
        return

    model = CNNModel()
    model.load_state_dict(torch.load("logs/model.pth", weights_only=True))
    model.eval()

    characters = detect_characters(model, canvas)
    contours, _, end_rows = get_characters_contours(canvas)
    bounds = [cv2.boundingRect(contour) for contour in contours]
    char_boxes = []
    space_boxes = []
    entries = []
    for i in range(len(bounds) - 1, -1, -1):
        if i in end_rows:
            bounds.insert(i + 1, None)

    detect_window = tk.Toplevel(root)
    detect_window_ref = detect_window
    detect_window.title("Detected Characters")
    if os.name == "nt":
        detect_window.iconbitmap("icon.ico")
    detect_window.geometry("880x300")
    detect_window.resizable(True, True)
    detect_window.update()
    detect_window.focus_set()

    container = tk.Frame(detect_window)
    container.pack(fill=tk.BOTH, expand=True)
    scroll_canvas = tk.Canvas(container)
    scroll_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    scroll_frame = tk.Frame(scroll_canvas)
    scroll_canvas.create_window((0, 0), window=scroll_frame, anchor="nw")
    scroll_frame.update_idletasks()

    menu_frame = tk.Frame(detect_window)
    menu_frame.pack(fill=tk.X, side=tk.BOTTOM)
    tk.Button(menu_frame, text="Save Results", command=lambda: save_results(detect_window, canvas, characters, bounds, entries), bg="lightgray", fg="black").pack(side=tk.LEFT, padx=5, pady=2)
    tk.Button(menu_frame, text="Copy Characters", command=lambda: copy_characters(characters), bg="lightgray", fg="black").pack(side=tk.LEFT, padx=5, pady=2)
    characters_label = tk.Label(menu_frame, text=f"Characters: {format_characters(characters)}", bg="lightgray", fg="black")
    characters_label.pack(side=tk.LEFT, padx=10, pady=3)
    tk.Button(menu_frame, text=">", command=lambda: scroll_canvas.xview_scroll(1, "units")).pack(side=tk.RIGHT, padx=5, pady=2)
    tk.Button(menu_frame, text="<", command=lambda: scroll_canvas.xview_scroll(-1, "units")).pack(side=tk.RIGHT, padx=5, pady=2)

    for i, character in enumerate(characters):
        if character != " ":
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
            char_box = tk.Frame(scroll_frame, width=120, height=210, borderwidth=2, relief="groove")
            char_box.pack_propagate(False)
            char_box.pack(side=tk.LEFT, padx=5, pady=5, anchor="n")

            char_img_pil = Image.fromarray(char_img_resized)
            char_img_tk = ImageTk.PhotoImage(char_img_pil)
            img_label = tk.Label(char_box, image=char_img_tk)
            img_label.image = char_img_tk
            img_label.pack(pady=(10, 5))
            tk.Label(char_box, text=character, font=("Arial", 14), fg="black").pack(pady=(0, 5))
            validate_cmd = detect_window.register(lambda x: len(x) == 0 or (len(x) == 1 and (x.isdigit() or x.isalpha())))
            entry = tk.Entry(char_box, font=("Arial", 12), width=5, validate="key", validatecommand=(validate_cmd, "%P"))
            entry.pack(pady=5)
            entry.bind("<FocusIn>", lambda e: scroll_canvas.config(highlightthickness=0))
            entry.bind("<FocusOut>", lambda e: scroll_canvas.config(highlightthickness=0))
            entry.bind("<Tab>", lambda event, entries=entries: focus_next_entry(event, entries))

            char_boxes.append(char_box)
            space_boxes.append(None)
            entries.append(entry)
            tk.Button(char_box, text="X", command=lambda i=i: delete_character(i, entries, char_boxes, space_boxes, characters, bounds, detect_window, characters_label), font=("Arial", 10)).pack(side=tk.BOTTOM, padx=5, pady=5)
        else:
            space_box = tk.Frame(scroll_frame, width=120, height=210, borderwidth=2, relief="groove")
            space_box.pack_propagate(False)
            space_box.pack(side=tk.LEFT, padx=5, pady=5, anchor="n")
            char_boxes.append(None)
            space_boxes.append(space_box)
            entries.append(None)
            tk.Button(space_box, text="X", command=lambda i=i: delete_space(i, space_boxes, characters, characters_label), font=("Arial", 10)).pack(side=tk.BOTTOM, padx=5, pady=5)
            tk.Label(space_box, text="[SPACE]", font=("Arial", 14), fg="black").pack(pady=(10, 5), side=tk.BOTTOM)

    detect_window.protocol("WM_DELETE_WINDOW", on_close)

if __name__ == "__main__":
    print("This file is not intended for direct use. Please run the 'main.py' file instead")
    input("Press Enter to exit...")
