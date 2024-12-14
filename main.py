import cv2
import numpy as np
import tkinter as tk
from tkinter import messagebox, filedialog
import threading
import os
from datetime import datetime
from PIL import Image, ImageTk
from video_stream import video_stream
from detect_screen import detect_screen
from train import start_training

if not os.path.exists("logs/model.pth"):
    threading.Thread(target=start_training).start()

root = tk.Tk()
root.title("HandSketch")
if os.name == "nt":
    root.iconbitmap("icon.ico")
root.attributes('-topmost', True)
root.update()
root.attributes('-topmost', False)
root.resizable(False, False)

video = cv2.VideoCapture(0)
allow_camera = messagebox.askyesno("Camera Permission", "Do you want to use the camera?") if video.isOpened() else False

if allow_camera:
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
else:
    width = 960
    height = 540
canvas = np.ones((height, width, 3), dtype="uint8") * 255

canvas_frame = tk.Frame(root, width=width, height=height)
canvas_frame.pack()
canvas_label = tk.Label(canvas_frame)
canvas_label.pack()
cursor_label = tk.Label(canvas_frame, width=1, height=1, bg="lightgrey")
cursor_label.place(x=-20, y=-20)

mouse_mode_var = tk.BooleanVar(value=not allow_camera)
mouse_drawing = False
mouse_erasing = False
prev_x, prev_y = None, None

def update_canvas():
    img = Image.fromarray(canvas)
    imgtk = ImageTk.PhotoImage(image=img)
    canvas_label.imgtk = imgtk
    canvas_label.configure(image=imgtk)
    canvas_label.after(10, update_canvas)

def mouse_press(event):
    global mouse_drawing, mouse_erasing, prev_x, prev_y
    if event.num == 1 and mouse_mode_var.get():
        mouse_drawing = True
        mouse_erasing = False
        prev_x, prev_y = event.x, event.y
    elif event.num == 3 and mouse_mode_var.get():
        mouse_drawing = False
        mouse_erasing = True

def mouse_release(event):
    global mouse_drawing, mouse_erasing
    mouse_drawing = False
    mouse_erasing = False

def mouse_motion(event):
    global mouse_drawing, mouse_erasing, prev_x, prev_y
    if mouse_mode_var.get():
        if mouse_drawing:
            if prev_x is not None and prev_y is not None:
                cv2.line(canvas, (prev_x, prev_y), (event.x, event.y), (0, 0, 0), 5)
            prev_x, prev_y = event.x, event.y
        elif mouse_erasing:
            cv2.circle(canvas, (event.x, event.y), 15, (255, 255, 255), -1)

def save_image():
    os.makedirs("images", exist_ok=True)
    file_path = filedialog.asksaveasfilename(
        defaultextension=".jpg",
        initialdir="images",
        initialfile=f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.jpg",
        filetypes=[("JPEG files", "*.jpg"), ("All files", "*.*")]
    )
    if file_path:
        img = Image.fromarray(canvas)
        img.save(file_path)

def toggle_mode():
    mouse_mode_var.set(not mouse_mode_var.get())
    if mouse_mode_var.get():
        camera_mode_label.config(text="Mode: MOUSE")
    else:
        camera_mode_label.config(text="Mode: CAMERA")

def train_again():
    if messagebox.askyesno("Train Again", "Are you sure you want to train again? You won't be able to close the application until the training is complete..."):
        threading.Thread(target=start_training).start()

menu_frame = tk.Frame(root)
menu_frame.pack(fill=tk.X)
tk.Button(menu_frame, text="Save Image", command=save_image, bg="lightgray", fg="black").pack(side=tk.LEFT, padx=5, pady=2)
tk.Button(menu_frame, text="Detect Characters", command=lambda: detect_screen(root, canvas, reopen=True) if np.any(canvas != 255) else None, bg="lightgray", fg="black").pack(side=tk.LEFT, padx=5, pady=2)
tk.Button(menu_frame, text="Clear Canvas", command=lambda: canvas.fill(255), bg="lightgray", fg="black").pack(side=tk.LEFT, padx=5, pady=2)
tk.Button(menu_frame, text="Train Again", command=train_again, bg="lightgray", fg="black").pack(side=tk.LEFT, padx=5, pady=2)
if allow_camera:
    tk.Button(menu_frame, text="Toggle Mode", command=toggle_mode, bg="lightgray", fg="black").pack(side=tk.LEFT, padx=5, pady=2)
    camera_mode_label = tk.Label(menu_frame, text="Mode: CAMERA", bg="lightgray", fg="black")
    camera_mode_label.pack(side=tk.LEFT, padx=10, pady=3)

canvas_label.bind("<ButtonPress>", mouse_press)
canvas_label.bind("<ButtonRelease>", mouse_release)
canvas_label.bind("<Motion>", mouse_motion)
root.protocol("WM_DELETE_WINDOW", lambda: root.quit())

if allow_camera:
    threading.Thread(target=lambda: video_stream(video, root, canvas, mouse_mode_var, prev_x, prev_y, cursor_label)).start()

update_canvas()

root.mainloop()
