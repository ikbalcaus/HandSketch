import cv2
import mediapipe as mp
import numpy as np
import threading
import tkinter as tk
import os
from datetime import datetime
from tkinter import filedialog
from PIL import Image, ImageTk
from detect_screen import detect_screen

mp_draw = mp.solutions.drawing_utils
mp_hand = mp.solutions.hands
video = cv2.VideoCapture(0)

if video.isOpened():
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
else:
    width = 640
    height = 480
canvas = np.ones((height, width, 3), dtype="uint8") * 255

root = tk.Tk()
root.title("Hand Drawing Canvas")

canvas_frame = tk.Frame(root, width=width, height=height)
canvas_frame.pack()

canvas_label = tk.Label(canvas_frame)
canvas_label.pack()

cursor_label = tk.Label(canvas_frame, width=1, height=1, bg="lightgrey")
cursor_label.place(x=-20, y=-20)

mouse_mode = True
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
    if event.num == 1:
        if mouse_mode:
            mouse_drawing = True
            mouse_erasing = False
            prev_x, prev_y = event.x, event.y
    elif event.num == 3:
        if mouse_mode:
            mouse_drawing = False
            mouse_erasing = True

def mouse_release(event):
    global mouse_drawing, mouse_erasing
    mouse_drawing = False
    mouse_erasing = False

def mouse_motion(event):
    global mouse_drawing, mouse_erasing, prev_x, prev_y
    if mouse_mode:
        if mouse_drawing:
            if prev_x is not None and prev_y is not None:
                cv2.line(canvas, (prev_x, prev_y), (event.x, event.y), (0, 0, 0), 5)
            prev_x, prev_y = event.x, event.y
        elif mouse_erasing:
            cv2.circle(canvas, (event.x, event.y), 15, (255, 255, 255), -1)

def video_stream():
    global canvas, mouse_mode, prev_x, prev_y

    with mp_hand.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
        while True:
            ret, frame = video.read()            
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            results = hands.process(rgb_frame)

            if results.multi_hand_landmarks and not mouse_mode:
                hand_landmark = results.multi_hand_landmarks[0]
                mp_draw.draw_landmarks(frame, hand_landmark, mp_hand.HAND_CONNECTIONS)

                landmark_3_horizontal = int(hand_landmark.landmark[3].x * frame.shape[1])
                landmark_4_horizontal = int(hand_landmark.landmark[4].x * frame.shape[1])
                landmark_5_horizontal = int(hand_landmark.landmark[5].x * frame.shape[1])
                landmark_7_vertical = int(hand_landmark.landmark[7].y * frame.shape[0])
                landmark_8_horizontal = int(hand_landmark.landmark[8].x * frame.shape[1])
                landmark_8_vertical = int(hand_landmark.landmark[8].y * frame.shape[0])
                landmark_11_vertical = int(hand_landmark.landmark[11].y * frame.shape[0])
                landmark_12_vertical = int(hand_landmark.landmark[12].y * frame.shape[0])
                landmark_15_vertical = int(hand_landmark.landmark[15].y * frame.shape[0])
                landmark_16_vertical = int(hand_landmark.landmark[16].y * frame.shape[0])
                landmark_17_horizontal = int(hand_landmark.landmark[17].x * frame.shape[1])
                landmark_19_vertical = int(hand_landmark.landmark[19].y * frame.shape[0])
                landmark_20_vertical = int(hand_landmark.landmark[20].y * frame.shape[0])

                cursor_label.place(x=landmark_8_horizontal, y=landmark_8_vertical)

                erasing_gesture = (
                    landmark_8_vertical < landmark_7_vertical
                    and landmark_12_vertical < landmark_11_vertical
                    and landmark_16_vertical < landmark_15_vertical
                    and landmark_20_vertical < landmark_19_vertical
                )

                writing_gesture = (
                    (
                        (
                            landmark_5_horizontal < landmark_17_horizontal
                            and landmark_3_horizontal < landmark_4_horizontal
                        )
                        or (
                            landmark_17_horizontal < landmark_5_horizontal
                            and landmark_4_horizontal < landmark_3_horizontal
                        )
                    )
                    and landmark_8_vertical < landmark_7_vertical
                    and not erasing_gesture
                )

                if writing_gesture:
                    if prev_x is not None and prev_y is not None:
                        cv2.line(canvas, (prev_x, prev_y), (landmark_8_horizontal, landmark_8_vertical), (0, 0, 0), 5)
                    prev_x, prev_y = landmark_8_horizontal, landmark_8_vertical
                else:
                    prev_x, prev_y = None, None
                    
                if erasing_gesture:
                    cv2.circle(canvas, (landmark_8_horizontal, landmark_8_vertical), 15, (255, 255, 255), -1)
                    prev_x, prev_y = None, None

                cv2.circle(frame, (landmark_8_horizontal, landmark_8_vertical), 5, (211, 211, 211), -1)

            else:
                prev_x, prev_y = None, None
                cursor_label.place(x=-20, y=-20)

            cv2.imshow("Frame", frame)

            if cv2.waitKey(1) and not cv2.getWindowProperty("Frame", cv2.WND_PROP_VISIBLE):
                break

    video.release()
    cv2.destroyAllWindows()

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

def clear_canvas():
    global canvas
    canvas.fill(255)

def toggle_mode():
    global video, mouse_mode, mouse_drawing, mouse_erasing, prev_x, prev_y
    mouse_mode = not mouse_mode
    mouse_drawing = False
    mouse_erasing = False
    prev_x, prev_y = None, None
    if mouse_mode:
        camera_mode_label.config(text="Mode: MOUSE")
    else:
        camera_mode_label.config(text="Mode: CAMERA")

def close_app():
    for image in os.listdir("images/temp"):
        os.remove(os.path.join("images/temp", image))
    os.rmdir("images/temp")
    video.release()
    cv2.destroyAllWindows()
    root.destroy()

menu_frame = tk.Frame(root)
menu_frame.pack(fill=tk.X)

tk.Button(menu_frame, text="Save Image", command=save_image, bg="lightgray", fg="black").pack(side=tk.LEFT, padx=10, pady=3)
tk.Button(menu_frame, text="Clear Canvas", command=clear_canvas, bg="lightgray", fg="black").pack(side=tk.LEFT, padx=10, pady=3)
tk.Button(menu_frame, text="Convert to Text", command=lambda: detect_screen(canvas, root), bg="lightgray", fg="black").pack(side=tk.LEFT, padx=10, pady=3)
tk.Button(menu_frame, text="Toggle Mode", command=toggle_mode, bg="lightgray", fg="black").pack(side=tk.LEFT, padx=10, pady=3)

canvas_label.bind("<ButtonPress>", mouse_press)
canvas_label.bind("<ButtonRelease>", mouse_release)
canvas_label.bind("<Motion>", mouse_motion)

camera_mode_label = tk.Label(menu_frame, text="Mode: MOUSE", bg="lightgray", fg="black")
camera_mode_label.pack(side=tk.LEFT, padx=10, pady=3)

root.protocol("WM_DELETE_WINDOW", close_app)

thread = threading.Thread(target=video_stream)
thread.start()

update_canvas()

root.mainloop()
