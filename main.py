import cv2
import mediapipe as mp
import numpy as np
import threading
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk

mp_draw = mp.solutions.drawing_utils
mp_hand = mp.solutions.hands
video = cv2.VideoCapture(0)

width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
canvas = np.ones((height, width, 3), dtype="uint8") * 255

prev_x, prev_y = None, None

root = tk.Tk()
root.title("Hand Drawing Canvas")

canvas_frame = tk.Frame(root, width=width, height=height)
canvas_frame.pack()

canvas_label = tk.Label(canvas_frame)
canvas_label.pack()

cursor_label = tk.Label(canvas_frame, width=1, height=1, bg="lightgrey")
cursor_label.place(x=-20, y=-20)

def update_canvas():
    img = Image.fromarray(canvas)
    imgtk = ImageTk.PhotoImage(image=img)
    canvas_label.imgtk = imgtk
    canvas_label.configure(image=imgtk)
    canvas_label.after(10, update_canvas)

def video_stream():
    global prev_x, prev_y, canvas

    with mp_hand.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
        while True:
            ret, frame = video.read()
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            results = hands.process(rgb_frame)

            if results.multi_hand_landmarks:
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

                cursor_label.place(x=landmark_8_horizontal, y=landmark_8_vertical)

                if writing_gesture:
                    global prev_x, prev_y
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

def close_app():
    video.release()
    cv2.destroyAllWindows()
    root.destroy()

def save_image():
    file_path = filedialog.asksaveasfilename(defaultextension=".png", initialfile="image.png", filetypes=[("PNG files", "*.png"), ("All files", "*.*")])
    if file_path:
        img = Image.fromarray(canvas)
        img.save(file_path)

def clear_canvas():
    global canvas
    canvas = np.ones((height, width, 3), dtype="uint8") * 255

menu_frame = tk.Frame(root)
menu_frame.pack(fill=tk.X)

save_button = tk.Button(menu_frame, text="Save", command=save_image, bg="lightgray", fg="black").pack(side=tk.LEFT, padx=10, pady=3)
clear_button = tk.Button(menu_frame, text="Clear", command=clear_canvas, bg="lightgray", fg="black").pack(side=tk.LEFT, padx=0, pady=3)
exit_button = tk.Button(menu_frame, text="Exit", command=close_app, bg="lightgray", fg="black").pack(side=tk.LEFT, padx=10, pady=3)

root.protocol("WM_DELETE_WINDOW", close_app)

thread = threading.Thread(target=video_stream)
thread.start()

update_canvas()

root.mainloop()
