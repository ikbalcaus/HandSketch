import cv2
import mediapipe as mp

mp_draw = mp.solutions.drawing_utils
mp_hand = mp.solutions.hands

def video_stream(video, canvas=None, mouse_mode_var=None, prev_x=None, prev_y=None, cursor_label=None):
    with mp_hand.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
        while True:
            ret, frame = video.read()            
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_frame)

            if results.multi_hand_landmarks and not mouse_mode_var.get():
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

                cv2.circle(frame, (landmark_8_horizontal, landmark_8_vertical), 5, (211, 211, 211), -1)
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

            else:
                prev_x, prev_y = None, None
                cursor_label.place(x=-20, y=-20)

            cv2.imshow("Frame", frame)

            if cv2.waitKey(1) and not cv2.getWindowProperty("Frame", cv2.WND_PROP_VISIBLE):
                break

    video.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    video = cv2.VideoCapture(0)
    class DummyClass():
        @staticmethod
        def get():
            return False
        @staticmethod
        def place(x, y):
            pass

    if not video.isOpened():
        print("Camera is not detected")
        input("Press Enter to exit...")
    else:
        video_stream(video=video, mouse_mode_var=DummyClass(), cursor_label=DummyClass())
