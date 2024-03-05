import tkinter as tk
from tkinter import ttk
import threading
import cv2
import mediapipe as mp
import numpy as np
from PIL import Image, ImageTk

webcam_thread = None
webcam_running = False
frame_label = None
hand_area_label = None

def Hand_Open(hand_landmarks,hand_label):
    if Victory_Sign(hand_landmarks,hand_label):
        return False
    fingertip_ids = [4, 8, 12, 16, 20]
    wrist_id = 0
    palm_base_id = 9
    wrist = np.array([hand_landmarks.landmark[wrist_id].x, hand_landmarks.landmark[wrist_id].y])
    palm_base = np.array([hand_landmarks.landmark[palm_base_id].x, hand_landmarks.landmark[palm_base_id].y])
    wrist_to_palm_base = np.linalg.norm(wrist - palm_base)
    open_hand_signals = 0
    for fingertip_id in fingertip_ids:
        fingertip = np.array([hand_landmarks.landmark[fingertip_id].x, hand_landmarks.landmark[fingertip_id].y])
        wrist_to_fingertip = np.linalg.norm(wrist - fingertip)
        ratio = wrist_to_fingertip / wrist_to_palm_base
        if ratio > 1:
            open_hand_signals += 1
    return open_hand_signals == 5

def Hand_Orientation(hand_landmarks, hand_label):
    ###TIPS
    thumb_tip = np.array([hand_landmarks.landmark[4].x, hand_landmarks.landmark[4].y, hand_landmarks.landmark[4].z])
    index_tip = np.array([hand_landmarks.landmark[8].x, hand_landmarks.landmark[8].y, hand_landmarks.landmark[8].z])
    middle_tip = np.array([hand_landmarks.landmark[12].x, hand_landmarks.landmark[12].y, hand_landmarks.landmark[12].z])
    pinky_tip = np.array([hand_landmarks.landmark[20].x, hand_landmarks.landmark[20].y, hand_landmarks.landmark[20].z])
    ###KNUCKLES AND FINGER BASES
    index_base = np.array([hand_landmarks.landmark[5].x, hand_landmarks.landmark[5].y, hand_landmarks.landmark[5].z])
    middle_base = np.array([hand_landmarks.landmark[9].x, hand_landmarks.landmark[9].y, hand_landmarks.landmark[9].z])
    pinky_base = np.array([hand_landmarks.landmark[17].x, hand_landmarks.landmark[17].y, hand_landmarks.landmark[17].z])
    index_knuckle = np.array([hand_landmarks.landmark[6].x, hand_landmarks.landmark[6].y, hand_landmarks.landmark[6].z])
    middle_knuckle = np.array([hand_landmarks.landmark[10].x, hand_landmarks.landmark[10].y, hand_landmarks.landmark[10].z])
    pinky_knuckle = np.array([hand_landmarks.landmark[14].x, hand_landmarks.landmark[14].y, hand_landmarks.landmark[14].z])
    
    thumb_to_index_base = index_base - thumb_tip
    thumb_to_middle_base = middle_base - thumb_tip
    thumb_to_pinky_base = pinky_base - thumb_tip
    thumb_to_index_knuckle = index_knuckle - thumb_tip
    thumb_to_middle_knuckle = middle_knuckle - thumb_tip
    thumb_to_pinky_knuckle = pinky_knuckle - thumb_tip
    
    #thumb_to_index = index_tip - thumb_tip
    #thumb_to_middle = middle_tip - thumb_tip
    #thumb_to_pinky = pinky_tip - thumb_tip
    
    cross_index_base = np.cross(thumb_to_index_base, thumb_to_pinky_base)
    cross_middle_base = np.cross(thumb_to_middle_base, thumb_to_pinky_base)
    cross_index_knuckle = np.cross(thumb_to_index_knuckle, thumb_to_pinky_knuckle)
    cross_middle_knuckle = np.cross(thumb_to_middle_knuckle, thumb_to_pinky_knuckle)
    
    avg_z_base = (cross_index_base[2] + cross_middle_base[2]) / 2
    avg_z_knuckle = (cross_index_knuckle[2] + cross_middle_knuckle[2]) / 2
    if hand_label == "Right":
        if avg_z_base and avg_z_knuckle > 0:
            return "Inside"
        else:
            return "Outside"
    else:
        if avg_z_base and avg_z_knuckle < 0:
            return "Inside"
        else:
            return "Outside"
        
def Victory_Sign(hand_landmarks, hand_label):
    thumb_tip = np.array([hand_landmarks.landmark[4].x, hand_landmarks.landmark[4].y, hand_landmarks.landmark[4].z])
    index_tip = np.array([hand_landmarks.landmark[8].x, hand_landmarks.landmark[8].y, hand_landmarks.landmark[8].z])
    middle_tip = np.array([hand_landmarks.landmark[12].x, hand_landmarks.landmark[12].y, hand_landmarks.landmark[12].z])
    ring_tip = np.array([hand_landmarks.landmark[16].x, hand_landmarks.landmark[16].y, hand_landmarks.landmark[16].z])
    pinky_tip = np.array([hand_landmarks.landmark[20].x, hand_landmarks.landmark[20].y, hand_landmarks.landmark[20].z])
    palm_base = np.array([hand_landmarks.landmark[0].x, hand_landmarks.landmark[0].y, hand_landmarks.landmark[0].z])
    
    thumb_base = np.array([hand_landmarks.landmark[1].x, hand_landmarks.landmark[1].y, hand_landmarks.landmark[1].z])
    index_base = np.array([hand_landmarks.landmark[5].x, hand_landmarks.landmark[5].y, hand_landmarks.landmark[5].z])
    middle_base = np.array([hand_landmarks.landmark[9].x, hand_landmarks.landmark[9].y, hand_landmarks.landmark[9].z])
    ring_base = np.array([hand_landmarks.landmark[13].x, hand_landmarks.landmark[13].y, hand_landmarks.landmark[13].z])
    pinky_base = np.array([hand_landmarks.landmark[17].x, hand_landmarks.landmark[17].y, hand_landmarks.landmark[17].z])
    
    thumb_to_palm = np.linalg.norm(thumb_tip - palm_base)
    thumb_base_to_palm = np.linalg.norm(thumb_base - palm_base)
    index_to_palm = np.linalg.norm(index_tip - palm_base)
    index_base_to_palm = np.linalg.norm(index_base - palm_base)
    middle_to_palm = np.linalg.norm(middle_tip - palm_base)
    middle_base_to_palm = np.linalg.norm(middle_base - palm_base)
    ring_to_palm = np.linalg.norm(ring_tip - palm_base)
    ring_base_to_palm = np.linalg.norm(ring_base - palm_base)
    pinky_to_palm = np.linalg.norm(pinky_tip - palm_base)
    pinky_base_to_palm = np.linalg.norm(pinky_base - palm_base)
    
    index_extended = index_to_palm > index_base_to_palm
    middle_extended = middle_to_palm > middle_base_to_palm
    
    #thumb_curled = thumb_to_palm < thumb_base_to_palm
    ring_curled = ring_to_palm < ring_base_to_palm
    pinky_curled = pinky_to_palm < pinky_base_to_palm
    
    orientation = Hand_Orientation(hand_landmarks, hand_label)
    
    return index_extended and middle_extended and ring_curled and pinky_curled and orientation == "Inside"
    
def Capture_Video():
    global webcam_running, frame_label
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.6, min_tracking_confidence=0.5)
    cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)  
    cam.set(cv2.CAP_PROP_FPS,  30)  
    frame_counter = 0
    pixel_size = 0.03
    scaling = pixel_size ** 2
    if not cam.isOpened():
        raise IOError("webcam not openable")
    open_hand_count = 0
    try:
        while webcam_running:
            ret, frame = cam.read()
            if not ret:
                print("no stream")
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)
            if results.multi_hand_landmarks:
                for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                    hand_label = handedness.classification[0].label
                    orientation = Hand_Orientation(hand_landmarks, hand_label)
                    x_min = min([lm.x for lm in hand_landmarks.landmark]) * frame.shape[1]
                    x_max = max([lm.x for lm in hand_landmarks.landmark]) * frame.shape[1]
                    y_min = min([lm.y for lm in hand_landmarks.landmark]) * frame.shape[0]
                    y_max = max([lm.y for lm in hand_landmarks.landmark]) * frame.shape[0]
                    x_min, x_max, y_min, y_max = int(x_min), int(x_max), int(y_min), int(y_max)
                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0,   255,   0),   2)
                    orientation = Hand_Orientation(hand_landmarks, hand_label)
                                    
                    if Victory_Sign(hand_landmarks, hand_label):
                        cv2.putText(frame, "Victory sign",(x_min, y_min - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
                        continue
                    
                    if Hand_Open(hand_landmarks, hand_label) and orientation == "Inside":
                        open_hand_count +=   1
                        print(f"open hand {open_hand_count}")
                        cv2.putText(frame, "Hand open", (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                        frame_counter +=1
                        if frame_counter == 15:
                            hand_area_pixels = (x_max - x_min) * (y_max - y_min)
                            hand_area = int(hand_area_pixels * scaling)
                            hand_area_label.config(text=f"Hand area: {hand_area} UNITS")
                            frame_counter = 0
                            hand_area = 0
                    elif not Hand_Open(hand_landmarks,hand_label):
                        cv2.putText(frame, "Hand closed", (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                    else:
                        cv2.putText(frame, "Outside", (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
            cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
            img = Image.fromarray(cv2image)
            imgtk = ImageTk.PhotoImage(image=img)
            frame_label.imgtk = imgtk
            frame_label.configure(image=imgtk)
            frame_label.image = imgtk
            if cv2.waitKey(1) & 0xFF == ord('q'):
                webcam_running = False
                break
    finally:
        cam.release()
        cv2.destroyAllWindows()


def start_webcam():
    global webcam_thread, webcam_running
    if not webcam_running:
        webcam_running = True
        webcam_thread = threading.Thread(target=Capture_Video)
        webcam_thread.start()
        start_button.config(state=tk.DISABLED)
        stop_button.config(state=tk.NORMAL)

def stop_webcam():
    global webcam_running
    if webcam_running:
        webcam_running = False
        start_button.config(state=tk.NORMAL)
        stop_button.config(state=tk.DISABLED)

def quit_app():
    global webcam_running
    webcam_running = False
    root.destroy()

root = tk.Tk()
root.title("Raspberry Pi Robot")

hand_area_label = tk.Label(root, text="Hand area: 0",font=("Arial, 20"))
hand_area_label.pack(side=tk.BOTTOM)

start_button = ttk.Button(root, text="START WEBCAM", command=start_webcam)
start_button.pack(side=tk.LEFT, padx=10, pady=10)

stop_button = ttk.Button(root, text="STOP", command=stop_webcam, state=tk.DISABLED)
stop_button.pack(side=tk.LEFT, padx=10, pady=10)

quit_button = ttk.Button(root, text="QUIT", command=quit_app)
quit_button.pack(side=tk.LEFT, padx=10, pady=10)

frame_label = tk.Label(root)
frame_label.pack(side=tk.LEFT)


root.mainloop()
