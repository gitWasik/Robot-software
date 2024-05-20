import tkinter as tk
from tkinter import ttk
import threading
import cv2
import mediapipe as mp
import numpy as np
from PIL import Image, ImageTk
import time
#import RPi.GPIO as GPIO
import Mock.GPIO as GPIO
from picamera2 import Picamera2
import sys

webcam_thread = None
webcam_running = False
frame_label = None
hand_area_label = None
black_image_tk = None
hands = None
picam2 = None
frame_counter = 0
gesture_counts = {}
frame_counter = 0
confirmed_gesture = None
last_confirmed_gesture = None
gesture_navigation_running = False

PWMA = None
PWMB = None
IN1 = 13
IN2 = 12
ENA = 6
IN3 = 21
IN4 = 20
ENB = 26
PA = 20
PB = 20

#====================================================================================================================
#MOTOR CONTROL

def setup_GPIO():
    global IN1, IN2, ENA, IN3, ENB, PA, PB, PWMA, PWMB
    GPIO.setmode(GPIO.BCM)
    GPIO.setwarnings(False)
    GPIO.setup(IN1, GPIO.OUT)
    GPIO.setup(IN2, GPIO.OUT)
    GPIO.setup(IN3, GPIO.OUT)
    GPIO.setup(IN4, GPIO.OUT)
    GPIO.setup(ENA, GPIO.OUT)
    GPIO.setup(ENB, GPIO.OUT)
    PWMA = GPIO.PWM(ENA, 500)
    PWMB = GPIO.PWM(ENB, 500)
    PWMA.start(PA)
    PWMB.start(PB)

def stop():
    global PWMA, PWMB, continuous_movement
    continuous_movement = False
    if PWMA is not None and PWMB is not None:
        PWMA.ChangeDutyCycle(0)
        PWMB.ChangeDutyCycle(0)
        GPIO.output(IN1, GPIO.LOW)
        GPIO.output(IN2, GPIO.LOW)
        GPIO.output(IN3, GPIO.LOW)
        GPIO.output(IN4, GPIO.LOW)

def forward():
    global PWMA, PWMB, continuous_movement
    continuous_movement = True
    
    def continuous_forward():
        while continuous_movement:
            GPIO.output(IN1, GPIO.HIGH)
            GPIO.output(IN2, GPIO.LOW)
            GPIO.output(IN3, GPIO.HIGH)
            GPIO.output(IN4, GPIO.LOW)
            PWMA.ChangeDutyCycle(PA)
            PWMB.ChangeDutyCycle(PB)
            time.sleep(0.03)  
    
    threading.Thread(target=continuous_forward).start()

def backward():
    global PWMA, PWMB, continuous_movement
    continuous_movement = True
    
    def continuous_backward():
        while continuous_movement:
            GPIO.output(IN1, GPIO.LOW)
            GPIO.output(IN2, GPIO.HIGH)
            GPIO.output(IN3, GPIO.LOW)
            GPIO.output(IN4, GPIO.HIGH)
            PWMA.ChangeDutyCycle(PA)
            PWMB.ChangeDutyCycle(PB)
            time.sleep(0.03)  
    
    threading.Thread(target=continuous_backward).start()

def left():
    global PWMA, PWMB
    PWMA.ChangeDutyCycle(30)
    PWMB.ChangeDutyCycle(30)
    GPIO.output(IN1, GPIO.LOW)
    GPIO.output(IN2, GPIO.HIGH)
    GPIO.output(IN3, GPIO.HIGH)
    GPIO.output(IN4, GPIO.LOW)

def right():
    global PWMA, PWMB
    PWMA.ChangeDutyCycle(30)
    PWMB.ChangeDutyCycle(30)
    GPIO.output(IN1, GPIO.HIGH)
    GPIO.output(IN2, GPIO.LOW)
    GPIO.output(IN3, GPIO.LOW)
    GPIO.output(IN4, GPIO.HIGH)

def setPWMA(value):
    global PA, PWMA
    PA = value
    PWMA.ChangeDutyCycle(PA)

def setPWMB(value):
    global PB, PWMB
    PB = value
    PWMB.ChangeDutyCycle(PB)

def setMotor(left, right):
    if 0 <= right <= 100:
        GPIO.output(IN1, GPIO.HIGH)
        GPIO.output(IN2, GPIO.LOW)
        PWMA.ChangeDutyCycle(right)
    elif -100 <= right < 0:
        GPIO.output(IN1, GPIO.LOW)
        GPIO.output(IN2, GPIO.HIGH)
        PWMA.ChangeDutyCycle(0 - right)
    if 0 <= left <= 100:
        GPIO.output(IN3, GPIO.HIGH)
        GPIO.output(IN4, GPIO.LOW)
        PWMB.ChangeDutyCycle(left)
    elif -100 <= left < 0:
        GPIO.output(IN3, GPIO.LOW)
        GPIO.output(IN4, GPIO.HIGH)
        PWMB.ChangeDutyCycle(0 - left)

#========================================================================================================================
#GESTY
        
def Hand_Orientation(hand_landmarks, hand_label):
    thumb_tip = np.array([hand_landmarks.landmark[4].x, hand_landmarks.landmark[4].y, hand_landmarks.landmark[4].z])
    index_tip = np.array([hand_landmarks.landmark[8].x, hand_landmarks.landmark[8].y, hand_landmarks.landmark[8].z])
    middle_tip = np.array([hand_landmarks.landmark[12].x, hand_landmarks.landmark[12].y, hand_landmarks.landmark[12].z])
    pinky_tip = np.array([hand_landmarks.landmark[20].x, hand_landmarks.landmark[20].y, hand_landmarks.landmark[20].z])
    
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
    
    cross_index_base = np.cross(thumb_to_index_base, thumb_to_pinky_base)
    cross_middle_base = np.cross(thumb_to_middle_base, thumb_to_pinky_base)
    cross_index_knuckle = np.cross(thumb_to_index_knuckle, thumb_to_pinky_knuckle)
    cross_middle_knuckle = np.cross(thumb_to_middle_knuckle, thumb_to_pinky_knuckle)
    
    avg_z_base = (cross_index_base[2] + cross_middle_base[2]) / 2
    avg_z_knuckle = (cross_index_knuckle[2] + cross_middle_knuckle[2]) / 2
    
    if hand_label == "Right":
        if avg_z_base > 0 and avg_z_knuckle > 0:
            return "Inside"
        else:
            return "Outside"
    else:
        if avg_z_base < 0 and avg_z_knuckle < 0:
            return "Inside"
        else:
            return "Outside"
        

def Znak_S(hand_landmarks, hand_label):
    if Hand_Orientation(hand_landmarks, hand_label) == "Outside":
        return False
    if Znak_R(hand_landmarks,hand_label):
        return False
    if Znak_F(hand_landmarks, hand_label):
        return False
    if Znak_L(hand_landmarks, hand_label):
        return False
    fingertip_ids = [4, 8, 12, 16, 20]  
    mcp_ids = [1, 5, 9, 13, 17]  
    wrist = np.array([hand_landmarks.landmark[0].x, hand_landmarks.landmark[0].y])
    mcps = [np.array([hand_landmarks.landmark[id].x, hand_landmarks.landmark[id].y]) for id in mcp_ids]

    wrist_to_mcps_vectors = [mcp - wrist for mcp in mcps]
    fingertips = [np.array([hand_landmarks.landmark[id].x, hand_landmarks.landmark[id].y]) for id in fingertip_ids]
    wrist_to_fingertips_vectors = [fingertip - wrist for fingertip in fingertips]
    closed_signals = 0
    for fingertip_vector, mcp_vector in zip(wrist_to_fingertips_vectors, wrist_to_mcps_vectors):
        if np.linalg.norm(fingertip_vector) < np.linalg.norm(mcp_vector):
            closed_signals += 1

    
    if closed_signals >= 4: 
        return True
    else:
        return False


def Znak_F(hand_landmarks, hand_label):
    if Hand_Orientation(hand_landmarks, hand_label) == "Outside":
        return False
    thumb_tip = np.array([hand_landmarks.landmark[4].x, hand_landmarks.landmark[4].y])
    index_tip = np.array([hand_landmarks.landmark[8].x, hand_landmarks.landmark[8].y])
    middle_tip = np.array([hand_landmarks.landmark[12].x, hand_landmarks.landmark[12].y])
    ring_tip = np.array([hand_landmarks.landmark[16].x, hand_landmarks.landmark[16].y])
    pinky_tip = np.array([hand_landmarks.landmark[20].x, hand_landmarks.landmark[20].y])
    middle_mcp = np.array([hand_landmarks.landmark[9].x, hand_landmarks.landmark[9].y])
    ring_mcp = np.array([hand_landmarks.landmark[13].x, hand_landmarks.landmark[13].y])
    pinky_mcp = np.array([hand_landmarks.landmark[17].x, hand_landmarks.landmark[17].y])
    
    thumb_index_distance = np.linalg.norm(thumb_tip - index_tip)
    touch_threshold = 0.05  
    thumb_index_touching = thumb_index_distance < touch_threshold
     
    extended_threshold = 0.02  
    middle_extended = (middle_tip[1] < middle_mcp[1] - extended_threshold)
    ring_extended = (ring_tip[1] < ring_mcp[1] - extended_threshold)
    pinky_extended = (pinky_tip[1] < pinky_mcp[1] - extended_threshold)
    
    if thumb_index_touching and middle_extended and ring_extended and pinky_extended:
        return True
    else:
        return False

def Znak_L(hand_landmarks, hand_label):
    if Hand_Orientation(hand_landmarks, hand_label) == "Outside":
        return False
    thumb_tip = np.array([hand_landmarks.landmark[4].x, hand_landmarks.landmark[4].y])
    index_tip = np.array([hand_landmarks.landmark[8].x, hand_landmarks.landmark[8].y])
    middle_tip = np.array([hand_landmarks.landmark[12].x, hand_landmarks.landmark[12].y])
    ring_tip = np.array([hand_landmarks.landmark[16].x, hand_landmarks.landmark[16].y])
    pinky_tip = np.array([hand_landmarks.landmark[20].x, hand_landmarks.landmark[20].y])
    palm_base = np.array([hand_landmarks.landmark[0].x, hand_landmarks.landmark[0].y])
   
    index_to_palm = np.linalg.norm(index_tip - palm_base)
    middle_to_palm = np.linalg.norm(middle_tip - palm_base)
    ring_to_palm = np.linalg.norm(ring_tip - palm_base)
    pinky_to_palm = np.linalg.norm(pinky_tip - palm_base)

    index_most_extended = index_to_palm > max(middle_to_palm, ring_to_palm, pinky_to_palm)

    thumb_position = (thumb_tip - palm_base) / np.linalg.norm(thumb_tip - palm_base)
    index_position = (index_tip - palm_base) / np.linalg.norm(index_tip - palm_base)
    thumb_index_angle = np.arccos(np.clip(np.dot(thumb_position, index_position), -1.0, 1.0))

    is_L_shape = np.degrees(thumb_index_angle) > 50 

    if index_most_extended and is_L_shape:
        return True
    else:
        return False

def Znak_B(hand_landmarks, hand_label):
    if Hand_Orientation(hand_landmarks, hand_label) == "Outside":
        return False
    fingertip_ids = [8, 12, 16, 20]
    mcp_ids = [5, 9, 13, 17]
    extended_fingers = 0
    for tip_id, mcp_id in zip(fingertip_ids, mcp_ids):
        tip = np.array([hand_landmarks.landmark[tip_id].x, hand_landmarks.landmark[tip_id].y])
        mcp = np.array([hand_landmarks.landmark[mcp_id].x, hand_landmarks.landmark[mcp_id].y])
        if tip[1] < mcp[1]:
            extended_fingers += 1

    if extended_fingers < 4:
        return False

    finger_distances = []
    for i in range(len(fingertip_ids) - 1):
        tip1 = np.array([hand_landmarks.landmark[fingertip_ids[i]].x, hand_landmarks.landmark[fingertip_ids[i]].y])
        tip2 = np.array([hand_landmarks.landmark[fingertip_ids[i+1]].x, hand_landmarks.landmark[fingertip_ids[i+1]].y])
        distance = np.linalg.norm(tip1 - tip2)
        finger_distances.append(distance)

    avg_distance = sum(finger_distances) / len(finger_distances)
    proximity_threshold = 0.08 
    if avg_distance < proximity_threshold:
        return True
    else:
        return False


def Znak_R(hand_landmarks, hand_label):
    if Hand_Orientation(hand_landmarks, hand_label) == "Outside":
        return False
    index_tip = np.array([hand_landmarks.landmark[8].x, hand_landmarks.landmark[8].y])
    middle_tip = np.array([hand_landmarks.landmark[12].x, hand_landmarks.landmark[12].y])
    index_pip = np.array([hand_landmarks.landmark[6].x, hand_landmarks.landmark[6].y])
    middle_pip = np.array([hand_landmarks.landmark[10].x, hand_landmarks.landmark[10].y])

    pip_distance = np.linalg.norm(index_pip - middle_pip)

    if hand_label == "Right":
        fingers_crossed = index_tip[0] > middle_tip[0]
    else:
        fingers_crossed = index_tip[0] < middle_tip[0]

    thumb_cmc = np.array([hand_landmarks.landmark[1].x, hand_landmarks.landmark[1].y])
    thumb_mcp = np.array([hand_landmarks.landmark[2].x, hand_landmarks.landmark[2].y])
    thumb_distance = np.linalg.norm(thumb_cmc - thumb_mcp)

    pips_close = pip_distance < thumb_distance

    if fingers_crossed and pips_close:
        return True
    else:
        return False
    
def Gesture_Confirmation(gesture_label):
    global gesture_counts, frame_counter

    if gesture_label not in gesture_counts:
        gesture_counts[gesture_label] = 1
    else:
        gesture_counts[gesture_label] += 1

    frame_counter += 1

    if frame_counter == 10:
        confirmed_gesture = None
        max_count = 0
        for gesture, count in gesture_counts.items():
            if count > max_count and count >= 5:
                confirmed_gesture = gesture
                max_count = count

        gesture_counts = {}
        frame_counter = 0

        if confirmed_gesture:
            print("Confirmed gesture:", confirmed_gesture)
            return confirmed_gesture

    return None
        
def start_gesture_navigation():
    global gesture_navigation_running, gesture_navigation_button
    gesture_navigation_running = True
    print("Gesture navigation started")
    gesture_navigation_button.config(text="STOP GESTURE NAVIGATION", style="Active.TButton")
    
def stop_gesture_navigation():
    global gesture_navigation_running, gesture_navigation_button
    gesture_navigation_running = False
    print("Gesture navigation stopped")
    gesture_navigation_button.config(text="START GESTURE NAVIGATION", style="TButton")
    
def Gesture_Navigation(confirmed_gesture):
    global webcam_running, last_confirmed_gesture, continuous_movement, gesture_navigation_running

    if not gesture_navigation_running:
        return
    
    gesture_commands = {
        "F Sign": forward,
        "B Sign": backward,
        "L Sign": left,
        "R Sign": right,
        "S Sign": stop
    }

    gesture_transitions = {
        ("F Sign", "L Sign"): (stop, 0.5, left),
        ("F Sign", "R Sign"): (stop, 0.5, right),
        ("B Sign", "L Sign"): (stop, 0.5, left),
        ("B Sign", "R Sign"): (stop, 0.5, right),
        ("L Sign", "R Sign"): (stop, 0.5, right),
        ("R Sign", "L Sign"): (stop, 0.5, left)
    }

    if confirmed_gesture:
        if (confirmed_gesture == "F Sign" and last_confirmed_gesture == "F Sign") or \
           (confirmed_gesture == "B Sign" and last_confirmed_gesture == "B Sign"):
            pass
        else:
            continuous_movement = False
            transition = (last_confirmed_gesture, confirmed_gesture)
            if transition in gesture_transitions:
                stop_func, delay, next_func = gesture_transitions[transition]
                stop_func()
                time.sleep(delay)
                next_func()
            else:
                gesture_commands[confirmed_gesture]()
        last_confirmed_gesture = confirmed_gesture
        print(f"Executing command: {confirmed_gesture}")
    else:
        continuous_movement = False
        stop()
        print("Stopping the robot")

    if not webcam_running:
        continuous_movement = False
        stop()


#==========================================================================================================================================
#APP

def Capture_Video():
    global webcam_running, frame_label, picam2
    frame_counter = 0
    pixel_size = 0.03
    scaling = pixel_size ** 2
    open_hand_count = 0
    try:
        while webcam_running:
            image = picam2.capture_array()
            time.sleep(0.03)
            frame_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)
            if results.multi_hand_landmarks:
                for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                    hand_label = handedness.classification[0].label
                    orientation = Hand_Orientation(hand_landmarks, hand_label)
                    x_min = min([lm.x for lm in hand_landmarks.landmark]) * image.shape[1]
                    x_max = max([lm.x for lm in hand_landmarks.landmark]) * image.shape[1]
                    y_min = min([lm.y for lm in hand_landmarks.landmark]) * image.shape[0]
                    y_max = max([lm.y for lm in hand_landmarks.landmark]) * image.shape[0]
                    x_min, x_max, y_min, y_max = int(x_min), int(x_max), int(y_min), int(y_max)
                    cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

                    gesture_functions = {
                        Znak_S: "S Sign",
                        Znak_F: "F Sign",
                        Znak_L: "L Sign",
                        Znak_R: "R Sign",
                        Znak_B: "B Sign"
                    }
                    time.sleep(0.03)
                    for gesture_func, gesture_label in gesture_functions.items():
                        if gesture_func(hand_landmarks, hand_label):
                            cv2.putText(image, gesture_label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                            frame_counter += 1
                            if frame_counter == 15:
                                hand_area_pixels = (x_max - x_min) * (y_max - y_min)
                                hand_area = int(hand_area_pixels * scaling)
                                hand_area_label.config(text=f"Hand area: {hand_area} UNITS")
                                frame_counter = 0
                            
                            confirmed_gesture = Gesture_Confirmation(gesture_label)
                            cv2.putText(image, gesture_label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                            if confirmed_gesture:
                                Gesture_Navigation(confirmed_gesture)
                                confirmed_gesture = None
                            break

            cv2image = cv2.cvtColor(image, cv2.COLOR_BGR2RGBA)
            img = Image.fromarray(cv2image)
            imgtk = ImageTk.PhotoImage(image=img)
            
            time.sleep(0.03)
            frame_label.imgtk = imgtk
            frame_label.configure(image=imgtk)
            frame_label.image = imgtk
    finally:
        picam2.stop()
        hands.close()
        cv2.destroyAllWindows()


def start_webcam():
    global webcam_thread, webcam_running, hands, picam2
    if not webcam_running:
        webcam_running = True
        mp_hands = mp.solutions.hands
        hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.8, min_tracking_confidence=0.7)
        picam2 = Picamera2()
        video_config = picam2.create_video_configuration(main={"size":(640,480)},controls={"FrameRate": 30.0})
        picam2.configure(video_config)
        #time.sleep(0.03)
        picam2.start()
        #time.sleep(0.03)
        webcam_thread = threading.Thread(target=Capture_Video)
        webcam_thread.start()
        start_button.config(state=tk.DISABLED)
        stop_button.config(state=tk.NORMAL)

def black_image():
    global frame_label
    width, height = 640, 480
    black_image = np.zeros((height, width, 3),dtype=np.uint8)
    black_image_pil = Image.fromarray(black_image)
    black_image_tk = ImageTk.PhotoImage(image=black_image_pil)
    time.sleep(0.03)
    frame_label.imgtk = black_image_tk
    frame_label.configure(image=black_image_tk)
    frame_label.image = black_image_tk

def stop_webcam():
    global webcam_running, gesture_navigation_running
    if webcam_running:
        webcam_running = False
        gesture_navigation_running = False
        start_button.config(state=tk.NORMAL)
        stop_button.config(state=tk.DISABLED)
        root.after(100, black_image)  

def quit_app():
    global webcam_running, picam2, webcam_thread, hands
    webcam_running = False
    gesture_navigation_running = False
    if picam2:
        picam2.close()
        
    if hands and getattr(hands, '_graph', None) is not None:
        hands.close()
        hands = None
    GPIO.cleanup()
    root.destroy()
    
def textbox_messenger(text_widget):
    def write (string):
        text_widget.insert(tk.END,string)
        time.sleep(0.03)
        text_widget.see(tk.END)
    sys.stdout.write = write
    sys.stderr.write = write

def forward_button_command(event):
    forward()
    print("Forward button pressed")
    root.after(5000, stop)

def forward_button_released(event):
    stop()
    print("Forward button released")
    
def backward_button_command(event):
    backward()
    print("Backward button pressed")
    root.after(5000, stop)
    
def backward_button_released(event):
    stop()
    print("Backward button released")
    
def left_button_command():
    left()
    print("Left button pressed")
    
def right_button_command():
    right()
    print("Right button pressed")

def stop_moving():
    stop()
    print("Stop moving button pressed")
    
if __name__ == "__main__":
    
    setup_GPIO()
    stop()
       
    root = tk.Tk()
    root.title("Raspberry Pi Robot")

    frame_label = tk.Label(root)   
    frame_label.grid(row=0, column=0, rowspan=6, padx=10, pady=10)
    black_image()
    
    style = ttk.Style()
    style.configure("TButton", background=root.cget("bg")) 
    style.configure("Active.TButton", background="green") 

    hand_area_label = tk.Label(root, text="Hand area: 0", font=("Arial", 15))
    hand_area_label.grid(row=6, column=0, padx=10, pady=10)

    console_text = tk.Text(root, height=10, width=50)
    console_text.grid(row=7, column=0, columnspan=2, padx=10, pady=10)

    movement_frame = ttk.Frame(root)
    movement_frame.grid(row=6, column=0, padx=10, pady=5)

    forward_button = ttk.Button(movement_frame, text="FORWARD")
    forward_button.grid(row=0, column=1, padx=5, pady=5)
    forward_button.bind("<ButtonPress-1>", forward_button_command)
    forward_button.bind("<ButtonRelease-1>", forward_button_released)

    left_button = ttk.Button(movement_frame, text="LEFT", command=left_button_command)
    left_button.grid(row=1, column=0, padx=5, pady=5)

    stop_moving_button = ttk.Button(movement_frame, text="STOP MOVING", command=stop_moving)
    stop_moving_button.grid(row=1, column=1, padx=5, pady=5)

    right_button = ttk.Button(movement_frame, text="RIGHT", command=right_button_command)
    right_button.grid(row=1, column=2, padx=5, pady=5)

    backward_button = ttk.Button(movement_frame, text="BACKWARD")
    backward_button.grid(row=2, column=1, padx=5, pady=5)
    backward_button.bind("<ButtonPress-1>", backward_button_command)
    backward_button.bind("<ButtonRelease-1>", backward_button_released)

    start_button = ttk.Button(root, text="START WEBCAM", command=start_webcam)
    start_button.grid(row=0, column=1, padx=10, pady=(10, 2))

    stop_button = ttk.Button(root, text="STOP", command=stop_webcam, state=tk.DISABLED)
    stop_button.grid(row=1, column=1, padx=10, pady=2)

    gesture_navigation_button = ttk.Button(root, text="START GESTURE NAVIGATION", 
                            command=lambda: start_gesture_navigation() if not gesture_navigation_running else stop_gesture_navigation())
    gesture_navigation_button.grid(row=2, column=1, padx=10, pady=(2, 10))

    quit_button = ttk.Button(root, text="QUIT", command=quit_app)
    quit_button.grid(row=3, column=1, padx=10, pady=(0, 10))
    
    textbox_messenger(console_text)

    root.mainloop()
