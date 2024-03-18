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
import sys

webcam_thread = None
webcam_running = False
frame_label = None
hand_area_label = None
black_image_tk = None
hands = None
cam = None

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
    
    global PWMA, PWMB
    if PWMA is not None and PWMB is not None:
        PWMA.ChangeDutyCycle(0)
        PWMB.ChangeDutyCycle(0)
        GPIO.output(IN1, GPIO.LOW)
        GPIO.output(IN2, GPIO.LOW)
        GPIO.output(IN3, GPIO.LOW)
        GPIO.output(IN4, GPIO.LOW)




def forward():
    
    global PWMA, PWMB
    PWMA.ChangeDutyCycle(PA)
    PWMB.ChangeDutyCycle(PB)
    GPIO.output(IN1, GPIO.HIGH)
    GPIO.output(IN2, GPIO.LOW)
    GPIO.output(IN3, GPIO.HIGH)
    GPIO.output(IN4, GPIO.LOW)

def backward():
    
    global PWMA, PWMB
    PWMA.ChangeDutyCycle(PA)
    PWMB.ChangeDutyCycle(PB)
    GPIO.output(IN1, GPIO.LOW)
    GPIO.output(IN2, GPIO.HIGH)
    GPIO.output(IN3, GPIO.LOW)
    GPIO.output(IN4, GPIO.HIGH)

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
        
def Hand_Open(hand_landmarks,hand_label):
    if Victory_Sign(hand_landmarks,hand_label):
        return False
    if Like(hand_landmarks, hand_label):
        return False
    fingertip_ids = [4, 8, 12, 16, 20]
    wrist_id = 0
    palm_base_id = 9
    wrist = np.array([hand_landmarks.landmark[wrist_id].x, hand_landmarks.landmark[wrist_id].y])
    palm_base = np.array([hand_landmarks.landmark[palm_base_id].x, hand_landmarks.landmark[palm_base_id].y])
    wrist_to_palm_base = np.linalg.norm(wrist - palm_base)

    open_hand_signals = 0
    closed_hand_signals = 0
    for fingertip_id in fingertip_ids:
        fingertip = np.array([hand_landmarks.landmark[fingertip_id].x, hand_landmarks.landmark[fingertip_id].y])
        wrist_to_fingertip = np.linalg.norm(wrist - fingertip)
        ratio = wrist_to_fingertip / wrist_to_palm_base
        if ratio > 1:
            open_hand_signals += 1
        elif ratio < 0.75: 
            closed_hand_signals += 1

    if open_hand_signals == 5:
        return "Open"
    elif closed_hand_signals >= 4: 
        return "Closed"
    else:
        return "Neither"

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
        

def Like(hand_landmarks, hand_label):

    thumb_tip = np.array([hand_landmarks.landmark[4].x, hand_landmarks.landmark[4].y])
    index_tip = np.array([hand_landmarks.landmark[8].x, hand_landmarks.landmark[8].y])
    middle_tip = np.array([hand_landmarks.landmark[12].x, hand_landmarks.landmark[12].y])
    ring_tip = np.array([hand_landmarks.landmark[16].x, hand_landmarks.landmark[16].y])
    pinky_tip = np.array([hand_landmarks.landmark[20].x, hand_landmarks.landmark[20].y])
    wrist = np.array([hand_landmarks.landmark[0].x, hand_landmarks.landmark[0].y])
  
    thumb_distance_wrist = np.linalg.norm(thumb_tip - wrist)
    index_distance_wrist = np.linalg.norm(index_tip - wrist)
    middle_distance_wrist = np.linalg.norm(middle_tip - wrist)
    ring_distance_wrist = np.linalg.norm(ring_tip - wrist)
    pinky_distance_wrist = np.linalg.norm(pinky_tip - wrist)
 
    thumb_index_distance = np.linalg.norm(thumb_tip - index_tip)
    thumb_middle_distance = np.linalg.norm(thumb_tip - middle_tip)
    thumb_ring_distance = np.linalg.norm(thumb_tip - ring_tip)
    thumb_pinky_distance = np.linalg.norm(thumb_tip - pinky_tip)

    thumb_extended = thumb_distance_wrist > 1.2 * min(index_distance_wrist, middle_distance_wrist, ring_distance_wrist, pinky_distance_wrist)

    thumb_far= all(distance > 0.15 for distance in [thumb_index_distance, thumb_middle_distance, thumb_ring_distance, thumb_pinky_distance])  
    
    fingers_close = all(distance < 0.2 for distance in [np.linalg.norm(index_tip - middle_tip), np.linalg.norm(middle_tip - ring_tip), np.linalg.norm(ring_tip - pinky_tip)])  

    return thumb_extended and thumb_far and fingers_close




        
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
    
    ring_curled = ring_to_palm < ring_base_to_palm
    pinky_curled = pinky_to_palm < pinky_base_to_palm
    
    thumb_tip_x = hand_landmarks.landmark[4].x
    thumb_mcp_x = hand_landmarks.landmark[1].x
    
    if hand_label == "Right":
        thumb_not_extended = thumb_tip_x > thumb_mcp_x
    else:
        thumb_not_extended = thumb_tip_x < thumb_mcp_x
        
        
    orientation = Hand_Orientation(hand_landmarks, hand_label)
    
    return index_extended and middle_extended and ring_curled and pinky_curled  and thumb_not_extended and orientation == "Inside"
    
def Gesture_Navigation():
    #tutaj cos kiedys bedzie
    print("Gesture navigation mode :)")

def Capture_Video():
    global webcam_running, frame_label, cam
    frame_counter = 0
    pixel_size = 0.03
    scaling = pixel_size ** 2
    open_hand_count = 0
    try:
        while webcam_running:
            if not cam.isOpened():
                cam.open(0)
                raise IOError("webcam not opened")
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
                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

                    gestures_recognized = []
                    if Like(hand_landmarks, hand_label):
                        gestures_recognized.append("Like")
                    if Victory_Sign(hand_landmarks, hand_label):
                        gestures_recognized.append("Victory sign")
                    hand_status = Hand_Open(hand_landmarks, hand_label)
                    if hand_status == "Open" and orientation == "Inside":
                        gestures_recognized.append("Hand open")
                        open_hand_count += 1
                        time.sleep(0.03)
                        print(f"open hand {open_hand_count}")
                        frame_counter += 1
                        if frame_counter == 15:
                            hand_area_pixels = (x_max - x_min) * (y_max - y_min)
                            hand_area = int(hand_area_pixels * scaling)
                            hand_area_label.config(text=f"Hand area: {hand_area} UNITS")
                            frame_counter = 0
                            hand_area = 0
                    elif hand_status == "Closed":
                        gestures_recognized.append("Hand closed")
                    if orientation == "Outside":
                        gestures_recognized.append("Outside")

                    
                    for gesture in gestures_recognized:
                        cv2.putText(frame, gesture, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                        y_min -= 30  

            cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
            img = Image.fromarray(cv2image)
            imgtk = ImageTk.PhotoImage(image=img)
            
            time.sleep(0.03)
            frame_label.imgtk = imgtk
            frame_label.configure(image=imgtk)
            frame_label.image = imgtk
    finally:
        if cam and cam.isOpened():
            cam.release()
        if hands:
            hands.close()
        cv2.destroyAllWindows()



def start_webcam():
    global webcam_thread, webcam_running, hands, cam
    if not webcam_running:
        webcam_running = True
        cam = cv2.VideoCapture(0)
        mp_hands = mp.solutions.hands
        hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)
        time.sleep(0.03)
        webcam_thread = threading.Thread(target=Capture_Video)
        #time.sleep(0.03)
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
    global webcam_running
    if webcam_running:
        webcam_running = False
        start_button.config(state=tk.NORMAL)
        stop_button.config(state=tk.DISABLED)

def quit_app():
    global webcam_running, webcam_thread, cam, hands
    webcam_running = False
    if cam and cam.isOpened():
        cam.release()  
    
    if hands and getattr(hands, '_graph', None) is not None:
        hands.close()  
        hands = None  
    GPIO.cleanup()
    root.destroy()  
    
def textbox_messenger(text_widget):
    def write (string):
        text_widget.insert(tk.END,string)
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
    frame_label.pack(side=tk.RIGHT)
    black_image()

    hand_area_label = tk.Label(root, text="Hand area: 0",font=("Arial, 20"))
    hand_area_label.pack(side=tk.BOTTOM)

    console_text = tk.Text(root, height=10, width=50)
    console_text.pack(anchor=tk.SW,padx=10, pady=10)

    forward_button = ttk.Button(root, text="FORWARD")
    forward_button.pack(side=tk.TOP, padx=10, pady=10)
    forward_button.bind("<ButtonPress-1>", forward_button_command)
    forward_button.bind("<ButtonRelease-1>", forward_button_released)

    backward_button = ttk.Button(root, text="BACKWARD")
    backward_button.pack(side=tk.TOP, padx=10, pady=10)
    backward_button.bind("<ButtonPress-1>", backward_button_command)
    backward_button.bind("<ButtonRelease-1>",)

    left_button = ttk.Button(root, text="LEFT",command=left_button_command)
    left_button.pack(side=tk.TOP, padx=10, pady=10)

    right_button = ttk.Button(root, text="RIGHT",command=right_button_command)
    right_button.pack(side=tk.TOP, padx=10, pady=10)
    
    stop_moving_button = ttk.Button(root, text="STOP MOVING",command=stop_moving)
    stop_moving_button.pack(side=tk.TOP, padx=10, pady=10)

    start_button = ttk.Button(root, text="START WEBCAM", command=start_webcam)
    start_button.pack(side=tk.LEFT, padx=10, pady=10)

    stop_button = ttk.Button(root, text="STOP", command=stop_webcam, state=tk.DISABLED)
    stop_button.pack(side=tk.LEFT, padx=10, pady=10)

    quit_button = ttk.Button(root, text="QUIT", command=quit_app)
    quit_button.pack(side=tk.LEFT, padx=10, pady=10)
    
    
    
    textbox_messenger(console_text)

    root.mainloop()
