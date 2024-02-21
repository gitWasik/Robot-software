import cv2
import mediapipe as mp
import numpy as np

def Hand_Open(hand_landmarks):
    fingertip_ids = [4,8,12,16,20]
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
        
def Hand_Orientation(hand_landmarks):
    thumb_tip = np.array([hand_landmarks.landmark[4].x, hand_landmarks.landmark[4].y, hand_landmarks.landmark[4].z])
    index_tip = np.array([hand_landmarks.landmark[8].x, hand_landmarks.landmark[8].y, hand_landmarks.landmark[8].z])
    pinky_tip = np.array([hand_landmarks.landmark[20].x, hand_landmarks.landmark[20].y, hand_landmarks.landmark[20].z])
    
    thumb_to_index = index_tip - thumb_tip
    thumb_to_pinky = pinky_tip - thumb_tip
    cross = np.cross(thumb_to_index, thumb_to_pinky)
    
    if cross[2] > 0:
        return "palm"
    else:
        return "wrong"
def Capture_Video(): 
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode = False, max_num_hands = 1,
                           min_detection_confidence = 0.7, 
                           min_tracking_confidence = 0.5)
    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        raise IOError ("webcam not openable") #error jesli kamera sie nie otworzy, od razu break programu
    
    open_hand_count = 0
    
    while True:
        ret, frame = cam.read()
        if not ret:
            print ("no stream")
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) #konwersja bgr2rgb bo opencv uzywa bgr xd
        results = hands.process(frame_rgb)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                if Hand_Open(hand_landmarks):
                    open_hand_count += 1
                    print(f"open hand {open_hand_count}")
                
                orientation = Hand_Orientation(hand_landmarks)
                x_min = min([lm.x for lm in hand_landmarks.landmark]) * frame.shape[1]
                x_max = max([lm.x for lm in hand_landmarks.landmark]) * frame.shape[1]
                y_min = min([lm.y for lm in hand_landmarks.landmark]) * frame.shape[0]
                y_max = max([lm.y for lm in hand_landmarks.landmark]) * frame.shape[0]
               
                x_min, x_max, y_min, y_max = int(x_min),int(x_max),int(y_min),int(y_max)
                cv2.rectangle(frame, (x_min,y_min),(x_max,y_max),(0,255,0),2)
                cv2.putText(frame, orientation, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
        cv2.imshow('Kamera', frame)
        
        if cv2.waitKey (1) & 0xFF == ord('q'):
            break
    cam.release()
    cv2.destroyAllWindows()
if __name__ == '__main__':
    Capture_Video()
