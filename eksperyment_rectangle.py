import cv2
import mediapipe as mp

def Hand_Open(hand_landmarks):
    fingertip_id = [4,8,12,16,20]
    palm_base_id = 0
    palm_base = hand_landmarks.landmark[palm_base_id]
    distances = []
    

def Capture_Video(): 
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode = False, max_num_hands = 4,
                           min_detection_confidence = 0.5, 
                           min_tracking_confidence = 0.5)
    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        raise IOError ("webcam not openable") #error jesli kamera sie nie otworzy, od razu break programu
    
    while True:
        ret, frame = cam.read()
        if not ret:
            print ("no stream")
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) #konwersja bgr2rgb bo opencv uzywa bgr xd
        results = hands.process(frame_rgb)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
               x_min = min([lm.x for lm in hand_landmarks.landmark]) * frame.shape[1]
               x_max = max([lm.x for lm in hand_landmarks.landmark]) * frame.shape[1]
               y_min = min([lm.y for lm in hand_landmarks.landmark]) * frame.shape[0]
               y_max = max([lm.y for lm in hand_landmarks.landmark]) * frame.shape[0]
               
               x_min, x_max, y_min, y_max = int(x_min),int(x_max),int(y_min),int(y_max)
               cv2.rectangle(frame, (x_min,y_min),(x_max,y_max),(0,255,0),2)
        cv2.imshow('webcam', frame)
        
        if cv2.waitKey (1) & 0xFF == ord('q'):
            break
    cam.release()
    cv2.destroyAllWindows()
if __name__ == '__main__':
    Capture_Video()