import cv2
import mediapipe as mp

def Capture_Video():
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode = False, max_num_hands = 1,
                           min_detection_confidence = 0.5, 
                           min_tracking_confidence = 0.5)
    mp_draw = mp.solutions.drawing_utils
    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        raise IOError ("webcam not openable")
    
    while True:
        ret, frame = cam.read()
        if not ret:
            print ("no stream")
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        cv2.imshow('webcam', frame)
        
        if cv2.waitKey (1) & 0xFF == ord('q'):
            break
    cam.release()
    cv2.destroyAllWindows()
if __name__ == '__main__':
    Capture_Video()