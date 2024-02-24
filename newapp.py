import tkinter as tk
import cv2
from PIL import Image, ImageTk

class App:
    def __init__(self, window, window_title, video_source = 0):
        self.window = window
        self.window.title(window_title)
        self.video_source = video_source
        self.vid = None
        
        self.canvas_frame = tk.Frame(window)
        self.canvas_frame.grid(row=0, column=0, sticky="nsew")
        self.button_frame = tk.Frame(window)
        self.button_frame.grid(row=1, column=0, sticky="nsew")
        
        self.window.grid_rowconfigure(0, weight=1)
        self.window.grid_rowconfigure(1, weight=0)
        self.window.grid_columnconfigure(0, weight=1)
        
        self.canvas = tk.Canvas(self.canvas_frame, width=1280, height=720)
        self.canvas.pack(fill="both", expand=True)
        
        self.photo = None
        
        self.quit_button = tk.Button(self.button_frame, text="QUIT", command=self.window.destroy)
        self.quit_button.pack(side="bottom")
        
        self.capture_button = tk.Button(self.button_frame, text="START WEBCAM", command=self.toggle_capture)
        self.capture_button.pack(side="bottom")
        
        
        self.is_capturing = False
    def toggle_capture (self):
        if not self.is_capturing:
            self.start_capture()
        else:
            self.stop_capture()

    def start_capture(self):
        self.vid = cv2.VideoCapture(self.video_source)
        if not self.vid.isOpened():
            print("Error opening webcam")
            return
        self.is_capturing = True
        self.capture_button.config(text="STOP")
        self.update()

    def stop_capture(self):
        self.is_capturing = False
        self.capture_button.config(text="START WEBCAM")
        if self.vid:
            self.vid.release()
            self.vid = None

    def update(self):
        if self.is_capturing:
            ret, frame = self.vid.read()
            if ret:
                self.photo = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
                self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)
            self.window.after(30, self.update)

    def __del__(self):
        self.stop_capture()


if __name__ == "__main__":
    window = tk.Tk()
    newapp = App(window, "Robot")
    window.mainloop()