from ultralytics import YOLO
from tkinter.filedialog import askopenfilename
from platform import system
from os import getcwd
import cv2
import easyocr
import numpy as np
import tkinter as tk

icon : str = "icon.ico"
title : str = "License plate recognition - Live feed"
size : str = "500x265"
spacing : str = "+500+200"
font : str = "Arial 15"
background : str = "gray25"
foreground : str = "AntiqueWhite2"
model_path : str = ""
filename : str = ""
error_msg : str = ""
capture_running : bool = False # This variable is used to limit the running capture instance to only one, as you can click the event button multiple times

win = tk.Tk()
win.title(title)
win.geometry(f"{size}{spacing}")
win.overrideredirect(False)
win.config(bg = background)
win.resizable(False, False)

#Place the icon, works only on Windows
if system() == "Windows":
    try :
        win.iconbitmap ( "r" , icon )
    except tk.TclError :
        pass

#This function is used to get the full PATH of the YOLO model file
def select_file(arg = None) -> None :
    global file_label, model_path, filename, capture_running

    if not capture_running:
        model_path = askopenfilename(initialdir = getcwd(), title = "Select File", filetypes = (("PT file", "*.pt"), ("all files", "*.*")))
        filename = model_path.split( "/" )[-1:][0]
        file_label.config(text = f"Chosen file: {filename}")
    return

#This function is used to exit the process of file selection
def exit_function(arg = None) -> None:
    win.destroy()

#This function is used to capture and evaluate the licence plate recognition in real-time
def capture(arg = None) -> None:
    global msg_label, model_path, capture_running

    if not capture_running:
        capture_running = True # Raising flag, start of capture
    else:
        return
    
    # Declaration of variables
    recognized_plates : set = set() # Stores license plate numbers
    clean_text : str = "" # Stores text from OCR result
    
    model = YOLO(model_path)			# Load YOLO model from path
    reader = easyocr.Reader(['en'])		# Initialize EasyOCR for english language
    cap = cv2.VideoCapture(0)  			# Open webcam

    if not cap.isOpened():
        msg_label.config(text = "MSG: Error, could not open webcam") # Printout to panel
        cap.release() # Release webcam
        cv2.destroyAllWindows() # Closing windows
        capture_running = False # Stopping capture
        return
    else:
        msg_label.config(text = "MSG: Starting capture") # Printout to panel
        while True:		# Continuous frame capture
            ret, frame = cap.read() # Read a frame from the webcam
            if not ret: # If the frame wasn't read properly, end loop
                msg_label.config(text = "MSG: Error during capture") # Printout to panel
                cap.release() # Release webcam
                cv2.destroyAllWindows() # Closing windows
                capture_running = False # Stopping capture
                break
            
            try:
                results = model(frame)
            except Exception as ImportError:
                msg_label.config(text = "MSG: Error with specified file") # Printout to panel
                cap.release() # Release webcam
                cv2.destroyAllWindows() # Closing windows
                capture_running = False # Stopping capture
                break

            # Annotate the frame with bounding boxes
            annotated_frame = results[0].plot()

            # Display the annotated frame
            cv2.imshow("YOLO Real-Time Detection", annotated_frame)

            for result in results:
                for box in result.boxes: # Each detected object has a bounding box
                    x1, y1, x2, y2 = map(int, box.xyxy[0]) # Get top-left and bottom-right coordinates

                    # Crop and preprocess (grayscale)
                    plate_box = frame[y1:y2, x1:x2] # Crop the detected plate region from the frame
                    gray_plate = cv2.cvtColor(plate_box, cv2.COLOR_BGR2GRAY) # Convert the cropped part to grayscale

                    # Displaying cropped and cropped + gray scaled plate
                    cv2.imshow("Cropped Plate (Color)", plate_box)
                    cv2.imshow("Cropped Plate (Grayscale)", gray_plate)

                    # Use EasyOCR on grayscale crop
                    ocr_results = reader.readtext(gray_plate)

                    for (bbox, text, prob) in ocr_results:
                        clean_text = text.strip() # Remove any leftover whitespace

                        # If new plate detected
                        if (clean_text != "") and (clean_text not in recognized_plates):
                            recognized_plates.add(clean_text) # Add to the set of recognized plates

                        # Draw main bounding box
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                        # Draw white-filled box above the plate
                        (text_width, text_height), _ = cv2.getTextSize(clean_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2) # Determine size of text
                        cv2.rectangle(frame, (x1, y1 - text_height - 10), (x1 + text_width + 10, y1), (255, 255, 255), -1) # Draw white box

                        # Put text inside white box
                        cv2.putText(frame, clean_text, (x1 + 5, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX,0.6, (0, 0, 0), 2)

            # Display detection and annotation
            cv2.imshow("License Plate Detection", frame)

            # Display recognized plate number
            plate_display = 255 * np.ones((150, 600, 3), dtype=np.uint8)

            # Write last recognized plate number
            cv2.putText(plate_display, f"Plate number: {clean_text}", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
            cv2.imshow("Recognized Plate", plate_display) # Show number in separate window

            # Exit on button 'Q'
            if (cv2.waitKey(1) & 0xFF) == ord('q'):
                msg_label.config(text = "MSG: Closed real-time capture") # Printout to panel
                cap.release() # Release webcam
                cv2.destroyAllWindows() # Closing windows
                capture_running = False # Stopping capture
                break


filename_in = tk.Button(win, text = "Select model file", bg = background, fg = foreground, bd = 1, anchor = "center", activebackground = foreground, activeforeground = background, font = font, command = select_file)
filename_in.place (x = 5, y = 5, width = 480, height = 60)

file_label = tk.Label(win, text = f"Chosen file: {filename}", bg = background, fg = foreground, anchor = "center", font = font)
file_label.place (x = 5, y = 70, width = 480, height = 60)

msg_label = tk.Label(win, text = f"{error_msg}", bg = background, fg = foreground, anchor = "center", font = font)
msg_label.place (x = 5, y = 135, width = 480, height = 60)

capture_start_button = tk.Button(win, text = "Start video capture", bg = background, fg = foreground, bd = 1, anchor = "center", activebackground = foreground, activeforeground = background, font = font, command = capture)
capture_start_button.place (x = 5, y = 200, width = 480, height = 60)

win.bind("<Escape>", exit_function)
win.bind("<Return>", capture)

win.mainloop()