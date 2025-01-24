import numpy as np
import cv2
import imutils
import datetime
import matplotlib.pyplot as plt  

# Load the gun detection cascade
gun_cascade = cv2.CascadeClassifier('cascade.xml')
camera = cv2.VideoCapture(0)
firstFrame = None
gun_exist = False

while True:
    ret, frame = camera.read()
    if frame is None:
        break
    
    frame = imutils.resize(frame, width=500)
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gun = gun_cascade.detectMultiScale(grey, 1.3, 60, minSize=(100, 100))
    
    if len(gun) > 0:
        gun_exist = True
    for (x, y, w, h) in gun:
        frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
    
    if firstFrame is None:
        firstFrame = gray
        continue
    
    cv2.putText(frame, datetime.datetime.now().strftime("%A %d %B %Y %I:%M:%S %p"),
                (10, frame.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35, (0, 0, 255), 1)
    
    if gun_exist:
        print("Guns detected")
        # Convert the frame from BGR to RGB for Matplotlib
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        plt.imshow(rgb_frame)
        plt.title("Detected Gun")
        plt.axis("off")
        plt.show()
        break
    else:
        cv2.imshow("Weapon Detection Cam", frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()
