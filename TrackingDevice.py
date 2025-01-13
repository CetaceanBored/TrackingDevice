from picamera2 import Picamera2, Preview
import cv2
import numpy as np

cam = Picamera2()
cam_config = cam.create_preview_configuration(main={"size": (1640, 1232)})
cam.configure(cam_config)
cam.start()

cv2.namedWindow("Camera", cv2.WINDOW_AUTOSIZE)

while True:
    frame = cv2.cvtColor(cam.capture_array(), cv2.COLOR_RGB2BGR)

    cv2.imshow("Camera", frame)
    if(cv2.waitKey(1) & 0xFF == ord('q')):
        break

cv2.destroyAllWindows()
cam.stop()

