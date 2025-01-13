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

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    blur1 = cv2.GaussianBlur(thresh, (5, 5), 0)
    # edges = cv2.Canny(blur, 50, 200)
    contours, _ = cv2.findContours(blur1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    
    for contour in contours:
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        if len(approx) == 4 and cv2.contourArea(approx) > 5000:
            cv2.drawContours(frame, [approx], -1, (255, 0, 0), 2)
            for i, point in enumerate(approx):
                x, y = point[0]
                cv2.putText(frame, f"{x},{y}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                x1, y1, w1, h1 = cv2.boundingRect(approx.reshape(4, 2))
                img = blur[y1:y1+h1, x1:x1+w1]
    edges = cv2.Canny(blur, 50, 200)
    contours2, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours2:
        epsilon = 0.04 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        if len(approx) == 4 and cv2.contourArea(approx) > 3000:
            x2, y2, w2, h2 = cv2.boundingRect(approx.reshape(4, 2))
            if x2 >= x1 and x2 + w2 <= x1 + w1 and y2 >= y1 and y2 + h2 <= y1 + h1:
                cv2.drawContours(frame, [approx], -1, (0, 255, 0), 2)
                for i, point in enumerate(approx):
                    x, y = point[0]
                    cv2.putText(frame, f"{x},{y}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                outer = edges[y2:y2 + h2, x2:x2+w2]
                dilate = cv2.dilate(outer, (15, 15), 1)
                erode = cv2.erode(dilate, (15, 15), 1)
                contours3, _ = cv2.findContours(erode, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for contour in contours3:
                    epsilon = 0.04 * cv2.arcLength(contour, True)
                    approx = cv2.approxPolyDP(contour, epsilon, True)
                    if len(approx) == 4 and cv2.contourArea(approx) > 3000:
                        approx[0] = (approx[0][0][0] + x2 , approx[0][0][1] + y2 )
                        approx[1] = (approx[1][0][0] + x2 , approx[1][0][1] + y2 )
                        approx[2] = (approx[2][0][0] + x2 , approx[2][0][1] + y2 )
                        approx[3] = (approx[3][0][0] + x2 , approx[3][0][1] + y2 )
                        cv2.drawContours(frame, [approx], -1, (0, 0, 255), 2)
                        for i, point in enumerate(approx):
                            x, y = point[0]
                            cv2.putText(frame, f"{x},{y}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    cv2.imshow("Camera", frame)
    # cv2.imshow("TEST", erode)
    if(cv2.waitKey(1) & 0xFF == ord('q')):
        break

cv2.destroyAllWindows()
cam.stop()
