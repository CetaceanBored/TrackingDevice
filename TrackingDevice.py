from picamera2 import Picamera2, Preview
import cv2
import numpy as np

cam = Picamera2()
cam_config = cam.create_preview_configuration(main={"size": (1640, 1232)})
cam.configure(cam_config)
cam.start()

cv2.namedWindow("Camera", cv2.WINDOW_AUTOSIZE)

outer_area = 0
inner_area = 0
pencilContour = None
innerContour = None
outerContour = None
redx = None
redy = None

while True:
    frame = cv2.cvtColor(cam.capture_array(), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    blur1 = cv2.GaussianBlur(thresh, (5, 5), 0)
    contours, _ = cv2.findContours(blur1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        if len(approx) == 4 and cv2.contourArea(approx) > 100000 and cv2.isContourConvex(approx):                     #PencilContour
            pencilContour = approx
            x1, y1, w1, h1 = cv2.boundingRect(approx.reshape(4, 2))
            edges = cv2.Canny(blur, 50, 200)
            contours2, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours2:
                epsilon = 0.04 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                area = cv2.contourArea(approx)
                if len(approx) == 4 and area > 10000 and area > inner_area + 1000 and cv2.isContourConvex(approx):                      #OuterContour
                    x, y, w, h = cv2.boundingRect(approx.reshape(4, 2))
                    if x >= x1 and x + w <= x1 + w1 and y >= y1 and y + h <= y1 + h1:
                        outer_area = area
                        outerContour = approx
                        x2 = x
                        y2 = y
                        w2 = w
                        h2 = h
                        outer = edges[y2:y2 + h2, x2:x2+w2]
                        outer = cv2.morphologyEx(outer, cv2.MORPH_CLOSE, (15, 15))
                        contours3, _ = cv2.findContours(outer, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        for contour in contours3:
                            epsilon = 0.04 * cv2.arcLength(contour, True)
                            approx = cv2.approxPolyDP(contour, epsilon, True)
                            area = cv2.contourArea(approx)
                            if len(approx) == 4 and area > 10000 and area < outer_area - 1000 and cv2.isContourConvex(approx):           #InnerContour
                                inner_area = area
                                approx[0] = (approx[0][0][0] + x2, approx[0][0][1] + y2)
                                approx[1] = (approx[1][0][0] + x2, approx[1][0][1] + y2)
                                approx[2] = (approx[2][0][0] + x2, approx[2][0][1] + y2)
                                approx[3] = (approx[3][0][0] + x2, approx[3][0][1] + y2)
                                innerContour = approx  
                                break                  
                    break

            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            lower_red1 = np.array([0, 60, 60])
            upper_red1 = np.array([20, 255, 255])
            lower_red2 = np.array([150, 60, 60])
            upper_red2 = np.array([180, 255, 255])
            mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
            mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
            mask = cv2.bitwise_or(mask1, mask2)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, (5, 5))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, (5, 5))
            contours4, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            max_area = 0
            
            for contour in contours4:
                area = cv2.contourArea(contour)
                if area > 50 and area < 1000:
                    M = cv2.moments(contour)
                    if M["m00"] != 0:   
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        if cx >= x1 and cx <= x1 + w1 and cy >= y1 and cy <= y1 + h1:
                            if area > max_area:
                                max_area = area
                                redx = cx
                                redy = cy
            
            print(max_area)

            cv2.drawContours(frame, [pencilContour], -1, (255, 0, 0), 2)
            for i, point in enumerate(pencilContour):
                x, y = point[0]
                cv2.putText(frame, f"{x},{y}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            cv2.drawContours(frame, [outerContour], -1, (0, 255, 0), 2)
            if outerContour.any() != None:
                for i, point in enumerate(outerContour):
                    x, y = point[0]
                    cv2.putText(frame, f"{x},{y}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                cv2.drawContours(frame, [innerContour], -1, (255, 0, 255), 2)
            if innerContour.any() != None:
                for i, point in enumerate(innerContour):
                    x, y = point[0]
                    cv2.putText(frame, f"{x},{y}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                cv2.circle(frame, (redx, redy), 5, (0, 255, 0), -1)
            break

    cv2.imshow("Camera", frame)
    #cv2.imshow("TEST", mask)
    if(cv2.waitKey(1) & 0xFF == ord('q')):
        break

cv2.destroyAllWindows()
cam.stop()
