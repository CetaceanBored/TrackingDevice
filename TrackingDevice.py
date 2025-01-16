from picamera2 import Picamera2, Preview
import cv2
import numpy as np
import serial
import time

ser = serial.Serial(
    port='/dev/ttyAMA0',
    baudrate=9600,
    parity=serial.PARITY_NONE,
    stopbits=serial.STOPBITS_ONE,
    bytesize=serial.EIGHTBITS,
    timeout=1
)

cam = Picamera2()
cam_config = cam.create_preview_configuration(main={"size": (1640, 1232)})
cam.configure(cam_config)
cam.start()

cv2.namedWindow("Camera", cv2.WINDOW_AUTOSIZE)

global frame
global outer_area, inner_area
outer_area = 0; inner_area = 0
originx = 0; originy = 0
redx = 0; redy = 0
greenx = 0; greeny = 0
RedAreaMin = 100; RedAreaMax = 10000
GreenAreaMin = 100; GreenAreaMax = 20000
global ServoX, ServoY
ServoX = 750; ServoY = 750

class Contour:
    x = 0; y = 0; w = 0; h = 0;
    area = 0;
    vertices = []
    confirm = False

    def GetVertices(self):
        global frame
        self.vertices.clear()
        for i, point in enumerate(self.rect):
            self.vertices.append(point[0])

    def GetArea(self):
        self.area = cv2.contourArea(self.rect)
        return self.area

    def GetLocation(self):
        self.x, self.y, self.w, self.h = cv2.boundingRect(self.rect.reshape(4, 2))
        return self.x, self.y, self.w, self.h

    def Draw(self, R, G, B):
        cv2.drawContours(frame, [self.rect], -1, (B, G, R), 2)
        for i, point in enumerate(self.rect):
            x, y = point[0]
            cv2.putText(frame, f"{x},{y}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    def Create(self, rect):
        self.rect = rect
        self.confirm = True
        self.GetArea()
        self.GetLocation()

PencilContour = Contour()
OuterContour = Contour()
InnerContour = Contour()

def DetectPencilContour():
    x = [0, 0, 0]; y = [0, 0, 0]; w = [0, 0, 0]; h = [0, 0, 0]
    flag = False
    for t in range(3):
        frame = cv2.cvtColor(cam.capture_array(), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
        blur1 = cv2.GaussianBlur(thresh, (5, 5), 0)
        contours, _ = cv2.findContours(blur1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            if len(approx) == 4 and cv2.contourArea(approx) > 100000 and cv2.isContourConvex(approx):
                cv2.drawContours(frame, [approx], -1, (255, 0, 0), 2)
                rect = approx
                x[t], y[t], w[t], h[t] = cv2.boundingRect(approx.reshape(4, 2))
    if abs(x[0] - x[1]) > 2 or abs(x[1] - x[2]) > 2 or abs(x[0] - x[2]) > 2 \
        or abs(y[0] - y[1]) > 2 or abs(y[1] - y[2]) > 2 or abs(y[0] - y[2]) > 2 \
        or abs(w[0] - w[1]) > 2 or abs(w[1] - w[2]) > 2 or abs(w[0] - w[2]) > 2 \
        or abs(h[0] - h[1]) > 2 or abs(h[1] - h[2]) > 2 or abs(h[0] - h[2]) > 2 \
        or w[0] == 0 or w[1] == 0 or w[2] == 0:
        flag = True
    else:
        PencilContour.Create(rect)
        print(cv2.contourArea(rect))
        flag = False
    return flag
                
def DetectOuterContour():
    global outer_area, inner_area
    x = [0, 0, 0]; y = [0, 0, 0]; w = [0, 0, 0]; h = [0, 0, 0]
    flag = False
    for t in range(3):
        frame = cv2.cvtColor(cam.capture_array(), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 200)
        contours2, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours2:
            epsilon = 0.04 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            area = cv2.contourArea(approx)
            if len(approx) == 4 and area > 10000 and area > inner_area + 1000 and cv2.isContourConvex(approx):
                x0, y0, w0, h0 = cv2.boundingRect(approx.reshape(4, 2))
                if x0 >= PencilContour.x and x0 + w0 <= PencilContour.x + PencilContour.w and y0 >= PencilContour.y and y0 + h0 <= PencilContour.y + PencilContour.h:
                    rect = approx
                    x[t] = x0; y[t] = y0; w[t] = w0; h[t] = h0
    if abs(x[0] - x[1]) > 2 or abs(x[1] - x[2]) > 2 or abs(x[0] - x[2]) > 2 \
        or abs(y[0] - y[1]) > 2 or abs(y[1] - y[2]) > 2 or abs(y[0] - y[2]) > 2 \
        or abs(w[0] - w[1]) > 2 or abs(w[1] - w[2]) > 2 or abs(w[0] - w[2]) > 2 \
        or abs(h[0] - h[1]) > 2 or abs(h[1] - h[2]) > 2 or abs(h[0] - h[2]) > 2 \
        or w[0] == 0 or w[1] == 0 or w[2] == 0:
        flag = True
    else:
        OuterContour.Create(rect)
        outer_area = OuterContour.area
        print(cv2.contourArea(rect))
        flag = False
    return flag

def DetectInnerContour():
    global outer_area, inner_area
    x = [0, 0, 0]; y = [0, 0, 0]; w = [0, 0, 0]; h = [0, 0, 0]
    flag = False
    for t in range(3):
        frame = cv2.cvtColor(cam.capture_array(), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 200)
        outer = edges[OuterContour.y:OuterContour.y + OuterContour.h, OuterContour.x:OuterContour.x+OuterContour.w]
        outer = cv2.morphologyEx(outer, cv2.MORPH_CLOSE, (15, 15))
        contours3, _ = cv2.findContours(outer, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours3:
            epsilon = 0.04 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            area = cv2.contourArea(approx)
            if len(approx) == 4 and area > 10000 and area < outer_area - 1000 and cv2.isContourConvex(approx):
                x0, y0, w0, h0 = cv2.boundingRect(approx.reshape(4, 2))
                x[t] = x0; y[t] = y0; w[t] = w0; h[t] = h0
                approx[0] = (approx[0][0][0] + OuterContour.x, approx[0][0][1] + OuterContour.y)
                approx[1] = (approx[1][0][0] + OuterContour.x, approx[1][0][1] + OuterContour.y)
                approx[2] = (approx[2][0][0] + OuterContour.x, approx[2][0][1] + OuterContour.y)
                approx[3] = (approx[3][0][0] + OuterContour.x, approx[3][0][1] + OuterContour.y)
                rect = approx
    if abs(x[0] - x[1]) > 2 or abs(x[1] - x[2]) > 2 or abs(x[0] - x[2]) > 2 \
        or abs(y[0] - y[1]) > 2 or abs(y[1] - y[2]) > 2 or abs(y[0] - y[2]) > 2 \
        or abs(w[0] - w[1]) > 2 or abs(w[1] - w[2]) > 2 or abs(w[0] - w[2]) > 2 \
        or abs(h[0] - h[1]) > 2 or abs(h[1] - h[2]) > 2 or abs(h[0] - h[2]) > 2 \
        or w[0] == 0 or w[1] == 0 or w[2] == 0:
        flag = True
    else:
        InnerContour.Create(rect)
        inner_area = InnerContour.area
        print(cv2.contourArea(rect))
        flag = False
    return flag

def DetectLaser():
    global frame
    global redx, greenx, redy, greeny
    # frame = cv2.cvtColor(cam.capture_array(), cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_red1 = np.array([0, 40, 40])
    upper_red1 = np.array([20, 255, 255])
    lower_red2 = np.array([160, 40, 40])
    upper_red2 = np.array([180, 255, 255])
    lower_green = np.array([35, 40, 40])
    upper_green = np.array([90, 255, 255])
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask3 = cv2.inRange(hsv, lower_green, upper_green)
    mask = cv2.bitwise_or(mask1, mask2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, (5, 5))
    mask3 = cv2.morphologyEx(mask3, cv2.MORPH_OPEN, (5, 5))
    mask3 = cv2.morphologyEx(mask3, cv2.MORPH_CLOSE, (5, 5))
    contours4, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours5, _ = cv2.findContours(mask3, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    max_area = 0; max_area2 = 0

    for contour in contours4:
        area = cv2.contourArea(contour)
        if area > RedAreaMin and area < RedAreaMax:
            M = cv2.moments(contour)
            if M["m00"] != 0:   
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                if cx >= PencilContour.x - 20 and cx <= PencilContour.x + PencilContour.w + 20 and cy >= PencilContour.y - 20 and cy <= PencilContour.y + PencilContour.h + 20:
                    if area > max_area:
                        max_area = area
                        redx = cx; redy = cy       
    print(max_area)

    for contour in contours5:
        area = cv2.contourArea(contour)
        if area > GreenAreaMin and area < GreenAreaMax:
            M = cv2.moments(contour)
            if M["m00"] != 0:   
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                if cx >= PencilContour.x - 20 and cx <= PencilContour.x + PencilContour.w + 20 and cy >= PencilContour.y - 20 and cy <= PencilContour.y + PencilContour.h + 20:
                    if area > max_area2:
                        max_area2 = area
                        greenx = cx; greeny = cy
    # print(max_area2)
    # cv2.imshow("TEST", mask)

class PIDcontroller:
    def __init__(self, Kp, Ki, Kd):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.previous_error = 0.0
        self.integral = 0.0

    def compute(self, target, current):
        error = target - current
        self.integral += error
        derivative = error - self.previous_error
        output = self.Kp * error + self.Ki * self.integral + self.Kd * derivative
        self.previous_error = error
        return output

    def changePID(self, Kp, Ki, Kd):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        
pid_x = PIDcontroller(0.03, 0.002, 0.2)
pid_y = PIDcontroller(0.03, 0.002, 0.2)

def ServoIncrease(x, y):
    global ServoX; global ServoY
    if ServoX + x >= 300 and ServoX + x <= 1200 and ServoY + y >= 300 and ServoY + y <= 1200:
        ServoX = ServoX + x; ServoY = ServoY + y
        signal = "@"
        signal += str(ServoY).zfill(4)
        signal += str(ServoX).zfill(4)
        signal += '#'
        signal = signal.encode("utf-8")
        ser.write(signal)
        print(signal)

def MoveNext(nextx, nexty):
    global redx, redy
    frame = cv2.cvtColor(cam.capture_array(), cv2.COLOR_RGB2BGR)
    DetectLaser()
    print(redx, "---", redy)
    outputX = pid_x.compute(nextx, redx)
    outputY = pid_y.compute(nexty, redy)
    print(outputX, "-#-", outputY)
    ServoIncrease(int(-outputX), int(-outputY))


def MoveTo(targetx, targety):
    global redx, redy
    distance = np.sqrt((targetx - redx) ** 2 + (targety - redy) ** 2)
    num_points = max(2, int(distance / 10))
    x_values = np.linspace(redx, targetx, num_points)
    y_values = np.linspace(redy, targety, num_points)
    interpolate = [((int(x)), int(y)) for x, y in zip(x_values, y_values)]
    for point in interpolate:
        # cv2.circle(frame, point, 2, (0, 0, 255), 2)
        print(point)
        MoveNext(point[0], point[1])
        

while (DetectPencilContour() == True):
    time.sleep(0.05)
#while (DetectOuterContour() == True):
#    time.sleep(0.05)
#while (DetectInnerContour() == True):
#    time.sleep(0.05)

while True:
    frame = cv2.cvtColor(cam.capture_array(), cv2.COLOR_RGB2BGR)
    cv2.imshow("Camera", frame)
    ServoIncrease(0, 0)
    # time.sleep(0.2)
    DetectLaser()

    
    
    M = cv2.moments(PencilContour.rect) 
    originx = int(M["m10"] / M["m00"])
    originy = int(M["m01"] / M["m00"])

    if redx and redy:
        MoveNext(originx, originy)
        # print("Finish.")

    if PencilContour.confirm:
        PencilContour.Draw(0, 0, 255)
    if OuterContour.confirm:
        OuterContour.Draw(255, 0, 0)
    if InnerContour.confirm:
        InnerContour.Draw(0, 255, 0)
    cv2.circle(frame, (originx, originy), 5, (255, 255, 0), -1)
    cv2.putText(frame, f"{originx},{originy}", (originx, originy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    cv2.circle(frame, (redx, redy), 5, (0, 255, 0), -1)
    cv2.putText(frame, f"{redx},{redy}", (redx, redy - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    if greenx and greeny:
        cv2.circle(frame, (greenx, greeny), 5, (0, 0, 255), -1)
        cv2.putText(frame, f"{greenx},{greeny}", (greenx, greeny - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    cv2.imshow("Camera", frame)
    
    if(cv2.waitKey(1) & 0xFF == ord('q')):
        break

cv2.destroyAllWindows()
cam.stop()
