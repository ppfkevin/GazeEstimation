import cv2
import numpy as np

img = cv2.imread('F:\images\SJTUGaze\Pang_data\P09\Eyetracking\GP4\Samples\p09_1.jpg')  #1, 100, 200
img = cv2.resize(img, (1352, 760))
def on_EVENT_LBUTTONDOWN(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        xy = "%d,%d" % (x, y)
        cv2.circle(img, (x, y), 1, (255, 255, 0), thickness = -1)
        cv2.putText(img, xy, (x, y), cv2.FONT_HERSHEY_PLAIN,
                    1.0, (255, 255, 0), thickness = 1)
        cv2.imshow("image", img)
cv2.namedWindow("image")
cv2.setMouseCallback("image", on_EVENT_LBUTTONDOWN)
while(1):
    cv2.imshow("image", img)
    if cv2.waitKey(0)&0xFF==27:
        break
cv2.destroyAllWindows()
