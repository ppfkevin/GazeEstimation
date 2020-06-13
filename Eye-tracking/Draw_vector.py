import cv2
import numpy as np

#496.0 71.55 494.09 91.7    509.34 433.19 1582.34 574.52 432.53 1584.38
#eyepos_left = (591, 418)
#eyepos_right = (615, 420)
#vector_left = (496 - 509.34, 71.55 - 433.19)
#vector_right = (494.09 - 574.52, 91.7 - 432.53)
img = cv2.imread('F:\images\SJTUGaze\Pang_data\P09\Eyetracking\GP3\Samples\p09_1.jpg')
pt1_left = (675*2 - 156, 293*2 + 216)
pt1_right = (675*2 + 3 - 130, 293*2 - 20 + 200)
pt2_left = (705*2 - 160, 295*2 + 210)
pt2_right = (705*2 + 3 - 130, 295*2 - 20 + 222)
# GP4
# pt1_left = (675*2 - 0, 293*2 + 5)
# pt1_right = (675*2 + 3 - 0, 293*2 - 20 + 5)
# pt2_left = (705*2 + 0, 295*2 + 0)
# pt2_right = (705*2 + 3 + 0, 295*2 - 20 + 0)
# GP1
# pt1_left = (656*2 - 17, 477*2 - 30)
# pt1_right = (656*2 - int((496 - 509)/3) - 17, 477*2 - int((71 - 433)/5) - 30)
# pt2_left = (690*2 - 20, 478*2 - 33)
# pt2_right = (690*2 - int((494 - 509)/3) - 20, 478*2 - int((91 - 452)/5) - 33)
point_color = (0, 255, 255)
thickness = 2
cv2.line(img, pt1_left, pt1_right, point_color, thickness)
cv2.line(img, pt2_left, pt2_right, point_color, thickness)

cv2.namedWindow("image")
cv2.imshow('image', img)
cv2.waitKey(10000)  # 显示 10000 ms 即 10s 后消失
cv2.destroyAllWindows()
cv2.imwrite('F:\images\SJTUGaze\Pang_data\P09\Eyetracking\\test.jpg', img)