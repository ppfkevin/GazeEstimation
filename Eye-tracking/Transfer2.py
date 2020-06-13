import os
import cv2
import numpy as np

R_wb = np.array([
    [0, -1, 0],
    [1,0, 0],
    [0, 0, 1]
])
T_wb = np.array([585, 0, 0])

Imgs_dir = 'F:\images\SJTUGaze\Pang_data\P05\Eyetracking\GP1\Samples'
Intrins_dir = 'E:\SJTUGaze\Pang\Intrinsic(photo)\GP1'
Extrin_dir = 'E:\SJTUGaze\Pang\P05@\Calibration\CalibrationGP1'
eye_loc_w = np.array([[477.34, 363.79, 1679.87], [542.9, 362.61, 1678.85], [472.45, 367.06, 1682.16], [538.04, 365.99, 1680.12], \
    [471.9, 364.21, 1680.27], [537.42, 363.96, 1679.23]])
eye_loc_b = np.zeros((1,3))
for i in range(len(eye_loc_w)):
    eye_loc_b = np.insert(eye_loc_b, i+1, np.dot(R_wb, eye_loc_w[i]) + T_wb, axis = 0)
eye_loc_b = np.delete(eye_loc_b, 0, axis=0)
print(1)

with open(os.path.join(Imgs_dir, 'annotation1.txt'), 'r') as f:
    a = f.readline().strip('\n').split(' ')[:-1]
    eye_loc_w = np.array([a[-3], a[-2], a[-1]]).astype(float)
    eye_loc_b = np.dot(R_wb, eye_loc_w) + T_wb
    intrinsic_raw = np.array(open(os.path.join(Intrins_dir, 'intrinsics.txt')).read().split('\n')[:-1])
    intrinsic = np.ones((1, 3))
    for i in intrinsic_raw:
        intrinsic = np.append(intrinsic, [np.array(i.split(',')).astype(float)], axis=0)
    intrinsic = np.delete(intrinsic, 0, 0)
    extrinsic_raw = open(os.path.join(Extrin_dir, 'extrinsic.txt')).read().split('\n')
    R_b = np.ones((1,3))
    T_b = np.array([])
    for k in range(len(extrinsic_raw)):
        if k in [4,5,6]:
            R_b = np.append(R_b, [np.array(extrinsic_raw[k].split(' ')[:-1]).astype(float)], axis=0)
        elif k in [8,9,10]:
            T_b = np.append(T_b, float(extrinsic_raw[k]))
    R_b = np.delete(R_b, 0, 0)
    #   1520, 2704
    pixel_length = np.array([[1/1520, 0, 0], [0, 1/2704, 0], [0, 0, 1]])
    # eye_loc_2D = np.dot(np.dot(pixel_length, intrinsic), np.dot(R_b, eye_loc_b) + T_b)
    # eye_loc_2D = np.dot(intrinsic, np.dot(R_b, eye_loc_b) + T_b)
    # 659*2，426*2
    print(1)