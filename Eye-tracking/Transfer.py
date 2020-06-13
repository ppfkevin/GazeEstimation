import os
import cv2
import numpy as np

R_wb = np.array([
    [0, -1, 0],
    [1,0, 0],
    [0, 0, 1]
])
T_wb = np.array([585, 0, 0])

Imgs_dir = 'F:\images\SJTUGaze\Pang_data\P09\Eyetracking\GP1\Samples'
Intrins_dir = 'E:\SJTUGaze\Pang\Intrinsic(video)\GP1'
Extrin_dir = 'E:\SJTUGaze\Pang\P-test\Samples'
with open(os.path.join(Imgs_dir, 'annotation1.txt'), 'r') as f:
    a = f.readline().strip('\n').split(' ')[:-1]
    # eye_loc_w = np.array([a[-3], a[-2], a[-1]]).astype(float)
    eye_loc_w = np.array([]).astype(float)
    eye_loc_b = np.insert(np.dot(R_wb, eye_loc_w) + T_wb, 3, 1)
    intrinsic_raw = np.array(open(os.path.join(Intrins_dir, 'intrinsics.txt')).read().split('\n')[:-1])
    intrinsic = np.ones((1, 3))
    for i in intrinsic_raw:
        intrinsic = np.append(intrinsic, [np.array(i.split(',')).astype(float)], axis=0)
    intrinsic = np.delete(intrinsic, 0, 0)
    # extrinsic_raw = open(os.path.join(Extrin_dir, 'extrinsic.txt')).read().split('\n')
    # R_b = np.ones((1,3))
    # T_b = np.array([])
    # for k in range(len(extrinsic_raw)):
    #     if k in [4,5,6]:
    #         R_b = np.append(R_b, [np.array(extrinsic_raw[k].split(' ')[:-1]).astype(float)], axis=0)
    #     elif k in [8,9,10]:
    #         T_b = np.append(T_b, float(extrinsic_raw[k]))
    # R_b = np.delete(R_b, 0, 0)
    # P05:
    # R_b = np.array([[-0.16984, 0.24265, 0.95513], [0.98075, 0.1364, 0.13975], [-0.09637, 0.96048, -0.26114]])
    # T_b = np.array([-1685.112, -405.0733,1385.443])
    #P06:
    # R_b = np.array([[0.52965, 0.4059, -0.74479], [0.83363, -0.08699, 0.54542], [0.1566, -0.90977, -0.38445]])
    # T_b = np.array([910.25714, -854.34579, 3061.2517])
    # P07:
    R_b = np.array([[0.03064,0.99951, 0.00695], [0.95346, -0.03131, 0.29989], [0.29996, -0.00256, -0.95395]])
    T_b = np.array([-534.63744568, -81.20317135, 4946.81112761])
    R_b = np.insert(R_b, 3, T_b, axis=1)
    #   2704, 1520
    pixel_length = np.array([[1/2704, 0, 0], [0, 1/1520, 0], [0, 0, 1]])
    eye_loc_2D = np.dot(np.dot(pixel_length, intrinsic), np.dot(R_b, eye_loc_b))
    # eye_loc_2D = np.dot(intrinsic, np.dot(R_b, eye_loc_b) + T_b)
    # P05:685*2，425*2
    #P06:699, 422
    #P07:640, 426
    #P09:680, 458
    print(1)