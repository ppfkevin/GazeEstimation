import cv2
import os
import glob
import numpy as np
from tqdm import tqdm

def transfer(intrinsic, R_b , T_b, eye_loc_w):
    '''
    :param intrinsic: 内参矩阵
    :param R_b: 旋转矩阵（相机坐标系下）
    :param T_b: 平移矩阵（相机坐标系下）
    :param eye_loc_w: tobii坐标系下眼睛3D坐标
    :return: 眼睛的2D坐标
    '''
    R_wb = np.array([
        [0, -1, 0],
        [1, 0, 0],
        [0, 0, 1]
    ])
    T_wb = np.array([585, 0, 0])
    R_b = np.insert(R_b, 3, T_b, axis=1)
    eye_loc_b = np.insert(np.dot(R_wb, eye_loc_w) + T_wb, 3, 1)
    #   2704, 1520
    pixel_length = np.array([[1 / 2704, 0, 0], [0, 1 / 1520, 0], [0, 0, 1]])
    tmp = np.dot(np.dot(pixel_length, intrinsic), np.dot(R_b, eye_loc_b))
    eye_loc_2D = [tmp[0]/tmp[2]*2704, tmp[1]/tmp[2]*1520]
    return eye_loc_2D

root_dir = 'F:\images\SJTUGaze\Pang_data\P16\Eyetracking\GP4'
Extrins_dir = 'E:\SJTUGaze\Pang\P16@\Calibration\CalibrationGP4'
Intrins_dir = 'E:\SJTUGaze\Pang\Intrinsic(video)\GP4'
intrinsic_raw = np.array(open(os.path.join(Intrins_dir, 'intrinsics.txt')).read().split('\n')[:-1])
Intrinsic = np.ones((1, 3))
for i in intrinsic_raw:
    Intrinsic = np.append(Intrinsic, [np.array(i.split(',')).astype(float)], axis=0)
Intrinsic = np.delete(Intrinsic, 0, 0)
f1 = open(os.path.join(Extrins_dir, 'extrinsic.txt'), 'r')
imgs_path = glob.glob(os.path.join(root_dir, 'Samples\*.jpg'))
a = f1.readlines()
R = np.array([a[1].strip('\n').split(' '), a[2].strip('\n').split(' '), a[3].strip('\n').split(' ')]).astype(float)
T = np.array([a[5].strip('\n'), a[6].strip('\n'), a[7]]).astype(float)
f1.close()

f2 = open(os.path.join(root_dir, 'Samples\\annotation1.txt'), 'r')
annotations = f2.readlines()
if not os.path.exists(os.path.join(root_dir, 'eye')):
    os.mkdir(os.path.join(root_dir, 'eye'))
for img_path in tqdm(imgs_path):
    img = cv2.imread(img_path)
    index = int(img_path.split('_')[-1].split('.')[0])
    tmp_anno = annotations[index - 1].strip('\n').split(' ')[:-1]
    eye_loc_3D = np.array([tmp_anno[-3], tmp_anno[-2], tmp_anno[-1]]).astype(float)
    if eye_loc_3D[0] + eye_loc_3D[0] + eye_loc_3D[0] == 0:
        continue
    eye_loc_2D = transfer(Intrinsic, R, T, eye_loc_3D)
    eye_img = img[int(eye_loc_2D[1])-90:int(eye_loc_2D[1])+130, int(eye_loc_2D[0])-100:int(eye_loc_2D[0])+120]
    # 130 , 90, 140, 80
    cv2.imwrite(img_path.replace('Samples', 'eye'), eye_img)