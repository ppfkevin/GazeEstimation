import h5py
import numpy as np
import os
import glob
import cv2
from tqdm import tqdm

def Gaze_vector(labels):
    tmp_list = labels.strip().split(' ')
    tmp_list = list(map(float, tmp_list))
    vector_left = [tmp_list[-10]-tmp_list[-6], tmp_list[-9]-tmp_list[-5], 0-tmp_list[-4]]
    vector_right = [tmp_list[-8]-tmp_list[-3], tmp_list[-7]-tmp_list[-2], 0-tmp_list[-1]]
    if vector_right == [0, 0, 0]:
        gaze_vector = vector_left
    elif vector_left == [0, 0, 0]:
        gaze_vector = vector_right
    else:
        gaze_vector = np.array([vector_left, vector_right]).mean(axis=0)
    return gaze_vector

root_dir = 'F:\images\SJTUGaze\\test2'
#for index in ['P01','P02','P04','P05','P06','P07','P08','P09','P10','P11','P12-2',\
#              'P13','P14','P15','P16']:
for index in ['P04']:
    person_path = os.path.join(root_dir, index, 'Eyetracking')
    print(person_path)
    f = h5py.File(index+'.h5', 'w')
    imgs_data = np.array([np.zeros((220, 220))])
    labels_data = np.array([np.zeros((3))])
    for root, dirs, files in os.walk(person_path):
        for dir in dirs:
            src_imgs_paths = glob.glob(os.path.join(root, dir, 'eye', '*.jpg'))
            if os.path.exists(os.path.join(root, dir, 'eye', 'annotation1.txt')):
              g = open(os.path.join(root, dir, 'eye', 'annotation1.txt'), 'r')
              txt = g.readlines()
              g.close()
            else:
              continue
            print('working on :%s, %s\n'%(index, dir))
            for src_img_path in tqdm(src_imgs_paths):
                tmp_img = cv2.imread(src_img_path, cv2.IMREAD_GRAYSCALE)
                if src_img_path.split('\\')[-1].split('_')[-1].strip('.jpg').isdigit():
                  try:
                    tmp_label = Gaze_vector(txt[int(src_img_path.split('\\')[-1].split('_')[-1].strip('.jpg'))-1])
                  except:
                    continue
                else:
                  continue
                imgs_data = np.insert(imgs_data, 0, tmp_img, axis=0)
                labels_data = np.insert(labels_data, 0, tmp_label, axis=0)
    imgs_data = np.delete(imgs_data, -1, axis=0)
    labels_data = np.delete(labels_data, -1, axis=0)
    f['data'] = imgs_data
    f['labels'] = labels_data
    f.close()
    print('%s finished' %index)