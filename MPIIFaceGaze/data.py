import os
import sys
import h5py
import glob
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.utils.data

# def images_labels_read(path):
#     imgs_data = np.array([np.zeros((220, 220))])
#     labels_data = np.array([np.zeros((3))])
#     for root, dirs, files in os.walk(path):
#         for dir in dirs:
#             src_imgs_paths = glob.glob(os.path.join(root, dir, 'eye', '*.jpg'))
#             if os.path.exists(os.path.join(root, dir, 'eye', 'annotation1.txt')):
#                 g = open(os.path.join(root, dir, 'eye', 'annotation1.txt'), 'r')
#             else:
#                 continue
#             txt = g.readlines()
#             g.close()
#             print('working on :%s, %s\n' % (path.split('/')[-2], dir))
#             for src_img_path in tqdm(src_imgs_paths):
#                 tmp_img = cv2.imread(src_img_path, cv2.IMREAD_GRAYSCALE)
#                 if src_img_path.split('/')[-1].split('_')[-1].strip('.jpg').isdigit():
#                     tmp_label = Gaze_vector(txt[int(src_img_path.split('/')[-1].split('_')[-1].strip('.jpg')) - 1])
#                 else:
#                     continue
#                 imgs_data = np.insert(imgs_data, 0, tmp_img, axis=0)
#                 labels_data = np.insert(labels_data, 0, tmp_label, axis=0)
#         imgs_data = np.delete(imgs_data, -1, axis=0)
#         labels_data = np.delete(labels_data, -1, axis=0)
#         print('%s finished' % path.split('/')[-2])
#     return imgs_data, labels_data


class MPIIFaceGazetestDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_dir):
        path = dataset_dir
        with h5py.File(path) as f:
            self.images = f['data'].value.astype(np.float32)
            self.gazes = f['labels'].value.astype(np.float32)
        self.length = len(self.images)

        self.images = torch.unsqueeze(torch.from_numpy(self.images), 1)
        self.gazes = torch.from_numpy(self.gazes)

    def __getitem__(self, index):
        return self.images[index, 0], self.gazes[index]

    def __len__(self):
        return self.length

    def __repr__(self):
        return self.__class__.__name__

class MPIIFaceGazeDataset(torch.utils.data.Dataset):
    def __init__(self, subject_id, dataset_dir):
        path = os.path.join(dataset_dir, subject_id+'.h5')
        with h5py.File(path) as f:
            self.images = f['data'].value.astype(np.float32)
            self.gazes = f['labels'].value.astype(np.float32)
        self.length = len(self.images)

        self.images = torch.unsqueeze(torch.from_numpy(self.images), 1)
        self.gazes = torch.from_numpy(self.gazes)
    def __getitem__(self, index):
        return self.images[index, 0], self.gazes[index]

    def __len__(self):
        return self.length

    def __repr__(self):
        return self.__class__.__name__
# class MPIIFaceGazeDataset(torch.utils.data.Dataset):
#     def __init__(self, subject_id, dataset_dir):
#         path = os.path.join(dataset_dir, '{}'.format(subject_id), 'Eyetracking')
#         self.images, self.gazes = images_labels_read(path)
#         self.length = len(self.images)
#
#         self.images = torch.unsqueeze(torch.from_numpy(self.images), 1)
#         self.gazes = torch.from_numpy(self.gazes)
#
#     def __getitem__(self, index):
#         return self.images[index].transpose(2, 0)[[2, 1, 0], :, :], self.gazes[index][0:2]
#
#     def __len__(self):
#         return self.length
#
#     def __repr__(self):
#         return self.__class__.__name__


def get_train_loader(dataset_dir, train_subject_id, batch_size, num_workers, use_gpu):
    assert os.path.exists(dataset_dir)
    # assert test_subject_id in range(15)

    subject_ids = ['P01', 'P02', 'P04', 'P05', 'P06', 'P07', 'P08', 'P09', 'P10', 'P11', 'P12-2', \
                   'P13', 'P14', 'P15', 'P16']
    train_subject_index = subject_ids[train_subject_id]
    print(train_subject_index)
    # train_dataset = torch.utils.data.ConcatDataset([
    #     MPIIFaceGazeDataset(subject_id, dataset_dir) for subject_id in subject_ids if subject_id == train_subject_id
    # ])
    train_dataset = MPIIFaceGazeDataset(train_subject_index, dataset_dir)

    # assert len(train_dataset) == 42000        #false触发异常
    # assert len(test_dataset) == 3000

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=use_gpu,
        drop_last=False,
    )

    return train_loader

def get_test_loader(dataset_dir, test_subject_id, batch_size, num_workers, use_gpu):
    assert os.path.exists(dataset_dir)
    # assert test_subject_id in range(15)

    subject_ids = ['P01', 'P02', 'P04', 'P05', 'P06', 'P07', 'P08', 'P09', 'P10', 'P11', 'P12-2', \
                   'P13', 'P14', 'P15', 'P16']
    test_subject_index = subject_ids[test_subject_id]
    # train_dataset = torch.utils.data.ConcatDataset([
    #     MPIIFaceGazeDataset(subject_id, dataset_dir) for subject_id in subject_ids if subject_id == train_subject_id
    # ])

    path = os.path.join(dataset_dir, '{}.h5'.format(test_subject_index))
    test_dataset = MPIIFaceGazetestDataset(path)

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=use_gpu,
        drop_last=False,
    )

    return test_loader