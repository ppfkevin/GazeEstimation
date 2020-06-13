import h5py
import argparse
import cv2
import torch
import torch.utils.data
from tqdm import tqdm
import numpy as np

class MPIIFaceGazetestDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_dir):
        path = dataset_dir
        with h5py.File(path) as f:
            self.images = f['data'].value
            self.gazes = f['labels'].value
        self.length = len(self.images)

        self.images = torch.unsqueeze(torch.from_numpy(self.images), 1)
        self.gazes = torch.from_numpy(self.gazes)

    def __getitem__(self, index):
        # return self.images[index, 0].transpose(2, 0)[[2, 1, 0], :, :], self.gazes[index]
        return self.images[index, 0], self.gazes[index]
    def __len__(self):
        return self.length

    def __repr__(self):
        return self.__class__.__name__

def get_test_loader(dataset_dir, test_subject_id, batch_size, num_workers):
    test_index = 'P05'
    path = './{}.h5'.format(test_index)
    test_dataset = MPIIFaceGazetestDataset(path)

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )
    return test_loader

data_dir = 'D:\Gitcode\Gaze\MPIIFaceGaze'
test_loader = get_test_loader(data_dir, 0, 4, 1)
for step, (images, gazes) in tqdm(enumerate(test_loader)):
    images = images.numpy()
    images = np.array([np.tile(image, (3, 1, 1)) for image in images])
    # images = cv2.cvtColor(images, cv2.COLOR_GRAY2BGR)
    images = torch.from_numpy(images)
    print(1)