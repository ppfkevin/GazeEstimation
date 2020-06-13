import os
import sys
import argparse
import json
import time
import random
import logging
import cv2
import numpy as np
from collections import OrderedDict
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim
import torch.utils.data
import torch.backends.cuda
import torchvision.utils
from torchvision import transforms

from models import GazeNet
from dataloader import get_train_loader, get_test_loader

logging.basicConfig(
    format='[%(asctime)s %(name)s %(levelname)s] - %(message)s',
    datefmt='%Y/%m/%d %H:%M:%S',
    level=logging.DEBUG)
logger = logging.getLogger(__name__)

global_step = 0


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, num):
        self.val = val
        self.sum += val * num
        self.count += num
        self.avg = self.sum / self.count

def Gaze_vector(labels):
    for i in range(len(labels)):
      std = np.sqrt(labels[i][0]**2 + labels[i][1]**2)
      labels[i] = torch.tensor([labels[i][0]/std, labels[i][1]/std])
    return labels

def convert_to_unit_vector(angles):
    x = -torch.cos(angles[:, 0]) * torch.sin(angles[:, 1])
    y = -torch.sin(angles[:, 0])
    # z = -torch.cos(angles[:, 1]) * torch.cos(angles[:, 1])
    # norm = torch.sqrt(x ** 2 + y ** 2 + z ** 2)
    norm = torch.sqrt(x ** 2 + y ** 2)
    x /= norm
    y /= norm
    # z /= norm
    return x, y


def compute_angle_error(preds, labels):
    pred_x, pred_y = convert_to_unit_vector(preds)
    label_x, label_y = convert_to_unit_vector(labels)
    angles = pred_x * label_x + pred_y * label_y
    return torch.acos(angles) * 180 / np.pi

def compute_mean_std(imgs):
    mean = np.mean(imgs.numpy())
    std = np.std(imgs.numpy())
    return mean, std

def train(epoch, model, optimizer, criterion, train_loader, config, writer):
    global global_step

    logger.info('Train {}'.format(epoch))

    model.train()

    loss_meter = AverageMeter()
    angle_error_meter = AverageMeter()
    start = time.time()
    for step, (images, gazes) in tqdm(enumerate(train_loader)):
        global_step += 1

        # if config['tensorboard_images'] and step == 0:
        #     image = torchvision.utils.make_grid(
        #         images, normalize=True, scale_each=True)
        #     writer.add_image('Train/Image', image, epoch)
        data_transform1 = transforms.Compose([transforms.ToTensor()])
        # transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                      std=[0.229, 0.224, 0.225])])
        images = images.numpy()
        images = np.array([cv2.resize(img, (224, 224)) for img in images])
        images = np.array([np.tile(image, (3, 1, 1)).transpose((2, 1, 0)) for image in images])
        images = torch.stack([data_transform1(img) for img in images], dim=0)
        meann, stdd = compute_mean_std(images)
        data_transform2 = transforms.Compose([transforms.Normalize(mean=[meann, meann, meann],
                                      std=[stdd, stdd, stdd])])
        images = torch.stack([torch.transpose(img, dim0=1, dim1=2) for img in images], dim=0)
        images = torch.tensor(images, dtype=torch.float32)
        gazes = Gaze_vector(gazes[:, :2])
        gazes = torch.tensor(gazes, dtype=torch.float32)
        images, gazes = images.cuda(), gazes.cuda()
        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, gazes)
        loss.backward()

        optimizer.step()

        angle_error = compute_angle_error(outputs, gazes).mean()

        num = images.size(0)
        loss_meter.update(loss.item(), num)
        angle_error_meter.update(angle_error.item(), num)

        # if config['tensorboard']:
        #     writer.add_scalar('Train/RunningLoss', loss_meter.val, global_step)

        if step % 100 == 0:
            logger.info('Epoch {} Step {}/{} '
                        'Loss {:.4f} ({:.4f}) '
                        'AngleError {:.2f} ({:.2f})'.format(
                epoch,
                step,
                len(train_loader),
                loss_meter.val,
                loss_meter.avg,
                angle_error_meter.val,
                angle_error_meter.avg,
            ))

    elapsed = time.time() - start
    logger.info('Elapsed {:.2f}'.format(elapsed))

    # if config['tensorboard']:
    #     writer.add_scalar('Train/Loss', loss_meter.avg, epoch)
    #     writer.add_scalar('Train/AngleError', angle_error_meter.avg, epoch)
    #     writer.add_scalar('Train/Time', elapsed, epoch)


def test(epoch, model, criterion, test_loader, config, writer):
    logger.info('Test {}'.format(epoch))

    model.eval()

    loss_meter = AverageMeter()
    angle_error_meter = AverageMeter()
    start = time.time()

    for step, (images, gazes) in tqdm(enumerate(test_loader)):
        # if config['tensorboard_images'] and epoch == 0 and step == 0:
        #     image = torchvision.utils.make_grid(
        #         images, normalize=True, scale_each=True)
        #     writer.add_image('Test/Image', image, epoch)
        data_transform1 = transforms.Compose([transforms.ToTensor()])
        # transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                      std=[0.229, 0.224, 0.225])])
        images = images.numpy()
        images = np.array([cv2.resize(img, (224, 224)) for img in images])
        images = np.array([np.tile(image, (3, 1, 1)).transpose((2, 1, 0)) for image in images])
        images = torch.stack([data_transform1(img) for img in images], dim=0)
        meann, stdd = compute_mean_std(images)
        data_transform2 = transforms.Compose([transforms.Normalize(mean=[meann, meann, meann],
                                        std=[stdd, stdd, stdd])])
        images = torch.stack([torch.transpose(img, dim0=1, dim1=2) for img in images], dim=0)
        images = torch.tensor(images, dtype=torch.float32)
        gazes = Gaze_vector(gazes[:, :2])
        gazes = torch.tensor(gazes, dtype=torch.float32)
        images, gazes = images.cuda(), gazes.cuda()

        with torch.no_grad():
            outputs = model(images)
        loss = criterion(outputs, gazes)

        angle_error = compute_angle_error(outputs, gazes).mean()

        num = images.size(0)
        loss_meter.update(loss.item(), num)
        angle_error_meter.update(angle_error.item(), num)

    logger.info('Epoch {} Loss {:.4f} AngleError {:.2f}'.format(
        epoch, loss_meter.avg, angle_error_meter.avg))

    elapsed = time.time() - start
    logger.info('Elapsed {:.2f}'.format(elapsed))
    print('test finished')

    # if config['tensorboard']:
    #     if epoch > 0:
    #         writer.add_scalar('Test/Loss', loss_meter.avg, epoch)
    #         writer.add_scalar('Test/AngleError', angle_error_meter.avg, epoch)
    #     writer.add_scalar('Test/Time', elapsed, epoch)

    # if config['tensorboard_parameters']:
    #     for name, param in model.named_parameters():
    #         writer.add_histogram(name, param, global_step)

    return angle_error_meter.avg


def process(args, train_id, test_loader):
    model = GazeNet()
    logger.info(json.dumps(vars(args), indent=2))

    # TensorBoard SummaryWriter
    # writer = SummaryWriter() if args.tensorboard else None
    writer = None

    # set random seed
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # create output directory
    outdir = args.outdir
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    outpath = os.path.join(outdir, 'config.json')
    with open(outpath, 'w') as fout:
        json.dump(vars(args), fout, indent=2)

    # data loaders
    train_loader = get_train_loader(
        args.dataset, train_id, args.batch_size, args.num_workers, True)
    print('data collected')
    # model
    model_path = os.path.join(outdir, 'model_state.pth')
    model = GazeNet()
    model = nn.DataParallel(model)
    model.cuda()
    if os.path.exists(os.path.join(outdir, 'model_state.pth')):
        model.load_state_dict(torch.load(model_path))
        model.eval()
    else:
        None

    criterion = nn.MSELoss(size_average=True)

    # optimizer
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.base_lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        nesterov=args.nesterov)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[20, 30], gamma=args.lr_decay)

    # Tensorboard Settings
    # config = {
    #     'tensorboard': args.tensorboard,
    #     'tensorboard_images': args.tensorboard_images,
    #     'tensorboard_parameters': args.tensorboard_parameters,
    # }
    config = None

    # run test before start training
    test(0, model, criterion, test_loader, config, writer)

    for epoch in range(1, args.epochs + 1):
        scheduler.step()

        train(epoch, model, optimizer, criterion, train_loader, config, writer)
        angle_error = test(epoch, model, criterion, test_loader, config,
                           writer)

        state = OrderedDict([
            ('args', vars(args)),
            ('state_dict', model.state_dict()),
            ('optimizer', optimizer.state_dict()),
            ('epoch', epoch),
            ('angle_error', angle_error),
        ])
        torch.save(state, model_path)

    # if args.tensorboard:
    #     outpath = os.path.join(outdir, 'all_scalars.json')
    #     writer.export_scalars_to_json(outpath)


def main(args):
    test_loaderr = get_test_loader(args.dataset, args.test_id, args.batch_size, args.num_workers, True)
    for i in range(1, 15):
        train_idd = i
        process(args, train_idd, test_loaderr)


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='/content/drive/My Drive/Colab_ppf/MPIIFaceGaze')
    parser.add_argument('--test_id', type=int, default=0)
    parser.add_argument('--train_id', type=int, default=1)
    parser.add_argument('--outdir', type=str, default='/content/drive/My Drive/Colab_ppf/MPIIFaceGaze/test')
    parser.add_argument('--seed', type=int, default=6)
    parser.add_argument('--num_workers', type=int, default=4)

    # optimizer
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--base_lr', type=float, default=0.01)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--nesterov', type=bool, default=True)
    parser.add_argument('--milestones', type=str, default='[20, 30]')
    parser.add_argument('--lr_decay', type=float, default=0.1)

    # TensorBoard
    # parser.add_argument(
    #     '--tensorboard', dest='tensorboard', action='store_true', default=True)
    # parser.add_argument(
    #     '--no-tensorboard', dest='tensorboard', action='store_false')
    # parser.add_argument('--tensorboard_images', action='store_true')
    # parser.add_argument('--tensorboard_parameters', action='store_true')

    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
