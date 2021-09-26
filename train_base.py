"""
@Filename : train_base.py
@Time : 2021/09/09 17:57:14
@Author : zhoubishu
@Email : zhoubs11@chinaunicom.cn
@Descript : 
"""

import argparse
# import kornia
import sobel_paddle
import paddle
import os
import random

# from paddle import nn
from paddle import vision
# from paddle.fluid import layers
from paddle.nn import functional as F
from paddle.amp import auto_cast, GradScaler
# from torch.utils.tensorboard import SummaryWriter
from paddle.io import DataLoader
# from paddle.fluid.optimizer import Adam
from paddle.optimizer import Adam
# from paddlevision import make_grid
from tqdm import tqdm
from paddle.vision import transforms as T
from PIL import Image


from data_path import DATA_PATH
from dataset import ImagesDataset, ZipDataset, SampleDataset
from dataset import augmentation as A
from models.model import MattingBase
from utils import load_matched_state_dict, resize, center_crop

# --------------- Arguments ---------------


parser = argparse.ArgumentParser()

parser.add_argument('--dataset-name', type=str,
                    default='videomatte240k', choices=DATA_PATH.keys())

parser.add_argument('--model-backbone', type=str, default='resnet50',
                    choices=['resnet101', 'resnet50', 'mobilenetv2'])
parser.add_argument('--model-name', type=str, default='bgm')
parser.add_argument('--model-pretrain-initialization', type=str, default=None)
parser.add_argument('--model-last-checkpoint', type=str, default=None)

parser.add_argument('--batch-size', type=int, default=8)
parser.add_argument('--num-workers', type=int, default=2)
parser.add_argument('--epoch-start', type=int, default=0)
parser.add_argument('--epoch-end', type=int, default=1000)

parser.add_argument('--log-train-loss-interval', type=int, default=1)
parser.add_argument('--log-train-images-interval', type=int, default=2000)
parser.add_argument('--log-valid-interval', type=int, default=5000)

parser.add_argument('--checkpoint-interval', type=int, default=50)

args = parser.parse_args()


# --------------- Loading ---------------


def train():

    # Training DataLoader
    dataset_train = ZipDataset([
        ZipDataset([
            ImagesDataset(DATA_PATH[args.dataset_name]
                          ['train']['pha'], mode='L'),
            ImagesDataset(DATA_PATH[args.dataset_name]
                          ['train']['fgr'], mode='RGB'),
        ], transforms=A.PairCompose([
            A.PairRandomAffineAndResize(
                (512, 512), degrees=(-5, 5), translate=(0.1, 0.1), scale=(0.4, 1), shear=(-5, 5)),
            A.PairRandomHorizontalFlip(),
            # A.PairRandomBoxBlur(0.1, 5),
            A.PairRandomSharpen(0.1),
            A.PairApplyOnlyAtIndices(
                [1], T.ColorJitter(0.15, 0.15, 0.15, 0.05)),
            A.PairApply(T.ToTensor())
        ]), assert_equal_length=True),
        ImagesDataset(DATA_PATH['backgrounds']['train'], mode='RGB', transforms=T.Compose([
            A.RandomAffineAndResize(
                (512, 512), degrees=(-5, 5), translate=(0.1, 0.1), scale=(1, 2), shear=(-5, 5)),
            T.RandomHorizontalFlip(),
            A.RandomBoxBlur(0.1, 5),
            A.RandomSharpen(0.1),
            T.ColorJitter(0.15, 0.15, 0.15, 0.05),
            T.ToTensor()
        ])),
    ])
    dataloader_train = DataLoader(dataset_train,
                                  shuffle=True,
                                  batch_size=args.batch_size,
                                  num_workers=args.num_workers,
                                  )

    # Validation DataLoader
    dataset_valid = ZipDataset([
        ZipDataset([
            ImagesDataset(DATA_PATH[args.dataset_name]
                          ['valid']['pha'], mode='L'),
            ImagesDataset(DATA_PATH[args.dataset_name]
                          ['valid']['fgr'], mode='RGB')
        ], transforms=A.PairCompose([
            A.PairRandomAffineAndResize(
                (512, 512), degrees=(-5, 5), translate=(0.1, 0.1), scale=(0.3, 1), shear=(-5, 5)),
            A.PairApply(T.ToTensor())
        ]), assert_equal_length=True),
        ImagesDataset(DATA_PATH['backgrounds']['valid'], mode='RGB', transforms=T.Compose([
            A.RandomAffineAndResize(
                (512, 512), degrees=(-5, 5), translate=(0.1, 0.1), scale=(1, 1.2), shear=(-5, 5)),
            T.ToTensor()
        ])),
    ])
    dataset_valid = SampleDataset(dataset_valid, 50)
    dataloader_valid = DataLoader(dataset_valid,
                                  batch_size=args.batch_size,
                                  num_workers=args.num_workers)

    # Model
    model = MattingBase(args.model_backbone)  # .cuda()
    print('model:', model)
    print('wether dynamic', model.aspp._init_in_dynamic_mode)

    if args.model_last_checkpoint is not None:
        load_matched_state_dict(model, paddle.load(args.model_last_checkpoint))
    # elif args.model_pretrain_initialization is not None:
    #     model.load_pretrained_deeplabv3_state_dict(paddle.load(args.model_pretrain_initialization)['model_state'])

    optimizer = Adam(learning_rate=1e-4,
                     parameters=model.backbone.parameters())

    optimizer_aspp = Adam(learning_rate=5e-4,
                          parameters=model.aspp.parameters())
    optimizer_decoder = Adam(
        learning_rate=5e-4, parameters=model.decoder.parameters())
    scaler = GradScaler()

    # Logging and checkpoints
    if not os.path.exists(f'checkpoint/{args.model_name}'):
        os.makedirs(f'checkpoint/{args.model_name}')
    # writer = SummaryWriter(f'log/{args.model_name}')

    # Run loop
    for epoch in range(args.epoch_start, args.epoch_end):
        for i, ((true_pha, true_fgr), true_bgr) in enumerate(dataloader_train):
            step = epoch * len(dataloader_train) + i

            true_pha = true_pha  # .cuda(non_blocking=True)
            true_fgr = true_fgr  # .cuda(non_blocking=True)
            true_bgr = true_bgr  # .cuda(non_blocking=True)
            true_pha, true_fgr, true_bgr = random_crop(
                true_pha, true_fgr, true_bgr)

            true_src = true_bgr.clone()

            # Augment with shadow
            # aug_shadow_idx = paddle.rand((len(true_src),)) < 0.3
            # if aug_shadow_idx.any():
            #     aug_shadow = true_pha[aug_shadow_idx].mul(0.3 * random.random())
            #     # aug_shadow = T.RandomAffine(degrees=(-5, 5), translate=(0.2, 0.2), scale=(0.5, 1.5), shear=(-5, 5))(aug_shadow)
            #     aug_shadow = T.RandomHorizontalFlip(prob=0.5)(aug_shadow)

            #     # aug_shadow = kornia.filters.box_blur(aug_shadow, (random.choice(range(20, 40)),) * 2)

            #     true_src[aug_shadow_idx] = true_src[aug_shadow_idx].sub_(aug_shadow).clamp_(0, 1)
            #     del aug_shadow
            # del aug_shadow_idx

            # Composite foreground onto source
            true_src = true_fgr * true_pha + true_src * (1 - true_pha)

            # Augment with noise
            # aug_noise_idx = paddle.rand(len(true_src)) < 0.4
            # if aug_noise_idx.any():
            #     true_src[aug_noise_idx] = true_src[aug_noise_idx].add_(paddle.randn_like(true_src[aug_noise_idx]).mul_(0.03 * random.random())).clamp_(0, 1)
            #     true_bgr[aug_noise_idx] = true_bgr[aug_noise_idx].add_(paddle.randn_like(true_bgr[aug_noise_idx]).mul_(0.03 * random.random())).clamp_(0, 1)
            # del aug_noise_idx

            # # Augment background with jitter
            # aug_jitter_idx = paddle.rand(len(true_src)) < 0.8
            # # if aug_jitter_idx.any():
            #     # true_bgr[aug_jitter_idx] = kornia.augmentation.ColorJitter(0.18, 0.18, 0.18, 0.1)(true_bgr[aug_jitter_idx])
            # del aug_jitter_idx

            # # Augment background with affine
            # aug_affine_idx = paddle.rand(len(true_bgr)) < 0.3
            # if aug_affine_idx.any():
            #     true_bgr[aug_affine_idx] = T.RandomAffine(degrees=(-1, 1), translate=(0.01, 0.01))(true_bgr[aug_affine_idx])
            # del aug_affine_idx

            with auto_cast():
                pred_pha, pred_fgr, pred_err = model(true_src, true_bgr)[:3]
                loss = compute_loss(pred_pha, pred_fgr,
                                    pred_err, true_pha, true_fgr)
                print('loss type:',loss)

            scled = scaler.scale(loss)
            print('loss type:',scled)
            loss.backward()
            optimizer.minimize(loss)
            optimizer_aspp.minimize(loss)
            optimizer_decoder.minimize(loss)
            # scled.backward()
            # scaler.minimize(optimizer, scled)
            # scaler.minimize(optimizer_aspp, scled)
            # scaler.minimize(optimizer_decoder, scled)
            optimizer.clear_grad()
            optimizer_aspp.clear_grad()
            optimizer_decoder.clear_grad()

            if (i + 1) % args.log_train_loss_interval == 0:
                print('loss', loss.item(), step)

            # if (i + 1) % args.log_train_images_interval == 0:
            #     writer.add_image('train_pred_pha', make_grid(pred_pha, nrow=5), step)
            #     writer.add_image('train_pred_fgr', make_grid(pred_fgr, nrow=5), step)
            #     writer.add_image('train_pred_com', make_grid(pred_fgr * pred_pha, nrow=5), step)
            #     writer.add_image('train_pred_err', make_grid(pred_err, nrow=5), step)
            #     writer.add_image('train_true_src', make_grid(true_src, nrow=5), step)
            #     writer.add_image('train_true_bgr', make_grid(true_bgr, nrow=5), step)

            del true_pha, true_fgr, true_bgr
            del pred_pha, pred_fgr, pred_err

            if (i + 1) % args.log_valid_interval == 0:
                valid(model, dataloader_valid, step)

            if (step + 1) % args.checkpoint_interval == 0:
                paddle.save(model.state_dict(
                ), f'checkpoint/{args.model_name}/epoch-{epoch}-iter-{step}.pdparams')

        paddle.save(model.state_dict(),
                    f'checkpoint/{args.model_name}/epoch-{epoch}.pdparams')


# --------------- Utils ---------------


def compute_loss(pred_pha, pred_fgr, pred_err, true_pha, true_fgr):
    true_err = paddle.abs(pred_pha.detach() - true_pha)
    true_msk = true_pha != 0
    return F.l1_loss(pred_pha, true_pha) + \
        F.l1_loss(sobel_paddle.sobel(pred_pha), sobel_paddle.sobel(true_pha)) + \
        F.l1_loss(pred_fgr * true_msk, true_fgr * true_msk) + \
        F.mse_loss(pred_err, true_err)


def random_crop(*imgs):
    w = random.choice(range(256, 512))
    h = random.choice(range(256, 512))
    results = []
    for img in imgs:
        img = resize(img, (max(h, w), max(h, w)))
        img = center_crop(img, (h, w))
        # img = kornia.resize(img, (max(h, w), max(h, w)))
        # img = kornia.center_crop(img, (h, w))
        results.append(img)
    return results


def valid(model, dataloader, step):
    model.eval()
    loss_total = 0
    loss_count = 0
    with paddle.no_grad():
        for (true_pha, true_fgr), true_bgr in dataloader:
            batch_size = true_pha.size(0)

            true_pha = true_pha.cuda(non_blocking=True)
            true_fgr = true_fgr.cuda(non_blocking=True)
            true_bgr = true_bgr.cuda(non_blocking=True)
            true_src = true_pha * true_fgr + (1 - true_pha) * true_bgr

            pred_pha, pred_fgr, pred_err = model(true_src, true_bgr)[:3]
            loss = compute_loss(pred_pha, pred_fgr,
                                pred_err, true_pha, true_fgr)
            loss_total += loss.cpu().item() * batch_size
            loss_count += batch_size

    # writer.add_scalar('valid_loss', loss_total / loss_count, step)
    print('valid_loss', loss_total / loss_count, step)
    model.train()


# --------------- Start ---------------


if __name__ == '__main__':
    train()
