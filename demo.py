from paddle.io import DataLoader
from data_path import DATA_PATH
from dataset import ImagesDataset, ZipDataset, SampleDataset
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--dataset-name', type=str, default='videomatte240k', choices=DATA_PATH.keys())

parser.add_argument('--model-backbone', type=str, default='resnet50', choices=['resnet101', 'resnet50', 'mobilenetv2'])
parser.add_argument('--model-name', type=str, default='bgm')
parser.add_argument('--model-pretrain-initialization', type=str, default=None)
parser.add_argument('--model-last-checkpoint', type=str, default=None)

parser.add_argument('--batch-size', type=int, default=8)
parser.add_argument('--num-workers', type=int, default=2)
parser.add_argument('--epoch-start', type=int, default=0)
parser.add_argument('--epoch-end', type=int, default=1000)

parser.add_argument('--log-train-loss-interval', type=int, default=10)
parser.add_argument('--log-train-images-interval', type=int, default=2000)
parser.add_argument('--log-valid-interval', type=int, default=5000)

parser.add_argument('--checkpoint-interval', type=int, default=5000)

args = parser.parse_args()


dataset_train = ZipDataset([
        ZipDataset([
            ImagesDataset(DATA_PATH[args.dataset_name]['train']['pha'], mode='L'),
            ImagesDataset(DATA_PATH[args.dataset_name]['train']['fgr'], mode='RGB'),
        ], assert_equal_length=True),
        ImagesDataset(DATA_PATH['backgrounds']['train'], mode='RGB'),
    ])
dataloader_train = DataLoader(dataset_train,
                                shuffle=True,
                                batch_size=args.batch_size,
                                num_workers=0,
                                )

# Validation DataLoader
# dataset_valid = ZipDataset([
#     ZipDataset([
#         ImagesDataset(DATA_PATH[args.dataset_name]['valid']['pha'], mode='L'),
#         ImagesDataset(DATA_PATH[args.dataset_name]['valid']['fgr'], mode='RGB')
#     ], assert_equal_length=True),
#     ImagesDataset(DATA_PATH['backgrounds']['valid'], mode='RGB'),
# ])
# dataset_valid = SampleDataset(dataset_valid, 50)
# dataloader_valid = DataLoader(dataset_valid,
#                                 batch_size=args.batch_size,
#                                 num_workers=args.num_workers)

for ((true_pha, true_fgr), true_bgr) in dataloader_train:
    ((true_pha, true_fgr), true_bgr) = ((true_pha, true_fgr), true_bgr)
    print('i')