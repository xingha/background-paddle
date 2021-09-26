import os
import glob
from paddle.io import Dataset
from PIL import Image
import numpy as np

class ImagesDataset(Dataset):
    def __init__(self, root, mode='RGB', transforms=None):
        self.transforms = transforms
        self.mode = mode
        self.filenames = sorted([*glob.glob(os.path.join(root, '**', '*.jpg'), recursive=True),
                                 *glob.glob(os.path.join(root, '**', '*.png'), recursive=True)])

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        with Image.open(self.filenames[idx]) as img:
            img = np.array(img.convert(self.mode))
        # if self.mode == 'L':
        # img = cv2.imread(self.filenames[idx],-1)
        
        if self.transforms:
            img = self.transforms(img)
        
        return img
