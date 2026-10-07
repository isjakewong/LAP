"""Paired RGB images and cached VAE moments, in the original sorted-stem order."""
import json
from pathlib import Path
import numpy as np
from PIL import Image, PngImagePlugin
import torch
from torch.utils.data import Dataset

PngImagePlugin.MAX_TEXT_CHUNK = 100 * 1024 * 1024
try:
    import pyspng
except ImportError:
    pyspng = None


class CustomDataset(Dataset):
    def __init__(self, data_dir, latents_dir='vae-sd'):
        self.images_dir = Path(data_dir) / 'images'
        self.features_dir = Path(data_dir) / latents_dir
        pairs = json.loads((self.features_dir/'dataset.json').read_text())['labels']
        if not pairs or len(dict(pairs)) != len(pairs):
            raise ValueError('Dataset manifest must contain unique, nonempty image entries.')
        label_map = dict(pairs)
        self.feature_fnames = sorted(label_map)
        self.image_fnames = []
        for name in self.feature_fnames:
            p = Path(name)
            if p.is_absolute() or '..' in p.parts or p.suffix != '.npy':
                raise ValueError(f'Invalid latent path in manifest: {name}')
            image = p.with_suffix('.png')
            self.image_fnames.append(str(image))
        self.labels = np.asarray([label_map[n] for n in self.feature_fnames], dtype=np.int64)
        if self.labels.ndim != 1 or np.any(self.labels < 0):
            raise ValueError('Expected nonnegative scalar class labels.')

    def __len__(self):
        return len(self.feature_fnames)

    def __getitem__(self, idx):
        path = self.images_dir/self.image_fnames[idx]
        if pyspng is not None:
            pixels = pyspng.load(path.read_bytes())
            if pixels.ndim != 3 or pixels.shape[2] != 3:
                raise ValueError(f'Expected RGB image: {path}')
        else:
            with Image.open(path) as image:
                pixels = np.array(image.convert('RGB'))
        moments = np.load(self.features_dir/self.feature_fnames[idx], allow_pickle=False)
        if moments.ndim == 4 and moments.shape[0] == 1:
            moments = moments[0]
        if moments.shape != (8, pixels.shape[0]//8, pixels.shape[1]//8):
            raise ValueError(f'Invalid VAE moments shape for {path}: {moments.shape}')
        return (torch.from_numpy(pixels.transpose(2, 0, 1).copy()),
                torch.from_numpy(moments), torch.tensor(self.labels[idx]))
