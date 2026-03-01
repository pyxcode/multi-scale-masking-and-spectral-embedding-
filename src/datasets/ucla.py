"""
Dataset UCLA — lit les matrices .npy parcellisées
Output : {'fmri': tensor [1, 400, 160]}
"""

import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from glob import glob


class UCLADataset(Dataset):
    def __init__(self, data_dir='data/timeseries', num_frames=160):
        self.files = sorted(glob(os.path.join(data_dir, 'sub-*.npy')))
        self.num_frames = num_frames
        assert len(self.files) > 0, f"Aucun fichier trouvé dans {data_dir}"
        print(f"UCLADataset : {len(self.files)} sujets")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        ts = np.load(self.files[idx])  # [400, T]

        # Clip aléatoire de num_frames TRs
        T = ts.shape[1]
        if T > self.num_frames:
            start = random.randint(0, T - self.num_frames)
            ts = ts[:, start:start + self.num_frames]  # [400, 160]
        else:
            # Padding si T < 160
            pad = np.zeros((ts.shape[0], self.num_frames))
            pad[:, :T] = ts
            ts = pad

        # Normalisation
        ts = torch.tensor(ts, dtype=torch.float32)
        ts = (ts - ts.mean()) / (ts.std() + 1e-6)

        return {'fmri': ts.unsqueeze(0)}  # [1, 400, 160]


def make_ucla(batch_size=4, collator=None, num_workers=4, data_dir='data/timeseries'):
    dataset = UCLADataset(data_dir=data_dir)
    loader = DataLoader(
        dataset,
        collate_fn=collator,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
        pin_memory=True,
    )
    return dataset, loader


if __name__ == '__main__':
    # Test rapide
    dataset = UCLADataset()
    sample = dataset[0]
    print(f"Shape : {sample['fmri'].shape}")   # attendu : [1, 400, 160]
    print(f"Mean  : {sample['fmri'].mean():.4f}")
    print(f"Std   : {sample['fmri'].std():.4f}")
    print("✅ Dataset OK")