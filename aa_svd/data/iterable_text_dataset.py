import logging
import os

import numpy as np
import torch

logger = logging.getLogger(__name__)


class TextCalibrationDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_dir,
        num_samples,
        seed=42,
        sampling='random',
        block_size=512,
        split='train',
    ):
        super().__init__()

        self.data_dir = data_dir
        self.split = split
        self.block_size = block_size
        self.num_samples = num_samples
        self.sampling = sampling
        self.seed = seed

        self.file_path = os.path.join(data_dir, f'{self.split}.bin')

        if not os.path.exists(self.file_path):
            self.file_path = None
            logger.warning(f"Could not find {self.split}.bin in {data_dir}")
            return

        len_data_stream = len(
            np.memmap(self.file_path, dtype=np.uint16, mode='r')
        )

        if self.sampling == 'random':
            torch.manual_seed(self.seed)
            self.sample_idxs = torch.randint(
                len_data_stream - self.block_size, (self.num_samples,)
            )
        elif self.sampling == 'sequential':
            self.sample_idxs = torch.arange(
                0, len_data_stream - self.block_size, self.block_size
            )
            torch.manual_seed(self.seed)
            if self.seed == -1:
                self.sample_idxs = self.sample_idxs[:self.num_samples]
            else:
                perm = torch.randperm(len(self.sample_idxs))
                self.sample_idxs = self.sample_idxs[perm][:self.num_samples]

        logger.info(
            f"Chosen indexes for {self.split} split: {self.sample_idxs}"
        )

    def __len__(self):
        return self.num_samples

    def __getitem__(self, item):
        ix = self.sample_idxs[item]
        return self.get_sample(ix)

    def get_sample(self, ix):
        data_stream = np.memmap(self.file_path, dtype=np.uint16, mode='r')
        X = torch.from_numpy(
            data_stream[ix:ix + self.block_size].astype(np.int64)
        )
        Y = torch.from_numpy(
            data_stream[ix + 1:ix + 1 + self.block_size].astype(np.int64)
        )
        return {'input_ids': X, 'targets': Y}
