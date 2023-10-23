import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.data.sampler import SubsetRandomSampler


class BaseDataLoader:
    """
    Base class for all data loaders
    """
    def __init__(self, config):
        self.batch_size = config["batch_size"]
        self.shuffle = config["shuffle"]
        self.collate_fn = default_collate
        self.num_workers = config["num_workers"]

        self.validation_split = config["validation_split"]
        self.mode = config["mode"]
        self.agent = config["agent"]
        self.seed = config["seed"]
        self.metric = config["metric"]

        self.min_seq_len = config["min_seq_len"] if "min_seq_len" in config else None
        self.max_seq_len = config["max_seq_len"] if "max_seq_len" in config else None
        self.stride = config["stride"] if "stride" in config else None
        self.max_subseq_len = config["max_subseq_len"] if "max_subseq_len" in config else None

        self.num_items = None
        self.train_data = None
        self.train_loader = None
        self.test_data = None
        self.test_loader = None

        self.init_kwargs = {
            'batch_size': self.batch_size,
            'shuffle': self.shuffle,
            'collate_fn': self.collate_fn,
            'num_workers': self.num_workers
        }

    def _split_sampler(self, n_samples, split, seed):
        if split == 0.0:
            return None, None

        idx_full = np.arange(n_samples)

        np.random.seed(seed)
        np.random.shuffle(idx_full)

        if isinstance(split, int):
            assert split > 0
            assert split < n_samples, \
                "validation set size is configured to be larger than entire dataset."
            len_valid = split
        else:
            len_valid = int(n_samples * split)

        valid_idx = idx_full[0:len_valid]
        train_idx = np.delete(idx_full, np.arange(0, len_valid))

        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)

        # turn off shuffle option which is mutually exclusive with sampler
        self.shuffle = False
        n_samples = len(train_idx)

        return train_sampler, valid_sampler

    def finalize(self):
        pass

    def split_validation(self):
        if self.valid_sampler is None:
            return None
        else:
            return DataLoader(sampler=self.valid_sampler, **self.init_kwargs)
