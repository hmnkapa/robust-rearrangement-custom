import itertools
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Sampler

from src.common.pytorch_util import dict_apply, dict_to_device


class FixedStepsDataloader(torch.utils.data.DataLoader):
    """
    Dataloader that always yields a fixed number of batches.
    If requested number of batches is smaller than available -> return a random subset
    If requested number is larger than available -> cycle through (like a new epoch, random order every time)
    """

    def __init__(self, *args, n_batches, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_batches = n_batches

    def __iter__(self):
        # Keep cycling over the first underlying iterator instead of recreating it
        # mid-epoch, which can be extremely expensive for lazy zarr-backed datasets.
        endless_dataloader = itertools.cycle(super().__iter__())
        for _ in range(self.n_batches):
            yield next(endless_dataloader)

    def __len__(self):
        return self.n_batches


class EndlessDataloader(torch.utils.data.DataLoader):
    """
    Dataloader that cycles through the dataset indefinitely.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __iter__(self):
        endless_dataloader = itertools.cycle(super().__iter__())
        for batch in endless_dataloader:
            yield batch

    def __len__(self):
        return float("inf")


class WeightedDataLoader:
    # Thanks to Lirui Wang for this code
    def __init__(self, dataloaders, weight_type="root"):
        """
        :param dataloaders: list of pytorch dataloaders
        :param weight_type: type of weighting, e.g., "square_root"
        """
        self.dataloaders = dataloaders
        if weight_type == "root":
            datasizes = [len(d) for d in dataloaders]
            datasizes = np.power(datasizes, 1.0 / 3)  # np.sqrt(datasizes)
            weights = datasizes / np.sum(datasizes)
            self.weights = weights
        else:
            print(f"weight type {weight_type} not defined")

        self.loader_iters = [iter(dataloader) for dataloader in self.dataloaders]

    def __iter__(self):
        return self

    def __next__(self):
        # Choose a dataloader based on weights
        chosen_dataloader_idx = np.random.choice(len(self.dataloaders), p=self.weights)
        chosen_loader_iter = self.loader_iters[chosen_dataloader_idx]
        try:
            data = next(chosen_loader_iter)
            return data
        except StopIteration:
            # Handle case where a dataloader is exhausted. Reinitialize the iterator.
            self.loader_iters[chosen_dataloader_idx] = iter(
                self.dataloaders[chosen_dataloader_idx]
            )
            return self.__next__()

    def __len__(self):
        return sum([len(dataloader) for dataloader in self.dataloaders])


class EpochShuffleSampler(Sampler[int]):
    def __init__(self, data_source, shuffle: bool = True, seed: int = 0):
        self.data_source = data_source
        self.shuffle = shuffle
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def __iter__(self):
        if not self.shuffle:
            return iter(range(len(self.data_source)))

        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        indices = torch.randperm(len(self.data_source), generator=generator).tolist()
        return iter(indices)

    def __len__(self):
        return len(self.data_source)


def build_dataloader(
    *,
    dataset,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    pin_memory: bool,
    drop_last: bool,
    persistent_workers: bool = False,
    prefetch_factor: Optional[int] = None,
    sampler=None,
    steps_per_epoch: int = -1,
):
    persistent_workers = persistent_workers and num_workers > 0

    dataloader_kwargs = dict(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle if sampler is None else False,
        pin_memory=pin_memory,
        drop_last=drop_last,
        persistent_workers=persistent_workers,
    )

    if sampler is not None:
        dataloader_kwargs["sampler"] = sampler

    if num_workers > 0 and prefetch_factor is not None:
        dataloader_kwargs["prefetch_factor"] = prefetch_factor

    if steps_per_epoch != -1:
        return FixedStepsDataloader(**dataloader_kwargs, n_batches=steps_per_epoch)

    return torch.utils.data.DataLoader(**dataloader_kwargs)


class _AsyncDevicePrefetchIterator:
    def __init__(self, dataloader, device: torch.device):
        self.loader_iter = iter(dataloader)
        self.device = device
        self.stream = torch.cuda.Stream(device=device)
        self.next_batch = None
        self._preload()

    def __iter__(self):
        return self

    def _record_stream(self, batch):
        current_stream = torch.cuda.current_stream(self.device)

        def record_tensor(tensor):
            if isinstance(tensor, torch.Tensor) and tensor.is_cuda:
                tensor.record_stream(current_stream)
            return tensor

        return dict_apply(batch, record_tensor)

    def _preload(self):
        try:
            batch = next(self.loader_iter)
        except StopIteration:
            self.next_batch = None
            return

        with torch.cuda.stream(self.stream):
            self.next_batch = dict_to_device(batch, self.device)

    def __next__(self):
        if self.next_batch is None:
            raise StopIteration

        torch.cuda.current_stream(self.device).wait_stream(self.stream)
        batch = self.next_batch
        self._record_stream(batch)
        self._preload()
        return batch


class AsyncDevicePrefetchLoader:
    def __init__(self, dataloader, device: torch.device):
        self.dataloader = dataloader
        self.device = torch.device(device)

    def __iter__(self):
        if self.device.type != "cuda":
            return iter(self.dataloader)
        return _AsyncDevicePrefetchIterator(self.dataloader, self.device)

    def __len__(self):
        return len(self.dataloader)

    def __getattr__(self, name):
        return getattr(self.dataloader, name)
