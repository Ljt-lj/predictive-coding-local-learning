from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


class FlattenTransform:
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return x.view(-1)


def build_mnist_loaders(
    data_dir: str | Path,
    batch_size: int = 128,
    num_workers: int = 0,
    train_subset: int | None = None,
    test_subset: int | None = None,
    pin_memory: bool | None = None,
) -> tuple[DataLoader, DataLoader]:
    root = Path(data_dir)
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            FlattenTransform(),
        ]
    )
    train_set = datasets.MNIST(root=str(root), train=True, download=True, transform=transform)
    test_set = datasets.MNIST(root=str(root), train=False, download=True, transform=transform)

    if train_subset is not None:
        train_set = Subset(train_set, list(range(min(train_subset, len(train_set)))))
    if test_subset is not None:
        test_set = Subset(test_set, list(range(min(test_subset, len(test_set)))))

    if pin_memory is None:
        pin_memory = torch.cuda.is_available()

    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "drop_last": False,
        "pin_memory": pin_memory,
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True

    train_loader = DataLoader(train_set, shuffle=True, **loader_kwargs)
    test_loader = DataLoader(test_set, shuffle=False, **loader_kwargs)
    return train_loader, test_loader
