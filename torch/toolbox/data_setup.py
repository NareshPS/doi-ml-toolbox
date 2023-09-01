import os

from pathlib import Path
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

WORKERS = os.cpu_count()


def create_dataloaders(
    train_path: Path,
    test_path: Path,
    transform: transforms.Compose,
    batch_size: int,
    workers: int = WORKERS,
    subset_size: int = None,
):
    """Creates train and test DataLoaders

    Creates DataLoaders to load data from train_path and test_path.

    Arguments:
        train_path: Path to training data directory.
        test_path: Path to testing data directory.
        transform: torchvision transform to apply over the training and testing data.
        batch_size: The number of samples in each batch of DataLoaders.
        workers: The number of workers per DataLoader to read the data.
        subset_size: The number of elements to load from the datasets.
    """
    # 1.  Create Datasets
    train_data = create_dataset(
        data_path=train_path, transform=transform, subset_size=subset_size, name="Train"
    )
    test_data = create_dataset(
        data_path=test_path, transform=transform, subset_size=subset_size, name="Test"
    )

    # 2. Get the classes in the input data.
    class_names = (
        train_data.classes if subset_size is None else train_data.dataset.classes
    )

    # 3. Batch the data using DataLoaders.
    train_dataloader = DataLoader(
        dataset=train_data,
        batch_size=batch_size,
        num_workers=workers,
        shuffle=True,
        pin_memory=True,
    )

    test_dataloader = DataLoader(
        dataset=test_data,
        batch_size=batch_size,
        num_workers=workers,
        shuffle=True,
        pin_memory=True,
    )

    return train_dataloader, test_dataloader, class_names


def create_dataset(
    data_path: Path,
    transform: transforms.Compose,
    subset_size: int = None,
    name: str = "Data",
):
    """Creates a torch dataset from image folder

    Arguments:
        data_path: Path to the data directory.
        transform: torchvision transform to apply over the training and testing data.
        subset_size: The number of elements to load from the dataset.
    """
    # 1. Create a dataset to load the data from the data path
    dataset = datasets.ImageFolder(root=data_path, transform=transform)
    dataset_summary(name, dataset)

    # 2. Pick subset of data, if required
    if subset_size is not None:
        print(f"\n[WARN] Using Subset Size: {subset_size}")
        dataset = Subset(dataset, list(range(subset_size)))

    return dataset


def dataset_summary(name, dataset):
    print(f"\n{name} Set")
    print("------------")
    print(f"{dataset}")
    print(f"Classes: {dataset.classes}")
    print(f"Class Dictionary: {dataset.class_to_idx}")
