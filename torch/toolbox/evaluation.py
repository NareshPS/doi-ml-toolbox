"""Support functions to evaluate torch models
"""
import torch
import random

from tqdm.auto import tqdm
from torch.utils.data import Dataset, DataLoader


def make_predictions(model: torch.nn.Module, data: list, device: torch.device):
    """
    It uses the model to compute predictions for the data. The model
    is expected to return raw logits.

    Returns:
    A torch tensor of predicted probabilities with shape [len(data)]

    Usage:
    make_predictions(model, [item_1, item_2...], 'cpu')
    """
    y_probs = []

    # 1. Prepare the model for evaluation
    model.eval()
    model = model.to(device)

    # 2. Setup inference mode
    with torch.inference_mode():
        for sample in data:
            sample = torch.unsqueeze(sample, dim=0).to(device)

            # 3. Compute logits and convert them into probabilities
            logits = model(sample)
            y_prob = torch.softmax(logits.squeeze(), dim=0)

            # 4. Add the probabilities to the collection
            y_probs.append(y_prob.cpu())

    # 5. Stack the y_probs to turn list into a tensor
    return torch.stack(y_probs)


def make_predictions_on_dataset(
    model: torch.nn.Module, dataloader: DataLoader, device: torch.device
):
    """
    It uses the model to compute predictions on an entire dataset. The model
    is expected to return raw logits.

    Returns:
    A torch tensor of predicted probabilities with shape [len(dataloader)]

    Usage:
    make_predictions(model, dataloader, 'cpu')
    """
    # 1. Create a container to collect predictions
    y_preds = []

    # 2. Prepare the model for evaluation
    model.eval()
    model = model.to(device)

    # 3. Setup inference mode
    with torch.inference_mode():
        for X, y in tqdm(dataloader, desc="Making predictions"):
            X, y = X.to(device), y.to(device)

            # 4. Compute logits and convert them into probabilities
            y_logit = model(X)
            y_pred = torch.softmax(y_logit, dim=1).argmax(dim=1)

            # 5. Add the probabilities to the collection
            y_preds.append(y_pred.cpu())

    # 6. Stack the y_preds to turn list into a tensor
    y_pred_tensor = torch.cat(y_preds)

    return y_pred_tensor


def pick_random_samples(dataset: Dataset, num_samples: int = 10, seed: int = None):
    """It picks num_samples random samples from the dataset.

    Returns:
    A tuple (xs, ys) of randomly selected samples from the dataset.

    Usage:
    pick_random_samples(dataset, num_samples=5)
    """
    if seed:
        random.seed(seed)

    xs = []
    ys = []

    for x, y in random.sample(list(dataset), k=num_samples):
        xs.append(x)
        ys.append(y)

    return xs, ys
