# %load 'toolbox/engine.py'

"""
Contains training functions.
"""

import torch
import os

import torch.nn as nn

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm
from typing import Dict, List, Tuple, Callable
from datetime import datetime


def train_step(model, dataloader, loss_fn, optimizer, accuracy_fn, device):
    ### Setup initial training state
    training_loss, training_accuracy = 0.0, 0.0
    model.train()

    for batch, (X, y) in enumerate(dataloader):
        # Send data to the appropriate device
        X, y = X.to(device), y.to(device)

        # 1. Forward Pass
        y_logits = model(X)
        y_pred = y_logits.softmax(1).argmax(1)

        # 2. Compute Loss and Accuracy
        loss = loss_fn(y_logits, y)
        accuracy = accuracy_fn(y, y_pred)

        training_loss += loss.item()
        training_accuracy += accuracy

        # 3. Clear Optimizer Gradients
        optimizer.zero_grad()

        # 4. Backward Pass
        loss.backward()

        # 5. Update Weights
        optimizer.step()

    ### Aggregate losses and accuracies
    training_loss /= len(dataloader)
    training_accuracy /= len(dataloader)

    return training_loss, training_accuracy


def test_step(model, dataloader, loss_fn, accuracy_fn, device):
    ### Setup epoch evaluation state
    test_loss, test_accuracy = 0.0, 0.0
    model.eval()

    with torch.inference_mode():
        for batch, (X, y) in enumerate(dataloader):
            # Send data to the appropriate device
            X, y = X.to(device), y.to(device)

            # 1. Forward Pass
            y_logits = model(X)
            y_pred = y_logits.softmax(1).argmax(1)

            # 2. Compute Loss and Accuracy
            loss = loss_fn(y_logits, y)
            accuracy = accuracy_fn(y, y_pred)

            test_loss += loss.item()
            test_accuracy += accuracy

        ### Aggregate losses and accuracies for the test set.
        test_loss /= len(dataloader)
        test_accuracy /= len(dataloader)

    return test_loss, test_accuracy


def create_writer(
    experiment_name: str, model_name: str, extra: str = None
) -> torch.utils.tensorboard.SummaryWriter:
    """Create an instance of torch.utils.tensorboard.SummaryWriter() saving to a specific directory.

    Usage Example:
        writer = create_writer(
            experiment_name='pizza_steak_sushi_classification',
            model_name='resnet_pretrained'
        )
    """
    # 1. Get current date in YYYY-MM-DD Format
    timestamp = datetime.now().strftime("%Y-%m-%d")

    # 2. Create a log_path for SummaryWriter
    if extra:
        log_path = os.path.join("runs", timestamp, experiment_name, model_name, extra)
    else:
        log_path = os.path.join("runs", timestamp, experiment_name, model_name)

    print(f"[INFO] Created a SummaryWriter writing to {log_path}")

    # 3. Create and return SummaryWriter
    return SummaryWriter(log_dir=log_path)


def train(
    model: torch.nn.Module,
    train_dataloader: DataLoader,
    test_dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    accuracy_fn: Callable,
    loss_fn: torch.nn.Module = nn.CrossEntropyLoss(),
    device: str = "cpu",
    epochs: int = 5,
    writer: torch.utils.tensorboard.SummaryWriter = None,
) -> Dict[str, List]:
    # 1. Move the model to right device
    model = model.to(device)

    # 2. Create containers for results
    results = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}

    # 3. Loop through training and testing steps for a number of epochs
    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_step(
            model=model,
            dataloader=train_dataloader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            accuracy_fn=accuracy_fn,
            device=device,
        )

        test_loss, test_acc = test_step(
            model=model,
            dataloader=test_dataloader,
            loss_fn=loss_fn,
            accuracy_fn=accuracy_fn,
            device=device,
        )

        # 4. Print out what's happening
        print(
            f"Epoch: {epoch + 1} | "
            f"train_loss: {train_loss:.4f} | "
            f"train_acc: {train_acc:.4f} | "
            f"test_loss: {test_loss:.4f} | "
            f"test_acc: {test_acc:.4f}"
        )

        # 5. Update results dictionary
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)

        ### 6. Experiment Tracking ###
        if writer:
            # 6.1. Add loss to SummaryWriter
            writer.add_scalars(
                main_tag="Loss",
                tag_scalar_dict={"train_loss": train_loss, "test_loss": test_loss},
                global_step=epoch,
            )

            # 6.2. Add accuracy to SummaryWriter
            writer.add_scalars(
                main_tag="Accuracy",
                tag_scalar_dict={"train_acc": train_acc, "test_acc": test_acc},
                global_step=epoch,
            )

            # 6.3. Track PyTorch Model Architecture
            writer.add_graph(
                model=model, input_to_model=torch.randn(32, 3, 224, 224).to(device)
            )
        ### End Experiment Tracking

    # 7. Close the SummaryWriter
    if writer:
        writer.close()

    # 8. Return the filled results at the end of the epochs
    return results


# # Create an example writer
# example_writer = create_writer(experiment_name="data_10_percent",
#                                model_name="effnetb0",
#                                extra="5_epochs")
