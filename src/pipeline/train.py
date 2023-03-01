import os

import torch
import wandb
from tqdm import tqdm
from config import *
from model import Model
from dataset import PushUpsDataset
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_recall_fscore_support,
)


def init_wandb():
    """
    The function initialize train monitor from WandB service and sends a training configuration.
    """

    wandb_config = {
        "learning_rate": LEARNING_RATE,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "num_workers": NUM_WORKERS,
    }

    wandb.init(project="Push-Ups-Counter", entity="ofevy", name=TRAIN_NAME)
    wandb.config = wandb_config


def load_dataloaders():
    """
    The function prepares, loads and returns all dataloaders.

    Returns:
        train_dataloader (DataLoader) - A train dataloader.
        val_dataloader (DataLoader) - A validation dataloader.
        test_dataloader (DataLoader) - A test dataloader.
    """

    train_csv_filepath = os.path.join(TRAIN_DATA_PATH, CSV_FILENAME)
    val_csv_filepath = os.path.join(VALID_DATA_PATH, CSV_FILENAME)
    test_csv_filepath = os.path.join(TEST_DATA_PATH, CSV_FILENAME)

    train_ds = PushUpsDataset(train_csv_filepath, TRAIN_DATA_PATH)
    val_ds = PushUpsDataset(val_csv_filepath, VALID_DATA_PATH)
    test_ds = PushUpsDataset(test_csv_filepath, TEST_DATA_PATH)

    train_dataloader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=True
    )
    val_dataloader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=True
    )
    test_dataloader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=True)

    return train_dataloader, val_dataloader, test_dataloader


def load_model():
    """
    The function loads a classification model.

    Returns:
        model (nn.Module) - The model.
    """
    model = Model()

    return model


def validate_model(model, val_dataloader):
    """
    The function validates model after each epoch and logs calculated metrics to the WandB service.

    Arguments:
        model (nn.Module) - The model.
        val_dataloader (DataLoader) - A validation dataloader.
    """

    running_loss = 0.0
    outputs_ = []
    targets_ = []

    pbar = tqdm(total=len(val_dataloader))

    for data in val_dataloader:
        inputs, targets = data[0].to(DEVICE), data[1].to(DEVICE)
        outputs = model(inputs)
        loss = model.criterion(outputs, targets)

        running_loss += loss.item()

        outputs_.extend(
            torch.softmax(outputs, dim=1).cpu().detach().numpy().argmax(1).tolist()
        )
        targets_.extend(targets.cpu().detach().numpy().tolist())

        pbar.update(1)

    # Calculate metrics
    mean_loss = running_loss / len(val_dataloader)
    accuracy = accuracy_score(targets_, outputs_)
    balanced_accuracy = balanced_accuracy_score(targets_, outputs_)

    precision, recall, f_beta, support = precision_recall_fscore_support(
        targets_, outputs_
    )

    mean_precision = sum(precision) / len(precision)
    mean_recall = sum(recall) / len(recall)
    mean_f_beta = sum(f_beta) / len(f_beta)
    mean_support = sum(support) / len(support)

    wandb.log(
        {
            "valid/loss": mean_loss,
            "valid/accuracy": accuracy,
            "valid/balanced_accuracy": balanced_accuracy,
            "valid/precision": mean_precision,
            "valid/recall": mean_recall,
            "valid/f_beta": mean_f_beta,
            "valid/support": mean_support,
        }
    )


def train_model(model, train_dataloader, val_dataloader, checkpoints_path):
    """
    The function trains model

    Arguments:
        model (nn.Module) - The model.
        train_dataloader (DataLoader) - A training dataloader.
        val_dataloader (DataLoader) - A validation dataloader.
        checkpoints_path (str) - A path to directory, where script will save all model checkpoints.
    """

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(EPOCHS):
        step_p_bar = tqdm(total=len(train_dataloader))

        running_loss = 0.0
        outputs_ = []
        targets_ = []

        for data in train_dataloader:
            inputs, targets = data[0].to(DEVICE), data[1].to(DEVICE)
            outputs = model(inputs)
            loss = model.criterion(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            outputs_.extend(
                torch.softmax(outputs, dim=1).cpu().detach().numpy().argmax(1).tolist()
            )
            targets_.extend(targets.cpu().detach().numpy().tolist())

            step_p_bar.update(1)

        # Calculate metrics
        mean_loss = running_loss / len(train_dataloader)
        accuracy = accuracy_score(targets_, outputs_)
        balanced_accuracy = balanced_accuracy_score(targets_, outputs_)

        precision, recall, f_beta, support = precision_recall_fscore_support(
            targets_, outputs_
        )

        mean_precision = sum(precision) / len(precision)
        mean_recall = sum(recall) / len(recall)
        mean_f_beta = sum(f_beta) / len(f_beta)
        mean_support = sum(support) / len(support)

        wandb.log(
            {
                "train/loss": mean_loss,
                "train/accuracy": accuracy,
                "train/balanced_accuracy": balanced_accuracy,
                "train/precision": mean_precision,
                "train/recall": mean_recall,
                "train/f_beta": mean_f_beta,
                "train/support": mean_support,
            }
        )
        validate_model(model, val_dataloader)

        checkpoint_model_path = os.path.join(
            checkpoints_path, "epoch-{}-model.pth".format(epoch + 1)
        )
        torch.save(model.state_dict(), checkpoint_model_path)


if __name__ == "__main__":
    CHECKPOINTS_PATH = os.path.join("src", "models", TRAIN_NAME)
    os.makedirs(CHECKPOINTS_PATH, exist_ok=True)

    train_loader, val_loader, test_loader = load_dataloaders()
    model = load_model().to(DEVICE)

    init_wandb()
    train_model(model, train_loader, val_loader, CHECKPOINTS_PATH)
