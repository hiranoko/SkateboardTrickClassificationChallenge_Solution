from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import accuracy_score
from src.config import CFG
from src.dataset.dataset import CustomDataset
from src.utils.util import AverageMeter, Snapshot
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


def run_one_epoch(loader, model, criterion, optimizer, device, training=True):
    losses = AverageMeter("Loss", ":.4e")
    preds = []
    labels = []

    if training:
        model.train()
    else:
        model.eval()

    for signal, label in loader:
        signal = signal.to(device)
        label = label.to(device)

        if training:
            optimizer.zero_grad()

        with torch.set_grad_enabled(training):
            pred = model(signal)
            loss = criterion(pred, label)
            if training:
                loss.backward()
                optimizer.step()

        losses.update(loss.item(), signal.size(0))

        pred = pred.detach().cpu()
        label = label.detach().cpu()
        preds.append(pred)
        labels.append(label)

    preds: torch.Tensor = torch.concatenate(preds, axis=0)
    labels: torch.Tensor = torch.concatenate(labels, axis=0)

    return losses, preds, labels


def run_eval(loader, model, device):
    preds = []
    labels = []

    model.eval()
    for signal, label in loader:
        signal = signal.to(device)
        label = label.to(device)

        with torch.inference_mode():
            pred = model(signal)

        pred = pred.detach().cpu()
        label = label.detach().cpu()
        preds.append(pred)
        labels.append(label)

    preds: torch.Tensor = torch.concatenate(preds, axis=0)
    labels: torch.Tensor = torch.concatenate(labels, axis=0)

    return preds, labels


def run_fold(
    cfg: CFG,
    model: nn.Module,
    df: pd.DataFrame,
    fold: int,
    working_dir: str = None,
    silent: bool = False,
    visualize: bool = False,
):
    if not silent:
        print(f"Fold: {fold}")

    if working_dir is not None:
        working_dir = Path(working_dir) / f"fold_{fold}"
        writer = SummaryWriter(log_dir=working_dir)
        snapshot = Snapshot(
            save_best_only=True,
            mode="max",
            initial_metric=None,
            name=cfg.exp_name,
            monitor="metric",
            output_dir=working_dir,
            silent=silent,
        )

    train_dataset = CustomDataset(
        df=df,
        target_col=cfg.target_col,
        fold_col=cfg.fold_col,
        data_col=cfg.data_col,
        fold=fold,
        mode="train",
        transform=cfg.train_transform,
        silent=silent,
    )
    val_dataset = CustomDataset(
        df=df,
        target_col=cfg.target_col,
        fold_col=cfg.fold_col,
        data_col=cfg.data_col,
        fold=fold,
        mode="valid",
        transform=cfg.test_transform,
        silent=silent,
    )

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=8)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)

    model.to(cfg.device)

    criteria = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.epochs, eta_min=0, last_epoch=-1
    )

    training_results = {}

    # if silent is False only using tqdm
    for epoch in range(cfg.epochs):
        train_losses, train_preds, train_labels = run_one_epoch(
            loader=train_loader,
            model=model,
            criterion=criteria,
            optimizer=optimizer,
            device=cfg.device,
            training=True,
        )

        val_losses, val_preds, val_labels = run_one_epoch(
            loader=val_loader,
            model=model,
            criterion=criteria,
            optimizer=optimizer,
            device=cfg.device,
            training=False,
        )

        # Numpy sigmoid
        train_accuracy = accuracy_score(
            np.argmax(train_preds.softmax(-1), axis=-1), train_labels
        )
        val_accuracy = accuracy_score(
            np.argmax(val_preds.softmax(-1), axis=-1), val_labels
        )

        scheduler.step()

        training_results[epoch] = {
            "train_loss": train_losses.avg,
            "val_loss": val_losses.avg,
            "train_accuracy": train_accuracy,
            "val_accuracy": val_accuracy,
            "train_preds": train_preds,
            "train_labels": train_labels,
            "val_preds": val_preds,
            "val_labels": val_labels,
        }

        if working_dir is not None:
            writer.add_scalar("Training_Loss", train_losses.avg, epoch)
            writer.add_scalar("Training_Metric", train_accuracy, epoch)
            writer.add_scalar("Validation_Loss", val_losses.avg, epoch)
            writer.add_scalar("Validation_Metric", val_accuracy, epoch)
            writer.add_scalar("Learning_Rate", optimizer.param_groups[0]["lr"], epoch)
            snapshot.snapshot(
                metric=val_accuracy, model=model, optimizer=optimizer, epoch=epoch
            )

        if not silent:
            print(f"Epoch: {epoch}")
            print(f"Loss(Train, Valid): {train_losses.avg} {val_losses.avg}")
            print(f"Accuracy(Train, Valid): {train_accuracy} {val_accuracy}")

    if working_dir is not None:
        training_results["snapshot"] = snapshot
        print(f"FOLD: {fold} Accuracy: {snapshot.best_metric}")
        # print("=" * 80)

    if visualize:
        epochs = range(cfg.epochs)
        train_losses = [training_results[i]["train_loss"] for i in epochs]
        val_losses = [training_results[i]["val_loss"] for i in epochs]
        train_accuracies = [training_results[i]["train_accuracy"] for i in epochs]
        val_accuracies = [training_results[i]["val_accuracy"] for i in epochs]

        plt.figure(figsize=(8, 6))

        # Plot for Loss
        plt.subplot(2, 1, 1)
        plt.plot(epochs, train_losses, label="train_loss")
        plt.plot(epochs, val_losses, label="val_loss")
        plt.xlim(0, cfg.epochs)
        plt.ylim(-0.05, 1.05)
        plt.xticks(epochs)
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)

        # Plot for Accuracy
        plt.subplot(2, 1, 2)
        plt.plot(epochs, train_accuracies, label="train_accuracy")
        plt.plot(epochs, val_accuracies, label="val_accuracy")
        plt.xlim(0, cfg.epochs)
        plt.ylim(0, 1.05)
        plt.xticks(epochs)
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.grid(True)

        # Title and layout adjustment
        plt.suptitle("Training and Validation Metrics Over Epochs", fontsize=14)
        plt.tight_layout()
        plt.show()

    return training_results


def run_oof(cfg: CFG, model, oof_df, working_dir, silent=False):
    for fold in range(len(cfg.fold_dict)):
        # working_dir = Path(
        #     f"../output/classification1/{cfg.exp_name}/{cfg.target_subjectid}/fold_{fold}"
        # )
        model_path = working_dir / f"fold_{fold}" / f"{cfg.exp_name}_best.pth"
        assert model_path.is_file(), f"{model_path} not found"

        val_dataset = CustomDataset(
            df=cfg.train_df,
            target_col=cfg.target_col,
            fold_col=cfg.fold_col,
            data_col=cfg.data_col,
            fold=fold,
            mode="valid",
            transform=cfg.test_transform,
            silent=silent,
        )
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0)

        model.load_state_dict(torch.load(model_path, map_location=cfg.device))
        model.to(cfg.device)

        val_preds, val_labels = run_eval(
            loader=val_loader,
            model=model,
            device=cfg.device,
        )

        indices = oof_df[cfg.fold_col] == fold
        val_preds_softmax = val_preds.softmax(-1).numpy().astype(float)

        for idx, label in enumerate(cfg.target_dict.keys()):
            oof_df.loc[indices, label] = val_preds_softmax[:, idx].astype(float)

        # 最も確率が高いクラスのインデックスを保存
        oof_df.loc[indices, "pred"] = np.argmax(val_preds_softmax, axis=-1)

        val_accuracy = accuracy_score(np.argmax(val_preds_softmax, axis=-1), val_labels)

        assert list(oof_df[indices][cfg.target_col]) == list(val_labels)

        if not silent:
            print(f"FOLD: {fold} - Accuracy: {val_accuracy}")
            print("=" * 80)

    return oof_df
