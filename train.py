import argparse
from statistics import mean
from typing import Type

import torch
from omegaconf import DictConfig, OmegaConf
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import wandb
from dataset import StockNetDataset
from models import BaseModel, MyModel
from tools import EarlyStopping, pbar_finish, val_loop


def train_loop(
    model: BaseModel,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    check_unit: int,
) -> tuple[tqdm, dict[str, float]]:
    device = next(model.parameters()).device
    train_loss = []
    pbar = tqdm(total=len(train_loader), desc=f"Epoch {epoch:2d}")
    pbar_update = pbar.update

    model.train()
    for batch in train_loader:
        batch: dict[str, Tensor]
        batch = {key: tensor.to(device) for key, tensor in batch.items()}

        optimizer.zero_grad()
        loss: Tensor = model(**batch)
        loss.backward()
        optimizer.step()

        train_loss.append(loss.item())
        if pbar.n % check_unit == 0:
            pbar.set_postfix({"loss": mean(train_loss)})
        pbar_update()

    train_metrics = {"train loss": mean(train_loss)}

    return pbar, train_metrics


def train(
    model: Type[BaseModel],
    config: DictConfig,
    train_set: Dataset,
    val_set: Dataset,
):
    if config.wandb.do is True:
        run = wandb.init(
            config=OmegaConf.to_container(config, resolve=True, throw_on_missing=True),
            **config.wandb.kwargs,
        )

    train_loader = DataLoader(train_set, **config.dataloader.train)
    val_loader = DataLoader(val_set, **config.dataloader.val)
    early_stopping = EarlyStopping(**config.early_stopping)
    check_unit = int(len(train_loader) * config.check_prop)
    assert check_unit != 0, "check_unit is 0. Please look into `len(train_loader)`, and `check_prop`."

    model: BaseModel
    model = model(config.model).to(config.device)
    optimizer = torch.optim.Adam(model.parameters(), **config.optimizer)
    print()

    for epoch in range(config.epochs):
        pbar, train_metrics = train_loop(model, train_loader, optimizer, epoch, check_unit)
        val_metrics = val_loop(model, val_loader, kind="val")
        postfix = pbar_finish(pbar, train_metrics, val_metrics)

        if config.wandb.do is True:
            run.log(postfix, step=epoch)

        early_stop, improved = early_stopping.check(postfix)
        if early_stop is True:
            break

        if improved is True:
            torch.save(
                model.state_dict(),
                config.model_save_path,
            )

    if config.wandb.do is True:
        run.finish()
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-path", type=str)
    args = parser.parse_args()

    model_type = MyModel
    config = OmegaConf.load(args.config_path)
    train_set = StockNetDataset("train", config.dataset)
    val_set = StockNetDataset("test", config.dataset)
    print("All settings are done. Start training.\n")
    train(model_type, config, train_set, val_set)
