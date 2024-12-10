from typing import Type

import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Dataset

from dataset import StockNetDataset
from models import BaseModel, MyModel
from tools import val_loop


@torch.no_grad()
def test(model: Type[BaseModel], config: DictConfig, test_set: Dataset):
    test_loader = DataLoader(test_set, **config.dataloader.test)
    check_unit = int(len(test_set) * config.check_prop)
    assert check_unit != 0, "check_unit is 0. Please look into `len(train_loader)`, and `check_prop`."

    model: BaseModel
    model = model(config.model).to(config.device)
    print()

    test_metrics = val_loop(model, test_loader, kind="test")
    return test_metrics


if __name__ == "__main__":
    config = OmegaConf.load("./configs/config.yaml")
    test_set = StockNetDataset("test", config.dataset)
    test_metrics = test(MyModel, config, test_set)
    print(test_metrics)
