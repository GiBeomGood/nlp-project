import argparse

from omegaconf import OmegaConf

from dataset import StockNetDataset
from models import DLinearModel
from train import train

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-path", type=str)
    args = parser.parse_args()

    config = OmegaConf.load(args.config_path)
    train_set = StockNetDataset("train", config.dataset)
    val_set = StockNetDataset("test", config.dataset)
    print("All settings are done. Start training.\n")
    train(DLinearModel, config, train_set, val_set)
