import pickle
import random
from glob import glob

import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import RobertaTokenizer


class StockNetDataset(Dataset):
    @torch.no_grad()
    def __init__(self, kind, config):
        super().__init__()
        tokenizer = RobertaTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")
        # padding index: 1
        tokenizer.model_max_length = config.max_words

        window_size = config.window_size  # 30
        forecast_size = config.forecast_size  # 10
        self.min_sents = config.min_sents  # 2
        self.max_sents = config.max_sents  # 5
        self.min_words = config.min_words  # 10
        self.max_words = config.max_words  # 20
        self.kind = kind  # train, val, test

        txt_fnames = sorted(glob("./data/my_stocknet/tweet/*.npy"))
        kw_fnames = sorted(glob("./data/my_stocknet/keywords/*.parquet"))
        price_fnames = sorted(glob("./data/my_stocknet/price/*.parquet"))
        price_fnames = [fname for fname in price_fnames if "GMRE" not in fname]
        fnames = list(zip(txt_fnames, kw_fnames, price_fnames))

        self.data = []
        self.changes = []
        for txt_fname, kw_fname, price_fname in tqdm(fnames):
            stock_name = price_fname.split("/")[-1][:-8]
            assert stock_name == txt_fname.split("/")[-1][:-4]

            prices = pd.read_parquet(price_fname)  # (~ x 7)
            tweets = np.load(txt_fname)  # (~ x 1+3+768)
            keywords = pd.read_parquet(kw_fname)  # (~ x 2)
            prices, tweets, keywords, price_dates, tweet_dates = self.process_stock_data(tweets, keywords, prices)

            if self.kind == "train":
                prices = prices[: int(prices.size(0) * config.train_prop)]
                price_dates = price_dates[: int(price_dates.shape[0] * config.train_prop)]
            elif self.kind == "val":
                prices = prices[int(prices.size(0) * config.train_prop) : int(prices.size(0) * config.val_prop)]
                price_dates = price_dates[
                    int(price_dates.shape[0] * config.train_prop) : int(price_dates.shape[0] * config.val_prop)
                ]
            elif self.kind == "test":
                prices = prices[int(prices.size(0) * config.val_prop) :]
                price_dates = price_dates[int(price_dates.shape[0] * config.val_prop) :]
            else:
                raise ValueError(f"Invalid kind: {self.kind}")

            data_num = prices.size(0) - window_size - forecast_size + 1
            for index in range(data_num):
                # slice price
                temp_dates = price_dates[index : index + window_size + forecast_size]
                price_input = prices[index : index + window_size]  # (T x 2)
                price_target = prices[index + window_size : index + window_size + forecast_size, 0]
                change, price_target = self.process_target(price_input, price_target)  # float
                self.changes.append(change)

                start_date = temp_dates[0]  # start date of input
                end_date = temp_dates[window_size - 1]  # end date of input

                # slice tweet
                indices = (tweet_dates >= start_date) & (tweet_dates <= end_date)
                txt = tweets[indices, :]  # (~ x 3+768)
                # pick sentences
                txt, use, sent_mask = self.pick_sentences(txt)  # TODO
                if use is False:
                    continue

                # slice keywords
                kws: list[str]
                kws = keywords.loc[indices].tolist()
                kws = [sent.split("\t") for sent in kws if len(sent) > 0]
                kws = list(set(word for sent in kws for word in sent))
                # pick keywords
                random.seed(42)
                kws, use = self.pick_keywords(kws)
                if use is False:
                    continue
                # tokenize keywords
                kws = tokenizer(
                    kws,
                    add_special_tokens=False,
                    padding="max_length",
                    truncation=True,
                    is_split_into_words=True,
                    return_tensors="pt",
                    return_attention_mask=True,
                )
                kws: Tensor
                kw_mask: Tensor
                kw_mask = ~kws.attention_mask.bool()  # (max_words)
                kws = kws.input_ids[0]  # (max_words)

                repeat_num = 1
                if (self.kind == "train") and (price_target in {0, 3}):
                    repeat_num = 5

                for _ in range(repeat_num):
                    self.data.append((price_input, txt, kws, sent_mask, kw_mask, price_target))

        self.length = len(self.data)
        return

    def process_stock_data(
        self, tweets: np.ndarray, keywords: pd.DataFrame, prices: pd.DataFrame
    ) -> tuple[Tensor, Tensor, pd.Series, np.ndarray, np.ndarray]:
        tweet_dates: np.ndarray[tuple[int], int]
        tweet_dates = tweets[:, 0].astype(int)

        start_date = max(prices["date"].min().item(), tweet_dates.min())
        end_date = min(prices["date"].max().item(), tweet_dates.max())

        prices = prices.loc[(prices["date"] >= start_date) & (prices["date"] <= end_date)]
        price_dates = prices["date"].values
        prices = torch.FloatTensor(prices.loc[:, ["high", "volume"]].values)  # (~ x 2) (high, volume)

        tweets = tweets[(tweet_dates >= start_date) & (tweet_dates <= end_date), :]
        tweet_dates = tweets[:, 0].astype(int)
        tweets = torch.FloatTensor(tweets[:, 1:])  # (~ x 3+768)

        keywords = keywords.loc[(keywords["date"] >= start_date) & (keywords["date"] <= end_date), :]
        keywords = keywords["keywords"].reset_index(drop=True)

        return prices, tweets, keywords, price_dates, tweet_dates

    def process_target(self, price_input: Tensor, price_target: Tensor):
        # x: (T x 2), target: (H)
        standard = price_input[-1, 0]
        change_min = (price_target.min() - standard) / standard
        change_min = change_min.item()
        change_max = (price_target.max() - standard) / standard
        change_max = change_max.item()

        change: float
        change = change_max if abs(change_max) >= abs(change_min) else change_min
        if change < -0.05:  # noqa: PLR2004
            result = 0
        elif change < 0.0:
            result = 1
        elif change < 0.05:  # noqa: PLR2004
            result = 2
        else:
            result = 3

        return change, result

    def pick_sentences(self, txt: Tensor):
        scores = txt[:, :3]  # (~ x 3)
        txt = txt[:, 3:]  # (~ x 768)
        values, labels = scores.max(1)  # (~), (~)
        mask = ~labels.eq(1)  # pick only positive and negative

        txt = txt[mask]  # (~ x 768)
        values = values[mask]
        sent_num = mask.sum().item()
        # sent_num = scores.size(0)
        sent_mask = torch.zeros(1, self.max_sents, dtype=torch.bool)
        use = True

        if sent_num >= self.max_sents:
            _, indices = values.topk(k=self.max_sents, dim=0)
            txt = txt[indices]  # (max_sents x 768)
        elif sent_num < self.min_sents:
            use = False
        else:
            txt = torch.cat([txt, torch.zeros(self.max_sents - sent_num, 768)], dim=0)
            sent_mask[0, sent_num:] = True

        return txt, use, sent_mask

    def pick_keywords(self, kws: list[str]):
        n_kws = len(kws)
        use = True

        if n_kws >= self.max_words:
            kws = random.sample(kws, k=self.max_words)
            kws = " ".join(kws)  # (max_words)
        elif n_kws < self.min_words:
            use = False
        else:
            kws = " ".join(kws)  # (n_words)

        return kws, use

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        input_ts, input_text, input_kw, sent_mask, kw_mask, target = self.data[index]
        return dict(
            input_ts=input_ts,
            input_text=input_text,
            input_kw=input_kw,
            sent_mask=sent_mask,
            kw_mask=kw_mask,
            target=target,
        )


if __name__ == "__main__":
    config: DictConfig
    config = OmegaConf.load("./configs/setting1.yaml")
    for kind in ("train", "val", "test"):
        dataset = StockNetDataset(kind, config.dataset)
        print(len(dataset))  # (24585, 957, 1148) (pos/neg sentences only) (26371, 1029, 1244)
        with open(f"./{kind}.target", "wb") as f:
            pickle.dump(dataset.changes, f)

        loader = DataLoader(dataset, **config.dataloader.val)
        for batch in loader:
            pass

        for batch in loader:
            batch: dict[str, Tensor]
            for name, data in batch.items():
                print(name, data.size())
            break
