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
        tokenizer.model_max_length = config.max_words
        random.seed(42)

        window_size = config.window_size  # 60
        forecast_size = config.forecast_size  # 5
        self.max_sents = config.max_sents  # 30
        self.kind = kind  # train, val, test

        txt_fnames = sorted(glob("./data/my_stocknet/tweet/*.npy"))
        kw_fnames = sorted(glob("./data/my_stocknet/keywords/*.parquet"))
        price_fnames = sorted(glob("./data/my_stocknet/price/*.parquet"))
        price_fnames = [fname for fname in price_fnames if "GMRE" not in fname]
        fnames = list(zip(txt_fnames, kw_fnames, price_fnames))

        self.pos = 0  #! to be deleted
        self.neg = 0  #! to be deleted

        self.data = []
        for txt_fname, kw_fname, price_fname in tqdm(fnames):
            stock_name = price_fname.split("/")[-1][:-8]
            assert stock_name == txt_fname.split("/")[-1][:-4]

            tweets: np.ndarray = np.load(txt_fname)  # (~ x 1+3+768)
            keywords = pd.read_parquet(kw_fname)  # (~ x 2)
            prices = pd.read_parquet(price_fname)  # (~ x 7)
            tweets, keywords, prices, tweet_dates, price_dates = self.process_stock_data(tweets, keywords, prices)

            data_num = price_dates.shape[0] - window_size - forecast_size + 1
            if self.kind == "train":
                start_index = 0
                end_index = int(data_num * 0.8)
            elif self.kind == "val":
                start_index = int(data_num * 0.8)
                end_index = int(data_num * 0.9)
            elif self.kind == "test":
                start_index = int(data_num * 0.9)
                end_index = data_num

            for index in range(start_index, end_index):
                start_date = price_dates[index]  # start of input date
                mid_date = price_dates[index + window_size]  # start of target date
                # end_date = price_dates[index + window_size + forecast_size - 1]  # end of target date, unused

                indices = (tweet_dates >= start_date) & (tweet_dates < mid_date)
                txt = tweets[indices, :]  # (~ x 3+768)
                kws = keywords.loc[indices].tolist()  # (~)
                kws = [sent.split("\t") for sent in kws if len(sent) > 0]
                kws = list(set(word for sent in kws for word in sent))
                price_target = prices[index : index + window_size + forecast_size, :]

                # text processing
                txt = self.pick_sentences(txt)

                # keyword processing
                kws = random.sample(kws, k=config.max_words)
                kws = " ".join(kws)  # (n_words)
                kws = tokenizer(
                    kws,
                    add_special_tokens=False,
                    padding=False,
                    truncation=True,
                    is_split_into_words=True,
                    return_tensors="pt",
                    return_attention_mask=False,
                ).input_ids[0]  # (n_words)

                # price processing
                # price_input = price_target[:window_size]
                price_input = price_target[:window_size, [1, -1]]  # (T x 2)
                price_target = price_target[window_size:, 1]  # (forecast_size)
                price_target = price_target.mean() >= price_input[:, 0].mean()
                price_target = price_target.float()

                if int(price_target.item()) == 1:
                    self.pos += 1
                else:
                    self.neg += 1

                # append
                self.data.append((price_input, txt, kws, price_target))

        self.length = len(self.data)
        return

    def process_stock_data(self, tweets: np.ndarray, keywords: pd.DataFrame, prices: pd.DataFrame):
        tweet_dates: np.ndarray[tuple[int], int]
        tweet_dates = tweets[:, 0].astype(int)

        # slice data by available dates
        both_dates = set(tweet_dates.tolist()).intersection(set(prices.date.tolist()))
        tweets = tweets[pd.Series(tweet_dates).isin(both_dates), :]
        keywords = keywords.loc[keywords.date.isin(both_dates), "keywords"].reset_index(drop=True)
        prices = prices.loc[prices.date.isin(both_dates), :]

        tweet_dates = tweets[:, 0].astype(int)

        # data processing
        price_dates = prices["date"].values
        prices = prices.values

        tweets: Tensor
        keywords: pd.Series
        prices: Tensor
        tweets = torch.FloatTensor(tweets[:, 1:])  # (~ x 3+768)
        prices = torch.FloatTensor(prices[:, 1:])  # (~ x 6)

        return tweets, keywords, prices, tweet_dates, price_dates

    def pick_sentences(self, txt: Tensor):
        scores = txt[:, :3]  # (~ x 3)
        txt = txt[:, 3:]  # (~ x 768)
        values, labels = scores.max(1)  # (~), (~)
        mask = ~labels.eq(1)  # pick only positive and negative

        txt = txt[mask]  # (~ x 768)
        values = values[mask]
        sent_num = mask.sum().item()
        if sent_num < self.max_sents:
            print(sent_num)

        else:
            _, indices = values.topk(k=self.max_sents, dim=0)
            txt = txt[indices]  # (max_sents x 768)
        return txt

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        input_ts, input_text, input_kw, target = self.data[index]
        return dict(input_ts=input_ts, input_text=input_text, input_kw=input_kw, target=target)


if __name__ == "__main__":
    config: DictConfig
    config = OmegaConf.load("./configs/config.yaml")
    for kind in ("train", "val", "test"):
        dataset = StockNetDataset(kind, config.dataset)
        pos_prop = dataset.pos / (dataset.pos + dataset.neg)
        print(f"{pos_prop:5.1%}", len(dataset))  # 1689

        loader = DataLoader(dataset, **config.dataloader.val)
        for batch in loader:
            pass
