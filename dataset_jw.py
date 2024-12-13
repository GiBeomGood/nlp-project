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

        self.data = []
        for txt_fname, kw_fname, price_fname in tqdm(fnames):
            stock_name = price_fname.split("/")[-1][:-8]
            assert stock_name == txt_fname.split("/")[-1][:-4]

            tweets: np.ndarray = np.load(txt_fname)  # (트윗수 x 1+3+768)
            keywords = pd.read_parquet(kw_fname)  # (트윗수 x 2)
            prices = pd.read_parquet(price_fname)  # (timeseries 길이 x 7)
            tweets, keywords, tweet_dates, price_dates = self.process_stock_data(tweets, keywords, prices)
            # 애플에서 트윗 나온 날만 정보 추출. 하지만 아직까지 트윗은 같은 날 여러 row가 있음.

            data_num = price_dates.shape[0]  # 수정
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
                start_date = prices[prices.index[prices["date"] == price_dates[index]][0] - window_size + 1]["date"]
                mid_date = price_dates[index]  # start of input date
                # end_date = price_dates[index + window_size + forecast_size - 1]  # end of target date, unused

                indices = (tweet_dates >= start_date) & (tweet_dates < mid_date)
                txt = tweets[indices, :]  # (~ x 3+768)
                kws = keywords.loc[indices].tolist()  # (~)
                kws = [sent.split("\t") for sent in kws if len(sent) > 0]
                kws = list(set(word for sent in kws for word in sent))  # 해당 기간 키워드 모두 set() 처리
                price_target = torch.FloatTensor(
                    prices[(prices["date"] >= start_date) & (prices["date"] <= mid_date)].values[:, 1:]
                )

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
                price_input = price_target[:window_size, [3, -1]]  # (T x 6)
                price_target = price_target[window_size:, 3]  # (forecast_size) 종가로 수정
                """ multilabel """
                last_close = price_input[-1, 3]  # 과거 데이터의 마지막 종가
                future_high = price_target[:, 3].max()  # 미래 데이터의 최고 종가
                future_low = price_target[:, 3].min()  # 미래 데이터의 최저 종가
                max_gain = ((future_high - last_close) / last_close) * 100  # 최고 상승률 (%)
                max_loss = ((future_low - last_close) / last_close) * 100  # 최고 하락률 (%)
                # 절댓값이 큰 변화 선택
                max_change = max_gain if abs(max_gain) > abs(max_loss) else max_loss

                # price_target 범위에 따라 지정 - 임시 지정
                if max_change < -10:
                    price_target = 0
                elif -10 <= max_change < -5:
                    price_target = 1
                elif -5 <= max_change < 5:
                    price_target = 2
                elif 5 <= max_change < 10:
                    price_target = 3
                else:
                    price_target = 4
                """ binary """
                # 과거 데이터 마지막 날 종가
                last_close = price_input[-1, 1]
                # 예측 데이터 마지막 날 종가
                future_last_close = price_target[-1, 1]
                # 비교하여 price_target 설정
                price_target = (future_last_close > last_close).float()  # True → 1.0, False → 0.0
                """ """
                price_target = price_target.float()

                # append
                self.data.append((price_input, txt, kws, price_target))  # ((60, 2), (30, 768), (n_keywords,), ())

        self.length = len(self.data)
        return

    def process_stock_data(self, tweets: np.ndarray, keywords: pd.DataFrame, prices: pd.DataFrame):
        tweet_dates: np.ndarray[tuple[int], int]
        tweet_dates = tweets[:, 0].astype(int)

        # slice data by available dates
        both_dates = set(tweet_dates.tolist()).intersection(
            set(prices.date.tolist()[self.window_size : -self.forecast_size])
        )  # 수정
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
        tweets = torch.FloatTensor(tweets)  # (~ x 3+768)
        prices = torch.FloatTensor(prices[:, 1:])  # (~ x 6)

        return tweets, keywords, tweet_dates, price_dates

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
        print(len(dataset))  # 1689

        loader = DataLoader(dataset, **config.dataloader.val)
        for batch in loader:
            pass
