import torch
from torch import Tensor, nn
from transformers import RobertaForSequenceClassification

from .base_model import BaseModel
from .dlinear import DLinear


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        embed_dim,
        num_heads,
        dropout,
        weight_kind,
    ):
        super().__init__()
        self.layer_q = nn.Linear(embed_dim, embed_dim)
        self.layer_k = nn.Linear(embed_dim, embed_dim)
        self.layer_v = nn.Linear(embed_dim, embed_dim)
        self.layer_output = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        if weight_kind == "softmax":
            self.mask_value = -1e10
            self.get_weight = nn.Softmax(3)
        elif weight_kind == "tanh":
            self.mask_value = 0
            self.get_weight = nn.Tanh()
        else:
            raise ValueError(f"Invalid value of `weight_kind`: {weight_kind}")
        return

    def forward(self, query: Tensor, key: Tensor, mask: Tensor = None) -> Tensor:
        # mask: (-1 x T1 x T2)
        query = self.layer_q(query)
        key = self.layer_k(key)
        value: Tensor
        value = self.layer_v(key)

        query = query.view(-1, query.size(1), self.num_heads, self.head_dim)
        key = key.view(-1, key.size(1), self.num_heads, self.head_dim)
        value = value.view_as(key)

        query = query.permute(0, 2, 1, 3).contiguous()  # (-1 x num_heads x T1 x d_h)
        key = key.permute(0, 2, 3, 1).contiguous()  # (-1 x num_heads x d_h x T2)
        value = value.permute(0, 2, 1, 3).contiguous()  # (-1 x num_heads x T2 x d_h)

        attention = query @ key  # (-1 x num_heads x T1 x T2)
        if mask is not None:
            mask = mask.unsqueeze(1)  # (-1 x 1 x T1 x T2)
            attention = attention.masked_fill(mask, self.mask_value)  # (-1 x num_heads x T1 x T2)
        attention = self.dropout(self.get_weight(attention))  # (-1 x num_heads x T1 x T2)

        output: Tensor
        output = attention @ value  # (-1 x num_heads x T1 x d_h)
        output = output.permute(0, 2, 1, 3).contiguous()  # (-1 x T1 x num_heads x d_h)
        output = output.view(-1, output.size(1), self.embed_dim)  # (-1 x T1 x d)
        output = self.layer_output(output)  # (-1 x T1 x d)

        return output, attention


class GeneralAttention(nn.Module):
    def __init__(self, query_dim, key_dim, dropout, weight_kind):
        super().__init__()
        self.layer_k = nn.Linear(key_dim, query_dim, bias=False)
        self.layer_output = nn.Linear(key_dim, key_dim)
        self.dropout = nn.Dropout(dropout)

        if weight_kind == "softmax":
            self.mask_value = -1e10
            self.get_weight = nn.Softmax(2)
        elif weight_kind == "tanh":
            self.mask_value = 0
            self.get_weight = nn.Tanh()
        else:
            raise ValueError(f"Invalid value of `weight_kind`: {weight_kind}")
        return

    def forward(self, query: Tensor, key: Tensor, mask: Tensor = None) -> Tensor:
        # query: (-1 x 1 x d1)
        # key: (-1 x T x d2)
        # mask: (-1 x 1 x T)
        key = self.layer_k(key)  # (-1 x T x d1)

        attention = query @ key.permute(0, 2, 1).contiguous()  # (-1 x 1 x T)
        if mask is not None:
            attention = attention.masked_fill(mask, self.mask_value)  # (-1 x 1 x T)
        attention = self.dropout(self.get_weight(attention))  # (-1 x 1 x T)

        output: Tensor
        output = attention @ key  # (-1 x 1 x d2)
        output = self.layer_output(output)  # (-1 x 1 x d2)
        return output, attention


class Aggregater(nn.Module):
    def __init__(self, kind, **kwargs):
        super().__init__()
        if kind == "multihead":
            self.layer = MultiHeadAttention(**kwargs)
        elif kind == "general":
            self.layer = GeneralAttention(**kwargs)
        else:
            raise ValueError(f"Invalid value of `kind`: {kind}")
        return

    def forward(self, query: Tensor, key: Tensor, mask: Tensor) -> Tensor:
        output, attention = self.layer(query, key, mask)
        return output, attention


class MyModel(BaseModel):
    train_keys: tuple[str] = ("loss",)
    # val_keys: tuple[str] = ("loss", "mae")
    val_keys: tuple[str] = ("loss", "acc")

    def __init__(self, config):
        super().__init__()
        self.do_kw_attention = config.do_kw_attention
        # time series part
        self.layer_norm1 = nn.BatchNorm1d(config.ts_feature_num)
        self.ts_encoder = DLinear(config.dlinear)
        self.layer_ts = nn.Linear(config.forecast_size * config.ts_feature_num, config.hidden_dim)

        # sentence part
        self.layer_text = nn.Sequential(
            nn.Linear(config.embed_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.hidden_dim),
        )

        # keyword part
        temp = RobertaForSequenceClassification.from_pretrained(config.roberta_pretrained_path)
        self.keyword_embedding = temp.get_input_embeddings()
        self.keyword_embedding.requires_grad_(False)
        self.layer_kw = nn.Sequential(
            nn.Linear(config.embed_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.hidden_dim),
        )
        del temp

        self.aggregater = Aggregater(**config.aggregater)
        self.layer_norm2 = nn.LayerNorm(config.hidden_dim)
        self.context_keyword_attention = Aggregater(**config.context_keyword_attention)
        self.layer_norm3 = nn.LayerNorm(config.hidden_dim)
        self.final_ffn = nn.Sequential(  # (-1 x d_embed) -> (-1 x H)
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            # nn.Linear(config.hidden_dim, config.forecast_size),
            nn.Linear(config.hidden_dim, config.num_classes),
        )
        self.dropout = nn.Dropout(config.dropout)

        # self.criterion = nn.MSELoss(**config.loss_kwargs)
        # self.val_criterion = nn.MSELoss(**config.val_loss_kwargs)
        loss_kwargs = dict(config.loss_kwargs)
        if config.loss_kwargs.weight is not None:
            loss_kwargs["weight"] = torch.FloatTensor(config.loss_kwargs.weight)
        # self.criterion = nn.CrossEntropyLoss(**loss_kwargs)
        # self.val_criterion = nn.CrossEntropyLoss(**config.val_loss_kwargs)
        self.criterion = nn.CrossEntropyLoss(reduction="none")  #! need to be fixed
        self.val_criterion = nn.CrossEntropyLoss(reduction="none")  #! need to be fixed
        return

    def get_output(self, input_ts: Tensor, input_text: Tensor, input_kw: Tensor, sent_mask: Tensor, kw_mask: Tensor):
        # `input_ts`: (-1 x T x d1) time series data (need to be normalized)
        # `input_text`: (-1 x max_sents x d_embed) output of RobertaForSequenceClassification.roberta
        # # good representation for sentiment analysis pretrained on twitter data
        # # max_sents: set as 30, padding of sentences applied
        # `input_kw`: (-1 x max_words) keyword data

        # step 1: make representations
        # # time series
        rep_ts: Tensor
        rep_ts = input_ts.permute(0, 2, 1).contiguous()  # (-1 x d_ts x T)
        rep_ts = self.layer_norm1(rep_ts)  # (-1 x d_ts x T)
        rep_ts = rep_ts.permute(0, 2, 1).contiguous()  # (-1 x T x d_ts)

        rep_ts = self.ts_encoder(input_ts)  # (-1 x H x d_ts)
        rep_ts = rep_ts.flatten(1)  # (-1 x H*d_ts)
        rep_ts = self.layer_ts(self.dropout(rep_ts))  # (-1 x d_embed)
        rep_ts = rep_ts.unsqueeze(1)  # (-1 x 1 x d_embed)

        # # sentence
        rep_text = self.layer_text(input_text)  # (-1 x max_sents x d_embed)

        # # keywords
        rep_kw = self.keyword_embedding(input_kw)  # (-1 x max_words x d_embed)
        rep_kw = self.layer_kw(rep_kw)  # (-1 x max_words x d_embed)

        # step 2: aggregate time series & text representations to make context vector
        # # considers relationship between time series history and sentimental representation of text
        rep_both, sent_attention = self.aggregater(rep_ts, rep_text, sent_mask)  # (-1 x 1 x d_embed)
        rep_both = self.layer_norm2(rep_ts + rep_both)  # (-1 x 1 x d_embed)
        output: Tensor

        if self.do_kw_attention is True:
            # step 3: aggregate context vector & keyword representations to make final output
            output, kw_attention = self.context_keyword_attention(rep_both, rep_kw, kw_mask)  # (-1 x 1 x d_embed)
            output = self.layer_norm3(rep_both + output)  # (-1 x 1 x d_embed)
            # sum of keywords with attention weights \in (-1, 1)
        else:
            kw_attention = None
            output = rep_both

        # step 4: feed forward network
        output = output.squeeze(1)  # (-1 x d_embed)
        output = self.final_ffn(output)  # (-1 x 4)

        return output, sent_attention, kw_attention

    @torch.no_grad()
    def predict(
        self, input_ts: Tensor, input_text: Tensor, input_kw: Tensor, sent_mask: Tensor, kw_mask: Tensor
    ) -> dict[str, Tensor]:
        output, sent_attention, kw_attention = self.get_output(input_ts, input_text, input_kw, sent_mask, kw_mask)
        return dict(output=output, sentence_attention=sent_attention, keyword_attention=kw_attention)

    def forward(
        self, input_ts: Tensor, input_text: Tensor, input_kw: Tensor, sent_mask: Tensor, kw_mask: Tensor, target: Tensor
    ) -> Tensor:
        output: Tensor
        loss: Tensor
        output, _, _ = self.get_output(input_ts, input_text, input_kw, sent_mask, kw_mask)
        loss = self.criterion(output, target)

        #! to be fixed
        pt = (-loss).exp()
        loss = (1 - pt).pow(2) * loss
        loss = loss.mean()
        return loss

    @torch.no_grad()
    def validate_batch(
        self, input_ts: Tensor, input_text: Tensor, input_kw: Tensor, sent_mask: Tensor, kw_mask: Tensor, target: Tensor
    ) -> dict[str, float]:
        loss: Tensor
        output: Tensor  # (-1 x 4)
        output, _, _ = self.get_output(input_ts, input_text, input_kw, sent_mask, kw_mask)
        loss = self.val_criterion(output, target)
        #! to be fixed
        pt = (-loss).exp()
        loss = (1 - pt).pow(2) * loss
        loss = loss.sum()
        loss = loss.item()

        acc = output.argmax(1) == target
        acc = acc.sum().item()

        return dict(zip(self.val_keys, (loss, acc)))


class DLinearModel(BaseModel):
    train_keys: tuple[str] = ("loss",)
    # val_keys: tuple[str] = ("loss", "mae")
    val_keys: tuple[str] = ("loss", "acc")

    def __init__(self, config):
        super().__init__()
        # time series part
        self.layer_norm1 = nn.BatchNorm1d(config.ts_feature_num)
        self.ts_encoder = DLinear(config.dlinear)
        self.layer_ts = nn.Sequential(
            DLinear(config.dlinear),
            nn.Flatten(1),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.forecast_size * config.ts_feature_num, config.num_classes),
        )

        self.criterion = nn.CrossEntropyLoss(reduction="none")  #! need to be fixed
        self.val_criterion = nn.CrossEntropyLoss(reduction="none")  #! need to be fixed
        return

    def get_output(self, input_ts: Tensor):
        # step 1: make representations
        # # time series
        output: Tensor
        output = input_ts.permute(0, 2, 1).contiguous()  # (-1 x d_ts x T)
        output = self.layer_norm1(output)  # (-1 x d_ts x T)
        output = output.permute(0, 2, 1).contiguous()  # (-1 x T x d_ts)

        output = self.layer_ts(output)  # (-1 x num_classes)
        return output

    def forward(self, input_ts: Tensor, target: Tensor, **kwargs) -> Tensor:
        output: Tensor
        loss: Tensor
        output = self.get_output(input_ts)
        loss = self.criterion(output, target)

        #! to be fixed
        pt = (-loss).exp()
        loss = (1 - pt).pow(2) * loss
        loss = loss.mean()
        return loss

    @torch.no_grad()
    def predict(self, input_ts: Tensor, **kwargs) -> dict[str, Tensor]:
        output = self.get_output(input_ts)
        return dict(output=output)

    @torch.no_grad()
    def validate_batch(self, input_ts: Tensor, target: Tensor, **kwargs) -> dict[str, float]:
        loss: Tensor
        output: Tensor  # (-1 x 4)
        output = self.get_output(input_ts)
        loss = self.val_criterion(output, target)

        #! to be fixed
        pt = (-loss).exp()
        loss = (1 - pt).pow(2) * loss
        loss = loss.sum()
        loss = loss.item()

        acc = output.argmax(1) == target
        acc = acc.sum().item()

        return dict(zip(self.val_keys, (loss, acc)))
