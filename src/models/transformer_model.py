import torch
import torch.nn as nn
import math
from .forecast_base import ForecastModel

class PositionalEncoding(nn.Module):
    """向序列中的词元（token）注入一些关于其相对或绝对位置的信息"""
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        参数:
            x: 张量, 形状为 [seq_len, batch_size, d_model]
        """
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class TransformerNet(nn.Module):
    """
    一个简化的基于 Transformer 的预测模型

    该模型使用 Transformer 编码器来处理输入序列
    """
    def __init__(
        self,
        d_feat: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        dropout: float,
        pred_len: int,
        seq_len: int,
    ):
        super(TransformerNet, self).__init__()
        self.feature_proj = nn.Linear(d_feat, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len=seq_len + 1)
        
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, d_model * 4, dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        
        self.decoder = nn.Linear(d_model, pred_len)
        self.d_model = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        参数:
            x: 张量, 形状为 [batch_size, seq_len, d_feat]
        """
        # 将特征投影到模型维度
        x = self.feature_proj(x) * math.sqrt(self.d_model) # 形状: [batch_size, seq_len, d_model]
        
        # 添加位置编码
        # TransformerEncoderLayer 期望形状为 [seq_len, batch_size, d_model] (如果 batch_first=False)
        # 但我们使用了 batch_first=True, 所以形状是 [batch_size, seq_len, d_model]
        x = self.pos_encoder(x.transpose(0, 1)).transpose(0, 1) # 这看起来有点笨拙，但符合 PE 的逻辑
        
        # 通过 transformer 编码器
        output = self.transformer_encoder(x) # 形状: [batch_size, seq_len, d_model]
        
        # 取最后一个时间步的输出
        output = output[:, -1, :] # 形状: [batch_size, d_model]
        
        # 解码到预测长度
        output = self.decoder(output) # 形状: [batch_size, pred_len]
        
        return output.squeeze()


class TransformerModel(ForecastModel):
    """
    Transformer 网络的 Qlib 兼容包装器

    与基类相比，此类需要额外的参数 (`d_model`, `nhead`)
    """
    def __init__(
        self,
        d_feat: int = 6,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
        n_epochs: int = 200,
        lr: float = 1e-3,
        batch_size: int = 2048,
        seq_len: int = 20,
        pred_len: int = 1,
        **kwargs
    ):
        self.d_model = d_model
        self.nhead = nhead
        super().__init__(
            d_feat=d_feat,
            hidden_size=d_model, # 使用 hidden_size 来存储 d_model 以保持一致性
            num_layers=num_layers,
            dropout=dropout,
            n_epochs=n_epochs,
            lr=lr,
            batch_size=batch_size,
            seq_len=seq_len,
            pred_len=pred_len,
            **kwargs,
        )

    def _build_model(self) -> TransformerNet:
        return TransformerNet(
            d_feat=self.d_feat,
            d_model=self.d_model,
            nhead=self.nhead,
            num_layers=self.num_layers,
            dropout=self.dropout,
            pred_len=self.pred_len,
            seq_len=self.seq_len,
        )

# 如何使用此模型的示例
if __name__ == "__main__":
    _d_feat = 6
    _d_model = 64
    _nhead = 4
    _num_layers = 2
    _dropout = 0.1
    _seq_len = 20
    _pred_len = 1
    _batch_size = 32

    dummy_input = torch.randn(_batch_size, _seq_len, _d_feat)

    qlib_model = TransformerModel(
        d_feat=_d_feat,
        d_model=_d_model,
        nhead=_nhead,
        num_layers=_num_layers,
        dropout=_dropout,
        seq_len=_seq_len,
        pred_len=_pred_len
    )

    pytorch_model = qlib_model.model
    output = pytorch_model(dummy_input)

    print("输入形状:", dummy_input.shape)
    print("输出形状:", output.shape)
    assert output.shape == (_batch_size, _pred_len) or output.shape == (_batch_size,)