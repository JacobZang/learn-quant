import torch
import torch.nn as nn
from .forecast_base import ForecastModel

class LSTMNet(nn.Module):
    """
    LSTM 网络架构
    该类定义了一个标准的 LSTM 模型
    后跟一个线性层以产生最终预测
    """
    def __init__(
        self,
        d_feat: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        pred_len: int,
    ):
        """
        初始化 LSTM 网络
        参数:
            d_feat (int): 输入特征的数量
            hidden_size (int): LSTM 隐藏状态中的特征数量
            num_layers (int): 循环层的数量
            dropout (float): dropout 概率
            pred_len (int): 预测序列的长度
        """
        super(LSTMNet, self).__init__()
        
        self.lstm = nn.LSTM(
            input_size=d_feat,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )
        self.fc = nn.Linear(hidden_size, pred_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        定义 LSTM 模型的前向传播

        参数:
            x (torch.Tensor): 输入张量，形状为 (batch_size, seq_len, d_feat)

        返回:
            torch.Tensor: 输出预测张量，形状为 (batch_size, pred_len)
        """
        # LSTM 层
        # x shape: (batch_size, seq_len, d_feat)
        # lstm_out shape: (batch_size, seq_len, hidden_size)
        # h_n, c_n shape: (num_layers, batch_size, hidden_size)
        lstm_out, (h_n, c_n) = self.lstm(x)

        # 使用最后一个时间步的隐藏状态进行预测
        # h_n 的形状是 (num_layers, batch_size, hidden_size)
        # 我们取最后一层的输出
        last_hidden_state = h_n[-1, :, :] # 形状: (batch_size, hidden_size)
        
        # 全连接层
        # 输出形状: (batch_size, pred_len)
        out = self.fc(last_hidden_state)
        
        return out.squeeze()


class LSTMModel(ForecastModel):
    """
    LSTM 网络的 Qlib 兼容包装器

    该类继承自 `ForecastModel` 并实现了 `_build_model` 方法
    来构建 `LSTMNet`
    """

    def _build_model(self) -> LSTMNet:
        """
        构建并返回 LSTMNet 模型

        返回:
            LSTMNet: LSTM 网络的一个实例
        """
        # 诸如 d_feat, hidden_size 等参数存储在 self 中，从基类的 __init__ 继承而来
        return LSTMNet(
            d_feat=self.d_feat,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
            pred_len=self.pred_len,
        )

# 示例
if __name__ == "__main__":
    # 模型参数
    _d_feat = 6
    _hidden_size = 64
    _num_layers = 2
    _dropout = 0.2
    _seq_len = 20
    _pred_len = 1
    _batch_size = 32

    # 创建一个虚拟输入张量
    dummy_input = torch.randn(_batch_size, _seq_len, _d_feat)

    # 实例化 Qlib 兼容的模型
    qlib_model = LSTMModel(
        d_feat=_d_feat,
        hidden_size=_hidden_size,
        num_layers=_num_layers,
        dropout=_dropout,
        pred_len=_pred_len
    )
    
    # 获取底层的 PyTorch 模型
    pytorch_model = qlib_model.model

    # 执行一次前向传播
    output = pytorch_model(dummy_input)

    # 打印形状以进行验证
    print("输入形状:", dummy_input.shape)
    print("输出形状:", output.shape)
    assert output.shape == (_batch_size, _pred_len) or output.shape == (_batch_size,)