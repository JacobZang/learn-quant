import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from typing import Tuple, List, Union

from qlib.data.dataset import DatasetH
from qlib.data.dataset.handler import DataHandlerLP

class TimeSeriesDataset(Dataset):
    """
    用于 Qlib 时间序列数据的自定义 PyTorch 数据集。

    该数据集将时间序列数据组织成固定长度（`seq_len`）的序列。
    数据集中的每个样本都是一个元组，包含：
    - 一个历史特征序列 (x)。
    - 对应的待预测未来值 (y)。
    """

    def __init__(self, df: pd.DataFrame, seq_len: int, pred_len: int):
        """
        初始化数据集。

        参数:
            df (pd.DataFrame):
                一个具有多级索引（datetime, instrument）和特征/标签列的 DataFrame。
                'feature' 列应命名为 'feature_0', 'feature_1', 等。
                'label' 列应命名为 'label_0', 'label_1', 等。
            seq_len (int): 输入序列的长度（回看窗口）。
            pred_len (int): 预测序列的长度（预测范围）。
        """
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len

        # 处理特征和标签
        features, labels = self._process_df(df)
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)

        # 创建索引映射
        self.index = df.index

    def _process_df(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        从 DataFrame 中提取特征和标签，并将它们对齐成序列。
        此方法处理按 instrument 分组和创建滚动窗口。
        """
        features_cols = [c for c in df.columns if c.startswith("feature")]
        label_cols = [c for c in df.columns if c.startswith("label")]

        x_list, y_list = [], []

        # 按 instrument 分组，为每只股票创建序列
        for instrument, group in df.groupby(level="instrument"):
            feature_data = group[features_cols].values
            label_data = group[label_cols].values

            if len(feature_data) < self.seq_len + self.pred_len:
                continue

            # 使用滑动窗口创建序列
            for i in range(len(feature_data) - self.seq_len - self.pred_len + 1):
                x_seq = feature_data[i : i + self.seq_len]
                # 标签是序列末尾的值
                y_seq = label_data[i + self.seq_len : i + self.seq_len + self.pred_len]

                x_list.append(x_seq)
                y_list.append(y_seq.squeeze())

        return np.array(x_list, dtype=np.float32), np.array(y_list, dtype=np.float32)

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.features[index], self.labels[index]


def get_ts_data_loader(
    dataset: DatasetH,
    batch_size: int,
    seq_len: int,
    pred_len: int,
    shuffle: bool = True,
    num_workers: int = 4,
) -> torch.utils.data.DataLoader:
    """
    为 Qlib 数据集分段（train, valid, test）创建一个 PyTorch DataLoader。

    参数:
        dataset (DatasetH): Qlib 历史数据集。
        batch_size (int): 每批的样本数。
        seq_len (int): 输入序列的长度。
        pred_len (int): 预测序列的长度。
        shuffle (bool): 是否打乱数据。
        num_workers (int): 用于数据加载的子进程数。

    返回:
        torch.utils.data.DataLoader: 一个 DataLoader 实例。
    """
    # 从 Qlib 数据集准备 DataFrame
    df = dataset.prepare("train", col_set=["feature", "label"], data_key=DataHandlerLP.DK_L)
    
    # 重命名列以保证清晰度和与 TimeSeriesDataset 的兼容性
    feature_cols = [f"feature_{i}" for i in range(df.shape[1] - df.columns.str.startswith("label").sum())]
    label_cols = [f"label_{i}" for i in range(df.columns.str.startswith("label").sum())]
    df.columns = feature_cols + label_cols

    # 创建自定义数据集
    ts_dataset = TimeSeriesDataset(df, seq_len=seq_len, pred_len=pred_len)

    # 创建并返回 DataLoader
    return torch.utils.data.DataLoader(
        dataset=ts_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=True, # 丢弃最后一个不完整的批次
    )

if __name__ == "__main__":
    # 使用示例:
    import qlib
    from qlib.constant import REG_CN

    # 1. 初始化 Qlib
    provider_uri = "./qlib_data/cn_data"
    qlib.init(provider_uri=provider_uri, region=REG_CN)

    # 2. 获取 Qlib 数据集
    from qlib.data import D
    dataset = D.features(D.instruments("csi300"), ["$close", "$volume"], start_time='2015-01-01', end_time='2017-01-01')
    dataset['label'] = D.features(D.instruments("csi300"), ["Ref($close, -1)/$close - 1"], start_time='2015-01-01', end_time='2017-01-01')
    dataset = dataset.dropna()

    # 3. 创建 TimeSeriesDataset
    # dataframe 必须格式化为带有 'feature' 和 'label' 列
    df_formatted = dataset.copy()
    feature_names = ["$close", "$volume"]
    label_name = "label"
    
    # 重命名列以适应预期格式
    df_formatted.columns = ["feature_0", "feature_1", "label_0"]

    seq_len = 20
    pred_len = 1
    
    ts_dataset = TimeSeriesDataset(df_formatted, seq_len=seq_len, pred_len=pred_len)

    # 4. 创建 DataLoader
    data_loader = torch.utils.data.DataLoader(ts_dataset, batch_size=32, shuffle=True)

    # 5. 遍历一个批次
    for x_batch, y_batch in data_loader:
        print("特征批次形状:", x_batch.shape)  # 应为 (batch_size, seq_len, num_features)
        print("标签批次形状:", y_batch.shape)    # 应为 (batch_size, pred_len)
        break