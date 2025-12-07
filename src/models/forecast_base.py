import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from qlib.data.dataset import DatasetH
from qlib.data.dataset.handler import DataHandlerLP
from qlib.log import get_module_logger
from qlib.model.base import Model
from typing import Union, List, Tuple, Any


class ForecastModel(Model):
    """
    Qlib 中基于 PyTorch 的预测模型的抽象基类。
    """

    def __init__(
        self,
        d_feat: int = 6,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.0,
        n_epochs: int = 200,
        lr: float = 1e-3,
        batch_size: int = 2048,
        early_stop: int = 20,
        loss: str = "mse",
        optimizer: str = "adam",
        seq_len: int = 20,
        pred_len: int = 1,
        seed: int = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # 模型架构参数
        self.d_feat = d_feat
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.seq_len = seq_len
        self.pred_len = pred_len

        # 训练参数
        self.n_epochs = n_epochs
        self.lr = lr
        self.batch_size = batch_size
        self.early_stop = early_stop
        self.loss_type = loss
        self.optimizer_type = optimizer
        self.seed = seed
        
        self.logger = get_module_logger(self.__class__.__name__)
        self.fitted = False

        if self.seed is not None:
            torch.manual_seed(self.seed)
            torch.cuda.manual_seed(self.seed)
            np.random.seed(self.seed)

        self.model = self._build_model()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def _build_model(self) -> nn.Module:
        raise NotImplementedError("子类必须实现 `_build_model` 方法。")

    def _get_loss(self):
        if self.loss_type == "mse":
            return nn.MSELoss()
        elif self.loss_type == "l1":
            return nn.L1Loss()
        else:
            raise NotImplementedError(f"不支持损失函数 `{self.loss_type}`。")

    def _get_optimizer(self):
        if self.optimizer_type.lower() == "adam":
            return optim.Adam(self.model.parameters(), lr=self.lr)
        elif self.optimizer_type.lower() == "sgd":
            return optim.SGD(self.model.parameters(), lr=self.lr)
        else:
            raise NotImplementedError(f"不支持优化器 `{self.optimizer_type}`。")

    def fit(self, dataset: DatasetH, evals_result: dict = None):
        df_train, df_valid = dataset.prepare(
            ["train", "valid"], col_set=["feature", "label"], data_key=DataHandlerLP.DK_L
        )
        if df_train.empty or df_valid.empty:
            raise ValueError("训练或验证数据集为空。请检查您的数据和日期范围。")

        train_loader, _ = self._prepare_data_loader(df_train, shuffle=True)
        valid_loader, _ = self._prepare_data_loader(df_valid, shuffle=False)

        optimizer = self._get_optimizer()
        loss_fn = self._get_loss()

        best_loss = float("inf")
        patience = self.early_stop
        best_model_state = self.model.state_dict()

        for epoch in range(self.n_epochs):
            self.model.train()
            train_loss = 0.0
            for features, labels in train_loader:
                features, labels = features.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                preds = self.model(features)
                loss = loss_fn(preds, labels)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            train_loss /= len(train_loader)

            self.model.eval()
            valid_loss = 0.0
            with torch.no_grad():
                for features, labels in valid_loader:
                    features, labels = features.to(self.device), labels.to(self.device)
                    preds = self.model(features)
                    loss = loss_fn(preds, labels)
                    valid_loss += loss.item()
            valid_loss /= len(valid_loader)
            self.logger.info(f"轮次 {epoch+1}/{self.n_epochs}, 训练损失: {train_loss:.6f}, 验证损失: {valid_loss:.6f}")

            if evals_result is not None:
                evals_result.setdefault("train", []).append(train_loss)
                evals_result.setdefault("valid", []).append(valid_loss)

            if valid_loss < best_loss:
                best_loss = valid_loss
                patience = self.early_stop
                best_model_state = self.model.state_dict()
            else:
                patience -= 1
                if patience == 0:
                    self.logger.info("早停已触发。")
                    break
        
        self.model.load_state_dict(best_model_state)
        self.fitted = True
        self.logger.info(f"训练完成。最佳验证损失: {best_loss:.6f}")

    def predict(self, dataset: DatasetH, segment: Union[str, slice] = "test") -> pd.DataFrame:
        if not self.fitted:
            raise RuntimeError("模型尚未训练。请先调用 `fit` 方法。")

        df_pred = dataset.prepare(segment, col_set=["feature"], data_key=DataHandlerLP.DK_I)
        if df_pred.empty:
            return pd.DataFrame()

        pred_loader, pred_indices = self._prepare_data_loader(df_pred, shuffle=False)
        
        self.model.eval()
        all_preds = []

        with torch.no_grad():
            for features in pred_loader:
                features = features[0].to(self.device)
                preds = self.model(features).cpu().numpy().squeeze()
                all_preds.append(preds)
        
        if not all_preds:
            return pd.DataFrame()

        preds = np.concatenate(all_preds)
        
        pred_df = pd.DataFrame(preds, index=pred_indices, columns=["score"])
        return pred_df

    def _prepare_data_loader(self, df: pd.DataFrame, shuffle: bool) -> Tuple[DataLoader, Any]:
        
        # In qlib, the column name of label is 'LABEL0', 'LABEL1', ...
        # It could also be multi-index, such as ('label', 'LABEL0')
        label_cols = [c for c in df.columns if (isinstance(c, str) and c.upper().startswith("LABEL")) or (isinstance(c, tuple) and c[0].upper() == "LABEL")]
        has_labels = len(label_cols) > 0

        x_all, y_all, index_all = [], [], []

        # Group by instrument to generate sequences
        for inst, group in df.groupby(level="instrument"):
            if len(group) < self.seq_len:
                continue

            features = group.drop(columns=label_cols).values
            
            if has_labels:
                labels = group[label_cols].values
            else:
                # Create dummy labels for prediction phase
                labels = np.zeros((len(group), self.pred_len))

            # The last valid index for a sequence start is `len(group) - self.seq_len`
            for i in range(len(group) - self.seq_len + 1):
                # The end of the feature sequence
                seq_end = i + self.seq_len

                x_all.append(features[i:seq_end])

                # The index for this sample is the last day of the sequence
                current_index = group.index[seq_end - 1]
                index_all.append(current_index)
                
                if has_labels:
                    # The label is at the same timestamp as `current_index`
                    y_all.append(labels[seq_end - 1].squeeze())

        x_tensor = torch.tensor(np.array(x_all), dtype=torch.float32)
        
        if has_labels:
            y_tensor = torch.tensor(np.array(y_all), dtype=torch.float32)
            if y_tensor.ndim == 1:
                y_tensor = y_tensor.unsqueeze(1)
            dataset = TensorDataset(x_tensor, y_tensor)
            return DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle), None
        else:
            # For prediction, we only need features. The indices are returned separately.
            dataset = TensorDataset(x_tensor)
            # Create the correct multi-index for predictions
            pred_indices = pd.MultiIndex.from_tuples(index_all, names=["datetime", "instrument"])
            return DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle), pred_indices
        
    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.fitted = True