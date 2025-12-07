import pytest
import torch
import sys
from pathlib import Path

# 将项目根目录添加到 Python 路径
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from src.models.lstm_model import LSTMModel
from src.models.transformer_model import TransformerModel

# 用于测试的通用参数
D_FEAT = 6
SEQ_LEN = 20
PRED_LEN = 1
BATCH_SIZE = 32

@pytest.fixture
def dummy_input_tensor():
    """为测试提供一个标准的虚拟输入张量。"""
    return torch.randn(BATCH_SIZE, SEQ_LEN, D_FEAT)

def test_lstm_model_forward_pass(dummy_input_tensor):
    """
    测试 LSTMModel 的前向传播。
    - 检查模型是否可以被实例化。
    - 检查输出形状是否正确。
    """
    model = LSTMModel(
        d_feat=D_FEAT,
        hidden_size=64,
        num_layers=2,
        dropout=0.1,
        pred_len=PRED_LEN,
        seq_len=SEQ_LEN
    )
    
    # 模型包装器在 `model` 属性中包含 pytorch nn.Module
    pytorch_model = model.model
    pytorch_model.eval()

    with torch.no_grad():
        output = pytorch_model(dummy_input_tensor)

    # 如果 PRED_LEN 为 1，输出形状可以是 (BATCH_SIZE)，或者 (BATCH_SIZE, PRED_LEN)
    expected_shape_squeezed = (BATCH_SIZE,)
    expected_shape_unsqueezed = (BATCH_SIZE, PRED_LEN)
    
    assert output.shape == expected_shape_squeezed or output.shape == expected_shape_unsqueezed, \
        f"LSTM 输出形状为 {output.shape}，但期望为 {expected_shape_squeezed} 或 {expected_shape_unsqueezed}"

def test_transformer_model_forward_pass(dummy_input_tensor):
    """
    测试 TransformerModel 的前向传播。
    - 检查模型是否可以被实例化。
    - 检查输出形状是否正确。
    """
    model = TransformerModel(
        d_feat=D_FEAT,
        d_model=64,
        nhead=4,
        num_layers=2,
        dropout=0.1,
        pred_len=PRED_LEN,
        seq_len=SEQ_LEN
    )

    # 模型包装器在 `model` 属性中包含 pytorch nn.Module
    pytorch_model = model.model
    pytorch_model.eval()

    with torch.no_grad():
        output = pytorch_model(dummy_input_tensor)

    # 如果 PRED_LEN 为 1，输出形状可以是 (BATCH_SIZE)，或者 (BATCH_SIZE, PRED_LEN)
    expected_shape_squeezed = (BATCH_SIZE,)
    expected_shape_unsqueezed = (BATCH_SIZE, PRED_LEN)

    assert output.shape == expected_shape_squeezed or output.shape == expected_shape_unsqueezed, \
        f"Transformer 输出形状为 {output.shape}，但期望为 {expected_shape_squeezed} 或 {expected_shape_unsqueezed}"

if __name__ == "__main__":
    # 直接运行此测试文件以进行调试
    pytest.main([__file__])