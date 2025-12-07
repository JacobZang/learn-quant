# Qlib 预测模型模板

本项目提供了一个完整、可运行的 Qlib 预测模型模板。它包括数据准备、模型实现（LSTM、Transformer）、训练、预测、评估和配置。项目结构设计为模块化且易于扩展。

## 1. 项目结构

```
.
├── .gitignore
├── Dockerfile
├── README.md
├── requirements.txt
├── configs
│   ├── experiment_example.yaml
│   └── predict_config.yaml
├── src
│   ├── data
│   │   └── data_utils.py
│   ├── eval
│   │   └── evaluator.py
│   ├── models
│   │   ├── forecast_base.py
│   │   ├── lstm_model.py
│   │   └── transformer_model.py
│   ├── predict.py
│   └── train.py
└── tests
    └── test_model_forward.py
```

## 2. 依赖与安装

### 环境设置

建议使用虚拟环境。

**使用 conda:**
```bash
conda create -n qlib-env python=3.10
conda activate qlib-env
```


### 安装 Qlib

最关键的依赖是 `pyqlib`。请遵循 [Qlib 官方安装指南](https://github.com/microsoft/qlib#installation) 以获取最稳定和最新的安装方法。通常的安装方式如下：

```bash
pip install pyqlib
```

### C. 安装其他依赖

从 `requirements.txt` 文件安装其余的依赖项。

```bash
pip install -r requirements.txt
```

## 3. 数据准备

此项目假设您已拥有一个 Qlib 兼容的数据集。如果尚未准备，请按照 [Qlib 官方数据准备指南](https://qlib.readthedocs.io/en/latest/component/data.html#data-preparation) 进行操作。

**示例：下载中国 A 股 CSI300 数据**

```bash
# 此命令将下载数据到指定目录
python -m qlib.cli.data qlib_data --target_dir ./qlib_data --region cn
```

数据下载完成后，您的 Qlib 数据将存储在 `./qlib_data/cn_data`。
**请务必更新 `.yaml` 配置文件中的 `provider_uri` 为此路径。**

## 4. 如何运行

有两种主要方式来运行实验：使用 `qrun` 或直接执行 Python 脚本。

### 使用 `qrun` 运行

`qrun` 工具能够基于单个配置文件处理整个工作流（数据、模型、训练、评估）。

1.  **配置：** 打开 `configs/experiment_example.yaml` 并验证 `qlib_init` 部分的 `provider_uri` 指向您的 Qlib 数据目录。
结构：
    task
    ├── model        - 用哪个模型、模型超参
    ├── dataset      - 用什么数据（handler + segment）
    └── record       - 怎么记录结果


2.  **运行实验：**
    ```bash
    qrun -c configs/experiment_example.yaml
    ```

    运行后，结果、模型检查点和预测记录将保存在 `mlruns` 目录中（这是 `mlflow` 创建的目录，`qrun` 默认使用它）。

### 手动运行 Python 脚本

这种方法为调试和自定义工作流提供了更大的灵活性。

1.  **配置：** 与上述相同，确保 `configs/predict_config.yaml` 具有正确的数据路径。

2.  **运行训练：**
    ```bash
    python src/train.py --config_path configs/predict_config.yaml
    ```
    此脚本将训练模型并将检查点保存到配置中指定的路径 (`output_dir`)。

3.  **运行预测：**
    ```bash
    python src/predict.py --config_path configs/predict_config.yaml --model_path 'model_checkpoints/model.pth' # 替换为您的实际模型路径
    ```
    此脚本加载训练好的模型并生成预测，将其保存到 `prediction_output/preds.csv`。

## 5. 运行测试

为确保模型结构正确，您可以运行单元测试：

```bash
pytest tests/
```