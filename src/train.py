import fire
import yaml
import sys
from pathlib import Path
import os

# 将项目根目录添加到 Python 路径
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

import qlib
from qlib.utils import init_instance_by_config, flatten_dict
from qlib.workflow import R
from qlib.log import get_module_logger

from src.utils import get_model_and_dataset_from_yaml


def train_model(config_path: str):
    """
    根据指定的 YAML 配置文件训练模型。

    此函数是为简化的 `predict_config.yaml` 设计的。
    1. 加载配置。
    2. 初始化 Qlib。
    3. 使用工具函数创建模型和数据集实例。
    4. 将模型拟合到数据集上。
    5. 将训练好的模型保存到配置中指定的路径。

    参数:
        config_path (str): YAML 配置文件的路径。
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    logger = get_module_logger(__name__)

    # 初始化 Qlib
    qlib.init(**config["qlib_init"])

    # 从简化配置中获取模型和数据集
    model, dataset = get_model_and_dataset_from_yaml(config_path)

    # 开始训练
    logger.info("--- 开始训练模型 ---")
    model.fit(dataset)
    logger.info("--- 模型训练完成 ---")

    # 保存训练好的模型
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "model.pth"
    
    logger.info(f"正在将模型保存到 {model_path}")
    model.save(model_path)
    logger.info("模型已保存。")


if __name__ == "__main__":
    fire.Fire(train_model)