import fire
import yaml
import sys
from pathlib import Path

# 将项目根目录添加到 Python 路径
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

import qlib
from qlib.log import get_module_logger
from src.utils import get_model_and_dataset_from_yaml

def predict(config_path: str, model_path: str):
    """
    使用训练好的模型生成预测
    该函数加载一个训练好的模型和数据集配置
    为指定的时间段生成预测分数

    参数:
        config_path (str):
            YAML 配置文件的路径 (例如, 'configs/predict_config.yaml')
            此文件提供 qlib 初始化配置和数据处理器设置
        model_path (str):
            已保存模型文件的路径 (例如, 'model_checkpoints/model.pth')
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    logger = get_module_logger(__name__)

    # 初始化 Qlib
    qlib.init(**config["qlib_init"])

    # 获取模型和数据集，这里我们只需要数据集定义
    # 模型将从指定路径加载
    model, dataset = get_model_and_dataset_from_yaml(config_path)

    # 加载训练好的模型
    logger.info(f"正在从 {model_path} 加载模型...")
    model.load(model_path)
    logger.info("模型已加载...")

    # 从配置中定义预测分段
    pred_start = config["prediction_dataset"]["start_time"]
    pred_end = config["prediction_dataset"]["end_time"]
    
    # 进行预测
    logger.info(f"--- 正在为分段 [{pred_start}, {pred_end}] 进行预测 ---")
    predictions = model.predict(dataset, segment=("test"))
    logger.info("--- 预测完成 ---")

    # 将预测保存到 CSV
    output_path = Path(config["prediction_path"])
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"正在将预测保存到 {output_path}...")
    predictions.to_csv(output_path)
    logger.info(f"预测已保存，预览:\n{predictions.head()}")

if __name__ == "__main__":
    fire.Fire(predict)