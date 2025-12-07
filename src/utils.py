import yaml
from pathlib import Path
import importlib
from typing import Tuple, Dict, Any

from qlib.data.dataset import Dataset, DatasetH
from qlib.contrib.model.base import Model

def get_model_and_dataset(config: Dict[str, Any]) -> Tuple[Model, Dataset]:
    """
    从配置字典中实例化模型和数据集
    该函数动态加载配置中指定的模型和数据集类
    并返回它们的实例

    参数:
        config (Dict[str, Any]): 一个包含 'model' 和 'dataset' 键的字典

    返回:
        Tuple[Model, Dataset]: 一个包含实例化的模型和数据集的元组
    """
    # 实例化数据集
    dataset_config = config["dataset"]
    try:
        module = importlib.import_module(dataset_config["module_path"])
        class_ = getattr(module, dataset_config["class"])
        dataset = class_(**dataset_config["kwargs"])
    except (ImportError, AttributeError) as e:
        raise type(e)(f"加载数据集失败: {e}")

    # 实例化模型
    model_config = config["task"]["model"]
    try:
        module = importlib.import_module(model_config["module_path"])
        class_ = getattr(module, model_config["class"])
        model = class_(**model_config["kwargs"])
    except (ImportError, AttributeError) as e:
        raise type(e)(f"加载模型失败: {e}")

    return model, dataset

def get_model_and_dataset_from_yaml(config_path: str) -> Tuple[Model, Dataset]:
    """
    从 yaml 文件中获取模型和数据集

    参数:
        config_path (str): yaml 文件的路径

    返回:
        Tuple[Model, Dataset]: 一个包含实例化的模型和数据集的元组
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # 对于 qrun 配置
    if "task" in config:
        return get_model_and_dataset(config)
    
    # 对于简化版配置
    model_class = _get_callable(config["model"]["module_path"], config["model"]["class"])
    model = model_class(**config["model"]["kwargs"])
    
    # 在简化版配置中，数据集从处理器创建
    from qlib.data import D
    handler = D.get_handler(config["data_handler_config"])
    dataset = D.get_dataset(
        handler=handler,
        segment={
            "train": (config["data_handler_config"]["fit_start_time"], config["data_handler_config"]["fit_end_time"]),
            "valid": (config["data_handler_config"]["fit_end_time"], config["data_handler_config"]["end_time"]),
            "test": (config["data_handler_config"]["end_time"], "2099-12-31") # 未来的占位符
        }
    )
    return model, dataset

def _get_callable(module_path: str, class_name: str):
    """从模块中动态加载一个类"""
    try:
        module = importlib.import_module(module_path)
        return getattr(module, class_name)
    except (ImportError, AttributeError) as e:
        raise type(e)(f"从 {module_path} 加载类 {class_name} 失败: {e}")