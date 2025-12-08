import os
import time
import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from typing import Dict, Union
import matplotlib.pyplot as plt
from qlib.contrib.evaluate import backtest_daily, risk_analysis
from qlib.contrib.strategy import TopkDropoutStrategy
from qlib.workflow import R
from qlib.data import D
from qlib.workflow.recorder import Recorder
from qlib.data.dataset import DatasetH
from qlib.utils import init_instance_by_config

def _visualize_results(report_df: pd.DataFrame, analysis: dict):
    """
    一个辅助函数，用于根据回测结果的 report_df 和 analysis 进行可视化
    """

    if "return" not in report_df.columns or "value" not in report_df.columns:
        print("report_df 中缺少 return 或 value 列，无法绘图")
        return

    dates = report_df.index
    daily_return = report_df["return"].fillna(0)
    account_value = report_df["value"].fillna(method="ffill")

    plt.figure(figsize=(14, 10))

    # === 子图1：每日收益率 ===
    ax1 = plt.subplot(2, 1, 1)
    ax1.plot(dates, daily_return, color="blue", linewidth=1)
    ax1.axhline(0, color="black", linestyle="--", alpha=0.5)
    ax1.set_title("Daily Return", fontsize=14)
    ax1.set_ylabel("Return")
    ax1.grid(True, linestyle="--", alpha=0.4)

    # === 子图2：资金曲线 ===
    ax2 = plt.subplot(2, 1, 2)
    ax2.plot(dates, account_value, color="red", linewidth=1.5)
    ax2.set_title("Equity Curve (Account Value)", fontsize=14)
    ax2.set_xlabel("Date")
    ax2.set_ylabel("Account Value")
    ax2.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()

    save_dir = "backtest_result"
    os.makedirs(save_dir, exist_ok=True)

    # 唯一文件名：时间戳 + 随机UUID
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    file_name = f"equity_curve_{timestamp}.png"
    save_path = os.path.join(save_dir, file_name)
    plt.savefig(save_path, dpi=300)

def _save_positions_to_file(positions: dict, output_dir="positions_log"):
    """
    将每天的持仓写入 txt 文件
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for date, pos_obj in positions.items():

        pos = pos_obj.position

        # 构造文件名
        date_str = str(date.date())
        file_path = os.path.join(output_dir, f"positions_{date_str}.txt")

        with open(file_path, "w", encoding="utf-8") as f:
            f.write(f"📅 日期: {date_str}\n\n")
            f.write(f"💰 现金: {float(pos.get('cash', 0)):.2f}\n")
            f.write(f"💼 账户总资产: {float(pos.get('now_account_value', 0)):.2f}\n\n")
            f.write("📊 持仓股票:\n")

            for k, v in pos.items():
                if k in ["cash", "now_account_value"]:
                    continue

                amount = float(v["amount"])
                price = float(v["price"])
                weight = float(v["weight"])
                days = int(v["count_day"])

                f.write(
                    f"{k} | 数量: {amount:.2f} | 价格: {price:.2f} "
                    f"| 权重: {weight:.4f} | 持仓天数: {days}\n"
                )


class CompleteEvaluator(Recorder):
    """
    一个完整、统一的评估器
    它在 qrun 工作流中被调用，通过接收配置来自己实例化所需的数据集
    然后一次性完成指标计算和投资组合回测
    """

    def __init__(self, dataset_config, strategy_config, backtest_config, **kwargs):
        experiment_id = kwargs.get("experiment_id")
        name = kwargs.get("name")
        self.recorder = kwargs.get("recorder")
        super().__init__(experiment_id, name)

        # 保存所有需要的配置
        self.dataset_config = dataset_config
        self.strategy_config = strategy_config
        self.backtest_config = backtest_config

    def generate(self, **kwargs):
        """qrun 工作流主入口"""
        # 加载预测结果
        try:
            pred_df = self.recorder.load_object("pred.pkl")
        except (FileNotFoundError, AttributeError) as e:
            raise ValueError(f"无法加载 pred.pkl：{e}")

        # 实例化数据集
        dataset: DatasetH = init_instance_by_config(self.dataset_config)
        if dataset is None:
            raise ValueError("数据集实例化失败，请检查 config")

        # 计算指标
        self.calculate_metrics(pred_df, dataset)

        # 执行回测
        self.run_backtest(pred_df)

    def calculate_metrics(self, pred_df: pd.DataFrame, dataset: DatasetH):
        """计算 IC、Rank IC 等指标"""
        df_test = dataset.prepare("test", col_set=["label"], data_key=dataset.handler.DK_R)

        label_cols = [
            c for c in df_test.columns
            if (isinstance(c, str) and c.upper().startswith("LABEL"))
            or (isinstance(c, tuple) and c[0].upper() == "LABEL")
        ]

        if not label_cols:
            print("未找到 label 列，跳过指标计算")
            return

        label_df = df_test[label_cols].copy()
        label_df.columns = ["label"]

        merged_df = pd.concat([pred_df, label_df], axis=1, join="inner").dropna()

        if merged_df.empty:
            print("预测与标签无法对齐，指标设为 0")
            metrics = {'IC': 0.0, 'Rank_IC': 0.0, 'ICIR': 0.0, 'Rank_ICIR': 0.0}
        else:
            df_cal = pd.DataFrame({
                'pred': merged_df['score'],
                'label': merged_df['label']
            })

            ic_by_day = df_cal.groupby(level='datetime').apply(
                lambda x: x['pred'].corr(x['label'])
            )
            rank_ic_by_day = df_cal.groupby(level='datetime').apply(
                lambda x: spearmanr(x['pred'], x['label'])[0]
            )

            metrics = {
                'IC': ic_by_day.mean(),
                'Rank_IC': rank_ic_by_day.mean(),
                'ICIR': ic_by_day.mean() / (ic_by_day.std() + 1e-9),
                'Rank_ICIR': rank_ic_by_day.mean() / (rank_ic_by_day.std() + 1e-9),
            }

        print("已计算指标:", {k: f"{v:.4f}" for k, v in metrics.items()})
        self.recorder.save_objects(**metrics)

    def run_backtest(self, pred_df: pd.DataFrame):
      
        strategy = TopkDropoutStrategy(
            signal=pred_df,
            topk=50,
            n_drop=5
        )

        # 获取交易区间
        start_time = pred_df.index.get_level_values("datetime").min()
        end_time = pred_df.index.get_level_values("datetime").max()

        print(f"回测区间: {start_time} ~ {end_time}")

        # 获取交易日历
        calendar = D.calendar(freq="day")

        if calendar is None or len(calendar) == 0:
            print("无法加载交易日历")
            return

        account = self.backtest_config.get("account", 100000000)
        benchmark = self.backtest_config.get("benchmark", "SH000300")
        try:
            report_df, positions = backtest_daily(
                start_time=start_time,
                end_time=end_time,
                strategy=strategy,
                account=account,
                benchmark=benchmark,
                exchange_kwargs={
                    "open_cost": 0.0005,
                    "close_cost": 0.0015,
                    "min_cost": 5
                },
            )
        except Exception as e:
            print(f"回测执行失败: {e}")
            return
        
        if report_df is None:
            print("回测未生成报告，无法进行分析和可视化。")
            return
        
        analysis = risk_analysis(report_df)

        self.recorder.save_objects(**{"portfolio_analysis": analysis})
                
        # 自动调用可视化函数
        _visualize_results(report_df, analysis)
        _save_positions_to_file(positions)


    def get_callable(self, module_path, class_name):
        """动态加载类"""
        import importlib
        module = importlib.import_module(module_path)
        return getattr(module, class_name)
