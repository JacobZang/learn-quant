# 使用官方 Python 运行时作为父镜像
FROM python:3.9-slim

# 在容器中设置工作目录
WORKDIR /app

# 将 requirements 文件复制到容器的 /app 目录下
COPY requirements.txt .

# 安装 requirements.txt 中指定的所有必需包
# 注意：这假设 pyqlib 要么在 requirements.txt 中列出，
# 要么已在基础镜像中预装。此示例依赖 pip。
# 为获得稳健的设置，请遵循 Qlib 官方安装指南。
RUN pip install --no-cache-dir -r requirements.txt

# 捆绑应用源码
COPY . .

# 如果需要，设置环境变量
# 对于 Qlib，您可能需要指定数据所在的位置。
# 这可以在运行时作为卷挂载。
# ENV QLIB_DATA_PATH /data/qlib_data

# 运行应用程序的命令
# 示例：运行训练脚本
# 实际命令将在运行容器时提供。
# CMD ["python", "src/train.py", "--config_path", "configs/experiment_example.yaml"]