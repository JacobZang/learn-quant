import fetch_data
import model

# ------- 参数 -------
WINDOW = 20

# 获取数据
df = fetch_data.get_qfq_data(symbol="600519", start_date="20200101", end_date="20250901")
X, y = fetch_data.make_dataset(df, window=WINDOW)
X_train, y_train, X_val, y_val, X_test, y_test = fetch_data.cut_dataset(X, y)

m = model.load_model(path="model/cnn_model.h5")
if m is None:
    model.train_evaluate_save_model(X_train, y_train, X_val, y_val, X_test, y_test)