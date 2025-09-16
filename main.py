import fetch_data
import model

# ------- 参数 -------
WINDOW = 20
RANDOM_STATE = 42

# 获取数据
df = fetch_data.get_qfq_data(symbol="600519", start_date="20200101", end_date="20250901")

X, y = fetch_data.make_dataset(df, window=WINDOW)

X_train, y_train, X_val, y_val, X_test, y_test = fetch_data.cut_dataset(X, y)

m = model.build_cnn(X_train.shape[1:])
m.summary()

model.train_model(m, X_train, y_train, X_val, y_val, epochs=150)
model.evaluate_model(m, X_test, y_test)