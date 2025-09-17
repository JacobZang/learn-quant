import fetch_data
import model

# ------- 参数 -------
WINDOW = 20
EPOCHS = 100


def get_cnn_model(is_train=True, path="model/cnn_model.h5", symbol=None, start_date=None, end_date=None, adjust=None, window=WINDOW, epochs=EPOCHS):
    if is_train:
        X_train, y_train, X_val, y_val, X_test, y_test = fetch_data.get_processed_data(symbol=symbol, start_date=start_date, end_date=end_date, adjust=adjust, window=window)
        m = model.build_train_evaluate_save_model(X_train, y_train, X_val, y_val, X_test, y_test, epochs=EPOCHS)
    else:
        m = model.load_model(path=path)
    return m


# 示例
if __name__ == "__main__":

    get_cnn_model(
        is_train=True,
        symbol="600519",
        start_date="20100101",
        end_date="20231231",
        adjust="hfq",  # 'qfq' 前复权, 'hfq' 后复权, None 不复权
        window=WINDOW,
        epochs=EPOCHS
    )