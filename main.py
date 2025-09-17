import fetch_data.fetch_adjusted_dataset as fetch_adjusted_dataset
import model.universal_func as universal_func
import model.cnn_model as cnn_model

# ------- 参数 -------
WINDOW = 20
EPOCHS = 100


def get_cnn_model(is_train=True, path="model_save/cnn_model.h5", symbol=None, start_date=None, end_date=None, adjust=None, window=WINDOW, epochs=EPOCHS):
    if is_train:
        fetch_adjusted_dataset.get_adjusted_stock_data_to_csv(symbol=symbol, start_date=start_date, end_date=end_date, adjust=adjust)
        X_train, y_train, X_val, y_val, X_test, y_test = fetch_adjusted_dataset.get_processed_adjusted_data(symbol=symbol, window=window)
        m = cnn_model.build_cnn(X_train.shape[1:])
        m = universal_func.train_evaluate_save_model(m, X_train, y_train, X_val, y_val, X_test, y_test, epochs=EPOCHS)
    else:
        try:
            m = universal_func.load_model(path=path)
        except:
            print("加载模型失败，尝试重新训练模型...")
            fetch_adjusted_dataset.get_adjusted_stock_data_to_csv(symbol=symbol, start_date=start_date, end_date=end_date, adjust=adjust)
            X_train, y_train, X_val, y_val, X_test, y_test = fetch_adjusted_dataset.get_processed_data(symbol=symbol, window=window)
            m = cnn_model.build_cnn(X_train.shape[1:])
            m = universal_func.train_evaluate_save_model(m, X_train, y_train, X_val, y_val, X_test, y_test, epochs=EPOCHS)
    return m


# 示例
if __name__ == "__main__":

    get_cnn_model(
        is_train=True,
        path="model_save/cnn_model.h5",
        symbol="600519",
        start_date="20100101",
        end_date="20231231",
        # 'qfq' 前复权, 'hfq' 后复权, None 不复权
        adjust="hfq",
        window=WINDOW,
        epochs=EPOCHS
    )