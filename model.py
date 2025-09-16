import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

def build_cnn(input_shape):

    model = models.Sequential([
        layers.Input(shape=(30, 10)),  # 指定输入形状
        layers.Conv1D(64, kernel_size=3, activation='relu'),
        layers.MaxPooling1D(pool_size=2),
        layers.Conv1D(128, kernel_size=3, activation='relu'),
        layers.GlobalAveragePooling1D(),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def train_model(model, X_train, y_train, X_val, y_val, epochs=100, batch_size=32):
    callbacks = [tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)]

    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val, y_val), callbacks=callbacks)
    
def evaluate_model(model, X_test, y_test):
    # 测试集评估
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
    print("测试集 Loss:", test_loss)
    print("测试集 Accuracy:", test_acc)

    probs = model.predict(X_test).ravel()
    preds = (probs > 0.5).astype(int)

    # 绘图
    plt.figure(figsize=(14,5))
    plt.plot(range(len(y_test)), y_test, label="real label", marker='o', linestyle='-', alpha=0.7)
    plt.plot(range(len(preds)), preds, label="predicted label", marker='s', linestyle=':', alpha=0.7)

    plt.title("true labels vs predicted labels")
    plt.xlabel("sample serial number")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    # 保存图片
    plt.savefig("image/test_prediction.png", dpi=300)