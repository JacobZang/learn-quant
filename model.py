import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
import matplotlib.pyplot as plt
import numpy as np

def build_cnn(input_shape):

    model = models.Sequential([
        layers.Input(shape=input_shape),
        
        # 第一卷积层
        layers.Conv1D(128, kernel_size=3, padding='same'),
        layers.LeakyReLU(alpha=0.1),
        layers.MaxPooling1D(pool_size=2),
        
        # 第二卷积层
        layers.Conv1D(128, kernel_size=3, padding='same'),
        layers.LeakyReLU(alpha=0.1),
        layers.MaxPooling1D(pool_size=2),
        
        # 第三卷积层
        layers.Conv1D(64, kernel_size=3, padding='same'),
        layers.LeakyReLU(alpha=0.1),
        
        layers.GlobalAveragePooling1D(),
        
        # 全连接层
        layers.Dense(128),
        layers.ReLU(),
        layers.Dropout(0.5),
        
        # 输出层
        layers.Dense(1, activation='sigmoid')
    ])
    
    # 可调整学习率
    optimizer = optimizers.Adam(learning_rate=0.001)
    
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return model

def train_model(model, X_train, y_train, X_val, y_val, epochs=100, batch_size=32):
    callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True)]
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val, y_val), callbacks=callbacks)

    
def evaluate_model(model, X_test, y_test):
    # 测试集评估
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
    print("测试集 Loss:", test_loss)
    print("测试集 Accuracy:", test_acc)
    # 概率和标签
    probs = model.predict(X_test).ravel()
    preds = (probs > 0.5).astype(int)

    # --- 可视化对比 ---
    plt.figure(figsize=(16,6))
    plt.plot(probs, label="Predicted Probability", color="blue", alpha=0.7)
    plt.scatter(range(len(preds)), preds, label="Predicted Label", color="blue", alpha=0.6, marker="x")
    plt.scatter(range(len(y_test)), y_test, label="True Label", color="red", alpha=0.6, marker="o")
    for i in range(len(y_test)):
        plt.plot([i, i], [y_test[i], preds[i]], color="gray", linestyle="--", linewidth=0.5, alpha=0.7)
    plt.axhline(y=0.5, color="green", linestyle="--", linewidth=1.5, alpha=0.8, label="threshold 0.5")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    # 保存图片
    plt.savefig("image/test_prediction.png", dpi=300)