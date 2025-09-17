import tensorflow as tf
import model.cnn_model as cnn_model
import matplotlib.pyplot as plt
import numpy as np

def train_evaluate_save_model(model, X_train, y_train, X_val, y_val, X_test, y_test, epochs=100):
    model.summary()
    train_model(model, X_train, y_train, X_val, y_val, epochs=epochs)
    evaluate_and_save_model(model, X_test, y_test)
    return model

def train_model(model, X_train, y_train, X_val, y_val, epochs=100, batch_size=32):
    callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True)]
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val, y_val), callbacks=callbacks)

def evaluate_and_save_model(model, X_test, y_test):
    # 测试集评估
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
    print("测试集 Loss:", test_loss)
    print("测试集 Accuracy:", test_acc)
    # 概率和标签
    probs = model.predict(X_test).ravel()
    preds = (probs > 0.5).astype(int)

    if test_acc >= 0.7:
        print("达到保存模型的准确率要求，保存模型...")
        save_model(model, path="model_save/cnn_model.h5")
    
    # 可视化对比
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
    filename = f"image/acc_{test_acc:.2f}.png"
    plt.savefig(filename, dpi=300)

def save_model(model, path="model_save/cnn_model.h5"):
    model.save(path)
    print(f"模型已保存到 {path}")

def load_model(path="model_save/cnn_model.h5"):
    model = tf.keras.models.load_model(path)
    print(f"模型已从 {path} 加载")
    return model