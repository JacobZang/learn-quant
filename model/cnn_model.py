from tensorflow.keras import layers, models, optimizers
import model.universal_func as universal_func

def build_train_save_cnn(X_train, y_train, X_val, y_val, X_test, y_test, epochs):
    m = build_cnn(X_train.shape[1:])
    m = universal_func.train_evaluate_save_model(m, X_train, y_train, X_val, y_val, X_test, y_test, epochs=epochs)
    return m

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