import tensorflow as tf
from tensorflow.keras import layers, Model
from typing import Dict, Any
import numpy as np

class MultiModalFusionModel:
    def __init__(self, config: Dict[str, Any] = None):
        if config is None:
            config = {}
        self.temporal_input_shape = config.get('temporal_input_shape', (72, 8))
        self.static_input_shape = config.get('static_input_shape', (35,))
        self.lstm_units = config.get('lstm_units', 64)
        self.dense_units = config.get('dense_units', 64)
        self.dropout_rate = config.get('dropout_rate', 0.3)
        self.model = self.build_model()

    def build_model(self) -> Model:
        temporal_input = layers.Input(shape=self.temporal_input_shape, name='temporal_input')
        temporal_out = layers.LSTM(self.lstm_units)(temporal_input)
        temporal_out = layers.Dense(32, activation='relu')(temporal_out)
        
        static_input = layers.Input(shape=self.static_input_shape, name='static_input')
        static_out = layers.Dense(64, activation='relu')(static_input)
        static_out = layers.BatchNormalization()(static_out)
        static_out = layers.Dense(32, activation='relu')(static_out)
        
        fused = layers.Concatenate()([temporal_out, static_out])
        fused = layers.Dense(64, activation='relu')(fused)
        fused = layers.Dropout(self.dropout_rate)(fused)
        fused = layers.Dense(32, activation='relu')(fused)
        output = layers.Dense(1, activation='sigmoid', name='output')(fused)
        
        model = Model(inputs=[temporal_input, static_input], outputs=output)
        
        model.compile(
            loss='binary_crossentropy',
            optimizer='adam',
            metrics=[
                tf.keras.metrics.AUC(name='AUC'),
                tf.keras.metrics.Precision(name='Precision'),
                tf.keras.metrics.Recall(name='Recall')
            ]
        )
        return model

    def train(self, X_temporal_train, X_static_train, y_train, X_temporal_val, X_static_val, y_val, epochs=100, batch_size=32):
        history = self.model.fit(
            [X_temporal_train, X_static_train], y_train,
            validation_data=([X_temporal_val, X_static_val], y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)]
        )
        return history

    def predict(self, X_temporal, X_static) -> np.ndarray:
        return self.model.predict([X_temporal, X_static])

    def save_model(self, path):
        self.model.save(path)

    def load_model(self, path):
        self.model = tf.keras.models.load_model(path)
