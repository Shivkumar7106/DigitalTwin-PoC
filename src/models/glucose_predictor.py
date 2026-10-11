import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from typing import List, Tuple, Dict, Any

class AttentionLayer(layers.Layer):
    """Custom attention mechanism that learns to weight different timesteps."""
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name='attention_weight', shape=(input_shape[-1], 1),
                                 initializer='random_normal', trainable=True)
        self.b = self.add_weight(name='attention_bias', shape=(input_shape[1], 1),
                                 initializer='zeros', trainable=True)
        super(AttentionLayer, self).build(input_shape)

    def call(self, x):
        e = tf.keras.backend.tanh(tf.keras.backend.dot(x, self.W) + self.b)
        a = tf.keras.backend.softmax(e, axis=1)
        output = x * a
        context = tf.keras.backend.sum(output, axis=1)
        return context, a

class GlucosePredictor:
    """Predicts glucose spikes using BiLSTM with Attention."""
    def __init__(self, config: Dict[str, Any] = None):
        if config is None:
            config = {}
        self.input_shape = config.get('input_shape', (72, 8))
        self.prediction_horizon = config.get('prediction_horizon', 24)
        self.lstm_units = config.get('lstm_units', 128)
        self.lstm_units_2 = config.get('lstm_units_2', 64)
        self.dense_units = config.get('dense_units', 32)
        self.dropout_rate = config.get('dropout_rate', 0.3)
        self.learning_rate = config.get('learning_rate', 0.001)
        self.model = self.build_model()

    def build_model(self) -> Model:
        inputs = layers.Input(shape=self.input_shape)
        x = layers.Bidirectional(layers.LSTM(self.lstm_units, return_sequences=True))(inputs)
        context, attn_weights = AttentionLayer()(x)
        x_reshaped = layers.Reshape((1, self.lstm_units * 2))(context)
        x_lstm = layers.LSTM(self.lstm_units_2)(x_reshaped)
        x_dense = layers.Dense(self.dense_units, activation='relu')(x_lstm)
        x_drop = layers.Dropout(self.dropout_rate)(x_dense)
        outputs = layers.Dense(self.prediction_horizon, activation='linear')(x_drop)
        
        model = Model(inputs=inputs, outputs=outputs)
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        model.compile(loss='mse', optimizer=optimizer, metrics=['mae'])
        return model

    def train(self, X_train, y_train, X_val, y_val, epochs=100, batch_size=32):
        callbacks = [
            EarlyStopping(patience=10, restore_best_weights=True),
            ReduceLROnPlateau(factor=0.5, patience=5),
            ModelCheckpoint('best_glucose_model.keras', save_best_only=True)
        ]
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks
        )
        return history

    def predict(self, X) -> np.ndarray:
        return self.model.predict(X)

    def predict_with_confidence(self, X, n_samples=50) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        predictions = []
        for _ in range(n_samples):
            pred = self.model(X, training=True)
            predictions.append(pred.numpy())
        predictions = np.array(predictions)
        
        mean_pred = np.mean(predictions, axis=0)
        std_pred = np.std(predictions, axis=0)
        
        lower_bound = mean_pred - 1.96 * std_pred
        upper_bound = mean_pred + 1.96 * std_pred
        return mean_pred, lower_bound, upper_bound

    def detect_spikes(self, predictions, threshold=180) -> List[Dict[str, Any]]:
        spikes = []
        for i, pred_seq in enumerate(predictions):
            for t, val in enumerate(pred_seq):
                if val > threshold:
                    spikes.append({
                        'sample_idx': i,
                        'timestep': t,
                        'predicted_value': float(val),
                        'time_to_spike_minutes': (t + 1) * 5
                    })
        return spikes

    def save_model(self, path):
        self.model.save(path)

    def load_model(self, path):
        self.model = tf.keras.models.load_model(path, custom_objects={'AttentionLayer': AttentionLayer})
