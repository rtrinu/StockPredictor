import yfinance as yf
import pandas as pd
import math
import numpy as np
from tensorflow.keras.models import Sequential #type: ignore
from tensorflow.keras.layers import Activation, Input, LSTM, Dense, Dropout, BatchNormalization #type: ignore
from tensorflow.keras.optimizers import Adam #type: ignore
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau #type: ignore
from src.stock_model.dualScalerPreprocessor import DualScalerPreprocessor
#from tensorflow.keras.utils import to_categorical



class LstmSignalModel():
    def __init__(self, df, stock_name, lookback=60):
        self.df = df
        self.stock_name = stock_name
        self.model = None
        self.model_file = 'models'
        self.dual_scaler = DualScalerPreprocessor()
        self.lookback = lookback
        self.x_train = []
        self.y_train = []
        self.x_test = []
        self.y_test = []
        self.scaled_data = None
        self.training_data_len = None

    @classmethod
    def create(cls, df, stock_name):
        model = cls(df, stock_name)
        model.load_and_preprocess_data()
        history = model.build_and_train_model()
        signals, metrics = model.predict_and_evaluate()
        return model, history, signals, metrics
    
    def _load_data(self):
        self.minmax_features = self.dual_scaler.minmax_features
        self.robust_features = self.dual_scaler.robust_features
        self.features = self.minmax_features + self.robust_features

        self.data = self.df[self.features]
        self.target = self.df['Signal'].values

        self.training_data_len = math.ceil(len(self.data)*0.8)
        self.dual_scaler.fit(self.data[:self.training_data_len])
        self.scaled_data = self.dual_scaler.transform(self.data)

    def _preprocess_training_data(self):
        train_data = self.scaled_data[:self.training_data_len]
        train_target = self.target[:self.training_data_len]

        x_train, y_train = [], []
        for i in range(self.lookback, len(train_data)):
            x_train.append(train_data[i-self.lookback:i])
            y_train.append(train_target[i])

        self.x_train = np.array(x_train)
        label_map = {-1: 0, 0: 1, 1: 2}
        self.y_train = np.array([label_map[y] for y in y_train])

    def _preprocess_testing_data(self):
        test_data = self.scaled_data[self.training_data_len - self.lookback:]
        test_target = self.target[self.training_data_len:]

        x_test = []
        for i in range(self.lookback, len(test_data)):
            x_test.append(test_data[i - self.lookback:i])
        
        self.x_test = np.array(x_test)
        label_map = {-1: 0, 0: 1, 1: 2}
        self.y_test = np.array([label_map[y] for y in test_target[:len(x_test)]])

    def _build_model(self):
        self.model = Sequential([
            Input(shape=(self.x_train.shape[1],self.x_train.shape[2])),
            LSTM(units=100, return_sequences=True),
            Dropout(0.2),
            LSTM(units=50, return_sequences=False),
            Dropout(0.2),
            Dense(units=3, activation='softmax')
        ])
        self.model.compile(optimizer=Adam(learning_rate=0.0001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])


    def _train(self, epochs=50, batch_size=16):
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=2,
            min_lr=0.0001
        )

        history = self.model.fit(
            self.x_train,
            self.y_train,
            batch_size=batch_size,
            epochs=epochs,
            shuffle=True,
            validation_split=0.2,
            callbacks=[ reduce_lr],
            verbose=1
        )
        
        return history
    
    def _predict(self):
        predictions = self.model.predict(self.x_test, verbose=0)
        signals = np.argmax(predictions, axis=1)

        # Map back to original labels
        reverse_map = {0: -1, 1: 0, 2: 1}
        signals = np.array([reverse_map[s] for s in signals])

        signal_map = {-1:'SELL', 0:'HOLD', 1:'BUY'}
        signal_labels = [signal_map[signal] for signal in signals]

        return signal_labels, predictions
    
    def _evaluate(self):
        loss, accuracy = self.model.evaluate(self.x_test, self.y_test, verbose=0)
        print(f"Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")
        return {'loss': loss, 'accuracy': accuracy}

    def load_and_preprocess_data(self):
        self._load_data()
        self._preprocess_training_data()
        self._preprocess_testing_data()

    def build_and_train_model(self):
        self._build_model()
        history = self._train()
        return history

    def predict_and_evaluate(self):
        signals, predictions = self._predict()
        metrics = self._evaluate()
        return signals, metrics