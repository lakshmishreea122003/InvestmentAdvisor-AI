import numpy as np
from sklearn.preprocessing import MinMaxScaler
from collections import deque

class Preprocessor():
    def __init__(self, data, pd):
        self.data = data
        self.scaler = MinMaxScaler() 
        self.processed_data = None
        self.n_steps = 2
        self.prediction_days = pd

    def clean_and_scale_data(self):
        """
        Clean the data by dropping unnecessary columns and scaling the 'close' column.
        """
        cleaned_data = self.data.drop(['open', 'high', 'low', 'adjclose', 'ticker', 'volume'], axis=1)
        cleaned_data['date'] = cleaned_data.index
        cleaned_data['close'] = self.scaler.fit_transform(np.expand_dims(cleaned_data['close'].values, axis=1))
        self.processed_data = cleaned_data

    def prepare_lstm_sequences(self):
        """
        Prepare the data for LSTM model training by generating sequences and targets.
        Args:
        - n_steps: Number of time steps to look back for creating sequences.
        - prediction_days: Number of days into the future to predict.
        Returns:
        - df: DataFrame with future values.
        - last_sequence: The last sequence for prediction.
        - X: Feature sequences for training.
        - y: Target values for training.
        """
        self.clean_and_scale_data()
        if self.processed_data is None:
            raise ValueError("Data has not been processed. Call 'clean_and_scale_data' first.")
        df = self.processed_data.copy()
        df['future'] = df['close'].shift(-self.prediction_days)
        last_sequence = np.array(df[['close']].tail(self.prediction_days))
        df.dropna(inplace=True)
        sequence_data = []
        sequences = deque(maxlen=self.n_steps)
        for entry, target in zip(df[['close', 'date']].values, df['future'].values):
            sequences.append(entry)
            if len(sequences) == self.n_steps:
                sequence_data.append([np.array(sequences), target])
        last_sequence = list([s[:1] for s in sequences]) + list(last_sequence)
        last_sequence = np.array(last_sequence).astype(np.float32)
        X, y = [], []
        for seq, target in sequence_data:
            X.append(seq)
            y.append(target)

        # Convert X and y to numpy arrays for model input
        X = np.array(X)
        y = np.array(y)

        return df, last_sequence, X, y
