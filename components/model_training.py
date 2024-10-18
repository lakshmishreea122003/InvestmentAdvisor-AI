from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM, Dropout

class Model():
    def __init__(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train
        self.batch_size = 1
        self.epochs = 1  # corrected typo from `epoch` to `epochs`
        self.n_steps = 2
    
    def build_model(self):
        model = Sequential()
        model.add(LSTM(60, return_sequences=True, input_shape=(self.n_steps, len(['close']))))
        model.add(Dropout(0.3))
        model.add(LSTM(120, return_sequences=False))
        model.add(Dropout(0.3))
        model.add(Dense(20))
        model.add(Dense(1))
        model.compile(loss='mean_squared_error', optimizer='adam')
        return model

    def train_model(self):
        model = self.build_model()
        model.fit(self.x_train, self.y_train, batch_size=self.batch_size, epochs=self.epochs, verbose=1)
        model.save('D:/Proti-on Energy/aws_practice/components/stock_prediction_model.h5')  
        print("Model saved successfully!")

    def load_trained_model(self):
        loaded_model = load_model('stock_prediction_model.h5')
        print("Model loaded successfully!")
        return loaded_model

# Example usage:
# x_train and y_train should be your prepared training data
# model_instance = Model(x_train, y_train)
# model_instance.train_model()

# To load the model later:
# loaded_model = model_instance.load_trained_model()
