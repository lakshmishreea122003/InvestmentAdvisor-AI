# import streamlit as st
from components.data_fetching import Fetch
from components.data_preprocess import Preprocessor
from components.model_training import Model
from components.prediction import Prediction


# st.title("Stock Price Predictor")

# fetch data
fetch = Fetch("GOOGL")
data = fetch.fetch_data()
print("data fetched")

# Preprocess
preprocessor = Preprocessor(data, 1)
df, last_sequence, X, y = preprocessor.prepare_lstm_sequences()
print(X)
print(type(X))
print("\n")
print(y)
print(type(y))

print("data preprocessed")

# model training
# model_training = Model(X,y)
# model = model_training.train_model()
# print("model trained")

# Predict
# predictor = Prediction()
