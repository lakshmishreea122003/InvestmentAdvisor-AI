import numpy as np
import datetime as dt
from sklearn.metrics import r2_score

class Prediction:
    def __init__(self, scalar, model, prepare_data_func, n_steps=60, lookup_steps=[1, 2]):
        self.scalar = scalar  # Scaler used for normalization
        self.model = model  # The trained model
        self.prepare_data_func = prepare_data_func  # Function to prepare the dataset
        self.n_steps = n_steps  # Number of steps for LSTM input
        self.lookup_steps = lookup_steps  # Lookup steps for future predictions

    # Prediction method
    def predict(self):
        predictions = []
        for step in self.lookup_steps:
            df, last_seq, x_train, y_train = self.prepare_data_func(step)
            x_train = x_train[:, :, :len(['close'])].astype(np.float32)

            # Fit the model
            self.model.fit(x_train, y_train)

            # Predict for last sequence
            last_sequence = last_seq[-self.n_steps:]
            last_sequence = np.expand_dims(last_sequence, axis=0)
            prediction = self.model.predict(last_sequence)
            predicted_price = self.scalar.inverse_transform(prediction)[0][0]

            # Append prediction
            predictions.append(round(float(predicted_price), 2))
        
        return predictions

    # Full history prediction and adding to dataframe
    def add_predictions_to_df(self, init_df, x_train, y_train):
        copy_df = init_df.copy()

        # Predict for the full historical data
        y_predicted = self.model.predict(x_train)
        y_predicted_transformed = np.squeeze(self.scalar.inverse_transform(y_predicted))
        
        # Adding historical first and last sequences
        first_seq = self.scalar.inverse_transform(np.expand_dims(y_train[:6], axis=1))
        last_seq = self.scalar.inverse_transform(np.expand_dims(y_train[-3:], axis=1))

        # Append predicted results
        y_predicted_transformed = np.append(first_seq, y_predicted_transformed)
        y_predicted_transformed = np.append(y_predicted_transformed, last_seq)
        copy_df['predicted_close'] = y_predicted_transformed

        # Add predicted future dates to the table (as an example)
        date_now = dt.date.today()
        date_tomorrow = dt.date.today() + dt.timedelta(days=1)
        date_after_tomorrow = dt.date.today() + dt.timedelta(days=2)

        return copy_df

    # Evaluation method using R2 score
    def evaluate(self, y_true, y_pred):
        r2 = r2_score(y_true, y_pred)
        print(f'R2 Score: {r2}')
        return r2

# # Example usage
# if __name__ == "__main__":
#     scalar = ...  # Your trained MinMaxScaler
#     model = ...   # Your pre-trained model
#     prepare_data_func = ...  # Function to prepare data
#     init_df = ...  # Your initial dataframe

#     predictor = Prediction(scalar, model, prepare_data_func)
    
#     # Get predictions
#     predictions = predictor.predict()
#     print("Predictions:", ', '.join([str(pred) for pred in predictions]))
    
#     # Add predictions to dataframe
#     copy_df = predictor.add_predictions_to_df(init_df, x_train, y_train)
    
#     # Evaluate R2 score
#     y_true = y_train  # Replace with actual values
#     y_pred = model.predict(x_train)  # Replace with predicted values
#     y_pred_transformed = predictor.scalar.inverse_transform(y_pred)
#     predictor.evaluate(y_true, y_pred_transformed)
