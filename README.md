## Project Overview:
This project is a time series forecasting endeavor that utilizes LSTM (Long Short-Term Memory) networks to predict the S&P 500 index prices. The primary objective is to develop a robust predictive model capable of capturing complex patterns in historical stock prices. By employing deep learning techniques, the project aims to enhance the accuracy of short-term predictions, providing valuable insights for potential investors and financial analysts.

## Technologies Used:
- **Programming Language:** Python
- **Deep Learning Libraries:** TensorFlow, Keras
- **Data Collection:** yfinance library for fetching historical stock price data
- **Data Preprocessing:** Pandas, NumPy, MinMaxScaler from scikit-learn
- **Model Development:** Sequential model with LSTM layers, Dense layers
- **Evaluation Metrics:** Mean Squared Error (MSE), Root Mean Squared Error (RMSE), Mean Absolute Error (MAE)
- **Visualization:** Matplotlib for plotting actual vs. predicted price graphs

## Results:
The model achieved promising results with the following average evaluation metrics across k-fold cross-validation:

- **Average Mean Squared Error (MSE) across folds:** 703.0468321908629
- **Average Root Mean Squared Error (RMSE) across folds:** 25.950545351615812
- **Average Mean Absolute Error (MAE) across folds:** 16.675288167602048

These metrics indicate that the model's predictions have an average deviation of around $25.95 from the actual S&P 500 index prices, within the range of 0 to 4500.

## Challenges and Solutions:
**Challenges:**
One of the primary challenges faced was handling the high volatility and non-linear patterns in stock prices. Additionally, finding an optimal sequence length for LSTM input data was crucial for model performance.

**Solutions:**
- Experimentation with different sequence lengths
- Increasing LSTM units
- Implementing k-fold cross-validation
- Fine-tuning hyperparameters

## Model Progression:

**Original Model:**
- **Mean Squared Error (MSE):** 3201.80831333263
- **Root Mean Squared Error (RMSE):** 56.58452362026767
- **Mean Absolute Error (MAE):** 46.36430483054091

**Added 50 LSTM Units for a total of 100:**
- **Mean Squared Error (MSE):** 1600.2962861860285
- **Root Mean Squared Error (RMSE):** 40.00370340588517
- **Mean Absolute Error (MAE):** 26.560644222211188

**Added Dropout 0.2:**
- **Mean Squared Error (MSE):** 2327.575799358953
- **Root Mean Squared Error (RMSE):** 48.24495620641554
- **Mean Absolute Error (MAE):** 32.18568748763839

**Removed Dropout and changed learning rate to 0.001:**
- **Mean Squared Error (MSE):** 1400.7363460330703
- **Root Mean Squared Error (RMSE):** 37.4264124119995
- **Mean Absolute Error (MAE):** 24.055243107552567

**Double Epochs and Batch Sizes:**
- **Mean Squared Error (MSE):** 2308.3255292215545
- **Root Mean Squared Error (RMSE):** 48.04503646810516
- **Mean Absolute Error (MAE):** 37.71658297602576

**Early Stopping:**
- **Mean Squared Error (MSE):** 1391.765760813871
- **Root Mean Squared Error (RMSE):** 37.30637694568947
- **Mean Absolute Error (MAE):** 23.976936878390596

**K Cross Validation:**
- **Average Mean Squared Error (MSE) across folds:** 703.0468321908629
- **Average Root Mean Squared Error (RMSE) across folds:** 25.950545351615812
- **Average Mean Absolute Error (MAE) across folds:** 16.675288167602048
