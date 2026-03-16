import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

data = pd.read_csv("/content/train.csv")   

# from sklearn.datasets import fetch_california_housing
# dataset = fetch_california_housing()
# data = pd.DataFrame(dataset.data, columns=dataset.feature_names)
# data["Price"] = dataset.target

print("Shape:", data.shape)
print(data.head())
print(data.dtypes)


print(data.isnull().sum())

data = data.dropna()


# data = pd.get_dummies(data, drop_first=True)

TARGET_COLUMN = "Price"

X = data.drop(TARGET_COLUMN, axis=1)
y = data[TARGET_COLUMN]

print("Target range:", y.min(), "to", y.max())


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)    
# KEY: Output layer has 1 neuron, NO activation (linear output)

model = keras.Sequential([
    layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    layers.Dense(64,  activation='relu'),
    layers.Dense(32,  activation='relu'),
    layers.Dense(1)          # <-- No activation for regression!
])

model.summary()


model.compile(
    optimizer='adam',
    loss='mse',              # mean squared error
    metrics=['mae']          # mean absolute error to track
)

# CASE B: Use MAE loss (more robust to outliers)
# model.compile(
#     optimizer='adam',
#     loss='mae',
#     metrics=['mae']
# )


history = model.fit(
    X_train, y_train,
    epochs=50,               
    batch_size=32,
    validation_split=0.1,
    verbose=1
)


loss, mae = model.evaluate(X_test, y_test)
print(f"\nTest MAE: {mae:.4f}")
print(f"Test Loss (MSE): {loss:.4f}")


# PLOT LOSS CURVE

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['mae'],     label='Train MAE')
plt.plot(history.history['val_mae'], label='Val MAE')
plt.title('Model MAE')
plt.xlabel('Epochs')
plt.ylabel('MAE')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'],     label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Model Loss (MSE)')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()


# PREDICTIONS & METRICS

y_pred = model.predict(X_test).flatten()   # flatten to 1D

# If you scaled y, inverse transform here:
# y_pred = y_scaler.inverse_transform(y_pred.reshape(-1,1)).flatten()
# y_test = y_scaler.inverse_transform(y_test.values.reshape(-1,1)).flatten()

mae  = mean_absolute_error(y_test, y_pred)
mse  = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2   = r2_score(y_test, y_pred)

print(f"\nMAE  (Mean Absolute Error) : {mae:.4f}")
print(f"MSE  (Mean Squared Error)  : {mse:.4f}")
print(f"RMSE (Root MSE)            : {rmse:.4f}")
print(f"R2 Score                   : {r2:.4f}")


#ACTUAL vs PREDICTED PLOT

plt.figure(figsize=(8, 5))
plt.scatter(y_test, y_pred, alpha=0.5, color='blue')
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()], 'r--', lw=2)   # perfect prediction line
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs Predicted')
plt.tight_layout()
plt.show()