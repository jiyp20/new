import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, SimpleRNN, GRU, Embedding, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


data = pd.read_csv('https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-min-temperatures.csv')

print("Shape:", data.shape)
print(data.head())
print(data.dtypes)

TARGET_COLUMN = "Temp"

# CASE A: Column has mixed types / strings → force numeric (your class code)
data[TARGET_COLUMN] = pd.to_numeric(data[TARGET_COLUMN], errors='coerce')
data.dropna(inplace=True)

# CASE B: Clean dataset, just drop nulls
# data.dropna(inplace=True)

# CASE C: Forward fill missing values (time series gaps)
# data.fillna(method='ffill', inplace=True)

print("After cleaning:", data.shape)

scaler = MinMaxScaler()
scaled = scaler.fit_transform(data[[TARGET_COLUMN]])


TIME_STEPS = 10

X, y = [], []

for i in range(len(scaled) - TIME_STEPS):
    X.append(scaled[i : i + TIME_STEPS])
    y.append(scaled[i + TIME_STEPS])            # next value to predict

X = np.array(X)
y = np.array(y)

print("X shape:", X.shape)


# shuffle=False is IMPORTANT for time series — order must be preserved
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

print("X_train:", X_train.shape)
print("X_test :", X_test.shape)


# CASE A: Stacked 3-layer LSTM with Dropout (your class code) ← default
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(TIME_STEPS, 1)),
    Dropout(0.2),
    LSTM(32, return_sequences=True),
    Dropout(0.2),
    LSTM(16),
    Dense(1)
])

# CASE B: Single LSTM layer (simpler, faster)
# model = Sequential([
#     LSTM(50, input_shape=(TIME_STEPS, 1)),
#     Dense(1)
# ])


model.compile(optimizer='adam', loss='mse')
model.summary()


history = model.fit(
    X_train, y_train,
    epochs=10,
    batch_size=32,
    validation_data=(X_test, y_test),   # use validation_data not split
    verbose=1                            # since we already have X_test
)


# STEP 9: EVALUATE

loss = model.evaluate(X_test, y_test)
print(f"\nTest MSE Loss: {loss:.6f}")


# STEP 10: PREDICT & INVERSE TRANSFORM

pred = model.predict(X_test)

# Convert scaled values back to original range
pred   = scaler.inverse_transform(pred)
y_test = scaler.inverse_transform(y_test)

# Extra metrics
mae  = np.mean(np.abs(y_test - pred))
rmse = np.sqrt(np.mean((y_test - pred) ** 2))
print(f"MAE  : {mae:.4f}")
print(f"RMSE : {rmse:.4f}")


# STEP 11: PLOT LOSS

plt.figure(figsize=(8, 4))
plt.plot(history.history['loss'],     label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Model Loss (MSE)')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


# ============================================================
# STEP 12: PLOT ACTUAL vs PREDICTED
# ============================================================

plt.figure(figsize=(12, 5))
plt.plot(y_test, label='Actual',    color='blue')
plt.plot(pred,   label='Predicted', color='red')
plt.title('Actual vs Predicted')
plt.xlabel('Time')
plt.ylabel(TARGET_COLUMN)
plt.legend()
plt.show()


# ============================================================
# ✅ TEXT / SENTIMENT CASE — comment out TIME SERIES above
#    and uncomment everything below if sir gives text data
# ============================================================

# import seaborn as sns
#
# # STEP 1: LOAD TEXT DATASET
# # data = pd.read_csv('/content/train.csv')   # <-- change filename
# # print(data.head())
# # print(data.dtypes)
#
# # STEP 2: SELECT TEXT & LABEL COLUMNS
# # TEXT_COLUMN  = "text"      # column with sentences/reviews
# # LABEL_COLUMN = "label"     # column with sentiment/category
# #
# # CASE B: IMDB-style
# # TEXT_COLUMN  = "review"
# # LABEL_COLUMN = "sentiment"
# #
# # X = data[TEXT_COLUMN].astype(str).values
# # y = data[LABEL_COLUMN].values
#
# # STEP 3: ENCODE LABELS
# # CASE A: Text labels (e.g. "positive"/"negative") → encode to 0/1
# # le = LabelEncoder()
# # y = le.fit_transform(y)
# # print("Classes:", le.classes_)
# #
# # CASE B: Already integers → skip above
#
# # STEP 4: TOKENIZE TEXT
# # MAX_WORDS = 10000   # vocabulary size
# # MAX_LEN   = 200     # max words per sentence
# #
# # tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token="<OOV>")
# # tokenizer.fit_on_texts(X)
# # sequences  = tokenizer.texts_to_sequences(X)
# # X_padded   = pad_sequences(sequences, maxlen=MAX_LEN, padding='post')
# # print("X_padded shape:", X_padded.shape)
#
# # STEP 5: TRAIN-TEST SPLIT
# # X_train, X_test, y_train, y_test = train_test_split(
# #     X_padded, y, test_size=0.2, random_state=42
# # )
#
# # STEP 6: BUILD MODEL
# # CASE A: Binary classification (positive/negative, spam/ham) ← default
# # model = Sequential([
# #     Embedding(MAX_WORDS, 64, input_length=MAX_LEN),
# #     LSTM(64, return_sequences=True),
# #     Dropout(0.2),
# #     LSTM(32),
# #     Dense(1, activation='sigmoid')    # binary output
# # ])
# #
# # CASE B: Multi-class (3+ categories)
# # NUM_CLASSES = len(np.unique(y))
# # model = Sequential([
# #     Embedding(MAX_WORDS, 64, input_length=MAX_LEN),
# #     LSTM(64, return_sequences=True),
# #     Dropout(0.2),
# #     LSTM(32),
# #     Dense(NUM_CLASSES, activation='softmax')
# # ])
# #
# # CASE C: Bidirectional LSTM (better accuracy)
# # model = Sequential([
# #     Embedding(MAX_WORDS, 64, input_length=MAX_LEN),
# #     Bidirectional(LSTM(64, return_sequences=True)),
# #     Dropout(0.2),
# #     Bidirectional(LSTM(32)),
# #     Dense(1, activation='sigmoid')
# # ])
#
# # STEP 7: COMPILE
# # CASE A: Binary
# # model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# #
# # CASE B: Multi-class integer labels
# # model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
#
# # model.summary()
#
# # STEP 8: TRAIN
# # history = model.fit(
# #     X_train, y_train,
# #     epochs=10,
# #     batch_size=32,
# #     validation_split=0.1,
# #     verbose=1
# # )
#
# # STEP 9: EVALUATE
# # loss, accuracy = model.evaluate(X_test, y_test)
# # print(f"Test Accuracy: {accuracy * 100:.2f}%")
#
# # STEP 10: PLOT ACCURACY & LOSS
# # plt.figure(figsize=(12, 4))
# # plt.subplot(1, 2, 1)
# # plt.plot(history.history['accuracy'],     label='Train Accuracy')
# # plt.plot(history.history['val_accuracy'], label='Val Accuracy')
# # plt.title('Model Accuracy')
# # plt.xlabel('Epochs')
# # plt.ylabel('Accuracy')
# # plt.legend()
# # plt.subplot(1, 2, 2)
# # plt.plot(history.history['loss'],     label='Train Loss')
# # plt.plot(history.history['val_loss'], label='Val Loss')
# # plt.title('Model Loss')
# # plt.xlabel('Epochs')
# # plt.ylabel('Loss')
# # plt.legend()
# # plt.tight_layout()
# # plt.show()
#
# # STEP 11: PREDICTIONS
# # CASE A: Binary
# # y_pred = (model.predict(X_test) > 0.5).astype(int).flatten()
# #
# # CASE B: Multi-class
# # y_pred = np.argmax(model.predict(X_test), axis=1)
# #
# # print(classification_report(y_test, y_pred))
# # cm = confusion_matrix(y_test, y_pred)
# # sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
# # plt.title('Confusion Matrix')
# # plt.show()
#
# # STEP 12: PREDICT ON CUSTOM TEXT (if sir asks)
# # def predict_text(text):
# #     seq    = tokenizer.texts_to_sequences([text])
# #     padded = pad_sequences(seq, maxlen=MAX_LEN, padding='post')
# #     pred   = model.predict(padded)[0][0]
# #     label  = "Positive" if pred > 0.5 else "Negative"
# #     print(f"Text: {text}")
# #     print(f"Prediction: {label} ({pred:.2f})")
# #
# # predict_text("This was an amazing experience!")
# # predict_text("Terrible, would not recommend.")