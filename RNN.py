import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, LSTM, GRU, Dense, Embedding, Bidirectional, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


df = pd.read_csv("/content/all_stocks_5yr.csv")   # <-- change if needed

print("Shape:", df.shape)
print(df.head())
print(df.dtypes)


print(df.isnull().sum())

# CASE A: Drop missing rows (default)
df = df.dropna()

# CASE B: Forward fill (better for time series gaps)
# df.fillna(method='ffill', inplace=True)

TARGET_COLUMN = "close"


plt.figure(figsize=(12, 6))
plt.plot(df[TARGET_COLUMN])
plt.title("Stock Prices")
plt.ylabel("Stock Price")
plt.show()


scaler = MinMaxScaler()
df[TARGET_COLUMN] = scaler.fit_transform(df[[TARGET_COLUMN]])


seq_len = 10

X, y = [], []

for i in range(len(df) - seq_len):
    X.append(df[TARGET_COLUMN].iloc[i : i + seq_len].values)
    y.append(df[TARGET_COLUMN].iloc[i + seq_len])

X = np.array(X)
y = np.array(y)

print("X shape:", X.shape)   # (samples, seq_len)
print("y shape:", y.shape)



X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
X_train = X_train.reshape(-1, seq_len, 1)
X_test  = X_test.reshape(-1, seq_len, 1)

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)


# CASE A: SimpleRNN (your class code - use this by default)
model = Sequential([
    SimpleRNN(50, activation='relu', input_shape=(seq_len, 1)),
    Dense(1)
])

# CASE B: LSTM (if sir asks for LSTM here or better accuracy needed)
# model = Sequential([
#     LSTM(50, activation='relu', input_shape=(seq_len, 1)),
#     Dense(1)
# ])

# CASE C: Stacked SimpleRNN (2 RNN layers)
# model = Sequential([
#     SimpleRNN(50, activation='relu', return_sequences=True,
#               input_shape=(seq_len, 1)),
#     SimpleRNN(50, activation='relu'),
#     Dense(1)
# ])

# CASE D: Stacked LSTM (2 LSTM layers)
# model = Sequential([
#     LSTM(50, activation='relu', return_sequences=True,
#          input_shape=(seq_len, 1)),
#     LSTM(50, activation='relu'),
#     Dense(1)
# ])

model.summary()

model.compile(
    optimizer='adam',
    loss='mse'       # mean squared error for regression
)

history = model.fit(
    X_train, y_train,
    epochs=20,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)

loss = model.evaluate(X_test, y_test)
print(f"\nMean Squared Error: {loss:.6f}")


# STEP 11: PREDICT & INVERSE TRANSFORM

pred = model.predict(X_test)

# Convert scaled values back to original price range
pred   = scaler.inverse_transform(pred)
y_test = scaler.inverse_transform(y_test.reshape(-1, 1))

# Extra metrics (optional but impress sir)
mae  = np.mean(np.abs(y_test - pred))
rmse = np.sqrt(np.mean((y_test - pred) ** 2))
print(f"MAE  : {mae:.4f}")
print(f"RMSE : {rmse:.4f}")


# ============================================================
# STEP 12: PLOT LOSS
# ============================================================

plt.figure(figsize=(8, 4))
plt.plot(history.history['loss'],     label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('MSE')
plt.legend()
plt.show()


# ============================================================
# STEP 13: PLOT ACTUAL vs PREDICTED

plt.figure(figsize=(12, 6))
plt.plot(pred,   label="Predicted", color='red')
plt.plot(y_test, label="Actual",    color='blue')
plt.title("Stock Price Prediction")
plt.ylabel("Stock Price")
plt.legend()
plt.show()


# ============================================================
# ✅ TEXT / SENTIMENT CASE — comment out TIME SERIES above
#    and uncomment everything below if sir gives text data
# ============================================================

# import seaborn as sns
#
# # STEP 1: LOAD TEXT DATASET
# # df = pd.read_csv('/content/train.csv')   # <-- change filename
# # print(df.head())
# # print(df.dtypes)
#
# # STEP 2: SELECT TEXT & LABEL COLUMNS
# # TEXT_COLUMN  = "text"      # column with sentences/reviews
# # LABEL_COLUMN = "label"     # column with sentiment/category
# #
# # CASE B: IMDB-style
# # TEXT_COLUMN  = "review"
# # LABEL_COLUMN = "sentiment"
# #
# # X = df[TEXT_COLUMN].astype(str).values
# # y = df[LABEL_COLUMN].values
#
# # STEP 3: ENCODE LABELS
# # CASE A: Text labels (e.g. "positive"/"negative") → encode to 0/1
# # le = LabelEncoder()
# # y  = le.fit_transform(y)
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
# #     SimpleRNN(64, return_sequences=True),
# #     Dropout(0.2),
# #     SimpleRNN(32),
# #     Dense(1, activation='sigmoid')    # binary output
# # ])
# #
# # CASE B: Multi-class (3+ categories)
# # NUM_CLASSES = len(np.unique(y))
# # model = Sequential([
# #     Embedding(MAX_WORDS, 64, input_length=MAX_LEN),
# #     SimpleRNN(64, return_sequences=True),
# #     Dropout(0.2),
# #     SimpleRNN(32),
# #     Dense(NUM_CLASSES, activation='softmax')
# # ])
# #
# # CASE C: Bidirectional RNN (better accuracy)
# # model = Sequential([
# #     Embedding(MAX_WORDS, 64, input_length=MAX_LEN),
# #     Bidirectional(SimpleRNN(64, return_sequences=True)),
# #     Dropout(0.2),
# #     Bidirectional(SimpleRNN(32)),
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