import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


data = pd.read_csv("/content/train.csv")  

print("Shape:", data.shape)
print(data.head())
print(data.dtypes)


print(data.isnull().sum())

# CASE A: Drop rows with missing values (use when very few rows are missing)
# data = data.dropna()

# CASE B: Fill missing values with column mean (use when many rows are missing)
# data.fillna(data.mean(numeric_only=True), inplace=True)

# CASE A: Dataset has text columns like Gender, City etc (one-hot encode)
# data = pd.get_dummies(data, drop_first=True)

# CASE B: No text columns - skip this (default)

# --- Change "Cover_Type" to your actual target column name ---
TARGET_COLUMN = "Cover_Type"

X = data.drop(TARGET_COLUMN, axis=1)
y = data[TARGET_COLUMN]


# CASE A: Labels are integers starting from 1 (e.g. 1,2,3,4...)
#         → shift to start from 0
y = y - 1   # e.g. 1-7 becomes 0-6

# CASE B: Labels are integers already starting from 0 (e.g. 0,1,2...)
#         → do nothing, comment out the line above

# CASE C: Labels are text strings (e.g. "cat","dog","bird")
#         → use LabelEncoder
# le = LabelEncoder()
# y = le.fit_transform(y)   # converts to 0,1,2...

# CASE D: Labels are one-hot encoded columns (e.g. class_0, class_1, class_2)
#         → select those columns as y, and handle separately in model section
# y = data[["class_0", "class_1", "class_2"]]
# X = data.drop(["class_0", "class_1", "class_2"], axis=1)

print("Classes:", y.nunique())
print("Label range:", y.min(), "to", y.max())


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)



scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)     


num_classes = y.nunique()

# --- CASE A: Multi-class classification (3+ classes) → softmax output ---
model = keras.Sequential([
    layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    layers.Dense(64,  activation='relu'),
    layers.Dense(32,  activation='relu'),
    layers.Dense(num_classes, activation='softmax')   # one neuron per class
])

# --- CASE B: Binary classification (2 classes only) → sigmoid output ---
# model = keras.Sequential([
#     layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
#     layers.Dense(64,  activation='relu'),
#     layers.Dense(32,  activation='relu'),
#     layers.Dense(1, activation='sigmoid')   # single output neuron
# ])

model.summary()


# --- CASE A: Multi-class, integer labels (most common) ---
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# --- CASE B: Binary classification ---
# model.compile(
#     optimizer='adam',
#     loss='binary_crossentropy',
#     metrics=['accuracy']
# )

# --- CASE C: Multi-class, one-hot encoded labels ---
# model.compile(
#     optimizer='adam',
#     loss='categorical_crossentropy',
#     metrics=['accuracy']
# )


history = model.fit(
    X_train, y_train,
    epochs=20,
    batch_size=32,
    validation_split=0.1,
    verbose=1
)



loss, accuracy = model.evaluate(X_test, y_test)
print(f"\nTest Accuracy: {accuracy * 100:.2f}%")


plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'],     label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'],     label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()


# --- CASE A: Multi-class (softmax output) ---
y_pred = np.argmax(model.predict(X_test), axis=1)

# --- CASE B: Binary (sigmoid output) ---
# y_pred = (model.predict(X_test) > 0.5).astype(int).flatten()

# --- CASE C: One-hot encoded labels ---
# y_pred = np.argmax(model.predict(X_test), axis=1)
# y_test = np.argmax(y_test.values, axis=1)   # convert y_test too

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))