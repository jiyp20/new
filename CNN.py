
import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix, classification_report



def preprocess_image(path):
    # CASE A: Grayscale image (MNIST-style digits) ← default
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (28, 28))
    img = cv2.bitwise_not(img)          # invert: white digit on black bg
    img = img.astype("float32") / 255.0
    return np.expand_dims(img, axis=(0, -1))   # shape: (1, 28, 28, 1)

    # CASE B: Color image (CIFAR-style)
    # img = cv2.imread(path)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)   # cv2 loads BGR → convert to RGB
    # img = cv2.resize(img, (32, 32))
    # img = img.astype("float32") / 255.0
    # return np.expand_dims(img, axis=0)            # shape: (1, 32, 32, 3)


# CASE A: MNIST - handwritten digits ← default
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
IMG_SHAPE   = (28, 28, 1)
NUM_CLASSES = 10

# CASE B: Fashion MNIST - same structure as MNIST, just different images
# (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.fashion_mnist.load_data()
# IMG_SHAPE   = (28, 28, 1)
# NUM_CLASSES = 10

# CASE C: CIFAR-10 - color images (cars, birds, dogs etc)
# (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()
# IMG_SHAPE   = (32, 32, 3)
# NUM_CLASSES = 10

print("Train shape:", train_images.shape)
print("Test shape :", test_images.shape)



plt.figure(figsize=(10, 4))
for i in range(10):
    plt.subplot(2, 5, i+1)
    # CASE A: Grayscale (MNIST / Fashion MNIST)
    plt.imshow(train_images[i], cmap='gray')
    # CASE B: Color (CIFAR-10)
    # plt.imshow(train_images[i])
    plt.title(f"Label: {train_labels[i]}")
    plt.axis('off')
plt.tight_layout()
plt.show()


# ============================================================
# STEP 4: NORMALIZE & RESHAPE
# ============================================================

# CASE A: Grayscale images (MNIST / Fashion MNIST) ← default
train_images = train_images.reshape(-1, 28, 28, 1) / 255.0
test_images  = test_images.reshape(-1, 28, 28, 1)  / 255.0

# CASE B: Color images (CIFAR-10)
# train_images = train_images.reshape(-1, 32, 32, 3) / 255.0
# test_images  = test_images.reshape(-1, 32, 32, 3)  / 255.0

print("train_images shape:", train_images.shape)  # (60000, 28, 28, 1)


train_labels_enc = to_categorical(train_labels)
test_labels_enc  = to_categorical(test_labels)

#(use sparse_categorical_crossentropy)
# train_labels_enc = train_labels
# test_labels_enc  = test_labels

model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=IMG_SHAPE),
    layers.MaxPooling2D(2, 2),

    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),

    layers.Conv2D(64, (3, 3), activation='relu'),

    layers.Flatten(),
    layers.Dense(64,          activation='relu'),
    layers.Dense(NUM_CLASSES, activation='softmax')
])

# CASE B: With Dropout (if overfitting)
# model = models.Sequential([
#     layers.Conv2D(32, (3, 3), activation='relu', input_shape=IMG_SHAPE),
#     layers.MaxPooling2D(2, 2),
#
#     layers.Conv2D(64, (3, 3), activation='relu'),
#     layers.MaxPooling2D(2, 2),
#
#     layers.Conv2D(64, (3, 3), activation='relu'),
#
#     layers.Flatten(),
#     layers.Dense(64,          activation='relu'),
#     layers.Dropout(0.5),
#     layers.Dense(NUM_CLASSES, activation='softmax')
# ])

model.summary()


model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# CASE B: Labels are integers
# model.compile(
#     optimizer='adam',
#     loss='sparse_categorical_crossentropy',
#     metrics=['accuracy']
# )


history = model.fit(
    train_images, train_labels_enc,
    epochs=5,
    batch_size=64,
    validation_split=0.2,
    verbose=1
)



loss, accuracy = model.evaluate(test_images, test_labels_enc)
print(f"\nTest Accuracy: {accuracy * 100:.2f}%")
print(f"Test Loss    : {loss:.4f}")


# STEP 10: PLOT ACCURACY & LOSS

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


# STEP 11: PREDICTIONS

y_pred         = model.predict(test_images)
y_pred_classes = np.argmax(y_pred,   axis=1)
y_true_classes = test_labels.flatten()   # original integer labels


# STEP 12: VISUALIZE PREDICTIONS
# 

plt.figure(figsize=(12, 4))
for i in range(10):
    plt.subplot(2, 5, i+1)
    plt.imshow(test_images[i].reshape(28, 28), cmap='gray')  # change shape for CIFAR
    color = 'green' if y_pred_classes[i] == y_true_classes[i] else 'red'
    plt.title(f"P:{y_pred_classes[i]} A:{y_true_classes[i]}", color=color)
    plt.axis('off')
plt.suptitle("Green=Correct  Red=Wrong")
plt.tight_layout()
plt.show()


# STEP 13: CONFUSION MATRIX & REPORT

cm = confusion_matrix(y_true_classes, y_pred_classes)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

print("\nClassification Report:")
print(classification_report(y_true_classes, y_pred_classes))


# STEP 14: 

# img = preprocess_image("/content/digit.png")   # <-- change path
# pred = model.predict(img)
# print("Predicted digit:", np.argmax(pred))
#
# plt.imshow(img.reshape(28, 28), cmap='gray')
# plt.title(f"Predicted: {np.argmax(pred)}")
# plt.axis('off')
# plt.show()