this sbavhajskskbegyeueiennn
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import numpy as np

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

x_train = x_train.reshape(-1, 28, 28, 1).astype("float32") / 255.0
x_test = x_test.reshape(-1, 28, 28, 1).astype("float32") / 255.0

print(f"Training data shape: {x_train.shape}")
print(f"Test data shape: {x_test.shape}")

model = keras.Sequential([
    keras.Input(shape=(28, 28, 1)),
    
    layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
    
    layers.MaxPooling2D(pool_size=(2, 2)),
    
    layers.Flatten(),
    
    layers.Dense(128, activation="relu"),
    
    layers.Dense(10, activation="softmax")
])

model.summary() 

model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer="adam",                 
    metrics=["accuracy"]              
)

print("\n--- Starting Training ---")
history = model.fit(
    x_train, 
    y_train, 
    epochs=5,
    validation_split=0.2 
)
print("--- Training Complete ---")

test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"\nFinal Test Accuracy: {test_acc * 100:.2f}%")

print("\nPlotting graphs...")
history_dict = history.history


loss = history_dict['loss']
val_loss = history_dict['val_loss']
accuracy = history_dict['accuracy']
val_accuracy = history_dict['val_accuracy']

epochs_range = range(1, len(loss) + 1)

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, loss, 'bo-', label='Training Loss')
plt.plot(epochs_range, val_loss, 'ro-', label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(epochs_range, accuracy, 'bo-', label='Training Accuracy')
plt.plot(epochs_range, val_accuracy, 'ro-', label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout() # Adjusts plots to prevent overlap
plt.show()
