import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt

vocab_size = 10000
(x_train, y_train), (x_test, y_test) = keras.datasets.imdb.load_data(num_words=vocab_size)

print(f"Training samples: {len(x_train)}")
print(f"Test samples: {len(x_test)}")

maxlen = 200
x_train = pad_sequences(x_train, maxlen=maxlen)
x_test = pad_sequences(x_test, maxlen=maxlen)

print(f"Padded training data shape: {x_train.shape}")


model = keras.Sequential([
   
    keras.Input(shape=(maxlen,)),
    layers.Embedding(input_dim=vocab_size, output_dim=128),
    
    layers.Conv1D(filters=64, kernel_size=5, activation="relu"),
    
    layers.GlobalMaxPooling1D(),

    layers.Dense(64, activation="relu"),
    
    layers.Dense(1, activation="sigmoid")
])

model.summary()

model.compile(
    loss="binary_crossentropy", 
    optimizer="adam",
    metrics=["accuracy"]
)

print("\n--- Starting Training ---")
history = model.fit(
    x_train,
    y_train,
    epochs=5,
    batch_size=64,
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

plt.tight_layout()
plt.show()
