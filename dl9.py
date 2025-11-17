import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense 
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

sentences = [
    "I love this product",
    "This is amazing",
    "What a great experience",
    "I am so happy",
    "This is the best",
    "I hate this",
    "This is terrible",
    "What a waste of time",
    "I am very disappointed",
    "This is the worst"
]
labels = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])

vocab_size = 100
max_length = 10
embedding_dim = 8

tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
tokenizer.fit_on_texts(sentences)
sequences = tokenizer.texts_to_sequences(sentences)
padded_sequences = pad_sequences(sequences, maxlen=max_length, padding='post')


model_lstm = Sequential([
   
    Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_length),
  
    LSTM(units=16),

    Dense(units=1, activation='sigmoid')
])

model_lstm.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print("--- LSTM Model Summary ---")
model_lstm.summary()

epochs = 50
history_lstm = model_lstm.fit(padded_sequences, labels, epochs=epochs, verbose=0)
print("\nLSTM Model training complete.")

plt.figure(figsize=(10, 4))


plt.subplot(1, 2, 1)
plt.plot(history_lstm.history['accuracy'], label='Training Accuracy')
plt.title('LSTM Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history_lstm.history['loss'], label='Training Loss')
plt.title('LSTM Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()
