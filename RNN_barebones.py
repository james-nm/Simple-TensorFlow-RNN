#https://www.tensorflow.org/guide/keras/rnn

from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import matplotlib.pyplot as plt

batch_size = 64
input_dimension = 28

LSTM_unit_qty = 64
output_size = 10  # labels are from 0 to 9

# Loat MNIST dataset
mnist_dataset = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist_dataset.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# Build and compile model
lstm_layer = tf.keras.layers.LSTM(LSTM_unit_qty,
                                  input_shape=(None, input_dimension))
model = tf.keras.models.Sequential([
    lstm_layer,
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(output_size, activation='softmax')])

model.compile(loss='sparse_categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])

model.summary()

# Train model
model.fit(x_train, y_train,
          validation_data=(x_test, y_test),
          batch_size=batch_size,
          epochs=5)

# Make predictions
for i in range(5):
  sample, sample_label = x_train[i], y_train[i]
  result = tf.argmax(model.predict_on_batch(
      tf.expand_dims(sample, 0)), axis=1)
  print(f'Predicted result: {result.numpy()}, sample label: {sample_label}')

  plt.imshow(sample, cmap=plt.get_cmap('gray'))