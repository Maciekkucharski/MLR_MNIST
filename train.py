import os
from matplotlib import pyplot as plt
# Disable TF warning messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf

class FullyConnectedForMnist:
    '''Simple NN for MNIST database. INPUT => FC/RELU => FC/SOFTMAX'''
    def build(hidden_units):
        # Initialize the model
        model = tf.keras.models.Sequential()
        # Flatten the input data of (x, y, 1) dimension
        model.add(tf.keras.layers.Flatten(input_shape=(28,28,1)))
        # FC/RELU layer
        model.add(tf.keras.layers.Dense(hidden_units, activation='relu'))
        # Softmax classifier (10 classes)
        model.add(tf.keras.layers.Dense(10, activation="softmax"))
        return model

EPOCHS = 10
MODEL_FILENAME = "model.h5"

# Load dataset as train and test sets
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Convert from uint8 to float32 and normalize to [0,1]
x_train = x_train.astype('float32')/255
x_test = x_test.astype('float32')/255

# Transform labels to 0-1 encoding, e.g.
# 2 -> [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
# 6 -> [0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

# Reshape the dataset into 4D array (required by Keras)
# First dimension is the number of samples
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

# Construct the model
model = FullyConnectedForMnist.build(500)

# Compile the model and print summary
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

# Train the model
history = model.fit(x=x_train, y=y_train, epochs=EPOCHS, validation_split=0.2)

plt.title("accuracy by epochs")
plt.ylabel("accuracy")
plt.xlabel("epoch")
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(['train split', 'validation split'])
plt.show()

plt.title("loss by epochs")
plt.ylabel("loss")
plt.xlabel("epoch")
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['train split', 'validation split'])
plt.show()



# Save model to a file
model.save(MODEL_FILENAME)

# Evaluate the model on the test data
model.evaluate(x_test, y_test)
