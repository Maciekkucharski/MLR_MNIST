import os
# Disable TF warning messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import numpy as np
import cv2     # pip install opencv-python

# Directory with test set
TEST_DATASET_DIR = 'mnist-test'

# Trained model name
MODEL = 'model.h5'

# Load trained model of neural network
model = tf.keras.models.load_model(MODEL)
(_, _), (x_test, _) = tf.keras.datasets.mnist.load_data()
for image in x_test:
    image_data = image
    image_data = image_data.astype('float32')/255
    # Make shape 4D for Keras, (28,28,1) -> (1,28,28,1)
    image_data = np.expand_dims(image_data, axis=0)
    # Classify the input image
    prediction = model.predict(image_data)
    # Find the winner class and the probability
    winner_class = np.argpartition(prediction, -1)[:, -1][0]
    second_class = np.argpartition(prediction, -2)[:, -2][0]
    third_class = np.argpartition(prediction, -3)[:, -3][0]

    winner_probability = np.partition(prediction, -1)[:, -1][0]*100
    second_probability = np.partition(prediction, -2)[:, -2][0]*100
    third_probability = np.partition(prediction, -3)[:, -3][0]*100

    # Build the text label
    label = "klasa = {}, p = {:.2f}%".format(winner_class, winner_probability)
    label2 = "klasa = {}, p = {:.2f}%".format(second_class, second_probability)
    label3 = "klasa = {}, p = {:.2f}%".format(third_class, third_probability)

    # Draw the label on the image
    output_image = cv2.resize(image, (500,500))
    cv2.putText(output_image, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.75, 255, 2)
    cv2.putText(output_image, label2, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, 255, 2)
    cv2.putText(output_image, label3, (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.75, 255, 2)

    # Show the output image        
    cv2.imshow("Output", output_image)
    cv2.waitKey(0)
