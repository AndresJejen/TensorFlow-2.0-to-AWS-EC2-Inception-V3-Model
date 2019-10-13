from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import os
import sys

from model import create_model

print("Tensorflow Version:", tf.__version__)

print("Downloading Data 📲")
mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

print("Importing Model 🙂")
model = create_model()

model.summary()

checkpoint_path = "training_1/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Train the model with the new callback
print("Training your Model 🤓")
model.fit(x_train, 
        y_train,  
        epochs=5,
        validation_data=(x_test,y_test))

print("Saving your weights Model")
model.save('./modelTF/1', save_format='tf')
print("Yeah!!")