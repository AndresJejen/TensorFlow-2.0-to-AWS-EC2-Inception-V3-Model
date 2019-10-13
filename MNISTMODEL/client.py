import argparse
import json

import numpy as np
import requests
import tensorflow as tf

print("Downloading Data 📲")
mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

print(x_test.shape)
index = 1
data = x_test[index]
print(type(data),data.shape)

data = {"instances": data.tolist()}
headers = {"content-type": "application/json"}

# sending post request to TensorFlow Serving server
r = requests.post('http://<YOUR AWS EC2 ROUTE>/v1/models/MNIST:predict', json=data, headers=headers)
pred = json.loads(r.content.decode())
print(r)
print(pred)
resultado = np.array(pred['predictions'])[0]
imagen = np.argmax(resultado)
print("El modelo predice el número:",imagen,", Con una probabilidad de",resultado[imagen])
print("La etiqueta real es:        ",y_test[index])