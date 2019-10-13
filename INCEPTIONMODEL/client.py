import argparse
import json

import numpy as np
import requests
from tensorflow.keras.applications import inception_v3
from tensorflow.keras.preprocessing import image

# Argument parser for giving input image_path from command line
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
                help="path of the image")
args = vars(ap.parse_args())

image_path = args['image']
# Preprocessing our input image
img = image.img_to_array(image.load_img(image_path, target_size=(224, 224))) / 255.

# this line is added because of a bug in tf_serving(1.10.0-dev)
img = img.astype('float16')

print(img.shape)

data = {"instances": [img.tolist()]}
print(type(img.tolist()))
headers = {"content-type": "application/json"}

# sending post request to TensorFlow Serving server
r = requests.post('http://<YOUR AWS EC2 ROUTE>:8080/v1/models/INCEPTION:predict', json=data, headers=headers)
print(r)
predictions = json.loads(r.content.decode('utf-8'))["predictions"]

# Decoding the response
# decode_predictions(preds, top=5) by default gives top 5 results
# You can pass "top=10" to get top 10 predicitons
resultados = inception_v3.decode_predictions(np.array(predictions))[0]

for i in resultados:
    print("Objeto predicho",i[1]," con una probabilidad de",i[2])