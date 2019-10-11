from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.layers import Input

print("Downloading Model ...")
inception_model = InceptionV3(weights='imagenet', input_tensor=Input(shape=(224, 224, 3)))
print("Saving Freeze Model...")
inception_model.save('./modelTF/1', save_format='tf')
print("Yeah! ... Every Little thing gonna be alright, Hillsong YF 😋")