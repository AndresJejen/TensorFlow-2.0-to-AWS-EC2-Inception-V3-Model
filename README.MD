# TensorFlow 2 to Production AWS EC2 DOCKER

## Description
Here you can find the code for "Ruta de aprendizaje en Deep Learning - Parte 2 - Modelo Tensorflow en producción AWS",
follow all these steps and enjoy tensorflow.

## Instructions

- 1. You must have installed virtual env and Python3
- 2. run this code on root project folder to create the virtual environment
    ```
        virtualenv env --python=python3
    ```
- 3. Install al dependencies
    ```
        pip install -r requirements.txt
    ```
- 4. Save the Model (format pb)
    ```
        python3 download-InceptionV3.py
    ```
- 5. run the Server with Docker (You must install docker)
    ```
        docker run -t --rm -p 8501:8501 -v $(pwd)\modelTF:/models/ImageClassifier --name "Test" -e MODEL_NAME=ImageClassifier tensorflow/serving
    ```
- 6. create your own querys
    ```
        python3 client.py -i vaso.jpg
    ```

## Deploy on AWS EC2

- 1. Create Your AWS ACCOUNT
- 2. Create a Virtual Machine and install docker
- 3. Clone your code fron GitHub to AWS EC2
- 4. Start your server
- 5. Query from your own pc

## Thanks

## Colombia AI

Twitter: @andres_jejen

## Mision de los setenta: "La mies a la verdad es mucha, mas los obreros pocos; por tanto, rogad al Señor de la mies que envíe obreros a su mies"