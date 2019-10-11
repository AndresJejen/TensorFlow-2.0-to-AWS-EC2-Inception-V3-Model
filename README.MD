# TensorFlow 2 to Production AWS EC2 DOCKER

## Description
Here you can find the code for "Ruta de aprendizaje en Deep Learning - Parte 2 - Modelo Tensorflow en producción AWS",
follow all these steps and enjoy tensorflow.

## Instructions

- 1. You must have installed ```virtualenv``` and ```Python3```  
        I´m using Ubuntu 18 on Windows Subsystem Linux.   
        clone this repo (I´m working on my disk partition D on WSL)  
        ```
        cd /mnt/d
        ```
        ```
        git clone https://github.com/AndresJejen/TensorFlow-2.0-to-AWS-EC2-Inception-V3-Model.git
        ```
        ```
        cd TensorFlow-2.0-to-AWS-EC2-Inception-V3-Model
        ```
- 2. run this code on root project folder to create the virtual environment
    ```
        virtualenv env --python=python3
    ```  
    and activate the virtual env
    ```
        source env/bin/activate
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
        docker run -t --rm -p 8501:8501 -v mnt/d/TensorFlow-2.0-to-AWS-EC2-Inception-V3-Model/modelTF:/models/ImageClassifier --name "Test" -e MODEL_NAME=ImageClassifier tensorflow/serving
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

## AWESOME LINKS
- https://www.tensorflow.org/tfx/serving/setup
- https://towardsdatascience.com/deploying-keras-models-using-tensorflow-serving-and-flask-508ba00f1037
- https://github.com/Hvass-Labs/TensorFlow-Tutorials/blob/master/01_Simple_Linear_Model.ipynb

## Thanks
If you like please clap the Medium Article. Share and let´s learn together.

## Colombia AI

Twitter: @andres_jejen

## Mision de los setenta: "La mies a la verdad es mucha, mas los obreros pocos; por tanto, rogad al Señor de la mies que envíe obreros a su mies"