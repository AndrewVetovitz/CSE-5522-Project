# CSE-5522-Project
Project code base for CSE 5522
Using python 3 for this project and pip 3

#### Ubuntu installation for cpu support
sudo apt-get install python3-pip python3-dev
pip3 install tensorflow

# Run this python code snipit to make sure tensor flow installed correctly
# Python
import tensorflow as tf
hello = tf.constant('Hello, TensorFlow!')
sess = tf.Session()
print(sess.run(hello))
