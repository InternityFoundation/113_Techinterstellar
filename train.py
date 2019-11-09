from __future__ import print_function
import tensorflow as tf
import keras 
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPool2D
from keras import backend as K 
from sklearn.model_selection import train_test_split
img_rows, img_cols = 28, 28
  train_data_dir = '/users/yashchaturvedi/Desktop/programming/ML/sign/asl-alphabet/asl_alphabet_train'
test_data_dir = '/users/yashchaturvedi/Desktop/programming/ML/sign/asl-alphabet/asl_alphabet_test'
X = X.reshape(124800, 28, 28)
y = y.reshape(124800, 1)
y = y-1
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=111)
x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
input_shape = (img_rows, img_cols, 1)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
batch_size = 128
num_classes = 26
epochs = 10


