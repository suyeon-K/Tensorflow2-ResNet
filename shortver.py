from keras.applications.resnet_v2 import ResNet50V2
from keras.datasets import cifar10
import numpy as np
import tensorflow as tf
import cv2
from tensorflow import keras
import matplotlib.pyplot as plt
# from keras.utils import to_categorical
# from keras.utils.vis_utils import plot_model
import ssl
ssl._create_default_https_context = ssl._create_unverified_context


(x_train, y_train), (x_test, y_test) = cifar10.load_data()

num_classes = 10
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

x_train=x_train / 255.0
x_test=x_test / 255.0

model = ResNet50V2(include_top=True, weights=None, input_shape=(32,32,3),classes=10)

model.compile(optimizer='adam', loss="categorical_crossentropy", metrics=['accuracy'])

print(model.summary())
# plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

history=model.fit(x_train, y_train, batch_size=32, epochs=10, verbose=1,validation_data=(x_test,y_test))

_, acc = model.evaluate(x_test, y_test)
print("Test Accuracy: {}%".format(acc*100))

y_vloss = history.history['val_loss']
y_loss = history.history['loss']

x_len = np.arange(len(y_loss))
plt.plot(x_len, y_vloss, marker='.', c='red', label="Validation-set Loss")
plt.plot(x_len, y_loss, marker='.', c='blue', label="Train-set Loss")

plt.legend(loc='upper right')
plt.grid()
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()