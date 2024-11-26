# coding: utf-8
import numpy as np
from dataset.mnist import load_mnist
# from deep_convnet import DeepConvNet
from neuralnet_mnist import init_network, predict
from common.image import img_show


(x_train, t_train), (x_test, t_test) = load_mnist(normalize=False, flatten=True)

index = 999
img = x_test[index][np.newaxis, ...]
label = t_test[index]


# 학습된 가중치
network = init_network("sample_weight.pkl")

prediction = np.argmax(predict(network, img / 255.0))
print("predict:", prediction)
print("answer:", label)

img_show(img.reshape(28, 28))