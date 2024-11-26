#coding: utf-8
# from common.layers import *
# import numpy as np
# import pickle


# class DeepConvNet:
#     """정확도 99% 이상의 고정밀 합성곱 신경망
#     네트워크 구성은 아래와 같음
#         conv - relu - conv- relu - pool -
#         conv - relu - conv- relu - pool -
#         conv - relu - conv- relu - pool -
#         affine - relu - dropout - affine - dropout - softmax
#     """

#     def __init__(self, input_dim=(1, 28, 28),
#                  conv_param_1={'filter_num': 16,
#                                'filter_size': 3, 'pad': 1, 'stride': 1},
#                  conv_param_2={'filter_num': 16,
#                                'filter_size': 3, 'pad': 1, 'stride': 1},
#                  conv_param_3={'filter_num': 32,
#                                'filter_size': 3, 'pad': 1, 'stride': 1},
#                  conv_param_4={'filter_num': 32,
#                                'filter_size': 3, 'pad': 2, 'stride': 1},
#                  conv_param_5={'filter_num': 64,
#                                'filter_size': 3, 'pad': 1, 'stride': 1},
#                  conv_param_6={'filter_num': 64,
#                                'filter_size': 3, 'pad': 1, 'stride': 1},
#                  hidden_size=50, output_size=10):
#         # 가중치 초기화===========
#         # 각 층의 뉴런 하나당 앞 층의 몇 개 뉴런과 연결되는가（TODO: 자동 계산되게 바꿀 것）
#         pre_node_nums = np.array(
#             [1*3*3, 16*3*3, 16*3*3, 32*3*3, 32*3*3, 64*3*3, 64*4*4, hidden_size])
#         wight_init_scales = np.sqrt(2.0 / pre_node_nums)  # ReLU를 사용할 때의 권장 초깃값

#         self.params = {}
#         pre_channel_num = input_dim[0]
#         for idx, conv_param in enumerate([conv_param_1, conv_param_2, conv_param_3, conv_param_4, conv_param_5, conv_param_6]):
#             self.params['W' + str(idx+1)] = wight_init_scales[idx] * np.random.randn(
#                 conv_param['filter_num'], pre_channel_num, conv_param['filter_size'], conv_param['filter_size'])
#             self.params['b' + str(idx+1)] = np.zeros(conv_param['filter_num'])
#             pre_channel_num = conv_param['filter_num']
#         self.params['W7'] = wight_init_scales[6] * \
#             np.random.randn(64*4*4, hidden_size)
#         self.params['b7'] = np.zeros(hidden_size)
#         self.params['W8'] = wight_init_scales[7] * \
#             np.random.randn(hidden_size, output_size)
#         self.params['b8'] = np.zeros(output_size)

#         # 계층 생성===========
#         self.layers = []
#         self.layers.append(Convolution(self.params['W1'], self.params['b1'],
#                            conv_param_1['stride'], conv_param_1['pad']))
#         self.layers.append(Relu())
#         self.layers.append(Convolution(self.params['W2'], self.params['b2'],
#                            conv_param_2['stride'], conv_param_2['pad']))
#         self.layers.append(Relu())
#         self.layers.append(Pooling(pool_h=2, pool_w=2, stride=2))
#         self.layers.append(Convolution(self.params['W3'], self.params['b3'],
#                            conv_param_3['stride'], conv_param_3['pad']))
#         self.layers.append(Relu())
#         self.layers.append(Convolution(self.params['W4'], self.params['b4'],
#                            conv_param_4['stride'], conv_param_4['pad']))
#         self.layers.append(Relu())
#         self.layers.append(Pooling(pool_h=2, pool_w=2, stride=2))
#         self.layers.append(Convolution(self.params['W5'], self.params['b5'],
#                            conv_param_5['stride'], conv_param_5['pad']))
#         self.layers.append(Relu())
#         self.layers.append(Convolution(self.params['W6'], self.params['b6'],
#                            conv_param_6['stride'], conv_param_6['pad']))
#         self.layers.append(Relu())
#         self.layers.append(Pooling(pool_h=2, pool_w=2, stride=2))
#         self.layers.append(Affine(self.params['W7'], self.params['b7']))
#         self.layers.append(Relu())
#         self.layers.append(Dropout(0.5))
#         self.layers.append(Affine(self.params['W8'], self.params['b8']))
#         self.layers.append(Dropout(0.5))

#         self.last_layer = SoftmaxWithLoss()

#     def predict(self, x, train_flg=False):
#         for layer in self.layers:
#             if isinstance(layer, Dropout):
#                 x = layer.forward(x, train_flg)
#             else:
#                 x = layer.forward(x)
#         return x

#     def loss(self, x, t):
#         y = self.predict(x, train_flg=True)
#         return self.last_layer.forward(y, t)

#     def accuracy(self, x, t, batch_size=100):
#         if t.ndim != 1:
#             t = np.argmax(t, axis=1)

#         acc = 0.0

#         for i in range(int(x.shape[0] / batch_size)):
#             tx = x[i*batch_size:(i+1)*batch_size]
#             tt = t[i*batch_size:(i+1)*batch_size]
#             y = self.predict(tx, train_flg=False)
#             y = np.argmax(y, axis=1)
#             acc += np.sum(y == tt)

#         return acc / x.shape[0]

#     def gradient(self, x, t):
#         # forward
#         self.loss(x, t)

#         # backward
#         dout = 1
#         dout = self.last_layer.backward(dout)

#         tmp_layers = self.layers.copy()
#         tmp_layers.reverse()
#         for layer in tmp_layers:
#             dout = layer.backward(dout)

#         # 결과 저장
#         grads = {}
#         for i, layer_idx in enumerate((0, 2, 5, 7, 10, 12, 15, 18)):
#             grads['W' + str(i+1)] = self.layers[layer_idx].dW
#             grads['b' + str(i+1)] = self.layers[layer_idx].db

#         return grads

#     def save_params(self, file_name="params.pkl"):
#         params = {}
#         for key, val in self.params.items():
#             params[key] = val
#         with open(file_name, 'wb') as f:
#             pickle.dump(params, f)

#     def load_params(self, file_name="params.pkl"):
#         with open(file_name, 'rb') as f:
#             params = pickle.load(f)
#         for key, val in params.items():
#             self.params[key] = val

#         for i, layer_idx in enumerate((0, 2, 5, 7, 10, 12, 15, 18)):
#             self.layers[layer_idx].W = self.params['W' + str(i+1)]
#             self.layers[layer_idx].b = self.params['b' + str(i+1)]

#-----------------------

# import numpy as np
# import pickle
# from common.layers import *

# class DeepConvNet:
#     def __init__(self, input_dim=(1, 28, 28), conv_params=[(16, 3, 1, 1), (16, 3, 1, 1), (32, 3, 1, 1), (32, 3, 1, 2), (64, 3, 1, 1), (64, 3, 1, 1)], hidden_size=50, output_size=10):
#         # 가중치 초기화
#         pre_channel_num = input_dim[0]
#         conv_param_list = []
#         self.params = {}

#         for i, (filter_num, filter_size, pad, stride) in enumerate(conv_params):
#             self.params['W' + str(i+1)] = np.random.randn(filter_num, pre_channel_num, filter_size, filter_size) / np.sqrt(filter_num / 2)
#             self.params['b' + str(i+1)] = np.zeros(filter_num)
#             conv_param_list.append({'stride': stride, 'pad': pad})
#             pre_channel_num = filter_num

#         conv_output_size = input_dim[1]
#         for param in conv_param_list:
#             conv_output_size = (conv_output_size - param['pad'] * 2) // param['stride'] + 1

#         self.params['W' + str(len(conv_params) + 1)] = np.random.randn(conv_output_size * conv_output_size * pre_channel_num, hidden_size) / np.sqrt(conv_output_size * conv_output_size * pre_channel_num / 2)
#         self.params['b' + str(len(conv_params) + 1)] = np.zeros(hidden_size)

#         self.params['W' + str(len(conv_params) + 2)] = np.random.randn(hidden_size, output_size) / np.sqrt(hidden_size / 2)
#         self.params['b' + str(len(conv_params) + 2)] = np.zeros(output_size)

#         # 계층 생성
#         self.layers = []
#         self.layers.append(Convolution(self.params['W1'], self.params['b1'], conv_param_list[0]['stride'], conv_param_list[0]['pad']))
#         self.layers.append(Relu())
#         self.layers.append(Convolution(self.params['W2'], self.params['b2'], conv_param_list[1]['stride'], conv_param_list[1]['pad']))
#         self.layers.append(Relu())
#         self.layers.append(Pooling(pool_h=2, pool_w=2, stride=2))
#         self.layers.append(Convolution(self.params['W3'], self.params['b3'], conv_param_list[2]['stride'], conv_param_list[2]['pad']))
#         self.layers.append(Relu())
#         self.layers.append(Convolution(self.params['W4'], self.params['b4'], conv_param_list[3]['stride'], conv_param_list[3]['pad']))
#         self.layers.append(Relu())
#         self.layers.append(Pooling(pool_h=2, pool_w=2, stride=2))
#         self.layers.append(Convolution(self.params['W5'], self.params['b5'], conv_param_list[4]['stride'], conv_param_list[4]['pad']))
#         self.layers.append(Relu())
#         self.layers.append(Convolution(self.params['W6'], self.params['b6'], conv_param_list[5]['stride'], conv_param_list[5]['pad']))
#         self.layers.append(Relu())
#         self.layers.append(Pooling(pool_h=2, pool_w=2, stride=2))
#         self.layers.append(Affine(self.params['W7'], self.params['b7']))
#         self.layers.append(Relu())
#         self.layers.append(Dropout(0.5))
#         self.layers.append(Affine(self.params['W8'], self.params['b8']))
#         self.layers.append(Dropout(0.5))

#         self.last_layer = SoftmaxWithLoss()

#     def predict(self, x, train_flg=False):
#         for layer in self.layers:
#             if isinstance(layer, Dropout):
#                 x = layer.forward(x, train_flg)
#             else:
#                 x = layer.forward(x)
#         return x

#     def loss(self, x, t):
#         y = self.predict(x, train_flg=True)
#         return self.last_layer.forward(y, t)

#     def accuracy(self, x, t, batch_size=100):
#         if t.ndim != 1:
#             t = np.argmax(t, axis=1)

#         acc = 0.0

#         for i in range(int(x.shape[0] / batch_size)):
#             tx = x[i*batch_size:(i+1)*batch_size]
#             tt = t[i*batch_size:(i+1)*batch_size]
#             y = self.predict(tx, train_flg=False)
#             y = np.argmax(y, axis=1)
#             acc += np.sum(y == tt)

#         return acc / x.shape[0]

#     def gradient(self, x, t):
#         # forward
#         self.loss(x, t)

#         # backward
#         dout = 1
#         dout = self.last_layer.backward(dout)

#         tmp_layers = self.layers.copy()
#         tmp_layers.reverse()
#         for layer in tmp_layers:
#             dout = layer.backward(dout)

#         # 결과 저장
#         grads = {}
#         for i, layer_idx in enumerate(range(0, len(self.layers), 2)):
#             grads['W' + str(i+1)] = self.layers[layer_idx].dW
#             grads['b' + str(i+1)] = self.layers[layer_idx].db

#         return grads

#     def save_params(self, file_name="params.pkl"):
#         params = {}
#         for key, val in self.params.items():
#             params[key] = val
#         with open(file_name, 'wb') as f:
#             pickle.dump(params, f)

#     def load_params(self, file_name="params.pkl"):
#         with open(file_name, 'rb') as f:
#             params = pickle.load(f)
#         for key, val in params.items():
#             self.params[key] = val

#         for i, layer_idx in enumerate(range(0, len(self.layers), 2)):
#             self.layers[layer_idx].W = self.params['W' + str(i+1)]
#             self.layers[layer_idx].b = self.params['b' + str(i+1)]

#--------------------

# coding: utf-8
from common.layers import *
import numpy as np
import pickle
import cv2

class DeepConvNet:
    """정확도 99% 이상의 고정밀 합성곱 신경망
    네트워크 구성은 아래와 같음
        conv - relu - conv- relu - pool -
        conv - relu - conv- relu - pool -
        conv - relu - conv- relu - pool -
        affine - relu - dropout - affine - dropout - softmax
    """
    def augment_image(self, image):
    # 회전, 이동, 확대/축소 등의 이미지 변환
        rows, cols = image.shape
        M_rotate = cv2.getRotationMatrix2D((cols/2, rows/2), np.random.uniform(-15, 15), 1)
        M_translate = np.float32([[1, 0, np.random.uniform(-3, 3)], [0, 1, np.random.uniform(-3, 3)]])
        rotated = cv2.warpAffine(image, M_rotate, (cols, rows))
        translated = cv2.warpAffine(rotated, M_translate, (cols, rows))
        return translated
    

        augmented_img = self.augment_image(img)
    


    def __init__(self, input_dim=(1, 28, 28),
                 conv_param_1={'filter_num': 16,
                               'filter_size': 3, 'pad': 1, 'stride': 1},
                 conv_param_2={'filter_num': 16,
                               'filter_size': 3, 'pad': 1, 'stride': 1},
                 conv_param_3={'filter_num': 32,
                               'filter_size': 3, 'pad': 1, 'stride': 1},
                 conv_param_4={'filter_num': 32,
                               'filter_size': 3, 'pad': 2, 'stride': 1},
                 conv_param_5={'filter_num': 64,
                               'filter_size': 3, 'pad': 1, 'stride': 1},
                 conv_param_6={'filter_num': 64,
                               'filter_size': 3, 'pad': 1, 'stride': 1},
                 hidden_size=50, output_size=10):
        # 가중치 초기화===========
        # 각 층의 뉴런 하나당 앞 층의 몇 개 뉴런과 연결되는가（TODO: 자동 계산되게 바꿀 것）
        pre_node_nums = np.array(
            [1*3*3, 16*3*3, 16*3*3, 32*3*3, 32*3*3, 64*3*3, 64*4*4, hidden_size])
        weight_init_scales = np.sqrt(2.0 / pre_node_nums)  # ReLU를 사용할 때의 권장 초깃값

        self.params = {}
        pre_channel_num = input_dim[0]
        for idx, conv_param in enumerate([conv_param_1, conv_param_2, conv_param_3, conv_param_4, conv_param_5, conv_param_6]):
            self.params['W' + str(idx+1)] = weight_init_scales[idx] * np.random.randn(
                conv_param['filter_num'], pre_channel_num, conv_param['filter_size'], conv_param['filter_size'])
            self.params['b' + str(idx+1)] = np.zeros(conv_param['filter_num'])
            pre_channel_num = conv_param['filter_num']
        self.params['W7'] = weight_init_scales[6] * \
            np.random.randn(64*4*4, hidden_size)
        self.params['b7'] = np.zeros(hidden_size)
        self.params['W8'] = weight_init_scales[7] * \
            np.random.randn(hidden_size, output_size)
        self.params['b8'] = np.zeros(output_size)

        # 계층 생성===========
        self.layers = []
        self.layers.append(Convolution(self.params['W1'], self.params['b1'],
                           conv_param_1['stride'], conv_param_1['pad']))
        self.layers.append(Relu())
        self.layers.append(Convolution(self.params['W2'], self.params['b2'],
                           conv_param_2['stride'], conv_param_2['pad']))
        self.layers.append(Relu())
        self.layers.append(Pooling(pool_h=2, pool_w=2, stride=2))
        self.layers.append(Convolution(self.params['W3'], self.params['b3'],
                           conv_param_3['stride'], conv_param_3['pad']))
        self.layers.append(Relu())
        self.layers.append(Convolution(self.params['W4'], self.params['b4'],
                           conv_param_4['stride'], conv_param_4['pad']))
        self.layers.append(Relu())
        self.layers.append(Pooling(pool_h=2, pool_w=2, stride=2))
        self.layers.append(Convolution(self.params['W5'], self.params['b5'],
                           conv_param_5['stride'], conv_param_5['pad']))
        self.layers.append(Relu())
        self.layers.append(Convolution(self.params['W6'], self.params['b6'],
                           conv_param_6['stride'], conv_param_6['pad']))
        self.layers.append(Relu())
        self.layers.append(Pooling(pool_h=2, pool_w=2, stride=2))
        self.layers.append(Affine(self.params['W7'], self.params['b7']))
        self.layers.append(Relu())
        self.layers.append(Dropout(0.5))
        self.layers.append(Affine(self.params['W8'], self.params['b8']))
        self.layers.append(Dropout(0.5))

        self.last_layer = SoftmaxWithLoss()

    def predict(self, x, train_flg=False):
        for layer in self.layers:
            if isinstance(layer, Dropout):
                x = layer.forward(x, train_flg)
            else:
                x = layer.forward(x)
        return x

    def loss(self, x, t):
        y = self.predict(x, train_flg=True)
        return self.last_layer.forward(y, t)

    def accuracy(self, x, t, batch_size=100):
        if t.ndim != 1:
            t = np.argmax(t, axis=1)

        acc = 0.0

        for i in range(int(x.shape[0] / batch_size)):
            tx = x[i*batch_size:(i+1)*batch_size]
            tt = t[i*batch_size:(i+1)*batch_size]
            y = self.predict(tx, train_flg=False)
            y = np.argmax(y, axis=1)
            acc += np.sum(y == tt)

        return acc / x.shape[0]

    def gradient(self, x, t):
        # forward
        self.loss(x, t)

        # backward
        dout = 1
        dout = self.last_layer.backward(dout)

        tmp_layers = self.layers.copy()
        tmp_layers.reverse()
        for layer in tmp_layers:
            dout = layer.backward(dout)

        # 결과 저장
        grads = {}
        for i, layer_idx in enumerate((0, 2, 5, 7, 10, 12, 15, 18)):
            grads['W' + str(i+1)] = self.layers[layer_idx].dW
            grads['b' + str(i+1)] = self.layers[layer_idx].db

        return grads

    def save_params(self, file_name="params.pkl"):
        params = {}
        for key, val in self.params.items():
            params[key] = val
        with open(file_name, 'wb') as f:
            pickle.dump(params, f)

    def load_params(self, file_name="params.pkl"):
        with open(file_name, 'rb') as f:
            params = pickle.load(f)
        for key, val in params.items():
            self.params[key] = val

        for i, layer_idx in enumerate((0, 2, 5, 7, 10, 12, 15, 18)):
            self.layers[layer_idx].W = self.params['W' + str(i+1)]
            self.layers[layer_idx].b = self.params['b' + str(i+1)]

# coding: utf-8
# from common.gradient import numerical_gradient
# from common.layers import *
# import numpy as np
# import pickle


# class DeepConvNet:
#     """정확도 99% 이상의 고정밀 합성곱 신경망
#     네트워크 구성은 아래와 같음
#         conv - relu - conv- relu - pool -
#         conv - relu - conv- relu - pool -
#         conv - relu - conv- relu - pool -
#         affine - relu - dropout - affine - dropout - softmax
#     """

#     def __init__(self, input_dim=(1, 28, 28),
#                  conv_param_1={'filter_num': 16,
#                                'filter_size': 3, 'pad': 1, 'stride': 1},
#                  conv_param_2={'filter_num': 16,
#                                'filter_size': 3, 'pad': 1, 'stride': 1},
#                  conv_param_3={'filter_num': 32,
#                                'filter_size': 3, 'pad': 1, 'stride': 1},
#                  conv_param_4={'filter_num': 32,
#                                'filter_size': 3, 'pad': 2, 'stride': 1},
#                  conv_param_5={'filter_num': 64,
#                                'filter_size': 3, 'pad': 1, 'stride': 1},
#                  conv_param_6={'filter_num': 64,
#                                'filter_size': 3, 'pad': 1, 'stride': 1},
#                  hidden_size=50, output_size=10):
#         # 가중치 초기화===========
#         # 각 층의 뉴런 하나당 앞 층의 몇 개 뉴런과 연결되는가（TODO: 자동 계산되게 바꿀 것）
#         pre_node_nums = np.array(
#             [1*3*3, 16*3*3, 16*3*3, 32*3*3, 32*3*3, 64*3*3, 64*4*4, hidden_size])
#         weight_init_scales = np.sqrt(2.0 / pre_node_nums)  # ReLU를 사용할 때의 권장 초깃값

#         self.params = {}
#         pre_channel_num = input_dim[0]
#         for idx, conv_param in enumerate([conv_param_1, conv_param_2, conv_param_3, conv_param_4, conv_param_5, conv_param_6]):
#             self.params['W' + str(idx+1)] = weight_init_scales[idx] * np.random.randn(
#                 conv_param['filter_num'], pre_channel_num, conv_param['filter_size'], conv_param['filter_size'])
#             self.params['b' + str(idx+1)] = np.zeros(conv_param['filter_num'])
#             pre_channel_num = conv_param['filter_num']
#         self.params['W7'] = weight_init_scales[6] * \
#             np.random.randn(64*4*4, hidden_size)
#         self.params['b7'] = np.zeros(hidden_size)
#         self.params['W8'] = weight_init_scales[7] * \
#             np.random.randn(hidden_size, output_size)
#         self.params['b8'] = np.zeros(output_size)


#         # 계층 생성===========
#         self.layers = []
#         self.layers.append(Convolution(self.params['W1'], self.params['b1'],
#                            conv_param_1['stride'], conv_param_1['pad']))
#         self.layers.append(Relu())
#         self.layers.append(Convolution(self.params['W2'], self.params['b2'],
#                            conv_param_2['stride'], conv_param_2['pad']))
#         self.layers.append(Relu())
#         self.layers.append(Pooling(pool_h=2, pool_w=2, stride=2))
#         self.layers.append(Convolution(self.params['W3'], self.params['b3'],
#                            conv_param_3['stride'], conv_param_3['pad']))
#         self.layers.append(Relu())
#         self.layers.append(Convolution(self.params['W4'], self.params['b4'],
#                            conv_param_4['stride'], conv_param_4['pad']))
#         self.layers.append(Relu())
#         self.layers.append(Pooling(pool_h=2, pool_w=2, stride=2))
#         self.layers.append(Convolution(self.params['W5'], self.params['b5'],
#                            conv_param_5['stride'], conv_param_5['pad']))
#         self.layers.append(Relu())
#         self.layers.append(Convolution(self.params['W6'], self.params['b6'],
#                            conv_param_6['stride'], conv_param_6['pad']))
#         self.layers.append(Relu())
#         self.layers.append(Pooling(pool_h=2, pool_w=2, stride=2))
#         self.layers.append(Affine(self.params['W7'], self.params['b7']))
#         self.layers.append(Relu())
#         self.layers.append(Dropout(0.5))
#         self.layers.append(Affine(self.params['W8'], self.params['b8']))

#         self.last_layer = SoftmaxWithLoss()

#     def predict(self, x, train_flg=False):
#         for layer in self.layers:
#             if isinstance(layer, Dropout):
#                 x = layer.forward(x, train_flg)
#             else:
#                 x = layer.forward(x)
#         return x

#     def loss(self, x, t):
#         y = self.predict(x, train_flg=True)
#         return self.last_layer.forward(y, t)

#     def accuracy(self, x, t, batch_size=100):
#         if t.ndim != 1:  # one-hot encoding이라면
#             t = np.argmax(t, axis=1)

#         acc = 0.0

#         for i in range(int(x.shape[0] / batch_size)):
#             tx = x[i*batch_size:(i+1)*batch_size]
#             tt = t[i*batch_size:(i+1)*batch_size]
#             y = self.predict(tx, train_flg=False)
#             y = np.argmax(y, axis=1)
#             acc += np.sum(y == tt)

#         return acc / x.shape[0]

#     def numerical_gradient(self, x, t):
#         loss_W = lambda W: self.loss(x, t)

#         grads = {}
#         for idx in range(1, 9):
#             grads['W' + str(idx)] = numerical_gradient(
#                 loss_W, self.params['W' + str(idx)])
#             grads['b' + str(idx)] = numerical_gradient(
#                 loss_W, self.params['b' + str(idx)])

#         return grads

#     def gradient(self, x, t):
#         # forward
#         self.loss(x, t)

#         # backward
#         dout = 1
#         dout = self.last_layer.backward(dout)

#         tmp_layers = self.layers.copy()
#         tmp_layers.reverse()
#         for layer in tmp_layers:
#             dout = layer.backward(dout)

#         # 결과 저장
#         grads = {}
#         for i, layer_idx in enumerate((0, 2, 5, 7, 10, 12, 15, 18)):
#             if isinstance(self.layers[layer_idx], Convolution):
#                 grads['W' + str(i+1)] = self.layers[layer_idx].dW
#                 grads['b' + str(i+1)] = self.layers[layer_idx].db
#             elif isinstance(self.layers[layer_idx], Affine):
#                 grads['W' + str(i+1)] = self.layers[layer_idx].dW
#                 grads['b' + str(i+1)] = self.layers[layer_idx].db

#         grads['W7'] = self.layers[15].dW
#         grads['b7'] = self.layers[15].db

#         return grads