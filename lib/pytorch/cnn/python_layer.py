import math
import numpy as np
import functools as ft

class Conv2D(object):
    def __init__(self, shape, output_channels, ksize=3, stride=1, method='VALID'):
        self.input_shape = shape
        self.output_channels = output_channels
        self.input_channels = shape[-1]
        self.batchsize = shape[0]
        self.stride = stride
        self.ksize = ksize
        self.method = method

        weights_scale = math.sqrt(ft.reduce(lambda x, y: x * y, shape) / self.output_channels)
        self.weights = np.random.standard_normal(
            (ksize, ksize, self.input_channels, self.output_channels)) / weights_scale
        self.bias = np.random.standard_normal(self.output_channels) / weights_scale

        if method == 'VALID':
            self.eta = np.zeros((shape[0],
                                 int((shape[1] - ksize + 1) / self.stride),
                                 int((shape[1] - ksize + 1) / self.stride),
                                 self.output_channels))

        if method == 'SAME':
            self.eta = np.zeros((shape[0],
                                 int(shape[1]/self.stride),
                                 int(shape[2]/self.stride),
                                 self.output_channels))

        self.w_gradient = np.zeros(self.weights.shape)
        self.b_gradient = np.zeros(self.bias.shape)
        self.output_shape = self.eta.shape

        if (shape[1] - ksize) % stride != 0:
            print('input tensor width can\'t fit stride')
        if (shape[2] - ksize) % stride != 0:
            print('input tensor height can\'t fit stride')

    def forward(self, x):
        col_weights = self.weights.reshape([-1, self.output_channels])
        if self.method == 'SAME':
            x = np.pad(x, (
                (0, 0), (int(self.ksize / 2), int(self.ksize / 2)), (int(self.ksize / 2), int(self.ksize / 2)), (0, 0)),
                       'constant', constant_values=0)

        self.col_image = []
        conv_out = np.zeros(self.eta.shape)
        for i in range(self.batchsize):
            img_i = x[i][np.newaxis, :]
            self.col_image_i = self._im2col(img_i, self.ksize, self.stride)
            conv_out[i] = np.reshape(np.dot(self.col_image_i, col_weights) + self.bias, self.eta[0].shape)
            self.col_image.append(self.col_image_i)
        self.col_image = np.array(self.col_image)
        return conv_out

    def gradient(self, eta):
        self.eta = eta
        col_eta = np.reshape(eta, [self.batchsize, -1, self.output_channels])

        for i in range(self.batchsize):
            self.w_gradient += np.dot(self.col_image[i].T, col_eta[i]).reshape(self.weights.shape)
        self.b_gradient += np.sum(col_eta, axis=(0, 1))

        # deconv of padded eta with flippd kernel to get next_eta
        if self.method == 'VALID':
            pad_eta = np.pad(self.eta, (
                (0, 0), (self.ksize - 1, self.ksize - 1), (self.ksize - 1, self.ksize - 1), (0, 0)),
                             'constant', constant_values=0)

        if self.method == 'SAME':
            pad_eta = np.pad(self.eta, (
                (0, 0), (self.ksize / 2, self.ksize / 2), (self.ksize / 2, self.ksize / 2), (0, 0)),
                             'constant', constant_values=0)

        flip_weights = np.flipud(np.fliplr(self.weights))
        flip_weights = flip_weights.swapaxes(2, 3)
        col_flip_weights = flip_weights.reshape([-1, self.input_channels])
        col_pad_eta = np.array([self._im2col(pad_eta[i][np.newaxis, :], self.ksize, self.stride) for i in range(self.batchsize)])
        next_eta = np.dot(col_pad_eta, col_flip_weights)
        next_eta = np.reshape(next_eta, self.input_shape)
        return next_eta

    def backward(self, alpha=0.00001, weight_decay=0.0004):
        # weight_decay = L2 regularization
        self.weights *= (1 - weight_decay)
        self.bias *= (1 - weight_decay)
        self.weights -= alpha * self.w_gradient
        self.bias -= alpha * self.bias

        self.w_gradient = np.zeros(self.weights.shape)
        self.b_gradient = np.zeros(self.bias.shape)

    def _im2col(self, image, ksize, stride):
        # image is a 4d tensor([batchsize, width ,height, channel])
        image_col = []
        for i in range(0, image.shape[1] - ksize + 1, stride):
            for j in range(0, image.shape[2] - ksize + 1, stride):
                col = image[:, i:i + ksize, j:j + ksize, :].reshape([-1])
                image_col.append(col)
        image_col = np.array(image_col)

        return image_col

class Relu(object):
    def __init__(self, shape):
        self.eta = np.zeros(shape)
        self.x = np.zeros(shape)
        self.output_shape = shape

    def forward(self, x):
        self.x = x
        return np.maximum(x, 0)

    def gradient(self, eta):
        self.eta = eta
        self.eta[self.x<0]=0
        return self.eta

class MaxPooling(object):
    def __init__(self, shape, ksize=2, stride=2):
        self.input_shape = shape
        self.ksize = ksize
        self.stride = stride
        self.output_channels = shape[-1]
        self.index = np.zeros(shape)
        self.output_shape = [shape[0], int(shape[1] / self.stride), int(shape[2] / self.stride), self.output_channels]

    def forward(self, x):
        out = np.zeros([x.shape[0], int(x.shape[1] / self.stride), int(x.shape[2] / self.stride), self.output_channels])

        for b in range(x.shape[0]):
            for c in range(self.output_channels):
                for i in range(0, x.shape[1], self.stride):
                    for j in range(0, x.shape[2], self.stride):
                        out[b, int(i / self.stride), int(j / self.stride), c] = np.max(
                            x[b, i:i + self.ksize, j:j + self.ksize, c])
                        index = np.argmax(x[b, i:i + self.ksize, j:j + self.ksize, c])
                        self.index[b, i+int(index/self.stride), j + index % self.stride, c] = 1
        return out

    def gradient(self, eta):
        return np.repeat(np.repeat(eta, self.stride, axis=1), self.stride, axis=2) * self.index

class FullyConnect(object):
    def __init__(self, shape, output_num=2):
        self.input_shape = shape
        self.batchsize = shape[0]

        input_len = ft.reduce(lambda x, y: x * y, shape[1:])

        self.weights = np.random.standard_normal((input_len, output_num))/100
        self.bias = np.random.standard_normal(output_num)/100

        self.output_shape = [self.batchsize, output_num]
        self.w_gradient = np.zeros(self.weights.shape)
        self.b_gradient = np.zeros(self.bias.shape)

    def forward(self, x):
        self.x = x.reshape([self.batchsize, -1])
        output = np.dot(self.x, self.weights)+self.bias
        return output

    def gradient(self, eta):
        for i in range(eta.shape[0]):
            col_x = self.x[i][:, np.newaxis]
            eta_i = eta[i][:, np.newaxis].T
            self.w_gradient += np.dot(col_x, eta_i)
            self.b_gradient += eta_i.reshape(self.bias.shape)

        next_eta = np.dot(eta, self.weights.T)
        next_eta = np.reshape(next_eta, self.input_shape)

        return next_eta

    def backward(self, alpha=0.00001, weight_decay=0.0004):
        # weight_decay = L2 regularization
        self.weights *= (1 - weight_decay)
        self.bias *= (1 - weight_decay)
        self.weights -= alpha * self.w_gradient
        self.bias -= alpha * self.bias
        # zero gradient
        self.w_gradient = np.zeros(self.weights.shape)
        self.b_gradient = np.zeros(self.bias.shape)

class Softmax(object):
    def __init__(self, shape):
        self.softmax = np.zeros(shape)
        self.eta = np.zeros(shape)
        self.batchsize = shape[0]

    def cal_loss(self, prediction, label):
        self.label = label
        self.prediction = prediction
        self.predict(prediction)
        self.loss = 0
        for i in range(self.batchsize):
            self.loss += np.log(np.sum(np.exp(prediction[i]))) - prediction[i, label[i]]

        return self.loss

    def predict(self, prediction):
        exp_prediction = np.zeros(prediction.shape)
        self.softmax = np.zeros(prediction.shape)
        for i in range(self.batchsize):
            prediction[i, :] -= np.max(prediction[i, :])
            exp_prediction[i] = np.exp(prediction[i])
            self.softmax[i] = exp_prediction[i]/np.sum(exp_prediction[i])
        return self.softmax

    def gradient(self):
        self.eta = self.softmax.copy()
        for i in range(self.batchsize):
            self.eta[i, self.label[i]] -= 1
        return self.eta