import numpy as np
import random as rand

class NeuralNewtwork:
    def __init__(
        self,
        in_shape,
        out_shape,
        activation_function,
        name=None,
        weight=None,
        use_bias=False,
        bias=None,
    ):
        self.__W = None
        self.__B = None
        self.__layes_params_info = {}

        self.construct_layer(
            in_shape=in_shape,
            out_shape=out_shape,
            activation_function=activation_function,
            name=name,
            weight=weight,
            use_bias=use_bias,
            bias=bias,
        )

        self.__key_paring_activating = {
            "relu": {0: self.__relu_backward, 1: self.__relu_forward},
            "sigmoid": {0: self.__sigmoid_backward, 1: self.__sigmoid_forward},
            "htanh": {0: self.__tanh_backward, 1: self.__tanh_forward},
        }

        self.__cache = {}

    def __relu_forward(self, z):
        return np.maximum(z, 0)

    def __sigmoid_forward(self, z):
        return 1 / (1 + np.exp(-z))

    def __tanh_forward(self, z):
        return np.tanh(z)

    def __relu_backward(self, z):
        dz = np.zeros_like(z)
        dz[z > 0] = 1
        return dz

    def __sigmoid_backward(self, z):
        s = self.__sigmoid_forward(z)
        return s * (1 - s)

    def __tanh_backward(self, z):
        return 1 - np.tanh(z) ** 2

    def __softmax(self, z):
        z_shifted = z - np.max(z, axis=1, keepdims=True)
        exp_z = np.exp(z_shifted)
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def __check_bias(self, bias, out_shape):
        return type(bias) != np.ndarray and bias.shape[0] == out_shape

    def __check_weights(self, weight, in_shape, out_shape):
        is_type_valid = type(weight) == np.ndarray

        if (is_type_valid):
            w, h = weight.shape
            if in_shape != h or out_shape != w:
                return False
            return True

        return False

    def get_weights(self):
        return self.__W

    def get_bias(self):
        return self.__B

    def layes_params_info(self):
        return self.__layes_params_info

    def construct_layer(
        self,
        in_shape,
        out_shape,
        activation_function,
        name=None,
        weight=None,
        use_bias=False,
        bias=None,
    ):
        if activation_function not in ['relu', 'sigmoid', 'htanh']:
            raise ValueError(
                "Invalid activation function: activation function must be one of theese relu, sigmoid or htanh")

        if name == None:
            name = f'nn_{int(rand.random() * 1000000)}'

        self.__layes_params_info['name'] = name
        self.__layes_params_info['in_shape'] = in_shape
        self.__layes_params_info['out_shape'] = out_shape
        self.__layes_params_info['activation_func'] = activation_function

        # initialize the weights and bias
        if weight is None:
            activation = activation_function

            if activation == "relu":
                # He Initialization
                self.__W = np.random.randn(out_shape, in_shape) * np.sqrt(2 / in_shape)

            elif activation in ["sigmoid", "htanh"]:
                # Xavier Initialization
                self.__W = np.random.randn(out_shape, in_shape) * np.sqrt(1 / in_shape)

            else:
                self.__W = np.random.randn(out_shape, in_shape) * 0.01

        else:
            if self.__check_weights(weight, in_shape, out_shape):
                self.__W = weight
            else:
                print("Invalid weights, generate random one")
                self.__W = np.random.randn(out_shape, in_shape) * 0.01

        self.__layes_params_info['weight'] = self.__W

        if use_bias:
            if bias == None:
                self.__B = np.random.rand(out_shape, 1)
            else:
                if self.__check_bias(bias, out_shape):
                    self.__B = bias
                else:
                    print("Invalid bias, generate random one")
                    self.__B = np.random.rand(out_shape, 1)
        
        self.__layes_params_info['bias'] = self.__B
        
    def calculate_z(self, x):
        if self.__B is None:
            z = np.dot(self.__W, x)
        else:
            z = np.dot(self.__W, x) + self.__B
        return z
    
    def forward(self, a):
        # previous a
        self.__cache['a'] = a
        
        Z = self.calculate_z(a)
        self.__cache['Z'] = Z
        
        act_func_name = self.__layes_params_info['activation_func']
        func = self.__key_paring_activating[act_func_name][1]
        A = func(Z)
        # calculated new a as A
        
        self.__cache['A'] = A
        
        return A
    
    def backward(self, dA):
        a_prev = self.__cache['a']
        Z = self.__cache['Z']
        
        act_func_name = self.__layes_params_info['activation_func']
        d_activation = self.__key_paring_activating[act_func_name][0]

        m = a_prev.shape[1]

        dZ = dA * d_activation(Z)
        dW = (1/m) * np.dot(dZ, a_prev.T)
        db = (1/m) * np.sum(dZ, axis=1, keepdims=True) if self.__B is not None else None
        dA_prev = np.dot(self.__W.T, dZ)

        return dA_prev, dW, db
    
    # [TODO: kerjain biar vanished grdient descentnya bisa ke solve lex]
    def get_dA_boosted(self, prefError):
        pass
    
class Convolution2D:
    def __init__(self, stride, kernel_size = (2,2), activation="relu"):
        
        self.stride = stride
        self.activation=activation
        
        self.__key_paring_activating = {
            "relu": {0: self.__relu_backward, 1: self.__relu_forward},
            "sigmoid": {0: self.__sigmoid_backward, 1: self.__sigmoid_forward},
            "htanh": {0: self.__tanh_backward, 1: self.__tanh_forward},
        }
        
        self.__cache = {}
        
        self.__kernel_W = np.random.random(size=kernel_size)
    
    def __relu_forward(self, z):
        return np.maximum(z, 0)

    def __sigmoid_forward(self, z):
        return 1 / (1 + np.exp(-z))

    def __tanh_forward(self, z):
        return np.tanh(z)

    def __relu_backward(self, z):
        dz = np.zeros_like(z)
        dz[z > 0] = 1
        return dz

    def __sigmoid_backward(self, z):
        s = self.__sigmoid_forward(z)
        return s * (1 - s)

    def __tanh_backward(self, z):
        return 1 - np.tanh(z) ** 2
    
    def __conv2d(self, input_matrix, stride):
        w_i, h_i = input_matrix.shape
        w_k, h_k = self.__kernel_W.shape
        # w_b, h_b = self.__kernel_B.shape

        if w_i < w_k or h_i < h_k:
            raise ValueError("Invalid size: kernel wight size larger than input matrix")

        out_w = ((w_i - w_k) // stride) + 1
        out_h = ((h_i - h_k) // stride) + 1
        
        output = np.zeros((out_w, out_h))

        for w in range(0, w_i - w_k + 1, stride):
            for h in range(0, h_i - h_k + 1, stride):
                selected_input_part = input_matrix[w:w+w_k, h:h+h_k]
                # conv_value = np.sum(selected_input_part * self.__kernel_W) + self.__kernel_B
                conv_value = np.sum(selected_input_part * self.__kernel_W)
                output[w // stride, h // stride] = conv_value

        return output
    
    def forward(self, a):
        # previous a
        self.__cache['a'] = a
        
        Z = self.__conv2d(a, self.stride)
        self.__cache['Z'] = Z
        
        act_func_name = self.activation
        func = self.__key_paring_activating[act_func_name][1]
        A = func(Z)
        # calculated new a as A
        
        self.__cache['A'] = A
        
        return A
    
    def backward(self, dA):
        a_prev = self.__cache['a']
        Z = self.__cache['Z']
        stride = self.stride

        # 🔹 activation backward
        act_func_name = self.activation
        act_backward = self.__key_paring_activating[act_func_name][0]
        dZ = dA * act_backward(Z)

        w_k, h_k = self.__kernel_W.shape
        
        # for padding letter
        # w_i, h_i = a_prev.shape

        dW = np.zeros_like(self.__kernel_W)
        dA_prev = np.zeros_like(a_prev)

        out_w, out_h = dZ.shape

        for w in range(out_w):
            for h in range(out_h):
                vert_start = w * stride
                vert_end = vert_start + w_k
                horiz_start = h * stride
                horiz_end = horiz_start + h_k

                input_patch = a_prev[vert_start:vert_end, horiz_start:horiz_end]

                dW += input_patch * dZ[w, h]

                dA_prev[vert_start:vert_end, horiz_start:horiz_end] += self.__kernel_W * dZ[w, h]

        self.__cache['dW'] = dW
        self.__cache['dA_prev'] = dA_prev

        return dA_prev

class Pooling2D:
    def __init__(self, kernel_size=(2,2), stride=2, mode="max"):
        self.kernel_size = kernel_size
        self.stride = stride
        # "max" atau "mean"
        self.mode = mode
        self.__cache = {}
        
    def forward(self, a):
        self.__cache['a'] = a
        
        w_i, h_i = a.shape
        w_k, h_k = self.kernel_size
        
        out_w = ((w_i - w_k) // self.stride) + 1
        out_h = ((h_i - h_k) // self.stride) + 1
        
        output = np.zeros((out_w, out_h))
        
        for w in range(out_w):
            for h in range(out_h):
                vert_start = w * self.stride
                vert_end = vert_start + w_k
                horiz_start = h * self.stride
                horiz_end = horiz_start + h_k
                
                window = a[vert_start:vert_end, horiz_start:horiz_end]
                
                if self.mode == "max":
                    output[w, h] = np.max(window)
                elif self.mode == "mean":
                    output[w, h] = np.mean(window)
        
        self.__cache['output'] = output
        return output
    
    def backward(self, dA):
        a_prev = self.__cache['a']
        
        # for padding letter
        # w_i, h_i = a_prev.shape
        
        w_k, h_k = self.kernel_size
        
        dA_prev = np.zeros_like(a_prev)
        
        out_w, out_h = dA.shape
        
        for w in range(out_w):
            for h in range(out_h):
                vert_start = w * self.stride
                vert_end = vert_start + w_k
                horiz_start = h * self.stride
                horiz_end = horiz_start + h_k
                
                if self.mode == "max":
                    window = a_prev[vert_start:vert_end, horiz_start:horiz_end]
                    max_val = np.max(window)
                    
                    mask = (window == max_val)
                    dA_prev[vert_start:vert_end, horiz_start:horiz_end] += mask * dA[w, h]
                
                elif self.mode == "mean":
                    gradient = dA[w, h] / (w_k * h_k)
                    dA_prev[vert_start:vert_end, horiz_start:horiz_end] += np.ones((w_k, h_k)) * gradient
        
        return dA_prev

class nnSystem:
    
    def __init__(self, learning_rate=0.01):
        self.__layers = []
        self.learning_rate = learning_rate
        self.loss_history = []

    def add(
        self,
        in_shape,
        out_shape,
        activation_function,
        name=None,
        use_bias=True,
    ):
        layer = NeuralNewtwork(
            in_shape=in_shape,
            out_shape=out_shape,
            activation_function=activation_function,
            name=name,
            use_bias=use_bias,
        )
        self.__layers.append(layer)

    def forward(self, X):
        A = X
        for layer in self.__layers:
            A = layer.forward(A)
        return A

    def loss(self, y_pred, y_true):
        return np.mean((y_pred - y_true) ** 2)

    def loss_derivative(self, y_pred, y_true):
        return 2 * (y_pred - y_true) / y_true.size

    def backward(self, y_pred, y_true):
        dA = self.loss_derivative(y_pred, y_true)

        grads = []

        for layer in reversed(self.__layers):
            dA, dW, db = layer.backward(dA)
            grads.append((layer, dW, db))

        for layer, dW, db in grads:
            layer._NeuralNewtwork__W -= self.learning_rate * dW
            if db is not None:
                layer._NeuralNewtwork__B -= self.learning_rate * db

    def train(self, X, Y, epochs=1000):
        for epoch in range(epochs):
            y_pred = self.forward(X)
            loss = self.loss(y_pred, Y)
            self.loss_history.append(loss)

            self.backward(y_pred, Y)

            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss}")
