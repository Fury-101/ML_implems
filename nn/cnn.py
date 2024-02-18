GPU = True
if GPU:
    from cupyx.scipy import signal
    import cupy as np
else:
    import numpy as np
    from scipy import signal

class Layer(object):
    def __init__():
        pass

    def forward(self, input):
        '''
        Forward propagation method -- overridden by child classes. This should provide the output of the layer given the input.
        '''
        pass

    def backward(self, output_gradient, lr):
        '''
        Backwards propagation method -- overridden by child classes. This should provide the input gradient to the layer given 
        the output gradient of the layer.
        '''
        pass

class Dense(Layer):
    '''
    Dense layer which only holds the weights.
    '''
    def __init__(self, input_size, output_size):
        '''
        Initializes the Dense layer given input and output sizes. This may be different than what is seen in other libraries 
        where you provide the neuron count, which can be thought of as the output size.
        '''
        self.weights = np.random.randn(output_size, input_size)
        self.biases = np.random.randn(output_size, 1)

    def forward(self, input):
        '''
        Dense forward propagation method. Returns w•a + b.
        '''
        self.input = input
        return np.dot(self.weights, self.input) + self.biases

    def backward(self, output_gradient, lr):
        '''
        Dense backwards propagation method. Updates the weights according to the gradient, calculated from the output
        gradient. Note that we use the transposed weight matrix to go backwards.
        '''
        weights_gradient = np.dot(output_gradient, np.transpose(self.input))
        input_gradient = np.dot(np.transpose(self.weights), output_gradient)
        
        self.weights -= lr * weights_gradient
        self.biases -= lr * output_gradient
        return input_gradient

# TODO
# class MaxPooling2D(Layer):
#     def __init__(self, input_shape, pool_size):

class Conv2D(Layer):
                       #(depth, width, height) 
    def __init__(self, input_shape, kernel_size, num_kernels):
        self.input_shape = input_shape
        self.output_shape = (num_kernels, input_shape[2] - kernel_size + 1, input_shape[1] - kernel_size + 1)
        self.kernel_shape = (num_kernels, input_shape[0], kernel_size, kernel_size)
        self.kernels = np.random.random_sample(self.kernel_shape)
        self.biases = np.random.random_sample(self.output_shape)

    def forward(self, inp):
        self.input = inp
        self.output = np.copy(self.biases)

        for i in range(self.kernel_shape[0]):
            for j in range(self.input_shape[0]):
                blockSize = 256
                threads = (self.input_shape[0]*self.input_shape[1] + blockSize - 1) / blockSize

                # this convolution kernel works in a google colab but not in python for some reason:
                # https://colab.research.google.com/drive/19Mh2-tw1LaBY_81jKP5aeI2zJkD_X-Ol
                # important note: expects img to be a square
                convolution_kernel = np.RawKernel(r'''
                __global__
                    void convolve2d(const int img_size, const int kernel_size, const int output_size, float *img, float *kernel, float *output, const bool rot180=0, const bool full=false)
                    {
                    int index = blockIdx.x * blockDim.x + threadIdx.x;
                    int stride = blockDim.x * gridDim.x;

                    auto getRotIdx = [&] (int r_in, int c_in) { // Rotates the kernel 180 by reflecting the kernel by x and y
                        return (kernel_size - r_in - 1) * kernel_size + kernel_size - c_in - 1;
                    };
                    
                    for (int i = index; i < img_size * img_size; i += stride) {
                        int r = i/img_size;
                        int c = i%img_size;

                        if (r > img_size - kernel_size || c > img_size - kernel_size) {
                            return;
                        }

                        for (int i = 0; i<kernel_size*kernel_size; i++) {
                            int rk = i/kernel_size;
                            int ck = i%kernel_size;

                            if (rot180)
                                output[r*output_size + c] += kernel[getRotIdx(rk, ck)] * img[(r + rk)*img_size + (c + ck)];
                            else
                                output[r*output_size + c] += kernel[rk * kernel_size + ck] * img[(r + rk)*img_size + (c + ck)];
                        }
                    }
                }
                ''', 'convolve2d')
                # output_temp = np.zeros(self.output[i].shape)
                # convolution_kernel((blockSize,), (threads,), 
                #                     (self.input_shape[-1], 
                #                     self.kernel_shape[-1],
                #                     self.output_shape[-1],
                #                     self.input[j],
                #                     self.kernels[i][j],
                #                     output_temp
                #                     ))
                # self.output[i] += output_temp
                self.output[i] += signal.correlate2d(self.input[j], self.kernels[i][j], 'valid')
        return self.output
    
    def backward(self, output_gradient, lr):
        dCdK = np.zeros(self.kernel_shape)
        dCdX = np.zeros(self.input_shape)
        dCdB = np.zeros(output_gradient.shape)

        for i in range(self.kernel_shape[0]):
            dCdB[i] = output_gradient[i]
            for j in range(self.input_shape[0]):
                dCdK[i][j] = signal.correlate2d(self.input[j], output_gradient[i], 'valid')
                dCdX[j] += signal.convolve2d(output_gradient[i], self.kernels[i][j], 'full')
        
        self.kernels -= lr * dCdK
        self.biases -= lr * dCdB

        return dCdX

class Reshape(Layer):
    def __init__(self, input_shape, output_shape):
        self.input_shape = input_shape
        self.output_shape = output_shape

    def forward(self, inp):
        return np.reshape(inp, self.output_shape)
    
    def backward(self, output_gradient, lr):
        return np.reshape(output_gradient, self.input_shape)
    
class Activation(Layer):
    def __init__(self, activation, activation_gradient):
        self.activation = activation
        self.activation_prime = activation_gradient

    def forward(self, input):
        self.input = input
        return self.activation(self.input)

    def backward(self, output_gradient, learning_rate):
        return np.multiply(output_gradient, self.activation_prime(self.input))
    
class Sigmoid(Activation):
    def __init__(self):
        super().__init__(self.sigmoid, self.gradient)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def gradient(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))

def binary_cross_entropy(y, y_pred):
    return -np.mean(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))

def binary_cross_entropy_gradient(y, y_pred):
    return ((1 - y) / (1 - y_pred) - y / y_pred) / np.size(y)