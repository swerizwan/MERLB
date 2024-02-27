import numpy as np
import keras.activations
from keras import backend as K
from tensorflow.keras.layers import Layer
from keras import activations
from keras import initializers
from keras import regularizers
from keras import constraints

def init_orthogonal_merlb_cores(list_shape_input, list_shape_output, list_ranks):
    """
    Initialize orthogonal tensor train (MERLB) cores.

    :param list_shape_input: List containing the input shapes for each dimension.
    :param list_shape_output: List containing the output shapes for each dimension.
    :param list_ranks: List containing the ranks for each dimension.
    :return: Initialized MERLB cores.
    """
    list_shape_input = np.array(list_shape_input)
    list_shape_output = np.array(list_shape_output)
    list_ranks = np.array(list_ranks)
    cores_arr_len = np.sum(list_shape_input * list_shape_output *
                           list_ranks[1:] * list_ranks[:-1])
    cores_arr = np.zeros(cores_arr_len)
    rv = 1

    d = list_shape_input.shape[0]
    rng = np.random
    shapes = [None] * d
    tall_shapes = [None] * d
    cores = [None] * d
    counter = 0

    for k in range(list_shape_input.shape[0]):
        shapes[k] = [list_ranks[k], list_shape_input[k], list_shape_output[k], list_ranks[k + 1]]
        tall_shapes[k] = (np.prod(shapes[k][:3]), shapes[k][3])
        cores[k] = np.dot(rv, rng.randn(shapes[k][0], np.prod(shapes[k][1:])))
        cores[k] = cores[k].reshape(tall_shapes[k])

        if k < list_shape_input.shape[0] - 1:
            cores[k], rv = np.linalg.qr(cores[k])
        cores_arr[counter:(counter + cores[k].size)] = cores[k].flatten()
        counter += cores[k].size

    glarot_style = (np.prod(list_shape_input) * np.prod(list_ranks)) ** (1.0 / list_shape_input.shape[0])
    return (0.1 / glarot_style) * cores_arr

class MERLB_Layer(Layer):
    """
    Tensor Train (MERLB) Layer implementation.
    """

    def __init__(self, list_shape_input, list_shape_output, list_ranks,
                 bias_use=True,
                 activation='linear',
                 initializer_kernel='glorot_uniform',
                 initializer_bias='zeros',
                 regularizer_kernel=None,
                 regularizer_bias=None,
                 regularizer_activity=None,
                 constraint_kernel=None,
                 constraint_bias=None,
                 debug=False,
                 seed_init=11111986,
                 **kwargs):
        """
        Initialize the MERLB layer.

        :param list_shape_input: List containing the input shapes for each dimension.
        :param list_shape_output: List containing the output shapes for each dimension.
        :param list_ranks: List containing the ranks for each dimension.
        :param bias_use: Boolean indicating whether to use bias.
        :param activation: Activation function for the layer.
        :param initializer_kernel: Initializer for the kernel weights.
        :param initializer_bias: Initializer for the bias.
        :param regularizer_kernel: Regularizer for the kernel weights.
        :param regularizer_bias: Regularizer for the bias.
        :param regularizer_activity: Regularizer for the layer activity.
        :param constraint_kernel: Constraint for the kernel weights.
        :param constraint_bias: Constraint for the bias.
        :param debug: Boolean indicating whether to enable debugging.
        :param seed_init: Seed for random initialization.
        :param kwargs: Additional keyword arguments.
        """
        list_shape_input = np.array(list_shape_input)
        list_shape_output = np.array(list_shape_output)
        list_ranks = np.array(list_ranks)

        self.list_shape_input = list_shape_input
        self.list_shape_output = list_shape_output
        self.list_ranks = list_ranks
        self.num_dim = list_shape_input.shape[0]  # length of the train
        self.bias_use = bias_use
        self.activation = keras.activations.get(activation)

        self.initializer_kernel = initializers.get(initializer_kernel)
        self.initializer_bias = initializers.get(initializer_bias)

        self.regularizer_kernel = regularizers.get(regularizer_kernel)
        self.regularizer_bias = regularizers.get(regularizer_bias)

        self.regularizer_activity = regularizers.get(regularizer_activity)

        self.constraint_kernel = constraints.get(constraint_kernel)
        self.constraint_bias = constraints.get(constraint_bias)

        self.debug = debug
        self.seed_init = seed_init

        super(MERLB_Layer, self).__init__(**kwargs)


    def build(self, input_shape):
        """
        Build method for the MERLB_Layer.

        :param input_shape: Shape of the input tensor.
        """
        num_inputs = int(np.prod(input_shape[1::]))

        # Validate input dimensions
        if np.prod(self.list_shape_input) != num_inputs:
            raise ValueError("The size of the input tensor should equal to the number of input neurons %d." % num_inputs)
        if self.list_shape_input.shape[0] != self.list_shape_output.shape[0]:
            raise ValueError("The number of input and output dimensions should be the same.")
        if self.list_ranks.shape[0] != self.list_shape_output.shape[0] + 1:
            raise ValueError("The number of MERLB-ranks should be 1 + the number of dimensions.")

        # Print information if in debug mode
        if self.debug:
            print('list_shape_input = ' + str(self.list_shape_input))
            print('list_shape_output = ' + str(self.list_shape_output))
            print('list_ranks = ' + str(self.list_ranks))

        # Set random seed for initialization
        if self.seed_init is None:
            self.seed_init = 11111986
        np.random.seed(self.seed_init)

        # Calculate total length of the MERLB cores
        total_length = np.sum(self.list_shape_input * self.list_shape_output *
                            self.list_ranks[1:] * self.list_ranks[:-1])

        # Initialize kernel weights as trainable parameters
        self.kernel = self.add_weight(name='kernel', shape=(total_length,))

        # Initialize bias if used
        if self.bias_use:
            self.bias = self.add_weight(shape=(np.prod(self.list_shape_output), ),
                                        initializer=self.initializer_bias,
                                        name='bias',
                                        regularizer=self.regularizer_bias,
                                        constraint=self.constraint_bias)

        # Initialize indices, shapes, and cores for MERLB decomposition
        self.inds = np.zeros(self.num_dim).astype('int32')
        self.shapes = np.zeros((self.num_dim, 2)).astype('int32')
        self.cores = [None] * self.num_dim

        # Iterate through dimensions to set up MERLB decomposition
        for k in range(self.num_dim - 1, -1, -1):
            self.shapes[k] = (self.list_shape_input[k] * self.list_ranks[k + 1],
                            self.list_ranks[k] * self.list_shape_output[k])
            self.cores[k] = self.kernel[self.inds[k]:self.inds[k] + np.prod(self.shapes[k])]
            if 0 < k:
                self.inds[k - 1] = self.inds[k] + np.prod(self.shapes[k])

        # Print shapes if in debug mode
        if self.debug:
            print('self.shapes = ' + str(self.shapes))

        # Set additional attributes for compression analysis
        self.MERLB_size = total_length
        self.full_size = (np.prod(self.list_shape_input) * np.prod(self.list_shape_output))
        self.compress_factor = 1. * self.MERLB_size / self.full_size
        print('Compression factor = ' + str(self.MERLB_size) + ' / ' \
            + str(self.full_size) + ' = ' + str(self.compress_factor))

    def call(self, x, mask=None):
        """
        Forward pass of the MERLB_Layer.

        :param x: Input tensor.
        :param mask: Not used.
        :return: Output tensor.
        """
        res = x
        for k in range(self.num_dim - 1, -1, -1):
            res = K.dot(K.reshape(res, (-1, self.shapes[k][0])),
                        K.reshape(self.cores[k], self.shapes[k])
                        )
            res = K.transpose(
                K.reshape(res, (-1, self.list_shape_output[k]))
            )

        res = K.transpose(K.reshape(res, (-1, K.shape(x)[0])))

        if self.bias_use:
            res = K.bias_add(res, self.bias)
        if self.activation is not None:
            res = self.activation(res)

        return res

    def generate_shape_output(self, input_shape):
        """
        Generate the shape of the output tensor based on the input shape.

        :param input_shape: Shape of the input tensor.
        :return: Shape of the output tensor.
        """
        return (input_shape[0], np.prod(self.list_shape_output))

    def generate_output_shape(self, input_shape):
        """
        Generate the output shape based on the input shape.

        :param input_shape: Shape of the input tensor.
        :return: Shape of the output tensor.
        """
        return (input_shape[0], np.prod(self.list_shape_output))

    def generate_orthogonal_value_list_cores(self):
        """
        Generate orthogonal values for the MERLB decomposition.

        :return: Orthogonal values for the MERLB decomposition.
        """
        cores_arr_len = np.sum(self.list_shape_input * self.list_shape_output *
                               self.list_ranks[1:] * self.list_ranks[:-1])
        cores_arr = np.zeros(cores_arr_len)
        rv = 1

        d = self.list_shape_input.shape[0]
        rng = np.random
        shapes = [None] * d
        tall_shapes = [None] * d
        cores = [None] * d
        counter = 0

        for k in range(self.list_shape_input.shape[0]):
            shapes[k] = [self.list_ranks[k], self.list_shape_input[k], self.list_shape_output[k], self.list_ranks[k + 1]]

            tall_shapes[k] = (np.prod(shapes[k][:3]), shapes[k][3])

            cores[k] = np.dot(rv, rng.randn(shapes[k][0], np.prod(shapes[k][1:])))

            cores[k] = cores[k].reshape(tall_shapes[k])

            if k < self.list_shape_input.shape[0] - 1:
                cores[k], rv = np.linalg.qr(cores[k])
            cores_arr[counter:(counter + cores[k].size)] = cores[k].flatten()
            counter += cores[k].size

        glarot_style = (np.prod(self.list_shape_input) * np.prod(self.list_ranks)) ** (1.0 / self.list_shape_input.shape[0])
        return (0.1 / glarot_style) * cores_arr


    def generate_weight_full(self):
        """
        Generate the full weight matrix using the MERLB decomposition.

        :return: Full weight matrix.
        """
        res = np.identity(np.prod(self.list_shape_input))
        for k in range(self.num_dim - 1, -1, -1):
            res = np.dot(np.reshape(res, (-1, self.shapes[k][0])),
                         np.reshape(self.cores[k], self.shapes[k])
                         )
            res = np.transpose(
                np.reshape(res, (-1, self.list_shape_output[k]))
            )
        res = np.transpose(np.reshape(res, (-1, np.shape(res)[0])))

        if self.bias_use:
            res = K.bias_add(res, self.bias)
        if self.activation is not None:
            res = self.activation(res)

        return res