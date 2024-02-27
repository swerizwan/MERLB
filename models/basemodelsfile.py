from tensorflow.keras.layers import Input, Dense, Conv1D, Conv2D, MaxPooling1D, MaxPooling2D, Dropout, LSTM, \
    BatchNormalization, Flatten, TimeDistributed, GlobalAveragePooling2D, Activation, Add, ZeroPadding2D
from tensorflow.keras.models import Model


def normalization_relu_add(x):
    """gives the input X a Batch Normalization and ReLU layer.

    Input to these layers, parameter x

    :return: x with applied Batch Norm and ReLU
    """
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x


def residual_block_add(x, filters_num_in, filters_num_out, id_block):
    """Using the specified number of filters, generated_values creates a residual block for x.

    Paragraph X What goes into these layers:filters_num_in parameter: How many bottleneck filters 
    should be used in the response block?Filters_num_out parameter: The number of filters in the 
    output should match the number of filters in x.id_block parameter: The block's identification 
    number (for saving and loading weight) is:return: x with the additional res block
    """
    tensor_input = x
    x = Conv2D(filters_num_in, (1, 1), name=id_block + '_1')(x)
    x = normalization_relu_add(x)
    x = Conv2D(filters_num_in, (3, 3), padding='same', name=id_block + '_2')(x)
    x = normalization_relu_add(x)
    x = Conv2D(filters_num_out, (1, 1), name=id_block + '_3')(x)
    x = BatchNormalization()(x)
    x = Add()([tensor_input, x])
    x = Activation('relu')(x)
    return x


def convolutional_block_add(x, filters_num_in, filters_num_out, id_block):
    """produced_values creates a conv block for x with the specified number of filters.

    Paragraph X What goes into these layers:filters_num_in parameter: How many bottleneck filters 
    should be used in the response block?Filters_num_out parameter: The number of filters in the 
    output should match the number of filters in x.id_block parameter: The block's identification
    number (for saving and loading weight) is:return: x with the additional res block
    """
    tensor_input = x
    x = Conv2D(filters_num_in, (1, 1), strides=(2, 2), name=id_block + '_1')(x)
    x = normalization_relu_add(x)
    x = Conv2D(filters_num_in, (3, 3), padding='same', name=id_block + '_2')(x)
    x = normalization_relu_add(x)
    x = Conv2D(filters_num_out, (1, 1), name=id_block + '_3')(x)
    shortcut = Conv2D(filters_num_out, (1, 1), strides=(2, 2), name=id_block + '_shortcut')(tensor_input)
    shortcut = BatchNormalization()(shortcut)
    x = Add()([x, shortcut])
    x = Activation('relu')(x)
    return x


def netaudio(input_shape, features_n):
    """creates and delivers an audio model

    :param features_n: Number of features to use at the network's end. :param input_shape: 
    Shape of the input audio.the model back (as a Keras model)
    """
    tensor_input = Input(input_shape)
    x = Conv1D(128, 5, padding='same', activation='relu')(tensor_input)
    x = MaxPooling1D(pool_size=8)(x)
    x = BatchNormalization()(x)

    x = Conv1D(128, 5, padding='same', activation='relu')(x)
    x = MaxPooling1D(pool_size=8)(x)
    x = BatchNormalization()(x)

    x = Conv1D(128, 5, padding='same', activation='relu')(x)
    x = MaxPooling1D(pool_size=8)(x)
    x = BatchNormalization()(x)

    x = Flatten()(x)
    x = Dense(features_n, activation='relu')(x)
    model = Model(tensor_input, x)
    return model


def time_distributed_netaudio(input_shape, features_n):
    """builds a time-distributed audio model and returns it.

    :param features_n: Number of features to use at the network's end. :param input_shape: 
    Shape of the input audio.the model back (as a Keras model)
    """
    return TimeDistributed(netaudio(input_shape, features_n))


def netface(input_image, network_id=""):
    """creates and gives back a residual face model.

    Shape of the input picture is specified by the :param input_image parameter. 
    The model is returned as a Keras model by the :param network_id parameter.
    """

    x = ZeroPadding2D(padding=(3, 3), name='%spre_conv_pad' % network_id)(input_image)
    x = Conv2D(64, (7, 7), strides=(2, 2), padding='valid', name='%sconv1' % network_id)(x)
    x = normalization_relu_add(x)
    x = ZeroPadding2D(padding=(1, 1), name='%spre_pool_pad' % network_id)(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), name='%spool1' % network_id)(x)

    x = convolutional_block_add(x, 64, 256, '%sres1_1' % network_id)
    x = residual_block_add(x, 64, 256, '%sres1_2' % network_id)

    x = convolutional_block_add(x, 256, 512, '%sres4_1' % network_id)
    x = residual_block_add(x, 256, 512, '%sres4_2' % network_id)
    x = GlobalAveragePooling2D()(x)

    model = Model(input_image, x, name='%s_resnet' % network_id)
    return model


def time_distributed_netface(input_image_s, network_id=""):
    """builds a time-distributed residual face model and returns it.

    Shape of the input picture is specified by the :param input_image parameter. 
    The model is returned as a Keras model by the :param network_id parameter.
    """
    return TimeDistributed(netface(Input(input_image_s), network_id))


def netlivestreaming(input_image, network_id=""):
    """builds a residual live_streaming model and returns it.

    Shape of the input picture is specified by the :param input_image parameter. 
    The model is returned as a Keras model by the :param network_id parameter.
    """
    x = ZeroPadding2D(padding=(3, 3), name='%spre_conv_pad' % network_id)(input_image)
    x = Conv2D(64, (7, 7), strides=(2, 2), padding='valid', name='%sconv1' % network_id)(x)
    x = normalization_relu_add(x)
    x = ZeroPadding2D(padding=(1, 1), name='%spre_pool_pad' % network_id)(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), name='%spool1' % network_id)(x)

    x = convolutional_block_add(x, 64, 128, '%sres1_1' % network_id)
    x = residual_block_add(x, 64, 128, '%sres1_2' % network_id)

    x = convolutional_block_add(x, 128, 256, '%sres2_1' % network_id)
    x = residual_block_add(x, 128, 256, '%sres2_2' % network_id)

    x = convolutional_block_add(x, 256, 512, '%sres4_1' % network_id)
    x = residual_block_add(x, 256, 512, '%sres4_2' % network_id)
    x = GlobalAveragePooling2D()(x)

    model = Model(input_image, x, name='%s_resnet' % network_id)
    return model


def time_distributed_netlivestreaming(input_image_s, network_id=""):
    """builds a time-distributed residual live-streaming model and returns it.

    Shape of the input picture is specified by the :param input_image parameter. 
    The model is returned as a Keras model by the :param network_id parameter.
    """
    return TimeDistributed(netlivestreaming(Input(input_image_s), network_id))
