#     <MSZNET by MNHUT0550. Github: github.com/mnhut0550>
#     Copyright (C) 2023  MNHUT0550

#     This program is free software: you can redistribute it and/or modify
#     it under the terms of the GNU General Public License as published by
#     the Free Software Foundation, either version 3 of the License, or
#     (at your option) any later version.

#     This program is distributed in the hope that it will be useful,
#     but WITHOUT ANY WARRANTY; without even the implied warranty of
#     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#     GNU General Public License for more details.

#     You should have received a copy of the GNU General Public License
#     along with this program.  If not, see <https://www.gnu.org/licenses/>.


#**NOTE:
#If you use this file, please make sure the Tensorflow version is higher or equal 2.7
##END NOTE



#-----------Import library
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input, Convolution2D, MaxPooling2D, Activation, concatenate,ReLU,Dropout, DepthwiseConv2D, GlobalAveragePooling2D, GlobalMaxPooling2D, BatchNormalization
from tensorflow.keras.models import Model
from keras import backend
from keras.utils import layer_utils
from keras.applications import imagenet_utils


#--------------

def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

#--------------

def fire_module(x, fire_id, squeeze=16, expand=64, alpha=1.0):
    s_id = "fire" + str(fire_id) + "/"

    if K.image_data_format() == 'channels_first':
        channel_axis = 1
    else:
        channel_axis = 3

    squeeze_filters = _make_divisible(squeeze * alpha, 8)
    expand_filters = _make_divisible(expand * alpha, 8)
    
    x = Convolution2D(squeeze_filters, (1, 1), padding='same', name=s_id + "squeeze_1x1")(x)
    x = BatchNormalization()(x)
    x = ReLU(6. , name=s_id + "relu_squeeze_1x1")(x)

    left = Convolution2D(expand_filters, (1, 1), padding='same', name=s_id + "expand_1x1")(x)
    left = ReLU(6. , name=s_id + "relu_expand_1x1")(left)

    right = DepthwiseConv2D((3,3), padding='same', name=s_id + "expand_3x3_dw")(x)
    right = ReLU(6. , name=s_id + "relu_expand_3x3_dw")(right)

    right = Convolution2D(expand_filters, (1,1), padding='same', name=s_id + "expand_3x3_conv")(right)
    right = ReLU(6. , name=s_id + "relu_expand_3x3_conv")(right)

    x = concatenate([left, right], axis=channel_axis, name=s_id + "concat")
    return x

#---------preprocessing_input

def preprocessing_input(image):
    image -= 127.5
    image /= 127.5

    return image


#----------model

def MSZNET(input_shape=None, alpha=1.0, include_top=True, input_tensor=None, pooling=None, classes=1000):
        
    assert alpha in [.25, .5, .75, 1.], "alpha must be 0.25, 0.5, 0.75 or 1.0 but found alpha = {}".format(alpha)
    
    if input_shape is not None and input_tensor is not None:
        try:
            is_input_t_tensor = backend.is_keras_tensor(input_tensor)
        except ValueError:
            try:
                is_input_t_tensor = backend.is_keras_tensor(layer_utils.get_source_inputs(input_tensor))
            except ValueError:
                raise ValueError(
                    f"input_tensor: {input_tensor}"
                    "is not type input_tensor. "
                    f"Received `type(input_tensor)={type(input_tensor)}`"
                )
        if is_input_t_tensor:
            if backend.image_data_format() == "channels_first":
                if backend.int_shape(input_tensor)[1] != input_shape[1]:
                    raise ValueError(
                        "input_shape[1] must equal shape(input_tensor)[1] "
                        "when `image_data_format` is `channels_first`; "
                        "Received `input_tensor.shape="
                        f"{input_tensor.shape}`"
                        f", `input_shape={input_shape}`"
                    )
            else:
                if backend.int_shape(input_tensor)[2] != input_shape[1]:
                    raise ValueError(
                        "input_tensor.shape[2] must equal input_shape[1]; "
                        "Received `input_tensor.shape="
                        f"{input_tensor.shape}`, "
                        f"`input_shape={input_shape}`"
                    )
        else:
            raise ValueError(
                "input_tensor is not a Keras tensor; "
                f"Received `input_tensor={input_tensor}`"
            )

    # If input_shape is None, infer shape from input_tensor.
    if input_shape is None and input_tensor is not None:

        try:
            backend.is_keras_tensor(input_tensor)
        except ValueError:
            raise ValueError(
                "input_tensor must be a valid Keras tensor type; "
                f"Received {input_tensor} of type {type(input_tensor)}"
            )

        if input_shape is None and not backend.is_keras_tensor(input_tensor):
            default_size = 224
        elif input_shape is None and backend.is_keras_tensor(input_tensor):
            if backend.image_data_format() == "channels_first":
                rows = backend.int_shape(input_tensor)[2]
                cols = backend.int_shape(input_tensor)[3]
            else:
                rows = backend.int_shape(input_tensor)[1]
                cols = backend.int_shape(input_tensor)[2]

            if rows == cols and rows in [96, 128, 160, 192, 224]:
                default_size = rows
            else:
                default_size = 224

    # If input_shape is None and no input_tensor
    elif input_shape is None:
        default_size = 224

    # If input_shape is not None, assume default size.
    else:
        if backend.image_data_format() == "channels_first":
            rows = input_shape[1]
            cols = input_shape[2]
        else:
            rows = input_shape[0]
            cols = input_shape[1]

        if rows == cols and rows in [96, 128, 160, 192, 224]:
            default_size = rows
        else:
            default_size = 224

    input_shape = imagenet_utils.obtain_input_shape(
        input_shape,
        default_size=default_size,
        min_size=32,
        data_format=backend.image_data_format(),
        require_flatten=include_top,
    )

    if backend.image_data_format() == "channels_last":
        row_axis, col_axis = (0, 1)
    else:
        row_axis, col_axis = (1, 2)
    rows = input_shape[row_axis]
    cols = input_shape[col_axis]

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not backend.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor




    #--------- model definition

    first_block_filters = _make_divisible(64 * alpha, 8)

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor


    x = Convolution2D(first_block_filters, (3, 3), strides=(2, 2), padding='same', name='conv_1')(img_input)
    x = ReLU(6. , name='relu_conv_1')(x)
    x = MaxPooling2D(name='pool_1', padding='same')(x)

    x = fire_module(x, fire_id=2, squeeze=16, expand=64, alpha=alpha)
    x = fire_module(x, fire_id=3, squeeze=16, expand=64, alpha=alpha)
    x = MaxPooling2D(name='pool_3', padding='same')(x)

    x = fire_module(x, fire_id=4, squeeze=32, expand=128, alpha=alpha)
    x = fire_module(x, fire_id=5, squeeze=32, expand=128, alpha=alpha)
    x = MaxPooling2D(name='pool_5', padding='same')(x)

    x = fire_module(x, fire_id=6, squeeze=48, expand=192, alpha=alpha)
    x = fire_module(x, fire_id=7, squeeze=48, expand=192, alpha=alpha)
    x = MaxPooling2D(name='pool_7', padding='same')(x)

    x = fire_module(x, fire_id=8, squeeze=64, expand=256, alpha=alpha)
    x = fire_module(x, fire_id=9, squeeze=64, expand=256, alpha=alpha)

    if include_top:
        x = Dropout(0.3, name='dropout_10')(x)
        x = Convolution2D(classes, (1, 1), padding='valid', name='conv_10')(x)
        x = ReLU(6. , name='relu_conv_10')(x)
        x = GlobalAveragePooling2D()(x)
        x = Activation('softmax', name='loss')(x)
    else:
        if pooling == 'avg':
            x = GlobalAveragePooling2D()(x)
        elif pooling=='max':
            x = GlobalMaxPooling2D()(x)
        elif pooling==None:
            pass
        else:
            raise ValueError("Unknown argument for 'pooling'=" + pooling)

    return Model(img_input, x, name='MSZNET_{}'.format(alpha))
