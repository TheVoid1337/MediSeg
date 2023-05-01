import numpy as np
from keras.layers import *
from keras.losses import CategoricalCrossentropy
from keras.models import Model
from keras.optimizers.optimizer_v2.rmsprop import RMSprop

from models.unet import UNet


class LSTMUnet(UNet):
    def __init__(self, input_shape: tuple, metrics: list,
                 filters: list, number_classes: int = 3,
                 batch_norm: bool = True,
                 summary: bool = True,
                 optimizer=RMSprop(10e-5, momentum=0.9),
                 dropout: float = 0.05,
                 loss=CategoricalCrossentropy()):
        super().__init__(input_shape,
                         metrics,
                         filters,
                         number_classes,
                         batch_norm,
                         summary,
                         optimizer,
                         dropout,
                         loss)

    def conv_block_lstm(self, layer, encoder, filter_index, input_shape):
        de_conv = Conv2DTranspose(self.filters[filter_index], (3, 3), (2, 2), padding="same",
                                  kernel_initializer="he_normal")(layer)
        de_conv = BatchNormalization()(de_conv)
        de_conv = Activation("relu")(de_conv)
        if self.dropout > 0:
            de_conv = Dropout(self.dropout)(de_conv)

        x1 = Reshape(target_shape=(1, np.int32(input_shape), np.int32(input_shape),
                                   self.filters[filter_index]))(encoder)
        x2 = Reshape(target_shape=(1, np.int32(input_shape), np.int32(input_shape),
                                   self.filters[filter_index]))(de_conv)
        merge = concatenate([x1, x2], axis=1)
        merge = ConvLSTM2D(self.filters[filter_index] // 2, (3, 3), padding="same", return_sequences=False,
                           go_backwards=True, kernel_initializer="he_normal")(merge)

        de_conv = Conv2D(self.filters[filter_index], 3, padding='same',
                         kernel_initializer='he_normal')(merge)
        de_conv = BatchNormalization()(de_conv)

        de_conv = Activation("relu")(de_conv)

        if self.dropout > 0:
            de_conv = Dropout(self.dropout)(de_conv)

        de_conv = Conv2D(self.filters[filter_index], 3, padding='same',
                         kernel_initializer='he_normal')(de_conv)

        de_conv = BatchNormalization()(de_conv)
        de_conv = Activation("relu")(de_conv)

        if self.dropout > 0:
            de_conv = Dropout(self.dropout)(de_conv)

        return de_conv

    def create_model(self) -> Model:
        input_layer = Input(shape=self.input_shape)

        conv1 = self.conv_block(input_layer, 0)
        pool1 = MaxPooling2D((2, 2))(conv1)

        conv2 = self.conv_block(pool1, 1)
        pool2 = MaxPooling2D((2, 2))(conv2)

        conv3 = self.conv_block(pool2, 2)
        pool3 = MaxPooling2D((2, 2))(conv3)

        conv4 = self.conv_block(pool3, 3)
        pool4 = MaxPooling2D((2, 2))(conv4)

        # Bridge
        bridge = self.conv_block(pool4, 4)

        de_conv_lstm1 = self.conv_block_lstm(bridge, conv4, 3, self.input_shape[0] // 8)

        de_conv_lstm2 = self.conv_block_lstm(de_conv_lstm1, conv3, 2, self.input_shape[0] // 4)

        de_conv_lstm3 = self.conv_block_lstm(de_conv_lstm2, conv2, 1, self.input_shape[0] // 2)

        de_conv_lstm4 = self.conv_block_lstm(de_conv_lstm3, conv1, 0, self.input_shape[0])

        output_layer = Conv2D(self.number_classes, 1, activation="softmax")(de_conv_lstm4)

        unet_model = Model(inputs=[input_layer], outputs=[output_layer])

        unet_model.compile(self.optimizer, self.loss, self.metrics)

        if self.summary:
            unet_model.summary()

        return unet_model
