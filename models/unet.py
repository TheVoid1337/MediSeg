from keras.layers import *
from keras.losses import CategoricalCrossentropy
from keras.models import Model
from keras.optimizers.optimizer_v2.adam import Adam


class UNet:
    def __init__(self,
                 input_shape: tuple,
                 metrics: list,
                 filters: list,
                 number_classes: int = 3,
                 batch_norm: bool = True,
                 summary: bool = True,
                 optimizer=Adam(10e-5),
                 dropout: float = 0.05,
                 loss=CategoricalCrossentropy()
                 ):
        self.input_shape = input_shape
        self.metrics = metrics
        self.filters = filters
        self.number_classes = number_classes
        self.batch_norm = batch_norm
        self.summary = summary
        self.optimizer = optimizer
        self.loss = loss
        self.dropout = dropout

    def conv_block(self, layer, filter_index):
        conv = Conv2D(self.filters[filter_index], (3, 3), kernel_initializer="he_normal",
                      padding="same")(layer)

        if self.batch_norm:
            conv = BatchNormalization()(conv)

        conv = Activation("relu")(conv)

        if self.dropout > 0:
            conv = Dropout(self.dropout)(conv)

        conv = Conv2D(self.filters[filter_index], (3, 3), kernel_initializer="he_normal",
                      padding="same")(conv)

        if self.batch_norm:
            conv = BatchNormalization()(conv)

        conv = Activation("relu")(conv)

        if self.dropout > 0:
            conv = Dropout(self.dropout)(conv)

        return conv

    def de_conv_block(self, layer, encoder, filter_index):
        de_conv = UpSampling2D(size=2, interpolation="bilinear")(layer)
        de_conv = concatenate([encoder, de_conv])
        de_conv = self.conv_block(de_conv, filter_index)
        return de_conv

    def create_model(self) -> Model:
        input_layer = Input(shape=self.input_shape)

        encoder1 = self.conv_block(input_layer, 0)
        pool1 = MaxPooling2D((2, 2))(encoder1)

        encoder2 = self.conv_block(pool1, 1)
        pool2 = MaxPooling2D((2, 2))(encoder2)

        encoder3 = self.conv_block(pool2, 2)
        pool3 = MaxPooling2D((2, 2))(encoder3)

        encoder4 = self.conv_block(pool3, 3)
        pool4 = MaxPooling2D((2, 2))(encoder4)

        bottle_nec = self.conv_block(pool4, 4)

        decoder1 = self.de_conv_block(bottle_nec, encoder4, 3)

        decoder2 = self.de_conv_block(decoder1, encoder3, 2)

        decoder3 = self.de_conv_block(decoder2, encoder2, 1)

        decoder4 = self.de_conv_block(decoder3, encoder1, 0)

        output_layer = Conv2D(self.number_classes, kernel_size=1, activation="softmax")(decoder4)

        model = Model(inputs=[input_layer], outputs=[output_layer])

        model.compile(self.optimizer, self.loss, self.metrics)

        if self.summary:
            model.summary()

        return model
