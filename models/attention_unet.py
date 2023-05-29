from keras.layers import *
from keras.losses import CategoricalCrossentropy
from keras.models import Model
from keras.optimizers.optimizer_v2.adam import Adam
from models.unet import UNet


class AttentionUnet(UNet):
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
        super().__init__(input_shape,
                         metrics,
                         filters,
                         number_classes,
                         batch_norm,
                         summary,
                         optimizer,
                         dropout,
                         loss)
    def gating_signal(self, layer, filter_index):
        """
        Gating signal implementation for the attention gate.
        :param layer: output of last decoder.
        :param filter_index: index of the filters array for the convolutional layer.
        :return: rescaled gating signal used in the attention gate.
        """
        gating_signal = Conv2D(self.filters[filter_index], kernel_size=1, padding="same")(layer)
        return gating_signal

    # attention gate
    def attention_gate(self, feature_layer, gating_signal, filter_index):
        """
        Recalculates weighted feature maps during training.
        :param feature_layer: Encoder output of the same level.
        :param gating_signal: gating output for the next lower level of the U-Net.
        :param filter_index: index of the filters array for the convolutional layers.
        :return: weighted feature maps by multiplying the attention scores with the feature maps.
        """
        x_l = Conv2D(self.filters[filter_index], kernel_size=1, strides=2, padding="same")(feature_layer)
        addition = add([x_l, gating_signal])

        psi = Activation("relu")(addition)

        psi_weights = Conv2D(1, 1, padding="same")(psi)

        psi_weights_scaled = Activation("sigmoid")(psi_weights)

        alpha_attention = UpSampling2D(size=2, interpolation="bilinear")(psi_weights_scaled)

        attention = multiply([alpha_attention, feature_layer])

        return attention

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

        gating_signal_1 = self.gating_signal(bottle_nec, 3)
        attention_1 = self.attention_gate(encoder4, gating_signal_1, 3)
        decoder_1 = self.de_conv_block(bottle_nec, attention_1, 3)

        gating_signal_2 = self.gating_signal(decoder_1, 2)
        attention_2 = self.attention_gate(encoder3, gating_signal_2, 2)
        decoder_2 = self.de_conv_block(decoder_1, attention_2, 2)

        gating_signal_3 = self.gating_signal(decoder_2, 1)
        attention_3 = self.attention_gate(encoder2, gating_signal_3, 1)
        decoder_3 = self.de_conv_block(decoder_2, attention_3, 1)

        gating_signal_4 = self.gating_signal(decoder_3, 0)
        attention_4 = self.attention_gate(encoder1, gating_signal_4, 0)
        decoder_4 = self.de_conv_block(decoder_3, attention_4, 0)

        output_layer = Conv2D(self.number_classes, kernel_size=1, activation="softmax")(decoder_4)

        model = Model(inputs=[input_layer], outputs=[output_layer])

        model.compile(self.optimizer, self.loss, self.metrics)

        if self.summary:
            model.summary()

        return model
