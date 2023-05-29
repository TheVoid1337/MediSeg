from keras.callbacks import CSVLogger
from keras.models import Model
from pathlib import Path
import os
from models.attention_lstm_unet import AttentionLSTMUnet
from models.lstm_unet import LSTMUnet
from metrics.metrics import dice_coefficient
from keras.metrics import OneHotIoU
from models.unet import UNet
from models.attention_unet import AttentionUnet

input_shape = (128, 128, 1)
target_shape = (input_shape[0], input_shape[1])
metrics = [OneHotIoU(3, [0, 1, 2], "IoU"), dice_coefficient]
filters = [16, 32, 64, 128, 256]


def train_model(train_images, train_masks, model: Model, **kwargs):
    """
    Template for model training. Uses kwargs to set the best parameters for model training.
    :param train_images: train images as numpy array
    :param train_masks: train masks as numpy array
    :param model: keras model
    :param kwargs: keyword args to set train params.
    :return: trained model as h5 file.
    """
    model_history = model.fit(
        train_images,
        train_masks,
        epochs=kwargs.get("epochs", 100),
        verbose=kwargs.get("verbose", True),
        callbacks=kwargs.get("callbacks", None),
        validation_split=kwargs.get("val_split", 0.2),  # 0.1 = 3d, 0.2 = 2d
        shuffle=kwargs.get("shuffle", True),
        batch_size=kwargs.get("batch_size", 16)  # 16 = 2d, 2 = 3d
    )

    weight_file = Path(kwargs.get("weight_filename"))
    if weight_file.is_file():
        os.remove(kwargs.get("weight_filename"))

    model.save(kwargs.get("weight_filename"))
    return model_history


def train_unet(cnn, net_model, train_images, train_masks, **kwargs):
    training_callbacks = kwargs.get("callbacks", [])
    training_callbacks = training_callbacks + [CSVLogger(f"nets/logs/{cnn.__class__.__name__}.log")]
    train_args = {"callbacks": training_callbacks,
                  "weight_filename": f"nets/weights/{cnn.__class__.__name__}.h5",
                  "batch_size": kwargs.get("batch_size", 16),
                  "val_split": kwargs.get("val_split", 0.2)
                  }
    train_model(train_images, train_masks, net_model, **train_args)


def create_models(train_images, train_masks, summary: bool = False, train: bool = False):
    """
    Creates all models and returns them. If train is true the models will be trained and returned. If summary is true
    the models will print a summary for each model.
    :param train_images: train images as numpy array
    :param train_masks: train masks as numpy array
    :param summary: if true summary is printed.
    :param train: if true models are trained.
    :return: model instances.
    """
    unet = UNet(input_shape, metrics, filters, summary=summary)
    unet_model = unet.create_model()

    att_unet = AttentionUnet(input_shape, metrics, filters, summary=summary)
    att_unet_model = att_unet.create_model()

    lstm_unet = LSTMUnet(input_shape, metrics, filters, summary=summary)
    lstm_unet_model = lstm_unet.create_model()

    att_lstm_unet = AttentionLSTMUnet(input_shape, metrics, filters, summary=summary)
    att_lstm_unet_model = att_lstm_unet.create_model()

    if train:
        train_unet(unet, unet_model, train_images, train_masks)
        train_unet(att_unet, att_unet_model, train_images, train_masks)
        train_unet(lstm_unet, lstm_unet_model, train_images, train_masks)
        train_unet(att_lstm_unet, att_lstm_unet_model, train_images, train_masks)
    else:
        unet_model.load_weights("nets/weights/UNet.h5")
        att_unet_model.load_weights("nets/weights/AttentionUnet.h5")
        lstm_unet_model.load_weights("nets/weights/LSTMUnet.h5")
        att_lstm_unet_model.load_weights("nets/weights/AttentionLSTMUnet.h5")

    return unet_model,att_unet_model,lstm_unet_model,att_lstm_unet_model
