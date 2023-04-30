from keras.callbacks import CSVLogger
from keras.models import Model
from pathlib import Path
import os


def train_model(train_images, train_masks, model: Model, **kwargs):
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
