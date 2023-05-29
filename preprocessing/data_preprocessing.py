import numpy as np
from keras.utils.np_utils import to_categorical, normalize
from sklearn.model_selection import train_test_split


def prepare(image_data, mask_data, num_classes, test_size=0.2):
    """
    This function splits the dataset into a train and test dataset with default size of 80/20.
    Additionally, is function normalizes the image data and One-Hot encoding the mask data for training.
    :param image_data: image data
    :param mask_data: mask data
    :param num_classes: number of classes (segmentations in one image).
    :param test_size: size of the test dataset.
    :return: train_images, train_masks, test_images, test_masks, test_images_to_plot (for plots only)
    """
    train_images, test_images, train_masks, test_masks = train_test_split(image_data, mask_data, test_size=test_size,
                                                                          shuffle=False)

    train_images = expand_dimensions(train_images)
    train_images = normalize(train_images)

    train_masks = expand_dimensions(train_masks)
    train_masks = categorize(train_masks, num_classes)

    test_images_to_plot = test_images
    test_images = expand_dimensions(test_images)
    test_images = normalize(test_images)

    test_masks = expand_dimensions(test_masks)
    test_masks = categorize(test_masks, num_classes)

    return train_images, train_masks, test_images, test_masks, test_images_to_plot


def expand_dimensions(data, axis=-1):
    return np.expand_dims(data, axis=axis)


def categorize(masks, num_classes):
    return to_categorical(masks, num_classes)
