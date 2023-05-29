import keras.backend as K


def dice_coef(y_true, y_pred, smooth=1):
    """
    Implementation of the dice coefficient for binary segmentation only. For more information see:
    https://arxiv.org/pdf/1606.04797v1.pdf and https://karan-jakhar.medium.com/100-days-of-code-day-7-84e4918cb72c
    :param y_true: true positives.
    :param y_pred: true predicted.
    :param smooth: smoothing factor.
    :return: dice coefficient.
    """
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coefficient(y_true, y_pred, num_labels=3):
    """
    Multiclass implementation for the dice coefficient.
    :param y_true: true positives.
    :param y_pred: true predicted.
    :param num_labels: number of classes.
    :return: mean dice coefficient for all classes.
    """
    dice = 0
    for index in range(num_labels):
        dice += dice_coef(y_true[:, :, :, index], y_pred[:, :, :, index])
    return dice / num_labels

