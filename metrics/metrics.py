import keras.backend as K


def dice_coef(y_true, y_pred, smooth=1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coefficient(y_true, y_pred, num_labels=3):
    dice = 0
    for index in range(num_labels):
        dice += dice_coef(y_true[:, :, :, index], y_pred[:, :, :, index])
    return dice / num_labels

