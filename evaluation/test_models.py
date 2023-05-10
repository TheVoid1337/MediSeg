import numpy as np
from keras.models import Model
from metrics.metrics import dice_coef
from keras.metrics.metrics import IoU
import pandas as pd


def calc_iou(y_true, y_pred):
    iou_liver = IoU(3, [1])
    iou_tumor = IoU(3, [2])
    if len(np.unique(y_true)) == 2:
        iou_liver.update_state(y_true, y_pred)
        return iou_liver.result().numpy()
    else:
        iou_liver.update_state(y_true, y_pred)
        iou_tumor.update_state(y_true, y_pred)
        return iou_liver.result().numpy(), iou_tumor.result().numpy()


def save_values(values: list[dict], filename: str):
    dataframe = pd.DataFrame(values,
                             columns=["unet_iou", "att_unet_iou", "lstm_unet_iou", "att_lstm_unet_iou", "unet_dice",
                                      "att_unet_dice",
                                      "lstm_unet_dice",
                                      "att_lstm_unet_dice"])
    dataframe.to_csv(f"results/test_data/{filename}")


def test_models(unet: Model, att_unet: Model, lstm_unet: Model, att_lstm_unet: Model, test_images, test_masks):
    liver_values = []
    tumor_values = []

    for i in range(len(test_images)):
        image = np.expand_dims(test_images[i], axis=0)
        mask = np.argmax(test_masks[i], axis=2)

        unet_pred = unet.predict(image,verbose=0)
        att_unet_pred = att_unet.predict(image,verbose=0)
        lstm_unet_pred = lstm_unet.predict(image,verbose=0)
        att_lstm_unet_pred = att_lstm_unet.predict(image,verbose=0)

        label = np.unique(mask)
        if len(label) == 2:
            temp = np.expand_dims(test_masks[i], axis=0)
            dice_unet_liver = dice_coef(temp[:, :, :, 1], unet_pred[:, :, :, 1]).numpy()
            dice_att_unet_liver = dice_coef(temp[:, :, :, 1], att_unet_pred[:, :, :, 1]).numpy()
            dice_lstm_unet_liver = dice_coef(temp[:, :, :, 1], lstm_unet_pred[:, :, :, 1]).numpy()
            dice_att_lstm_unet_liver = dice_coef(temp[:, :, :, 1], att_lstm_unet_pred[:, :, :, 1]).numpy()

            unet_pred = np.argmax(unet_pred, axis=3)[0, :, :]
            att_unet_pred = np.argmax(att_unet_pred, axis=3)[0, :, :]
            lstm_unet_pred = np.argmax(lstm_unet_pred, axis=3)[0, :, :]
            att_lstm_unet_pred = np.argmax(att_lstm_unet_pred, axis=3)[0, :, :]

            unet_liver = calc_iou(mask, unet_pred)
            att_unet_liver = calc_iou(mask, att_unet_pred)
            lstm_unet_liver = calc_iou(mask, lstm_unet_pred)
            att_lstm_unet_liver = calc_iou(mask, att_lstm_unet_pred)

            liver_values.append({"unet_iou": unet_liver, "att_unet_iou": att_unet_liver,
                                 "lstm_unet_iou": lstm_unet_liver,
                                 "att_lstm_unet_iou": att_lstm_unet_liver, "unet_dice": dice_unet_liver,
                                 "att_unet_dice": dice_att_unet_liver,
                                 "lstm_unet_dice": dice_lstm_unet_liver,
                                 "att_lstm_unet_dice": dice_att_lstm_unet_liver})
        else:
            temp = np.expand_dims(test_masks[i], axis=0)
            dice_unet_liver = dice_coef(temp[:, :, :, 1], unet_pred[:, :, :, 1]).numpy()
            dice_att_unet_liver = dice_coef(temp[:, :, :, 1], att_unet_pred[:, :, :, 1]).numpy()
            dice_lstm_unet_liver = dice_coef(temp[:, :, :, 1], lstm_unet_pred[:, :, :, 1]).numpy()
            dice_att_lstm_unet_liver = dice_coef(temp[:, :, :, 1], att_lstm_unet_pred[:, :, :, 1]).numpy()

            dice_unet_tumor = dice_coef(temp[:, :, :, 2], unet_pred[:, :, :, 2]).numpy()
            dice_att_unet_tumor = dice_coef(temp[:, :, :, 2], att_unet_pred[:, :, :, 2]).numpy()
            dice_lstm_unet_tumor = dice_coef(temp[:, :, :, 2], lstm_unet_pred[:, :, :, 2]).numpy()
            dice_att_lstm_unet_tumor = dice_coef(temp[:, :, :, 2], att_lstm_unet_pred[:, :, :, 2]).numpy()

            unet_pred = np.argmax(unet_pred, axis=3)[0, :, :]
            att_unet_pred = np.argmax(att_unet_pred, axis=3)[0, :, :]
            lstm_unet_pred = np.argmax(lstm_unet_pred, axis=3)[0, :, :]
            att_lstm_unet_pred = np.argmax(att_lstm_unet_pred, axis=3)[0, :, :]

            unet_liver = calc_iou(mask, unet_pred)
            att_unet_liver = calc_iou(mask, att_unet_pred)
            lstm_unet_liver = calc_iou(mask, lstm_unet_pred)
            att_lstm_unet_liver = calc_iou(mask, att_lstm_unet_pred)

            liver_values.append({"unet_iou": unet_liver[0], "att_unet_iou": att_unet_liver[0],
                                 "lstm_unet_iou": lstm_unet_liver[0],
                                 "att_lstm_unet_iou": att_lstm_unet_liver[0], "unet_dice": dice_unet_liver,
                                 "att_unet_dice": dice_att_unet_liver,
                                 "lstm_unet_dice": dice_lstm_unet_liver,
                                 "att_lstm_unet_dice": dice_att_lstm_unet_liver})

            tumor_values.append({"unet_iou": unet_liver[1], "att_unet_iou": att_unet_liver[1],
                                 "lstm_unet_iou": lstm_unet_liver[1],
                                 "att_lstm_unet_iou": att_lstm_unet_liver[1], "unet_dice": dice_unet_tumor,
                                 "att_unet_dice": dice_att_unet_tumor,
                                 "lstm_unet_dice": dice_lstm_unet_tumor,
                                 "att_lstm_unet_dice": dice_att_lstm_unet_tumor})

    save_values(liver_values, "liver_test_results.csv")
    save_values(tumor_values, "tumor_test_results.csv")
